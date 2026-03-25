from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Callable

from autoresearch_rl.checkpoint import LoopCheckpoint, load_checkpoint, save_checkpoint
from autoresearch_rl.config import (
    ComparabilityConfig,
    ControllerConfig,
    ObjectiveConfig,
    TelemetryConfig,
)
from autoresearch_rl.controller.executor import Evaluator, Executor, Outcome
from autoresearch_rl.controller.helpers import (
    check_failure_rate,
    check_no_improve,
    current_commit,
)
from autoresearch_rl.controller.shutdown import ShutdownHandler
from autoresearch_rl.controller.types import LoopResult
from autoresearch_rl.forecasting import should_early_stop
from autoresearch_rl.policy.interface import Learnable, Proposal
from autoresearch_rl.promotion import PromotionTracker
from autoresearch_rl.telemetry.aggregation import compute_episode_stats
from autoresearch_rl.telemetry.comparability import (
    ComparabilityPolicy,
    check_comparability,
    hardware_fingerprint,
)
from autoresearch_rl.telemetry.events import emit
from autoresearch_rl.telemetry.ledger import append_result_row, ensure_results_tsv
from autoresearch_rl.telemetry.manifest import new_run_id, write_manifest
from autoresearch_rl.telemetry.run import write_run_manifest
from autoresearch_rl.tracking import LocalFileTracker

logger = logging.getLogger(__name__)

# Callback type for iteration hooks (e.g. distillation)
IterationCallback = Callable[
    [int, Proposal, Outcome, float | None, str],  # iter, proposal, outcome, score, decision
    None,
]


def _objective_value(
    metrics: dict[str, float], objective: ObjectiveConfig
) -> float | None:
    if objective.metric not in metrics:
        return None
    return float(metrics[objective.metric])


def _score(value: float, objective: ObjectiveConfig) -> float:
    return value if objective.direction == "min" else -value


def _save_version(
    versions_dir: str, iter_idx: int, outcome: Outcome, params: dict,
    model_dir: str | None = None,
) -> str:
    target_dir = Path(versions_dir) / f"v{iter_idx:04d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "iter": iter_idx,
        "metrics": outcome.metrics,
        "params": params,
        "status": outcome.status,
        "run_dir": outcome.run_dir,
    }
    if model_dir:
        meta["model_dir"] = model_dir
    (target_dir / "version.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return str(target_dir)


def run_experiment(
    *,
    executor: Executor,
    evaluator: Evaluator,
    policy,
    objective: ObjectiveConfig,
    controller: ControllerConfig,
    telemetry: TelemetryConfig,
    comparability_cfg: ComparabilityConfig,
    proposal_state_builder: Callable[[list[dict], str], dict],
    proposal_params_extractor: Callable[[Proposal], dict],
    program: str = "",
    description_label: str = "experiment",
    on_iteration: IterationCallback | None = None,
    enable_shutdown_handler: bool = True,
    enable_versions: bool = True,
    enable_tracker: bool = True,
    enable_forecasting: bool = True,
    enable_run_manifest: bool = True,
    manifest_config: dict | None = None,
) -> LoopResult:
    shutdown = ShutdownHandler()
    if enable_shutdown_handler:
        shutdown.register()

    ensure_results_tsv(telemetry.ledger_path)
    Path(telemetry.artifacts_dir).mkdir(parents=True, exist_ok=True)
    if enable_versions:
        Path(telemetry.versions_dir).mkdir(parents=True, exist_ok=True)
    Path(telemetry.trace_path).parent.mkdir(parents=True, exist_ok=True)

    episode_id = new_run_id()
    history: list[dict] = []
    best_score = float("inf")
    best_value: float | None = None
    no_improve_streak = 0
    recent_statuses: list[str] = []
    iter_idx = 0
    iterations_done = 0
    start_ts = time.monotonic()
    elapsed_offset = 0.0

    # Checkpoint restore
    if controller.checkpoint_path:
        ckpt = load_checkpoint(controller.checkpoint_path)
        if ckpt is not None:
            episode_id = ckpt.episode_id
            iter_idx = ckpt.iteration + 1
            iterations_done = ckpt.iteration + 1
            best_score = ckpt.best_score
            best_value = ckpt.best_value
            no_improve_streak = ckpt.no_improve_streak
            history = ckpt.history
            recent_statuses = ckpt.recent_statuses
            elapsed_offset = ckpt.elapsed_s

    # Seeding
    if controller.seed is not None:
        random.seed(controller.seed)
        os.environ["PYTHONHASHSEED"] = str(controller.seed)
        os.environ["AR_SEED"] = str(controller.seed)

    # Run manifest
    if enable_run_manifest:
        run_manifest_path = str(
            Path(telemetry.artifacts_dir) / "run-manifest.json"
        )
        write_run_manifest(
            run_manifest_path,
            config=manifest_config or {},
            run_id=episode_id,
        )

    # Tracker
    tracker = LocalFileTracker(telemetry.artifacts_dir, episode_id) if enable_tracker else None
    if tracker:
        tracker.set_status("running")

    score_history: list[float] = []
    promotion = PromotionTracker()

    # Comparability
    comp_policy = ComparabilityPolicy(
        budget_mode=comparability_cfg.budget_mode,
        expected_budget_s=comparability_cfg.expected_budget_s,
        expected_hardware_fingerprint=comparability_cfg.expected_hardware_fingerprint,
        strict=comparability_cfg.strict,
    )
    hw_fp = hardware_fingerprint()
    run_budget_s = controller.max_wall_time_s or comparability_cfg.expected_budget_s
    comparable, non_comparable_reason = check_comparability(
        comp_policy, run_budget_s, hw_fp
    )
    if comp_policy.strict and not comparable:
        raise ValueError(f"Non-comparable run blocked: {non_comparable_reason}")

    try:
        while True:
            if shutdown.requested:
                logger.info("Graceful shutdown: finishing loop")
                break

            elapsed = (time.monotonic() - start_ts) + elapsed_offset
            if controller.max_wall_time_s is not None and elapsed >= controller.max_wall_time_s:
                break

            if controller.max_iterations is not None and iterations_done >= controller.max_iterations:
                break

            state = proposal_state_builder(history, program)
            proposal = policy.propose(state)
            params = proposal_params_extractor(proposal)

            run_dir = str(Path(telemetry.artifacts_dir) / f"run-{iter_idx:04d}")

            # Inject model output directory if configured
            model_dir: str | None = None
            if telemetry.model_output_dir:
                model_dir = str(
                    Path(telemetry.model_output_dir) / f"v{iter_idx:04d}"
                )
                params["AR_MODEL_DIR"] = model_dir

            emit(
                telemetry.trace_path,
                {
                    "type": "proposal",
                    "episode_id": episode_id,
                    "iter": iter_idx,
                    "params": params,
                },
                run_id=episode_id,
                max_file_size_bytes=telemetry.max_file_size_bytes,
                max_rotated_files=telemetry.max_rotated_files,
            )

            outcome = executor.execute(proposal, run_dir)

            # Pick up model dir from target (Basilica downloads it)
            # or from the injected AR_MODEL_DIR
            downloaded_model = outcome.metrics.pop("_model_dir", None)
            effective_model_dir = str(downloaded_model) if downloaded_model else model_dir

            value = _objective_value(outcome.metrics, objective)
            status = outcome.status
            if value is None:
                status = "failed"

            # Evaluate score
            decision = "discard"
            improved = False
            if value is not None:
                score = _score(value, objective)
                if score < best_score:
                    best_score = score
                    best_value = value
                    decision = "keep"
                    improved = True
                    no_improve_streak = 0
                    if enable_versions:
                        _save_version(
                            telemetry.versions_dir, iter_idx, outcome, params,
                            model_dir=effective_model_dir,
                        )
                else:
                    no_improve_streak += 1
            else:
                no_improve_streak += 1

            promotion.record_result(
                float(value) if value is not None else float("inf"),
                improved=improved,
            )

            # Learnable policy feedback
            if isinstance(policy, Learnable):
                reward = 1.0 if decision == "keep" else (
                    -0.1 if status == "failed" else 0.0
                )
                policy.record_reward(reward)

            # Tracker metrics
            if tracker:
                tracker.log_metrics(
                    {
                        objective.metric: float(value) if value is not None else 0.0,
                        "elapsed_s": outcome.elapsed_s,
                    },
                    step=iter_idx,
                )
            if value is not None:
                score_history.append(float(value))

            # Telemetry: iteration event
            emit(
                telemetry.trace_path,
                {
                    "type": "iteration",
                    "episode_id": episode_id,
                    "iter": iter_idx,
                    "status": status,
                    "decision": decision,
                    "metrics": outcome.metrics,
                    "params": params,
                    "elapsed_s": outcome.elapsed_s,
                },
                run_id=episode_id,
                max_file_size_bytes=telemetry.max_file_size_bytes,
                max_rotated_files=telemetry.max_rotated_files,
            )

            # Telemetry: manifest
            write_manifest(
                telemetry.artifacts_dir,
                {
                    "episode_id": episode_id,
                    "iter": iter_idx,
                    "status": status,
                    "decision": decision,
                    "metrics": outcome.metrics,
                    "params": params,
                    "stdout": outcome.stdout,
                    "stderr": outcome.stderr,
                    "run_dir": outcome.run_dir,
                },
            )

            # Telemetry: ledger
            append_result_row(
                path=telemetry.ledger_path,
                commit=current_commit(),
                metric_name=objective.metric,
                metric_value=float(value if value is not None else 0.0),
                memory_gb=0.0,
                status=decision,
                description=description_label,
                episode_id=str(episode_id),
                iter_idx=int(iter_idx),
                score=float(
                    best_score if best_score < float("inf") else 0.0
                ),
                budget_mode=comp_policy.budget_mode,
                budget_s=run_budget_s,
                hardware_fingerprint=hw_fp,
                comparable=comparable,
                non_comparable_reason=non_comparable_reason,
                max_file_size_bytes=telemetry.max_file_size_bytes,
                max_rotated_files=telemetry.max_rotated_files,
            )

            # History
            history.append(
                {
                    "iter": iter_idx,
                    "status": status,
                    "decision": decision,
                    "metrics": outcome.metrics,
                    "params": params,
                    "stdout_tail": outcome.stdout[-500:] if outcome.stdout else "",
                    "stderr_tail": outcome.stderr[-300:] if outcome.stderr else "",
                }
            )

            # Iteration callback (distillation, etc.)
            if on_iteration is not None:
                on_iteration(iter_idx, proposal, outcome, value, decision)

            # Status tracking for failure rate
            recent_statuses.append(status)
            if len(recent_statuses) > max(1, controller.failure_window):
                recent_statuses.pop(0)

            iterations_done += 1
            iter_idx += 1

            # Stop guards
            if check_no_improve(no_improve_streak, controller.no_improve_limit):
                break

            if check_failure_rate(
                recent_statuses,
                controller.failure_rate_limit,
                controller.failure_window,
            ):
                break

            if (
                enable_forecasting
                and best_value is not None
                and len(score_history) >= 5
                and should_early_stop(score_history, float(best_value))
            ):
                logger.info("Early stop: forecast does not beat best")
                break

    finally:
        if score_history:
            stats = compute_episode_stats(score_history)
            emit(
                telemetry.trace_path,
                {
                    "type": "episode_summary",
                    "episode_id": episode_id,
                    "mean": stats.mean,
                    "median": stats.median,
                    "min": stats.min,
                    "max": stats.max,
                    "stdev": stats.stdev,
                    "count": stats.count,
                    "trend_slope": stats.trend_slope,
                },
                run_id=episode_id,
            )
        if tracker:
            tracker.set_status("completed")

        if controller.checkpoint_path and iterations_done > 0:
            save_checkpoint(
                controller.checkpoint_path,
                LoopCheckpoint(
                    episode_id=episode_id,
                    iteration=iter_idx - 1,
                    best_score=best_score,
                    best_value=best_value,
                    no_improve_streak=no_improve_streak,
                    history=history,
                    recent_statuses=recent_statuses,
                    policy_state={},
                    elapsed_s=(time.monotonic() - start_ts) + elapsed_offset,
                    timestamp=time.time(),
                ),
            )

    return LoopResult(
        best_score=best_score, best_value=best_value, iterations=iterations_done
    )
