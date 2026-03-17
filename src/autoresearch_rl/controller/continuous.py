from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path

from autoresearch_rl.checkpoint import LoopCheckpoint, load_checkpoint, save_checkpoint
from autoresearch_rl.config import ComparabilityConfig, ControllerConfig, ObjectiveConfig, TelemetryConfig
from autoresearch_rl.controller.helpers import (
    check_failure_rate,
    check_no_improve,
    current_commit,
)
from autoresearch_rl.controller.shutdown import ShutdownHandler
from autoresearch_rl.controller.types import LoopResult
from autoresearch_rl.forecasting import should_early_stop
from autoresearch_rl.policy.interface import Learnable, ParamProposal
from autoresearch_rl.policy.learned_search import LearnedParamPolicy, LearnedSearchConfig
from autoresearch_rl.policy.llm_search import LLMParamPolicy
from autoresearch_rl.policy.search import GridPolicy, RandomPolicy, StaticPolicy
from autoresearch_rl.promotion import PromotionTracker
from autoresearch_rl.target.interface import RunOutcome, TargetAdapter
from autoresearch_rl.telemetry.aggregation import compute_episode_stats
from autoresearch_rl.telemetry.comparability import ComparabilityPolicy, check_comparability, hardware_fingerprint
from autoresearch_rl.telemetry.events import emit
from autoresearch_rl.telemetry.ledger import append_result_row, ensure_results_tsv
from autoresearch_rl.telemetry.manifest import new_run_id, write_manifest
from autoresearch_rl.telemetry.run import write_run_manifest
from autoresearch_rl.tracking import LocalFileTracker


def _objective_value(metrics: dict[str, float], objective: ObjectiveConfig) -> float | None:
    if objective.metric not in metrics:
        return None
    return float(metrics[objective.metric])


def _score(value: float, objective: ObjectiveConfig) -> float:
    return value if objective.direction == "min" else -value


def _policy_from_config(policy_cfg, objective: ObjectiveConfig | None = None):
    if policy_cfg.type == "grid":
        return GridPolicy(policy_cfg.params)
    if policy_cfg.type == "random":
        return RandomPolicy(policy_cfg.params, seed=policy_cfg.seed)
    if policy_cfg.type == "learned":
        return LearnedParamPolicy(policy_cfg.params, LearnedSearchConfig())
    if policy_cfg.type == "llm":
        return LLMParamPolicy(
            policy_cfg.params,
            api_url=policy_cfg.llm_api_url,
            model=policy_cfg.llm_model,
            api_key_env=policy_cfg.llm_api_key_env,
            timeout_s=policy_cfg.llm_timeout_s,
            metric=objective.metric if objective else "val_bpb",
            direction=objective.direction if objective else "min",
            seed=policy_cfg.seed,
        )
    return StaticPolicy()


def _save_version(versions_dir: str, iter_idx: int, outcome: RunOutcome, params: dict[str, object]) -> str:
    target_dir = Path(versions_dir) / f"v{iter_idx:04d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "iter": iter_idx,
        "metrics": outcome.metrics,
        "params": params,
        "status": outcome.status,
        "run_dir": outcome.run_dir,
    }
    (target_dir / "version.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return str(target_dir)


def run_continuous(
    *,
    target: TargetAdapter,
    objective: ObjectiveConfig,
    controller: ControllerConfig,
    telemetry: TelemetryConfig,
    policy_cfg,
    comparability_cfg: ComparabilityConfig,
    program: str = "",
) -> LoopResult:
    logger = logging.getLogger(__name__)
    shutdown = ShutdownHandler()
    shutdown.register()

    ensure_results_tsv(telemetry.ledger_path)
    Path(telemetry.artifacts_dir).mkdir(parents=True, exist_ok=True)
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

    policy = _policy_from_config(policy_cfg, objective)

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

    if controller.seed is not None:
        random.seed(controller.seed)
        os.environ["PYTHONHASHSEED"] = str(controller.seed)
        os.environ["AR_SEED"] = str(controller.seed)

    run_manifest_path = str(Path(telemetry.artifacts_dir) / "run-manifest.json")
    manifest_config: dict = {
        "objective": objective.model_dump(),
        "controller": controller.model_dump(),
        "telemetry": telemetry.model_dump(),
        "policy": policy_cfg.model_dump(),
        "comparability": comparability_cfg.model_dump(),
    }
    if program:
        manifest_config["program"] = program
    write_run_manifest(run_manifest_path, config=manifest_config, run_id=episode_id)

    tracker = LocalFileTracker(telemetry.artifacts_dir, episode_id)
    tracker.log_params({
        "objective": objective.model_dump(),
        "policy": policy_cfg.model_dump(),
        "comparability": comparability_cfg.model_dump(),
    })
    tracker.set_status("running")
    score_history: list[float] = []
    promotion = PromotionTracker()

    comp_policy = ComparabilityPolicy(
        budget_mode=comparability_cfg.budget_mode,
        expected_budget_s=comparability_cfg.expected_budget_s,
        expected_hardware_fingerprint=comparability_cfg.expected_hardware_fingerprint,
        strict=comparability_cfg.strict,
    )
    hw_fp = hardware_fingerprint()
    run_budget_s = controller.max_wall_time_s or comparability_cfg.expected_budget_s
    comparable, non_comparable_reason = check_comparability(comp_policy, run_budget_s, hw_fp)
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

            proposal = policy.propose({"history": history, "program": program})
            assert isinstance(proposal, ParamProposal)
            run_dir = str(Path(telemetry.artifacts_dir) / f"run-{iter_idx:04d}")
            Path(run_dir).mkdir(parents=True, exist_ok=True)

            emit(
                telemetry.trace_path,
                {
                    "type": "proposal",
                    "episode_id": episode_id,
                    "iter": iter_idx,
                    "params": proposal.params,
                },
                run_id=episode_id,
                max_file_size_bytes=telemetry.max_file_size_bytes,
                max_rotated_files=telemetry.max_rotated_files,
            )

            try:
                train_out = target.run(run_dir=run_dir, params=proposal.params)
                if train_out.status != "ok":
                    outcome = train_out
                else:
                    outcome = target.eval(run_dir=run_dir, params=proposal.params)
            except Exception as exc:
                outcome = RunOutcome(
                    status="failed", metrics={}, stdout="",
                    stderr=str(exc), elapsed_s=0.0, run_dir=run_dir,
                )

            value = _objective_value(outcome.metrics, objective)
            status = outcome.status
            if value is None:
                status = "failed"

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
                    _save_version(
                        telemetry.versions_dir, iter_idx, outcome, proposal.params
                    )
                else:
                    no_improve_streak += 1
            else:
                no_improve_streak += 1

            promotion.record_result(
                float(value) if value is not None else float("inf"),
                improved=improved,
            )

            if isinstance(policy, Learnable):
                reward = 1.0 if decision == "keep" else (
                    -0.1 if status == "failed" else 0.0
                )
                policy.record_reward(reward)

            tracker.log_metrics(
                {objective.metric: float(value) if value is not None else 0.0,
                 "elapsed_s": outcome.elapsed_s},
                step=iter_idx,
            )
            if value is not None:
                score_history.append(float(value))

            emit(
                telemetry.trace_path,
                {
                    "type": "iteration",
                    "episode_id": episode_id,
                    "iter": iter_idx,
                    "status": status,
                    "decision": decision,
                    "metrics": outcome.metrics,
                    "params": proposal.params,
                    "elapsed_s": outcome.elapsed_s,
                },
                run_id=episode_id,
                max_file_size_bytes=telemetry.max_file_size_bytes,
                max_rotated_files=telemetry.max_rotated_files,
            )

            write_manifest(
                telemetry.artifacts_dir,
                {
                    "episode_id": episode_id,
                    "iter": iter_idx,
                    "status": status,
                    "decision": decision,
                    "metrics": outcome.metrics,
                    "params": proposal.params,
                    "stdout": outcome.stdout,
                    "stderr": outcome.stderr,
                    "run_dir": outcome.run_dir,
                },
            )

            append_result_row(
                path=telemetry.ledger_path,
                commit=current_commit(),
                metric_name=objective.metric,
                metric_value=float(value if value is not None else 0.0),
                memory_gb=0.0,
                status=decision,
                description="continuous",
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

            history.append(
                {
                    "iter": iter_idx,
                    "status": status,
                    "decision": decision,
                    "metrics": outcome.metrics,
                    "params": proposal.params,
                }
            )

            recent_statuses.append(status)
            if len(recent_statuses) > max(1, controller.failure_window):
                recent_statuses.pop(0)

            iterations_done += 1
            iter_idx += 1

            if check_no_improve(no_improve_streak, controller.no_improve_limit):
                break

            if check_failure_rate(
                recent_statuses,
                controller.failure_rate_limit,
                controller.failure_window,
            ):
                break

            if (
                best_value is not None
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
