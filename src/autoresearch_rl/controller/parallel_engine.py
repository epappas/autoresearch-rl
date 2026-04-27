"""Parallel iteration engine (Phase 4).

Sibling to controller/engine.py::run_experiment. Same observable contract
(LoopResult, telemetry events, ledger rows, checkpoint, keep/discard) but
multiple iterations execute concurrently inside a ThreadPoolExecutor,
admitted by a ResourcePool, with results processed in submission order.

Why a separate module instead of refactoring engine.py:
- engine.py has 387+ tests behind it; an in-place refactor risks regressions.
- Code duplication is real but bounded — the per-iteration body is mirrored
  here. Future cleanup can extract _run_one_iteration if/when both paths
  prove stable.

R3.a (reward ordering): Learnable.record_reward is called in monotonic
iter_idx order regardless of completion order, by buffering rewards in
pending_rewards and draining when next_unflushed matches.

R3.b (parallel_wallclock comparability): when comparability.budget_mode is
'parallel_wallclock', the ledger's budget_s column carries per-trial
elapsed_s instead of loop wall time. The description column is annotated
with concurrency at submission so post-hoc analysis can filter.

Cancellation: each worker spawns its own IntraIterationGuard pointed at
$run_dir/{progress.jsonl,control.json}. CommandTarget honours these paths
via its per-call env dict (no os.environ races), so the guard is race-free
even at high concurrency.
"""
from __future__ import annotations

import logging
import os
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from autoresearch_rl.checkpoint import LoopCheckpoint, load_checkpoint, save_checkpoint
from autoresearch_rl.config import (
    ComparabilityConfig,
    ControllerConfig,
    ObjectiveConfig,
    TelemetryConfig,
)
from autoresearch_rl.controller.engine import (
    _emit_progress_events,
    _read_progress_series,
    _save_version,
)
from autoresearch_rl.controller.executor import Outcome
from autoresearch_rl.controller.helpers import (
    check_failure_rate,
    check_no_improve,
    current_commit,
)
from autoresearch_rl.controller.intra_iteration import (
    BestValueRef,
    GuardConfig,
    IntraIterationGuard,
)
from autoresearch_rl.controller.resource_pool import ResourcePool
from autoresearch_rl.controller.shutdown import ShutdownHandler
from autoresearch_rl.controller.types import LoopResult
from autoresearch_rl.policy.interface import Learnable, Proposal, propose_batch
from autoresearch_rl.promotion import PromotionTracker
from autoresearch_rl.target.interface import resource_cost as compute_resource_cost
from autoresearch_rl.target.progress_reader import ProgressReader
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
from autoresearch_rl.telemetry.timeline import TimelineRecorder, set_global
from autoresearch_rl.tracking import LocalFileTracker

logger = logging.getLogger(__name__)


@dataclass
class _SubmittedTrial:
    iter_idx: int
    proposal: Proposal
    params: dict
    run_dir: str
    model_dir: str | None
    submitted_at: float
    cost: dict[str, int]
    future: Future


def _objective_value(metrics: dict, objective: ObjectiveConfig) -> float | None:
    if objective.metric not in metrics:
        return None
    return float(metrics[objective.metric])


def _score(value: float, objective: ObjectiveConfig) -> float:
    return value if objective.direction == "min" else -value


def run_experiment_parallel(
    *,
    executor,
    policy,
    objective: ObjectiveConfig,
    controller: ControllerConfig,
    telemetry: TelemetryConfig,
    comparability_cfg: ComparabilityConfig,
    proposal_state_builder,
    proposal_params_extractor,
    program: str = "",
    description_label: str = "parallel",
    target=None,
    enable_versions: bool = True,
    enable_tracker: bool = True,
    enable_run_manifest: bool = True,
    manifest_config: dict | None = None,
) -> LoopResult:
    """Parallel iteration loop. See module docstring for design notes."""
    pcfg = controller.parallel
    assert pcfg.enabled, "run_experiment_parallel requires controller.parallel.enabled"
    assert pcfg.max_concurrency >= 1
    pool = ResourcePool(pcfg.resources)

    shutdown = ShutdownHandler()
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

    if enable_run_manifest:
        write_run_manifest(
            str(Path(telemetry.artifacts_dir) / "run-manifest.json"),
            config=manifest_config or {}, run_id=episode_id,
        )

    tracker = LocalFileTracker(telemetry.artifacts_dir, episode_id) if enable_tracker else None
    if tracker:
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
    comparable, non_comparable_reason = check_comparability(
        comp_policy, run_budget_s, hw_fp,
    )
    if comp_policy.strict and not comparable:
        raise ValueError(f"Non-comparable run blocked: {non_comparable_reason}")

    timeline = TimelineRecorder(telemetry.timeline_path)
    set_global(timeline)

    in_flight: dict[int, _SubmittedTrial] = {}
    completed: dict[int, tuple[Outcome, _SubmittedTrial]] = {}
    pending_rewards: dict[int, float] = {}
    next_unflushed_reward = iter_idx
    next_to_process = iter_idx
    best_value_ref = BestValueRef(best_value)
    pool_executor = ThreadPoolExecutor(max_workers=pcfg.max_concurrency)

    def _stop_requested() -> bool:
        if shutdown.requested:
            return True
        elapsed = (time.monotonic() - start_ts) + elapsed_offset
        if controller.max_wall_time_s is not None and elapsed >= controller.max_wall_time_s:
            return True
        if controller.max_iterations is not None and iterations_done >= controller.max_iterations:
            return True
        if check_no_improve(no_improve_streak, controller.no_improve_limit):
            return True
        if check_failure_rate(
            recent_statuses, controller.failure_rate_limit, controller.failure_window,
        ):
            return True
        return False

    def _drain_rewards() -> None:
        nonlocal next_unflushed_reward
        if not isinstance(policy, Learnable):
            pending_rewards.clear()
            return
        while next_unflushed_reward in pending_rewards:
            policy.record_reward(pending_rewards.pop(next_unflushed_reward))
            next_unflushed_reward += 1

    try:
        while not _stop_requested() or in_flight or completed:
            # Submit phase: fill the pool up to max_concurrency.
            slots_open = pcfg.max_concurrency - len(in_flight)
            if slots_open > 0 and not _stop_requested():
                with timeline.span(
                    "policy.propose_batch",
                    category="policy",
                    args={"k": slots_open, "policy": type(policy).__name__},
                ):
                    proposals = propose_batch(
                        policy, proposal_state_builder(history, program), slots_open,
                    )
                for prop in proposals:
                    if _stop_requested():
                        break
                    if controller.max_iterations is not None and (
                        iterations_done + len(in_flight) + len(completed)
                        >= controller.max_iterations
                    ):
                        break

                    params = proposal_params_extractor(prop)
                    run_dir = str(Path(telemetry.artifacts_dir) / f"run-{iter_idx:04d}")
                    Path(run_dir).mkdir(parents=True, exist_ok=True)
                    model_dir: str | None = None
                    if telemetry.model_output_dir:
                        model_dir = str(Path(telemetry.model_output_dir) / f"v{iter_idx:04d}")
                        params["AR_MODEL_DIR"] = model_dir

                    cost = compute_resource_cost(target, params) if target else {"gpu": 1}
                    if not pool.try_acquire(iter_idx=iter_idx, cost=cost):
                        # Doesn't fit alongside in-flight load. Stop submitting this round.
                        break

                    emit(
                        telemetry.trace_path,
                        {
                            "type": "proposal",
                            "episode_id": episode_id,
                            "iter": iter_idx,
                            "params": params,
                            "concurrency": pcfg.max_concurrency,
                        },
                        run_id=episode_id,
                        max_file_size_bytes=telemetry.max_file_size_bytes,
                        max_rotated_files=telemetry.max_rotated_files,
                    )

                    submit_iter = iter_idx
                    fut = pool_executor.submit(
                        _execute_one_timed,
                        timeline=timeline,
                        executor=executor, proposal=prop, run_dir=run_dir,
                        objective=objective, controller=controller,
                        best_value_ref=best_value_ref,
                        iter_idx=submit_iter,
                        executor_name=type(executor).__name__,
                    )
                    in_flight[iter_idx] = _SubmittedTrial(
                        iter_idx=iter_idx, proposal=prop, params=params,
                        run_dir=run_dir, model_dir=model_dir,
                        submitted_at=time.monotonic(), cost=cost, future=fut,
                    )
                    iter_idx += 1

            # Drain phase: move every done future from in_flight to completed.
            # Holding completed-out-of-order trials in `completed` buffer is
            # how we guarantee monotonic processing order (R3.a, ledger order).
            for idx in sorted(in_flight):
                if in_flight[idx].future.done():
                    trial_done = in_flight.pop(idx)
                    try:
                        out = trial_done.future.result()
                    except Exception as exc:
                        out = Outcome(
                            status="failed", metrics={}, stdout="",
                            stderr=str(exc), elapsed_s=0.0, run_dir=trial_done.run_dir,
                        )
                    pool.release(iter_idx=trial_done.iter_idx)
                    completed[trial_done.iter_idx] = (out, trial_done)

            # If nothing is processable yet, sleep briefly (don't burn CPU)
            # unless next_to_process is in completed (drained immediately).
            if next_to_process not in completed:
                if not in_flight and not completed:
                    break
                time.sleep(pcfg.submit_poll_interval_s)
                continue

            # Process exactly one trial per loop tick, in submission order.
            outcome, trial = completed.pop(next_to_process)
            next_to_process += 1

            downloaded_model = outcome.metrics.pop("_model_dir", None)
            effective_model_dir = (
                str(downloaded_model) if downloaded_model else trial.model_dir
            )

            _emit_progress_events(
                trace_path=telemetry.trace_path,
                run_dir=trial.run_dir,
                episode_id=episode_id,
                iter_idx=trial.iter_idx,
                max_file_size_bytes=telemetry.max_file_size_bytes,
                max_rotated_files=telemetry.max_rotated_files,
            )

            value = _objective_value(outcome.metrics, objective)
            status = outcome.status
            if status != "cancelled" and value is None:
                status = "failed"

            decision = "discard"
            improved = False
            reward = 0.0
            if status == "cancelled":
                decision = "cancelled"
                no_improve_streak += 1
                reward = -0.05
            elif value is not None:
                score = _score(value, objective)
                if score < best_score:
                    best_score = score
                    best_value = value
                    best_value_ref.set(value)  # publish to live workers
                    decision = "keep"
                    improved = True
                    no_improve_streak = 0
                    reward = 1.0
                    if enable_versions:
                        _save_version(
                            telemetry.versions_dir, trial.iter_idx, outcome,
                            trial.params, model_dir=effective_model_dir,
                        )
                else:
                    no_improve_streak += 1
            else:
                no_improve_streak += 1
                reward = -0.1

            promotion.record_result(
                float(value) if value is not None else float("inf"),
                improved=improved,
            )

            pending_rewards[trial.iter_idx] = reward
            _drain_rewards()

            if tracker:
                tracker.log_metrics(
                    {
                        objective.metric: float(value) if value is not None else 0.0,
                        "elapsed_s": outcome.elapsed_s,
                    },
                    step=trial.iter_idx,
                )
            if value is not None:
                score_history.append(float(value))

            emit(
                telemetry.trace_path,
                {
                    "type": "iteration",
                    "episode_id": episode_id,
                    "iter": trial.iter_idx,
                    "status": status,
                    "decision": decision,
                    "metrics": outcome.metrics,
                    "params": trial.params,
                    "elapsed_s": outcome.elapsed_s,
                    "concurrency": pcfg.max_concurrency,
                },
                run_id=episode_id,
                max_file_size_bytes=telemetry.max_file_size_bytes,
                max_rotated_files=telemetry.max_rotated_files,
            )

            write_manifest(
                telemetry.artifacts_dir,
                {
                    "episode_id": episode_id,
                    "iter": trial.iter_idx,
                    "status": status,
                    "decision": decision,
                    "metrics": outcome.metrics,
                    "params": trial.params,
                    "stdout": outcome.stdout,
                    "stderr": outcome.stderr,
                    "run_dir": outcome.run_dir,
                },
            )

            # R3.b: under parallel_wallclock the budget_s is per-trial elapsed,
            # not loop wall, so the ledger remains comparable across runs even
            # though k trials share the same wall window.
            if comp_policy.budget_mode == "parallel_wallclock":
                ledger_budget_s = int(outcome.elapsed_s)
            else:
                ledger_budget_s = run_budget_s
            ledger_desc = f"{description_label}|conc={pcfg.max_concurrency}"

            append_result_row(
                path=telemetry.ledger_path,
                commit=current_commit(),
                metric_name=objective.metric,
                metric_value=float(value if value is not None else 0.0),
                memory_gb=0.0,
                status=decision,
                description=ledger_desc,
                episode_id=str(episode_id),
                iter_idx=int(trial.iter_idx),
                score=float(best_score if best_score < float("inf") else 0.0),
                budget_mode=comp_policy.budget_mode,
                budget_s=ledger_budget_s,
                hardware_fingerprint=hw_fp,
                comparable=comparable,
                non_comparable_reason=non_comparable_reason,
                max_file_size_bytes=telemetry.max_file_size_bytes,
                max_rotated_files=telemetry.max_rotated_files,
            )

            history.append(
                {
                    "iter": trial.iter_idx,
                    "status": status,
                    "decision": decision,
                    "metrics": outcome.metrics,
                    "params": trial.params,
                    "stdout_tail": outcome.stdout[-500:] if outcome.stdout else "",
                    "stderr_tail": outcome.stderr[-300:] if outcome.stderr else "",
                    "progress_series": _read_progress_series(
                        trial.run_dir, objective.metric, max_points=20,
                    ),
                }
            )

            recent_statuses.append(status)
            if len(recent_statuses) > max(1, controller.failure_window):
                recent_statuses.pop(0)

            iterations_done += 1

    finally:
        # Cancel any still-pending futures and drain.
        for trial in list(in_flight.values()):
            trial.future.cancel()
        pool_executor.shutdown(wait=True)
        timeline.close()
        set_global(None)

        if score_history:
            stats = compute_episode_stats(score_history)
            emit(
                telemetry.trace_path,
                {
                    "type": "episode_summary",
                    "episode_id": episode_id,
                    "mean": stats.mean, "median": stats.median,
                    "min": stats.min, "max": stats.max, "stdev": stats.stdev,
                    "count": stats.count, "trend_slope": stats.trend_slope,
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
        best_score=best_score, best_value=best_value, iterations=iterations_done,
    )


def _execute_one_timed(
    *,
    timeline,
    executor,
    proposal: Proposal,
    run_dir: str,
    objective: ObjectiveConfig,
    controller: ControllerConfig,
    best_value_ref: BestValueRef,
    iter_idx: int,
    executor_name: str,
) -> Outcome:
    """Worker entrypoint that wraps _execute_one in a timeline span."""
    with timeline.span(
        "executor.execute",
        category="execute",
        args={"iter": iter_idx, "executor": executor_name},
    ) as span_args:
        outcome = _execute_one(
            executor=executor, proposal=proposal, run_dir=run_dir,
            objective=objective, controller=controller,
            best_value_ref=best_value_ref,
        )
        span_args["status"] = outcome.status
        span_args["elapsed_s"] = outcome.elapsed_s
    return outcome


def _execute_one(
    *,
    executor,
    proposal: Proposal,
    run_dir: str,
    objective: ObjectiveConfig,
    controller: ControllerConfig,
    best_value_ref: BestValueRef,
) -> Outcome:
    """Per-trial worker.

    Per-worker IntraIterationGuard is wired safely because CommandTarget
    derives its progress/control paths from run_dir via its per-call env
    dict (no os.environ writes; see CommandTarget._run). The guard reads
    and writes the same per-iter paths. The guard always starts when
    cancel is enabled — it consults best_value_ref live, so a worker
    submitted before any best exists will activate as soon as a sibling
    trial's keep updates the ref.
    """
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    progress_path = str(Path(run_dir) / "progress.jsonl")
    control_path = str(Path(run_dir) / "control.json")

    cancel_cfg = controller.intra_iteration_cancel
    guard: IntraIterationGuard | None = None
    guard_reader: ProgressReader | None = None
    if cancel_cfg.enabled:
        # Faster reader poll than the engine default (0.5s) so cancel
        # decisions land quickly; the guard's own poll throttles further.
        reader_poll = max(0.05, min(0.2, cancel_cfg.poll_interval_s))
        guard_reader = ProgressReader(progress_path, poll_interval_s=reader_poll)
        guard_reader.start()
        guard = IntraIterationGuard(
            reader=guard_reader,
            control_path=control_path,
            metric=objective.metric,
            direction=objective.direction,
            best_value_ref=best_value_ref,
            config=GuardConfig(
                enabled=True,
                min_steps=cancel_cfg.min_steps,
                poll_interval_s=cancel_cfg.poll_interval_s,
                min_reports_before_decide=cancel_cfg.min_reports_before_decide,
            ),
        )
        guard.start()

    try:
        outcome = executor.execute(proposal, run_dir)
    finally:
        if guard is not None:
            guard.stop()
        if guard_reader is not None:
            guard_reader.stop()

    if guard is not None and guard.cancelled:
        outcome = Outcome(
            status="cancelled",
            metrics=outcome.metrics, stdout=outcome.stdout,
            stderr=outcome.stderr or guard.cancel_reason,
            elapsed_s=outcome.elapsed_s, run_dir=outcome.run_dir,
            judge_signals=outcome.judge_signals,
        )
    return outcome
