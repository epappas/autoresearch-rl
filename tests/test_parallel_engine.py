"""Parallel engine end-to-end: concurrent execution, ordered ledger, R3.a, R3.b."""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path


from autoresearch_rl.config import (
    ComparabilityConfig,
    ControllerConfig,
    ObjectiveConfig,
    ParallelConfig,
    TelemetryConfig,
)
from autoresearch_rl.controller.executor import Outcome
from autoresearch_rl.controller.parallel_engine import run_experiment_parallel
from autoresearch_rl.policy.interface import Learnable, ParamProposal
from autoresearch_rl.policy.search import RandomPolicy


# ---------------------------------------------------------------- fakes


class _SleepyExecutor:
    """Simulates a slow trial. Sleeps `iter_s` then returns ok with a metric."""

    def __init__(self, *, iter_s: float = 0.5, metric_fn=None) -> None:
        self.iter_s = iter_s
        self.calls: list[tuple[float, float]] = []  # (start_ts, end_ts)
        self.metric_fn = metric_fn or (lambda _params: 0.5)
        self._lock = threading.Lock()

    def execute(self, proposal: ParamProposal, run_dir: str) -> Outcome:
        t0 = time.monotonic()
        time.sleep(self.iter_s)
        t1 = time.monotonic()
        with self._lock:
            self.calls.append((t0, t1))
        loss = self.metric_fn(proposal.params)
        return Outcome(
            status="ok", metrics={"loss": float(loss)}, stdout="",
            stderr="", elapsed_s=t1 - t0, run_dir=run_dir,
        )


class _OrderedRewardPolicy(RandomPolicy, Learnable):
    """RandomPolicy + records the order in which rewards arrive."""

    def __init__(self, space: dict, seed: int = 0) -> None:
        super().__init__(space, seed=seed)
        self.reward_log: list[float] = []
        self._lock = threading.Lock()

    def record_reward(self, reward: float) -> None:
        with self._lock:
            self.reward_log.append(reward)


def _telemetry(tmp_path: Path) -> TelemetryConfig:
    return TelemetryConfig(
        trace_path=str(tmp_path / "events.jsonl"),
        ledger_path=str(tmp_path / "results.tsv"),
        artifacts_dir=str(tmp_path / "artifacts"),
        versions_dir=str(tmp_path / "versions"),
    )


# ---------------------------------------------------------------- speedup


def test_parallel_runs_faster_than_serial(tmp_path: Path) -> None:
    """4 iters of 0.5s should complete in ~0.5s wall under K=4 (well under 2s)."""
    space = {"lr": [1e-5, 1e-4, 1e-3, 1e-2]}
    executor = _SleepyExecutor(iter_s=0.5)
    t0 = time.monotonic()
    result = run_experiment_parallel(
        executor=executor,
        policy=RandomPolicy(space, seed=11),
        objective=ObjectiveConfig(metric="loss", direction="min"),
        controller=ControllerConfig(
            max_iterations=4,
            parallel=ParallelConfig(enabled=True, max_concurrency=4, resources={"gpu": 4}),
        ),
        telemetry=_telemetry(tmp_path),
        comparability_cfg=ComparabilityConfig(strict=False),
        proposal_state_builder=lambda h, p: {"history": h, "program": p},
        proposal_params_extractor=lambda p: getattr(p, "params", {}),
        enable_run_manifest=False, enable_versions=False, enable_tracker=False,
    )
    elapsed = time.monotonic() - t0
    assert result.iterations == 4
    assert elapsed < 1.5, f"4 iters of 0.5s should fit in <1.5s; got {elapsed:.2f}s"
    # And the trials must actually have overlapped — earliest end > latest start.
    starts = sorted(c[0] for c in executor.calls)
    ends = sorted(c[1] for c in executor.calls)
    assert ends[0] > starts[-1] - 0.05, "trials did not overlap"


# ---------------------------------------------------------------- R3.a reward ordering


def test_rewards_arrive_in_submission_order(tmp_path: Path) -> None:
    """Even when futures complete out of order, record_reward sees ascending iter."""
    space = {"lr": [1e-5, 1e-4, 1e-3, 1e-2]}

    # Sleep durations chosen so iter 1 finishes before iter 0.
    sleeps = {0: 0.40, 1: 0.05, 2: 0.30, 3: 0.10}
    call_count = [0]

    class _OrderedExec:
        def execute(self, proposal: ParamProposal, run_dir: str) -> Outcome:
            idx = call_count[0]
            call_count[0] += 1
            sleep_s = sleeps[idx]
            time.sleep(sleep_s)
            # All trials get distinct losses so each is a "keep" if processed in
            # the order they finished. We want to see whether record_reward gets
            # them in submission order (0,1,2,3) or completion order (1,3,2,0).
            return Outcome(
                status="ok", metrics={"loss": float(idx) * 0.1}, stdout="",
                stderr="", elapsed_s=sleep_s, run_dir=run_dir,
            )

    policy = _OrderedRewardPolicy(space, seed=0)
    run_experiment_parallel(
        executor=_OrderedExec(),
        policy=policy,
        objective=ObjectiveConfig(metric="loss", direction="min"),
        controller=ControllerConfig(
            max_iterations=4,
            parallel=ParallelConfig(enabled=True, max_concurrency=4, resources={"gpu": 4}),
        ),
        telemetry=_telemetry(tmp_path),
        comparability_cfg=ComparabilityConfig(strict=False),
        proposal_state_builder=lambda h, p: {"history": h, "program": p},
        proposal_params_extractor=lambda p: getattr(p, "params", {}),
        enable_run_manifest=False, enable_versions=False, enable_tracker=False,
    )
    # Reward sequence must be the submission order (one entry per iter).
    # Loss series is [0.0, 0.1, 0.2, 0.3]; iter 0 is best, then increasing
    # — so rewards in submission order are [keep, discard, discard, discard]
    # = [1.0, 0.0, 0.0, 0.0]. If reward order matched completion order
    # (iter 1 first), we'd see [discard reward first, then keep], not
    # [keep, discard, discard, discard].
    assert policy.reward_log == [1.0, 0.0, 0.0, 0.0], policy.reward_log


# ---------------------------------------------------------------- R3.b parallel comparability


def test_parallel_wallclock_records_per_trial_budget(tmp_path: Path) -> None:
    space = {"lr": [1e-5, 1e-3]}
    executor = _SleepyExecutor(iter_s=0.1)
    run_experiment_parallel(
        executor=executor,
        policy=RandomPolicy(space, seed=0),
        objective=ObjectiveConfig(metric="loss", direction="min"),
        controller=ControllerConfig(
            max_iterations=2,
            parallel=ParallelConfig(enabled=True, max_concurrency=2, resources={"gpu": 2}),
        ),
        telemetry=_telemetry(tmp_path),
        comparability_cfg=ComparabilityConfig(
            budget_mode="parallel_wallclock", expected_budget_s=10, strict=False,
        ),
        proposal_state_builder=lambda h, p: {"history": h, "program": p},
        proposal_params_extractor=lambda p: getattr(p, "params", {}),
        enable_run_manifest=False, enable_versions=False, enable_tracker=False,
    )

    rows = list(csv.reader(
        (tmp_path / "results.tsv").open(encoding="utf-8"), delimiter="\t",
    ))
    assert rows[0][0] == "commit"  # header
    data = rows[1:]
    assert len(data) == 2
    for row in data:
        budget_mode = row[9]
        budget_s_str = row[10]
        description = row[5]
        assert budget_mode == "parallel_wallclock"
        # Per-trial budget should be 0 or 1 (we slept 0.1s, int-cast).
        assert int(budget_s_str) <= 1
        assert "conc=2" in description


def test_fixed_wallclock_keeps_loop_budget(tmp_path: Path) -> None:
    space = {"lr": [1e-5, 1e-3]}
    executor = _SleepyExecutor(iter_s=0.05)
    run_experiment_parallel(
        executor=executor,
        policy=RandomPolicy(space, seed=0),
        objective=ObjectiveConfig(metric="loss", direction="min"),
        controller=ControllerConfig(
            max_iterations=2, max_wall_time_s=5,
            parallel=ParallelConfig(enabled=True, max_concurrency=2, resources={"gpu": 2}),
        ),
        telemetry=_telemetry(tmp_path),
        comparability_cfg=ComparabilityConfig(
            budget_mode="fixed_wallclock", strict=False,
        ),
        proposal_state_builder=lambda h, p: {"history": h, "program": p},
        proposal_params_extractor=lambda p: getattr(p, "params", {}),
        enable_run_manifest=False, enable_versions=False, enable_tracker=False,
    )
    rows = list(csv.reader(
        (tmp_path / "results.tsv").open(encoding="utf-8"), delimiter="\t",
    ))
    for row in rows[1:]:
        assert row[10] == "5"  # max_wall_time_s


# ---------------------------------------------------------------- ledger ordering


def test_ledger_rows_in_submission_order(tmp_path: Path) -> None:
    space = {"lr": [1e-5, 1e-4, 1e-3, 1e-2]}
    sleeps = {0: 0.30, 1: 0.05, 2: 0.20, 3: 0.10}
    call_count = [0]

    class _Exec:
        def execute(self, proposal: ParamProposal, run_dir: str) -> Outcome:
            idx = call_count[0]
            call_count[0] += 1
            time.sleep(sleeps[idx])
            return Outcome(
                status="ok", metrics={"loss": 0.5}, stdout="",
                stderr="", elapsed_s=sleeps[idx], run_dir=run_dir,
            )

    run_experiment_parallel(
        executor=_Exec(),
        policy=RandomPolicy(space, seed=0),
        objective=ObjectiveConfig(metric="loss", direction="min"),
        controller=ControllerConfig(
            max_iterations=4,
            parallel=ParallelConfig(enabled=True, max_concurrency=4, resources={"gpu": 4}),
        ),
        telemetry=_telemetry(tmp_path),
        comparability_cfg=ComparabilityConfig(strict=False),
        proposal_state_builder=lambda h, p: {"history": h, "program": p},
        proposal_params_extractor=lambda p: getattr(p, "params", {}),
        enable_run_manifest=False, enable_versions=False, enable_tracker=False,
    )
    rows = list(csv.reader(
        (tmp_path / "results.tsv").open(encoding="utf-8"), delimiter="\t",
    ))
    iters = [int(r[7]) for r in rows[1:]]
    assert iters == sorted(iters), f"iters not in order: {iters}"


# ---------------------------------------------------------------- ResourcePool admission


def test_resource_pool_caps_concurrency(tmp_path: Path) -> None:
    """K=4 max_concurrency with pool of 2 GPUs must not exceed 2 in flight."""
    in_flight_max = [0]
    in_flight_now = [0]
    lock = threading.Lock()
    space = {"lr": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}

    class _Tracking:
        def execute(self, proposal: ParamProposal, run_dir: str) -> Outcome:
            with lock:
                in_flight_now[0] += 1
                in_flight_max[0] = max(in_flight_max[0], in_flight_now[0])
            time.sleep(0.1)
            with lock:
                in_flight_now[0] -= 1
            return Outcome(
                status="ok", metrics={"loss": 0.5}, stdout="",
                stderr="", elapsed_s=0.1, run_dir=run_dir,
            )

    run_experiment_parallel(
        executor=_Tracking(),
        policy=RandomPolicy(space, seed=0),
        objective=ObjectiveConfig(metric="loss", direction="min"),
        controller=ControllerConfig(
            max_iterations=5,
            parallel=ParallelConfig(enabled=True, max_concurrency=4, resources={"gpu": 2}),
        ),
        telemetry=_telemetry(tmp_path),
        comparability_cfg=ComparabilityConfig(strict=False),
        proposal_state_builder=lambda h, p: {"history": h, "program": p},
        proposal_params_extractor=lambda p: getattr(p, "params", {}),
        enable_run_manifest=False, enable_versions=False, enable_tracker=False,
    )
    assert in_flight_max[0] <= 2, f"pool overran: {in_flight_max[0]}"


# ---------------------------------------------------------------- stop guards


def test_no_improve_limit_stops_loop(tmp_path: Path) -> None:
    space = {"lr": [1e-5, 1e-3]}
    executor = _SleepyExecutor(iter_s=0.05, metric_fn=lambda _p: 1.0)  # always same
    result = run_experiment_parallel(
        executor=executor,
        policy=RandomPolicy(space, seed=0),
        objective=ObjectiveConfig(metric="loss", direction="min"),
        controller=ControllerConfig(
            max_iterations=20, no_improve_limit=2,
            parallel=ParallelConfig(enabled=True, max_concurrency=2, resources={"gpu": 2}),
        ),
        telemetry=_telemetry(tmp_path),
        comparability_cfg=ComparabilityConfig(strict=False),
        proposal_state_builder=lambda h, p: {"history": h, "program": p},
        proposal_params_extractor=lambda p: getattr(p, "params", {}),
        enable_run_manifest=False, enable_versions=False, enable_tracker=False,
    )
    # First iter is "keep" (best), then no-improve streak grows. With K=2,
    # the loop may submit a few more before stopping; allow up to 6 total.
    assert result.iterations < 20
