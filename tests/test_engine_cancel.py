"""Engine-level cooperative cancellation: guard flips outcome.status."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from autoresearch_rl.config import (
    ComparabilityConfig,
    ControllerConfig,
    IntraIterationCancelConfig,
    ObjectiveConfig,
    TelemetryConfig,
)
from autoresearch_rl.controller.engine import run_experiment
from autoresearch_rl.controller.executor import Outcome
from autoresearch_rl.policy.interface import ParamProposal
from autoresearch_rl.policy.search import StaticPolicy
from autoresearch_rl.target.progress import ProgressReport


class _FakeExecutor:
    """Executor that simulates a slow trial and emits worsening progress.

    On each call, writes N progress reports into run_dir/progress.jsonl with
    increasing loss values, then returns ok with a final value. The engine's
    IntraIterationGuard should detect the doomed series and write the cancel
    file mid-run.
    """

    def __init__(self, *, return_status: str = "ok", final_loss: float = 2.0) -> None:
        self.calls = 0
        self.return_status = return_status
        self.final_loss = final_loss
        self.status_history: list[str] = []

    def execute(self, proposal: ParamProposal, run_dir: str) -> Outcome:  # noqa: ARG002
        self.calls += 1
        progress_path = Path(os.environ["AR_PROGRESS_FILE"])
        control_path = Path(os.environ["AR_CONTROL_FILE"])
        progress_path.parent.mkdir(parents=True, exist_ok=True)

        # Simulate worsening loss over 10 fast steps.
        with open(progress_path, "w", encoding="utf-8") as f:
            for i in range(10):
                if control_path.exists():
                    break
                losses = 1.5 + 0.05 * i  # always above best=0.5 → forecast doomed
                f.write(
                    ProgressReport(
                        iter=0, step=i + 1, step_target=10,
                        elapsed_s=float(i + 1), metrics={"loss": losses},
                    ).to_json_line() + "\n"
                )
                f.flush()
                time.sleep(0.06)

        # Trial finishes (or would have); the engine sees status=ok but the
        # guard's flag is what matters.
        return Outcome(
            status=self.return_status,
            metrics={"loss": self.final_loss},
            stdout="", stderr="", elapsed_s=1.0, run_dir=run_dir,
        )


class _FixedPolicy(StaticPolicy):
    """Returns a fixed param proposal each call."""

    def propose(self, state):  # type: ignore[override]
        return ParamProposal(params={"lr": 1e-3})


def test_engine_marks_cancelled_when_guard_fires(tmp_path: Path) -> None:
    """First iter sets best=0.4; second iter is doomed (loss stuck at 1.5+)."""
    artifacts = tmp_path / "artifacts"
    traces = tmp_path / "traces"
    versions = tmp_path / "versions"

    # Pre-seed best by running one good iteration first.
    good_executor = _FakeExecutor(return_status="ok", final_loss=0.4)
    run_experiment(
        executor=good_executor,
        evaluator=None,  # type: ignore[arg-type]
        policy=_FixedPolicy(),
        objective=ObjectiveConfig(metric="loss", direction="min"),
        controller=ControllerConfig(max_iterations=1, intra_iteration_cancel=IntraIterationCancelConfig(enabled=False)),
        telemetry=TelemetryConfig(
            trace_path=str(traces / "events.jsonl"),
            ledger_path=str(artifacts / "results.tsv"),
            artifacts_dir=str(artifacts),
            versions_dir=str(versions),
        ),
        comparability_cfg=ComparabilityConfig(strict=False),
        proposal_state_builder=lambda h, p: {"history": h, "program": p},
        proposal_params_extractor=lambda p: getattr(p, "params", {}),
        enable_run_manifest=False,
        enable_versions=False,
        enable_tracker=False,
        enable_forecasting=False,
    )

    # Now drive a 2-iter run where iter 0 sets best=0.4 and iter 1 emits
    # a worsening series → guard cancels mid-iter.
    bad_traces = tmp_path / "traces2"
    bad_artifacts = tmp_path / "artifacts2"
    bad_versions = tmp_path / "versions2"

    # Two iterations: first sets best=0.4 (good_executor would, but we use
    # bad_executor which returns 1.5). Switch: first iter succeeds with
    # final_loss tweaked to 0.4, second iter returns 1.5 and should cancel.
    class _Switch(_FakeExecutor):
        def __init__(self) -> None:
            super().__init__(return_status="ok", final_loss=0.0)

        def execute(self, proposal, run_dir):  # type: ignore[override]
            if self.calls == 0:
                self.final_loss = 0.4
                # First iter: do NOT emit worsening progress; just return ok.
                self.calls += 1
                return Outcome(
                    status="ok", metrics={"loss": 0.4}, stdout="",
                    stderr="", elapsed_s=0.1, run_dir=run_dir,
                )
            # Second iter: emit worsening progress so guard cancels.
            self.final_loss = 1.5
            outcome = super().execute(proposal, run_dir)
            self.status_history.append(outcome.status)
            return outcome

    switch_executor = _Switch()
    result = run_experiment(
        executor=switch_executor,
        evaluator=None,  # type: ignore[arg-type]
        policy=_FixedPolicy(),
        objective=ObjectiveConfig(metric="loss", direction="min"),
        controller=ControllerConfig(
            max_iterations=2,
            intra_iteration_cancel=IntraIterationCancelConfig(
                enabled=True, min_steps=1, poll_interval_s=0.05,
                min_reports_before_decide=5,
            ),
        ),
        telemetry=TelemetryConfig(
            trace_path=str(bad_traces / "events.jsonl"),
            ledger_path=str(bad_artifacts / "results.tsv"),
            artifacts_dir=str(bad_artifacts),
            versions_dir=str(bad_versions),
        ),
        comparability_cfg=ComparabilityConfig(strict=False),
        proposal_state_builder=lambda h, p: {"history": h, "program": p},
        proposal_params_extractor=lambda p: getattr(p, "params", {}),
        enable_run_manifest=False,
        enable_versions=False,
        enable_tracker=False,
        enable_forecasting=False,
    )

    # Confirm: 2 iterations, best_value=0.4 from first, second cancelled.
    assert switch_executor.calls == 2
    assert result.iterations == 2
    assert result.best_value == 0.4

    # The control file for iter 1 should exist.
    iter1_control = bad_artifacts / "run-0001" / "control.json"
    assert iter1_control.exists(), "guard must have written cancel file"
    payload = json.loads(iter1_control.read_text())
    assert payload["action"] == "cancel"
