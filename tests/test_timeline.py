"""TimelineRecorder unit + end-to-end (engine produces parseable JSON)."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from autoresearch_rl.config import (
    ComparabilityConfig,
    ControllerConfig,
    ObjectiveConfig,
    TelemetryConfig,
)
from autoresearch_rl.controller.engine import run_experiment
from autoresearch_rl.controller.executor import Outcome
from autoresearch_rl.policy.interface import ParamProposal
from autoresearch_rl.policy.search import StaticPolicy
from autoresearch_rl.telemetry.timeline import TimelineRecorder, global_span, set_global


# ---------------------------------------------------------------- recorder


def test_disabled_when_path_is_none() -> None:
    rec = TimelineRecorder(None)
    assert rec.enabled is False
    rec.slice("x", category="c", start_ts_us=0, duration_us=10)
    rec.close()


def test_writes_valid_json_array(tmp_path: Path) -> None:
    path = tmp_path / "timeline.json"
    rec = TimelineRecorder(str(path))
    rec.slice("a", category="cat", start_ts_us=0, duration_us=100, args={"k": 1})
    rec.slice("b", category="cat", start_ts_us=100, duration_us=200)
    rec.close()
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["name"] == "a"
    assert data[0]["ph"] == "X"
    assert data[0]["dur"] == 100
    assert data[0]["args"] == {"k": 1}
    assert data[1]["name"] == "b"


def test_empty_writer_produces_empty_array(tmp_path: Path) -> None:
    path = tmp_path / "timeline.json"
    rec = TimelineRecorder(str(path))
    rec.close()
    assert json.loads(path.read_text()) == []


def test_span_records_duration(tmp_path: Path) -> None:
    path = tmp_path / "timeline.json"
    rec = TimelineRecorder(str(path))
    with rec.span("work", category="test") as args:
        time.sleep(0.05)
        args["result"] = "done"
    rec.close()
    data = json.loads(path.read_text())
    assert len(data) == 1
    event = data[0]
    assert event["name"] == "work"
    assert event["dur"] >= 50_000  # >= 50 ms
    assert event["args"] == {"result": "done"}


def test_thread_safe_concurrent_writes(tmp_path: Path) -> None:
    path = tmp_path / "timeline.json"
    rec = TimelineRecorder(str(path))

    def burst() -> None:
        for i in range(20):
            rec.slice(f"e{i}", category="c", start_ts_us=i, duration_us=1)

    threads = [threading.Thread(target=burst) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rec.close()

    data = json.loads(path.read_text())
    assert len(data) == 80
    assert all(e["name"].startswith("e") for e in data)


def test_global_span_no_op_when_unset(tmp_path: Path) -> None:
    set_global(None)
    with global_span("x", category="c") as args:
        args["k"] = "v"  # must not raise even when no recorder
    set_global(None)


def test_global_span_records_through_global(tmp_path: Path) -> None:
    path = tmp_path / "timeline.json"
    rec = TimelineRecorder(str(path))
    set_global(rec)
    try:
        with global_span("g", category="c"):
            time.sleep(0.01)
    finally:
        set_global(None)
    rec.close()
    data = json.loads(path.read_text())
    assert len(data) == 1
    assert data[0]["name"] == "g"


# ---------------------------------------------------------------- engine produces timeline


class _FastExecutor:
    def execute(self, proposal: ParamProposal, run_dir: str) -> Outcome:  # noqa: ARG002
        return Outcome(
            status="ok", metrics={"loss": 0.5}, stdout="", stderr="",
            elapsed_s=0.01, run_dir=run_dir,
        )


class _FixedPolicy(StaticPolicy):
    def propose(self, state):  # type: ignore[override]
        return ParamProposal(params={"lr": 1e-3})


def test_engine_emits_timeline_events(tmp_path: Path) -> None:
    timeline_path = tmp_path / "timeline.json"
    run_experiment(
        executor=_FastExecutor(),
        evaluator=None,  # type: ignore[arg-type]
        policy=_FixedPolicy(),
        objective=ObjectiveConfig(metric="loss", direction="min"),
        controller=ControllerConfig(max_iterations=2),
        telemetry=TelemetryConfig(
            trace_path=str(tmp_path / "events.jsonl"),
            ledger_path=str(tmp_path / "results.tsv"),
            artifacts_dir=str(tmp_path / "artifacts"),
            versions_dir=str(tmp_path / "versions"),
            timeline_path=str(timeline_path),
        ),
        comparability_cfg=ComparabilityConfig(strict=False),
        proposal_state_builder=lambda h, p: {"history": h, "program": p},
        proposal_params_extractor=lambda p: getattr(p, "params", {}),
        enable_run_manifest=False,
        enable_versions=False,
        enable_tracker=False,
        enable_forecasting=False,
    )

    data = json.loads(timeline_path.read_text())
    names = [e["name"] for e in data]
    # Each iteration should produce policy.propose + executor.execute spans.
    assert names.count("policy.propose") == 2, names
    assert names.count("executor.execute") == 2, names
    # All events have correct ph/dur fields.
    for event in data:
        assert event["ph"] == "X"
        assert "dur" in event
        assert event["dur"] >= 0
