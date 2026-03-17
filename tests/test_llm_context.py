from __future__ import annotations

from autoresearch_rl.policy.llm_context import (
    extract_recent_errors,
    extract_recent_logs,
    format_history_section,
    summarize_history,
)


def _make_entry(
    i: int,
    status: str = "ok",
    val_bpb: float | None = 1.5,
    decision: str = "discard",
    stdout_tail: str = "",
    stderr_tail: str = "",
) -> dict:
    metrics = {"val_bpb": val_bpb} if val_bpb is not None else {}
    return {
        "iter": i,
        "status": status,
        "decision": decision,
        "metrics": metrics,
        "params": {},
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


# --- summarize_history ---


def test_summarize_small_history_no_summary():
    history = [_make_entry(i) for i in range(10)]
    summary, recent = summarize_history(history, "val_bpb")
    assert summary == ""
    assert len(recent) == 10


def test_summarize_large_history():
    history = [_make_entry(i, val_bpb=float(i)) for i in range(80)]
    summary, recent = summarize_history(history, "val_bpb", max_full=50)
    assert len(recent) == 50
    assert "Summary of first 30 entries" in summary
    assert "Best val_bpb:" in summary
    assert "Failure rate:" in summary


def test_summarize_with_failures_and_errors():
    history = []
    for i in range(60):
        if i % 3 == 0:
            history.append(
                _make_entry(i, status="failed", val_bpb=None, stderr_tail="CUDA OOM")
            )
        else:
            history.append(_make_entry(i, val_bpb=1.5 - i * 0.01))

    summary, recent = summarize_history(history, "val_bpb", max_full=50)
    assert "CUDA OOM" in summary
    assert len(recent) == 50


def test_summarize_exact_threshold():
    history = [_make_entry(i) for i in range(50)]
    summary, recent = summarize_history(history, "val_bpb", max_full=50)
    assert summary == ""
    assert len(recent) == 50


# --- extract_recent_logs ---


def test_extract_recent_logs():
    history = [
        _make_entry(0, status="ok", stdout_tail="epoch=1 val_bpb=1.5"),
        _make_entry(1, status="failed", stdout_tail=""),
        _make_entry(2, status="ok", stdout_tail="epoch=2 val_bpb=1.3"),
        _make_entry(3, status="ok", stdout_tail="epoch=3 val_bpb=1.1"),
    ]
    logs = extract_recent_logs(history, n=2)
    assert len(logs) == 2
    assert "epoch=2" in logs[0]
    assert "epoch=3" in logs[1]


def test_extract_recent_logs_empty_history():
    assert extract_recent_logs([], n=3) == []


def test_extract_recent_logs_no_successful():
    history = [_make_entry(0, status="failed")]
    assert extract_recent_logs(history, n=3) == []


def test_extract_recent_logs_skips_empty_stdout():
    history = [
        _make_entry(0, status="ok", stdout_tail=""),
        _make_entry(1, status="ok", stdout_tail="real log"),
    ]
    logs = extract_recent_logs(history, n=3)
    assert len(logs) == 1
    assert "real log" in logs[0]


# --- extract_recent_errors ---


def test_extract_recent_errors():
    history = [
        _make_entry(0, status="failed", stderr_tail="OOM error"),
        _make_entry(1, status="ok"),
        _make_entry(2, status="timeout", stderr_tail="timeout after 3600s"),
        _make_entry(3, status="rejected", stderr_tail="forbidden token"),
    ]
    errors = extract_recent_errors(history, n=5)
    assert len(errors) == 3
    assert "OOM error" in errors[0]
    assert "timeout" in errors[1]
    assert "forbidden" in errors[2]


def test_extract_recent_errors_limits_n():
    history = [
        _make_entry(i, status="failed", stderr_tail=f"error {i}")
        for i in range(10)
    ]
    errors = extract_recent_errors(history, n=3)
    assert len(errors) == 3
    # Most recent 3
    assert "error 7" in errors[0]
    assert "error 8" in errors[1]
    assert "error 9" in errors[2]


def test_extract_recent_errors_empty_history():
    assert extract_recent_errors([], n=5) == []


# --- format_history_section ---


def test_format_history_section_small():
    history = [
        _make_entry(0, val_bpb=1.5, decision="keep"),
        _make_entry(1, val_bpb=1.3, decision="keep"),
    ]
    text = format_history_section(history, "val_bpb")
    assert "iter=0" in text
    assert "iter=1" in text
    assert "1.5" in text
    assert "1.3" in text


def test_format_history_section_large():
    history = [_make_entry(i, val_bpb=float(i)) for i in range(80)]
    text = format_history_section(history, "val_bpb", max_full=50)
    assert "Summary of first 30 entries" in text
    assert "Recent iterations (50)" in text


def test_format_history_section_empty():
    text = format_history_section([], "val_bpb")
    assert "No experiment history" in text
