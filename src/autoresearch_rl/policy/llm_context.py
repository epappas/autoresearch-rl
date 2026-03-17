"""Shared context utilities for LLM-based policies.

Provides history summarization, log extraction, and error extraction
used by both llm_diff and llm_search policies.
"""
from __future__ import annotations


def summarize_history(
    history: list[dict],
    metric: str,
    max_full: int = 50,
) -> tuple[str, list[dict]]:
    """Split history into a compact summary + recent full entries.

    Returns (summary_text, recent_entries). If len(history) <= max_full,
    summary_text is empty and all entries are returned as recent.
    """
    if len(history) <= max_full:
        return "", history

    old = history[:-max_full]
    recent = history[-max_full:]

    values = [
        e["metrics"][metric]
        for e in old
        if isinstance(e.get("metrics"), dict) and metric in e["metrics"]
    ]
    failures = sum(
        1 for e in old if e.get("status") in ("failed", "timeout", "rejected")
    )

    lines: list[str] = [f"Summary of first {len(old)} entries:"]
    if values:
        lines.append(f"  Best {metric}: {min(values):.4f}, Worst: {max(values):.4f}")
    lines.append(f"  Failure rate: {failures}/{len(old)} ({100 * failures / len(old):.0f}%)")

    errors: dict[str, int] = {}
    for e in old:
        if e.get("status") in ("failed", "timeout", "rejected"):
            err = (e.get("stderr_tail") or "")[:80]
            if err:
                errors[err] = errors.get(err, 0) + 1
    if errors:
        top = sorted(errors.items(), key=lambda x: -x[1])[:3]
        for err, count in top:
            lines.append(f"  Common error ({count}x): {err}")

    return "\n".join(lines), recent


def extract_recent_logs(history: list[dict], n: int = 3) -> list[str]:
    """Extract stdout tails from the last N successful iterations."""
    logs: list[str] = []
    for entry in reversed(history):
        if entry.get("status") == "ok" and entry.get("stdout_tail"):
            logs.append(entry["stdout_tail"])
            if len(logs) >= n:
                break
    return list(reversed(logs))


def extract_recent_errors(history: list[dict], n: int = 5) -> list[str]:
    """Extract stderr tails from the last N failed iterations."""
    errors: list[str] = []
    for entry in reversed(history):
        if entry.get("status") in ("failed", "timeout", "rejected"):
            tail = entry.get("stderr_tail", "")
            if tail:
                errors.append(tail)
                if len(errors) >= n:
                    break
    return list(reversed(errors))


def format_history_section(
    history: list[dict],
    metric: str,
    max_full: int = 50,
) -> str:
    """Format experiment history for an LLM prompt with optional summarization."""
    summary, recent = summarize_history(history, metric, max_full)
    lines: list[str] = []
    if summary:
        lines.append(summary)
        lines.append("")

    if recent:
        lines.append(f"Recent iterations ({len(recent)}):")
        for entry in recent:
            metrics = entry.get("metrics", {})
            status = entry.get("status", "unknown")
            val = metrics.get(metric, "N/A")
            decision = entry.get("decision", "")
            lines.append(
                f"  iter={entry.get('iter', '?')} -> {metric}={val} "
                f"(status={status}, decision={decision})"
            )
    else:
        lines.append("No experiment history yet.")

    return "\n".join(lines)
