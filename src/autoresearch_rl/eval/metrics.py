from __future__ import annotations

import re
from dataclasses import dataclass

_FLOAT = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?"
VAL_BPB_RE = re.compile(rf"val[_-]?bpb\s*[:=]\s*({_FLOAT})", re.IGNORECASE)
LOSS_RE = re.compile(rf"(?:^|\s)loss\s*[:=]\s*({_FLOAT})", re.IGNORECASE)
TIME_RE = re.compile(rf"(?:time|elapsed|training_seconds)\s*[:=]\s*({_FLOAT})", re.IGNORECASE)
STEP_RE = re.compile(rf"(?:step|num_steps)\s*[:=]\s*(\d+)", re.IGNORECASE)


@dataclass
class ParsedMetrics:
    val_bpb: float | None = None
    loss: float | None = None


@dataclass
class MetricPoint:
    t: float
    val_bpb: float | None
    loss: float | None


def parse_metrics(text: str) -> ParsedMetrics:
    """Parse metrics from stdout/stderr text.

    Uses last-seen values so progressive logs resolve to final metric lines.
    Supports scientific notation.
    """
    val_bpb: float | None = None
    loss: float | None = None

    for line in text.splitlines():
        m = VAL_BPB_RE.search(line)
        if m:
            val_bpb = float(m.group(1))

        m2 = LOSS_RE.search(line)
        if m2:
            loss = float(m2.group(1))

    return ParsedMetrics(val_bpb=val_bpb, loss=loss)


def parse_metric_series(text: str) -> list[MetricPoint]:
    """Parse a time/step series from logs. Best-effort with partial signals."""
    points: list[MetricPoint] = []
    current_time: float | None = None

    for line in text.splitlines():
        t_match = TIME_RE.search(line)
        if t_match:
            current_time = float(t_match.group(1))

        step_match = STEP_RE.search(line)
        if step_match and current_time is None:
            # treat step index as time proxy when no time present
            current_time = float(step_match.group(1))

        val_match = VAL_BPB_RE.search(line)
        loss_match = LOSS_RE.search(line)

        if (val_match or loss_match) and current_time is not None:
            points.append(
                MetricPoint(
                    t=current_time,
                    val_bpb=float(val_match.group(1)) if val_match else None,
                    loss=float(loss_match.group(1)) if loss_match else None,
                )
            )
            current_time = None

    return points
