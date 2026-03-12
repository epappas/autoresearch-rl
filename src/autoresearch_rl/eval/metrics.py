from __future__ import annotations

import re
from dataclasses import dataclass

_FLOAT = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?"
VAL_BPB_RE = re.compile(rf"val[_-]?bpb\s*[:=]\s*({_FLOAT})", re.IGNORECASE)
LOSS_RE = re.compile(rf"(?:^|\s)loss\s*[:=]\s*({_FLOAT})", re.IGNORECASE)


@dataclass
class ParsedMetrics:
    val_bpb: float | None = None
    loss: float | None = None


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
