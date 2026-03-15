from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoopResult:
    best_score: float
    best_value: float | None
    iterations: int
