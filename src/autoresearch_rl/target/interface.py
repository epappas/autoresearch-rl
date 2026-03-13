from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RunOutcome:
    status: str
    metrics: dict[str, float]
    stdout: str
    stderr: str
    elapsed_s: float
    run_dir: str


class TargetAdapter(Protocol):
    def run(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome: ...
    def eval(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome: ...
