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


def resource_cost(target: object, params: dict[str, object]) -> dict[str, int]:
    """Return the resource cost of running `target` with `params`.

    Targets may declare a `resource_cost(self, params)` method. If absent,
    defaults to {"gpu": 1}. Used by the parallel engine's ResourcePool.
    """
    fn = getattr(target, "resource_cost", None)
    if callable(fn):
        return dict(fn(params))
    return {"gpu": 1}
