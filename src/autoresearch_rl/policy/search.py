from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterable


@dataclass
class ParamProposal:
    params: dict[str, object]
    rationale: str


class ParamPolicy:
    def next(self, *, history: list[dict]) -> ParamProposal:  # pragma: no cover - interface
        raise NotImplementedError


class StaticPolicy(ParamPolicy):
    def next(self, *, history: list[dict]) -> ParamProposal:
        return ParamProposal(params={}, rationale="static")


class GridPolicy(ParamPolicy):
    def __init__(self, grid: dict[str, Iterable[object]]):
        keys = list(grid.keys())
        values = [list(grid[k]) for k in keys]
        self._keys = keys
        self._iter = itertools.cycle(list(itertools.product(*values)) or [()])

    def next(self, *, history: list[dict]) -> ParamProposal:
        combo = next(self._iter)
        params = {k: v for k, v in zip(self._keys, combo)}
        return ParamProposal(params=params, rationale="grid")


class RandomPolicy(ParamPolicy):
    def __init__(self, space: dict[str, Iterable[object]], seed: int = 7):
        self._rng = random.Random(seed)
        self._space = {k: list(v) for k, v in space.items()}

    def next(self, *, history: list[dict]) -> ParamProposal:
        params = {k: self._rng.choice(v) for k, v in self._space.items() if v}
        return ParamProposal(params=params, rationale="random")
