from __future__ import annotations

import itertools
import random
from typing import Iterable

from autoresearch_rl.policy.interface import ParamProposal


class StaticPolicy:
    def propose(self, state: dict) -> ParamProposal:
        return ParamProposal(params={}, rationale="static")


class GridPolicy:
    def __init__(self, grid: dict[str, Iterable[object]]):
        keys = list(grid.keys())
        values = [list(grid[k]) for k in keys]
        self._keys = keys
        self._iter = itertools.cycle(list(itertools.product(*values)) or [()])

    def propose(self, state: dict) -> ParamProposal:
        combo = next(self._iter)
        params = {k: v for k, v in zip(self._keys, combo)}
        return ParamProposal(params=params, rationale="grid")


class RandomPolicy:
    def __init__(self, space: dict[str, Iterable[object]], seed: int = 7):
        self._rng = random.Random(seed)
        self._space = {k: list(v) for k, v in space.items()}

    def propose(self, state: dict) -> ParamProposal:
        params = {k: self._rng.choice(v) for k, v in self._space.items() if v}
        return ParamProposal(params=params, rationale="random")
