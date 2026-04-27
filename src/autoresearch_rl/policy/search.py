from __future__ import annotations

import itertools
import random
from typing import Iterable

from autoresearch_rl.policy.interface import ParamProposal


class StaticPolicy:
    def propose(self, state: dict) -> ParamProposal:
        return ParamProposal(params={}, rationale="static")

    def propose_batch(self, state: dict, k: int) -> list[ParamProposal]:
        return [self.propose(state) for _ in range(max(0, k))]


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

    def propose_batch(self, state: dict, k: int) -> list[ParamProposal]:
        return [self.propose(state) for _ in range(max(0, k))]


class RandomPolicy:
    def __init__(self, space: dict[str, Iterable[object]], seed: int = 7):
        self._rng = random.Random(seed)
        self._space = {k: list(v) for k, v in space.items()}

    def propose(self, state: dict) -> ParamProposal:
        params = {k: self._rng.choice(v) for k, v in self._space.items() if v}
        return ParamProposal(params=params, rationale="random")

    def propose_batch(self, state: dict, k: int) -> list[ParamProposal]:
        """k seeded-random draws.

        Uses the same RNG so the sequence is reproducible across batched
        and serial runs (a serial run of K iterations and a batch of K
        produce the same K params in the same order).
        """
        return [self.propose(state) for _ in range(max(0, k))]
