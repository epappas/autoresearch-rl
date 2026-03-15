from __future__ import annotations

from dataclasses import dataclass

from autoresearch_rl.mdp import Action, Reward, State


@dataclass
class Transition:
    state: State
    action: Action
    reward: Reward
    next_state: State
    log_prob: float
    value_estimate: float


class TrajectoryBuffer:
    """Fixed-size circular buffer for storing transitions."""

    def __init__(self, max_size: int = 1024) -> None:
        self._buffer: list[Transition] = []
        self._max_size = max_size

    def add(self, transition: Transition) -> None:
        if len(self._buffer) >= self._max_size:
            self._buffer.pop(0)
        self._buffer.append(transition)

    def get_batch(self, batch_size: int) -> list[Transition]:
        return self._buffer[-batch_size:]

    def get_episode(
        self, start: int = 0, end: int | None = None
    ) -> list[Transition]:
        return self._buffer[start:end]

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def rewards(self) -> list[float]:
        return [t.reward.value for t in self._buffer]

    @property
    def values(self) -> list[float]:
        return [t.value_estimate for t in self._buffer]
