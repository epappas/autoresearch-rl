"""Bin-packing resource admission for the parallel engine (Phase 4).

Inspired by RLix's ResourceManager but stripped to the essentials we need
for autoresearch-rl: a flat dict of resource counts, threadsafe try-acquire
/ release, and bookkeeping of in-flight reservations keyed by iteration id.

Design choices (kept deliberately small):
- No partial allocation. A reservation either fits or it doesn't.
- No fairness layer. Submission order is FIFO; the parallel engine
  decides when to ask for k more by checking pool.available().
- No types beyond what's needed. Resources are str -> int (e.g. {"gpu":4}).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Reservation:
    iter_idx: int
    cost: dict[str, int]


@dataclass
class _PoolState:
    capacity: dict[str, int]
    in_use: dict[str, int] = field(default_factory=dict)
    reservations: dict[int, Reservation] = field(default_factory=dict)


class ResourcePool:
    """Threadsafe bounded resource bin.

    Usage:
        pool = ResourcePool({"gpu": 4, "memory_gb": 128})
        if pool.try_acquire(iter_idx=7, cost={"gpu": 1, "memory_gb": 16}):
            ... run ...
            pool.release(iter_idx=7)
    """

    def __init__(self, capacity: dict[str, int]) -> None:
        if not capacity:
            raise ValueError("capacity must be non-empty")
        for k, v in capacity.items():
            if v < 0:
                raise ValueError(f"capacity[{k!r}] must be >= 0, got {v}")
        self._state = _PoolState(capacity=dict(capacity))
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def capacity(self) -> dict[str, int]:
        return dict(self._state.capacity)

    def in_use(self) -> dict[str, int]:
        with self._lock:
            return dict(self._state.in_use)

    def available(self) -> dict[str, int]:
        with self._lock:
            return self._available_locked()

    def in_flight_count(self) -> int:
        with self._lock:
            return len(self._state.reservations)

    def try_acquire(self, *, iter_idx: int, cost: dict[str, int]) -> bool:
        """Reserve `cost` for `iter_idx`. Returns False if it doesn't fit.

        Re-acquiring the same iter_idx is an error (caller bug); raises.
        """
        with self._lock:
            if iter_idx in self._state.reservations:
                raise ValueError(
                    f"iter_idx {iter_idx} already has an active reservation"
                )
            available = self._available_locked()
            for k, v in cost.items():
                if v < 0:
                    raise ValueError(f"cost[{k!r}] must be >= 0, got {v}")
                if available.get(k, 0) < v:
                    return False
            for k, v in cost.items():
                self._state.in_use[k] = self._state.in_use.get(k, 0) + v
            self._state.reservations[iter_idx] = Reservation(
                iter_idx=iter_idx, cost=dict(cost),
            )
            return True

    def release(self, iter_idx: int) -> None:
        with self._cond:
            res = self._state.reservations.pop(iter_idx, None)
            if res is None:
                return
            for k, v in res.cost.items():
                current = self._state.in_use.get(k, 0)
                self._state.in_use[k] = max(0, current - v)
            self._cond.notify_all()

    def wait_for_capacity(
        self, *, cost: dict[str, int], timeout_s: float | None = None,
    ) -> bool:
        """Block until `cost` would fit, or timeout. Does NOT acquire.

        Caller still must call try_acquire() after this returns True; the
        slot may be claimed by another thread in between. Use the
        try_acquire / wait pair only when you want to back off the
        submission loop.
        """
        with self._cond:
            deadline = None
            if timeout_s is not None:
                import time
                deadline = time.monotonic() + timeout_s
            while True:
                available = self._available_locked()
                if all(available.get(k, 0) >= v for k, v in cost.items()):
                    return True
                if deadline is None:
                    self._cond.wait()
                else:
                    import time
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cond.wait(timeout=remaining)

    # ----- internal -----

    def _available_locked(self) -> dict[str, int]:
        return {
            k: cap - self._state.in_use.get(k, 0)
            for k, cap in self._state.capacity.items()
        }
