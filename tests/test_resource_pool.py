"""ResourcePool admission + release semantics."""
from __future__ import annotations

import threading
import time

import pytest

from autoresearch_rl.controller.resource_pool import ResourcePool


def test_capacity_must_be_non_empty() -> None:
    with pytest.raises(ValueError):
        ResourcePool({})


def test_capacity_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        ResourcePool({"gpu": -1})


def test_acquire_within_capacity() -> None:
    pool = ResourcePool({"gpu": 4})
    assert pool.try_acquire(iter_idx=0, cost={"gpu": 1}) is True
    assert pool.try_acquire(iter_idx=1, cost={"gpu": 2}) is True
    assert pool.in_flight_count() == 2
    assert pool.available() == {"gpu": 1}


def test_acquire_rejects_over_capacity() -> None:
    pool = ResourcePool({"gpu": 4})
    assert pool.try_acquire(iter_idx=0, cost={"gpu": 3}) is True
    assert pool.try_acquire(iter_idx=1, cost={"gpu": 2}) is False
    assert pool.in_flight_count() == 1


def test_release_frees_capacity() -> None:
    pool = ResourcePool({"gpu": 2})
    pool.try_acquire(iter_idx=0, cost={"gpu": 2})
    assert pool.try_acquire(iter_idx=1, cost={"gpu": 1}) is False
    pool.release(iter_idx=0)
    assert pool.available() == {"gpu": 2}
    assert pool.try_acquire(iter_idx=1, cost={"gpu": 1}) is True


def test_release_unknown_iter_is_noop() -> None:
    pool = ResourcePool({"gpu": 1})
    pool.release(iter_idx=999)  # must not raise


def test_double_acquire_same_iter_raises() -> None:
    pool = ResourcePool({"gpu": 4})
    pool.try_acquire(iter_idx=0, cost={"gpu": 1})
    with pytest.raises(ValueError, match="already has an active reservation"):
        pool.try_acquire(iter_idx=0, cost={"gpu": 1})


def test_multi_resource_admission() -> None:
    pool = ResourcePool({"gpu": 4, "memory_gb": 64})
    assert pool.try_acquire(iter_idx=0, cost={"gpu": 2, "memory_gb": 32}) is True
    assert pool.try_acquire(iter_idx=1, cost={"gpu": 2, "memory_gb": 40}) is False
    assert pool.try_acquire(iter_idx=2, cost={"gpu": 2, "memory_gb": 16}) is True


def test_negative_cost_rejected() -> None:
    pool = ResourcePool({"gpu": 4})
    with pytest.raises(ValueError, match=">= 0"):
        pool.try_acquire(iter_idx=0, cost={"gpu": -1})


def test_wait_for_capacity_returns_false_on_timeout() -> None:
    pool = ResourcePool({"gpu": 1})
    pool.try_acquire(iter_idx=0, cost={"gpu": 1})
    t0 = time.monotonic()
    ok = pool.wait_for_capacity(cost={"gpu": 1}, timeout_s=0.2)
    elapsed = time.monotonic() - t0
    assert ok is False
    assert 0.15 < elapsed < 0.5


def test_wait_for_capacity_unblocks_on_release() -> None:
    pool = ResourcePool({"gpu": 1})
    pool.try_acquire(iter_idx=0, cost={"gpu": 1})

    def _release_after_delay() -> None:
        time.sleep(0.1)
        pool.release(iter_idx=0)

    threading.Thread(target=_release_after_delay, daemon=True).start()
    t0 = time.monotonic()
    ok = pool.wait_for_capacity(cost={"gpu": 1}, timeout_s=2.0)
    elapsed = time.monotonic() - t0
    assert ok is True
    assert elapsed < 1.0


def test_resource_cost_helper_default() -> None:
    from autoresearch_rl.target.interface import resource_cost

    class Bare:
        pass

    assert resource_cost(Bare(), params={}) == {"gpu": 1}


def test_resource_cost_helper_uses_method_when_present() -> None:
    from autoresearch_rl.target.interface import resource_cost

    class Custom:
        def resource_cost(self, params: dict) -> dict[str, int]:  # noqa: ARG002
            return {"gpu": 2, "memory_gb": 32}

    assert resource_cost(Custom(), params={}) == {"gpu": 2, "memory_gb": 32}
