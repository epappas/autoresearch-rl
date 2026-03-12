"""Mutable training script (autoresearch-like example)."""

from __future__ import annotations

import time

from prepare import SEED, TIME_BUDGET_S


def run() -> tuple[float, float, int]:
    # simple deterministic loop bounded by fixed wall-clock budget
    start = time.monotonic()
    step = 0
    val_bpb = 1.35

    while time.monotonic() - start < TIME_BUDGET_S:
        step += 1
        # deterministic improvement curve
        val_bpb = max(1.10, val_bpb - 0.0004)
        time.sleep(0.01)

    loss = 2.0 + (val_bpb - 1.1) * 0.7
    return loss, val_bpb, step


def main() -> None:
    _ = SEED
    t0 = time.monotonic()
    loss, val_bpb, steps = run()
    elapsed = time.monotonic() - t0
    print(f"loss={loss:.6f}")
    print(f"val_bpb={val_bpb:.6f}")
    print(f"training_seconds={elapsed:.2f}")
    print(f"num_steps={steps}")


if __name__ == "__main__":
    main()
