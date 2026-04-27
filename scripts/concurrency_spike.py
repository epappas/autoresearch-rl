"""Threads-vs-asyncio spike for Phase 4 (parallel iterations).

Mimics the I/O profile of `BasilicaTarget._poll_for_metrics`: long sleep loops
broken by short polls. Measures wall-time and per-call overhead under
ThreadPoolExecutor with concurrency K against the same workload run serially.

Acceptance from RLix-Adoption-Remediation.md R3.c:
    wall time of K concurrent fake targets <= 1.3x single-iter time.

Run:
    uv run python scripts/concurrency_spike.py [--n 8] [--iter-s 5]
"""
from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor


def fake_basilica_iteration(iter_s: float, poll_interval_s: float = 0.5) -> dict:
    """Mimic Basilica polling: sleep iter_s in poll_interval_s chunks."""
    t0 = time.monotonic()
    waited = 0.0
    polls = 0
    while waited < iter_s:
        time.sleep(poll_interval_s)
        waited += poll_interval_s
        polls += 1
    return {"elapsed_s": time.monotonic() - t0, "polls": polls}


def run_serial(n: int, iter_s: float) -> dict:
    t0 = time.monotonic()
    results = [fake_basilica_iteration(iter_s) for _ in range(n)]
    return {
        "wall_s": time.monotonic() - t0,
        "per_iter_mean_s": statistics.mean(r["elapsed_s"] for r in results),
        "per_iter_max_s": max(r["elapsed_s"] for r in results),
    }


def run_parallel(n: int, iter_s: float, workers: int) -> dict:
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fake_basilica_iteration, iter_s) for _ in range(n)]
        results = [f.result() for f in futures]
    return {
        "wall_s": time.monotonic() - t0,
        "per_iter_mean_s": statistics.mean(r["elapsed_s"] for r in results),
        "per_iter_max_s": max(r["elapsed_s"] for r in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="number of iterations")
    parser.add_argument("--iter-s", type=float, default=5.0, help="seconds per iteration")
    args = parser.parse_args()

    print(f"Spike: n={args.n}, iter_s={args.iter_s}")
    print()

    print("Serial run...")
    serial = run_serial(args.n, args.iter_s)
    print(f"  wall_s = {serial['wall_s']:.2f}")
    print(f"  per_iter_mean_s = {serial['per_iter_mean_s']:.3f}")
    print(f"  per_iter_max_s  = {serial['per_iter_max_s']:.3f}")
    print()

    for k in (2, 4, 8):
        print(f"Parallel run (workers={k})...")
        par = run_parallel(args.n, args.iter_s, workers=k)
        ideal = serial["wall_s"] / k
        overhead_pct = 100 * (par["wall_s"] - ideal) / ideal if ideal > 0 else 0.0
        single_iter_time = args.iter_s
        ratio_to_single = par["wall_s"] / single_iter_time
        print(f"  wall_s          = {par['wall_s']:.2f}")
        print(f"  per_iter_mean_s = {par['per_iter_mean_s']:.3f}")
        print(f"  per_iter_max_s  = {par['per_iter_max_s']:.3f}")
        print(f"  ideal_wall_s    = {ideal:.2f}  (serial / {k})")
        print(f"  overhead_pct    = {overhead_pct:+.1f}%")
        print(f"  ratio_to_single_iter = {ratio_to_single:.2f}x")
        print()


if __name__ == "__main__":
    main()
