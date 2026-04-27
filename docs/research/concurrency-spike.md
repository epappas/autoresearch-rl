# Concurrency Spike — Threads vs asyncio for Phase 4

**Date**: 2026-04-27
**Question**: Does `ThreadPoolExecutor` give us the parallelism Phase 4 needs without resorting to asyncio?
**Acceptance from R3.c**: wall time of K concurrent fake `BasilicaTarget`s ≤ 1.3× single-iter time.

## Method

`scripts/concurrency_spike.py` runs a synthetic `fake_basilica_iteration(iter_s)` that sleeps for `iter_s` seconds in `0.5 s` poll chunks — mimicking `target/basilica.py:_poll_for_metrics`'s 20 s sleep loop. Compared serial vs `ThreadPoolExecutor(max_workers=K)` for K ∈ {2, 4, 8}, n=8 iterations, iter_s=3 s.

## Results

```
Serial:        wall = 24.26 s, per_iter ≈ 3.03 s
Workers = 2:   wall = 12.07 s,  overhead = -0.5%, ratio_to_single = 4.02x
Workers = 4:   wall =  6.04 s,  overhead = -0.4%, ratio_to_single = 2.01x
Workers = 8:   wall =  3.01 s,  overhead = -0.8%, ratio_to_single = 1.00x
```

`overhead = (parallel_wall - serial_wall/K) / (serial_wall/K)` — i.e. how much worse than ideal.

## Verdict

**Threads pass.** With workers=8, wall time equals single-iter time (1.00× — far inside the 1.3× budget). Overhead is negative-to-zero (within measurement noise) across all worker counts. The GIL is irrelevant here because the workload is `time.sleep`, which releases the GIL.

**Decision**: Phase 4 uses `concurrent.futures.ThreadPoolExecutor`. No asyncio. No `aiohttp`. No new dependencies.

## What this does NOT prove

- Real `BasilicaTarget` does HTTP calls (`urllib.request.urlopen`) and JSON-decoding alongside the sleeps. urllib also releases the GIL on the socket read, so this should be fine — but **re-run this spike** against a real `BasilicaTarget` against a stubbed Basilica API before Phase 4 ships.
- We did not test K > 8. Production likely caps at ~4 (Basilica GPU pool size); larger K is academic.
- We did not test under CPU-bound user code in the trial — that's outside the controller, runs in containers.

## Follow-up

Add `tests/test_concurrency_spike.py` that runs a tiny version (n=4, iter_s=0.2) as part of CI, asserting `overhead_pct < 30`. Catches future regressions if someone introduces a controller-side lock that serializes I/O.
