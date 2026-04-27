# Velocity Log

Per RLix-Adoption-Remediation.md R4.a: track wall-clock per phase to recalibrate t-shirt sizes.

| Date | Phase | Size estimate | Actual sweep | Surprises |
|---|---|---|---|---|
| 2026-04-27 | R0 (restore target/) | S | trivial — `git restore` + verify | none |
| 2026-04-27 | R1 (baseline) | S | mypy red 10 errors, all pre-existing; pytest+ruff green | filed M1/M2/M3 follow-ups |
| 2026-04-27 | R1.b (verify claims) | S | trivial via `gh api` | all 3 claims verified correct |
| 2026-04-27 | R1.c (AST audit) | S | trivial — existing AST machinery is sufficient | downgraded `required_calls` from 1d to ~2h |
| 2026-04-27 | R2 (re-ground plan) | S | trivial — 7 plan edits | discovered sandbox/validator IS on canonical path |
| 2026-04-27 | R3.d (Template refactor) | S | trivial — single-file refactor + 3 tests | 28 basilica tests still green |
| 2026-04-27 | R3.c (concurrency spike) | S | trivial — script + report | overhead = 0% even at K=8 |
| 2026-04-27 | R4 (calibration + harness) | M | ~1h — 5 fixtures + 6 assertions + xfail/xpass discipline | xfail flipped to xpass after Phase 7.2 — caught it |
| 2026-04-27 | Phase 6 (config validate) | S | ~1h — 8 checks, 12 tests, CLI wired | clean |
| 2026-04-27 | Phase 1 + 7.1/7.2/7.5 | M | ~1.5h — protocol + reader + CommandTarget + Basilica bootstrap + engine drain + state builder + prompt fragments + example | smoothest yet — modular extension points held up |
| 2026-04-27 | Phase 2 + 7.3 | M | ~1.5h — IntraIterationGuard + engine wiring + cancelled status + required_calls AST guardrail + 15 tests | Basilica cancel propagation deferred to follow-up; failure_rate now ignores cancelled |
| 2026-04-27 | Phase 3 (timeline) | S | ~45m — TimelineRecorder + engine + Basilica + LLM spans + 8 tests + e2e proof | one bad multi-line edit corrupted basilica indentation; caught on next test run, fixed in 2 edits |
| 2026-04-27 | M1+M2+M3 (mypy debt) | S | ~30m — Protocol for Deployment, type-ignore for SDK stubs, narrow runner.py fixes | mypy now green (0 errors in 63 files) for the first time |
| 2026-04-27 | Phase 4A propose_batch | S | ~45m — Protocol method + native impls + LLMParamPolicy.propose_batch + 14 tests | seeded random batch is bit-identical to k serial draws — important for reproducibility |
| 2026-04-27 | Phase 4B ResourcePool | S | ~30m — bin-packing pool + ParallelConfig + resource_cost helper + 13 tests | concise; took the simple path |
| 2026-04-27 | Phase 4C parallel_engine | M | ~2.5h — parallel_engine.py + R3.a + R3.b + 7 tests + e2e CLI proof | first attempt serialized via env-lock; tests caught it; restructured into in_flight + completed dicts with min-iter draining |

## Recalibration after first 10 PRs

Original M (Phase 1 + 7.1/7.2/7.5): predicted 1–3 days. Actual: ~1.5h. Either the scaffolding was already very good, or the PR is being undersized — tested only via unit/integration and not against a real LLM. **Plan: keep M for Phase 2 (cancellation) since real LLM eval is the part that bites.**
