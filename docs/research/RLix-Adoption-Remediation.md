# Remediation & Execution Plan

Reconciles the gaps surfaced after [`RLix-Adoption-Plan.md`](./RLix-Adoption-Plan.md) was drafted. Five remediation phases (**R0–R4**) precede a re-pointed execution of the original plan's **Phase 1–7**. Every step has a measurable acceptance test; nothing here is "trust me."

## Map: 14 pointers → remediation tasks

| # | Pointer | Remediation phase |
|---|---|---|
| 1 | Working tree imports broken (`target/` deleted) | R0 |
| 8 | Never ran tests / ruff / mypy | R1.a |
| 5 | `LLMParamPolicy` 50-entry cap unverified | R1.b |
| 6 | Basilica polling intervals (15 s / 20 s) unverified | R1.b |
| 7 | RLix file/class names are second-hand | R1.b |
| 14 | `required_calls` AST machinery feasibility unknown | R1.c |
| 2 | Loop lives in `engine.py`, not `continuous.py` | R2 |
| 3 | Diff validation is `controller/diff_executor.py`, not sandbox | R2 |
| 4 | Richer `controller/` and `policy/` module set | R2 |
| 11a | Reward ordering under concurrent execution | R3.a → folded into Phase 4 |
| 11b | Comparability strict mode breaks under parallelism | R3.b → folded into Phase 4 |
| 11c | Threads vs asyncio decision unmeasured | R3.c → spike before Phase 4 |
| 13 | `_BOOTSTRAP` `.format()` brace escaping risk | R3.d → before Phase 1 |
| 12 | Guardrail blind spot when source never had `emit_progress` | R3.e → folded into Phase 6 |
| 9 | Day-estimates are gut-feel | R4 |
| 10 | Phase 7 acceptance criteria are hopes | R4 (prompt eval harness) |

---

## R0 — Unblock: restore the working tree

**Status**: hard blocker; nothing imports without it.

### R0.1 Resolve the deleted `target/` files

Two options, decide with user:
- **(a) Restore from HEAD**: `git restore src/autoresearch_rl/target/` — fastest, preserves the existing six modules.
- **(b) Finish the in-progress refactor** that deleted them — find the replacement (likely a single-file `target.py` or new `adapters/` package), rewire the three live imports (`cli.py:17`, `controller/continuous.py:22`, `controller/executor.py:15`), commit.

### Acceptance R0
- `uv sync --extra dev` succeeds.
- `uv run python -c "import autoresearch_rl.cli"` succeeds.
- `uv run autoresearch-rl --help` prints help.

---

## R1 — Verified baseline + verify second-hand claims

### R1.a Run the green-baseline triplet

```bash
uv run pytest -q              | tee docs/research/baseline-pytest.txt
uv run ruff check src/ tests/ | tee docs/research/baseline-ruff.txt
uv run mypy src/              | tee docs/research/baseline-mypy.txt
```

Write `docs/research/baseline-2026-04-27.md` with summary counts and the first 50 lines of any failure output.

**If anything is red**: each failure becomes its own blocker task. Do not proceed to R1.b until the baseline is documented (red or green is OK; *unknown* is not).

### R1.b Verify the three second-hand claims

| Claim | How to verify | If wrong, action |
|---|---|---|
| `LLMParamPolicy` caps history at 50 entries | `grep -n "history\[:50\]\|history\[-50:\]\|len(history)" src/autoresearch_rl/policy/llm_search.py` | Patch `RLix-Comparison.md` and `RLix-Adoption-Plan.md` |
| Basilica polls at 15 s + 20 s | Read `target/basilica.py:_wait_and_collect`, `_poll_for_metrics` | Patch Phase 1.3 to use the actual numbers |
| RLix `SchedulerImpl`, `_GapRatioDPWorker`, `plan_generation_gap_ratio` exist as named | `gh api repos/rlops/rlix/contents/rlix/scheduler/scheduler.py --jq .content \| base64 -d \| grep -E "class SchedulerImpl\|_GapRatioDPWorker\|def plan_generation"` | Patch comparison; not load-bearing for the plan |

### R1.c Audit the AST validator surface

Read in this order: `controller/diff_executor.py`, `controller/contract.py`, `sandbox/validator.py`. Answer two questions:
1. Is there an existing AST walker we can reuse?
2. Can we count function-name occurrences pre/post diff in <30 lines of new code?

If yes → `policy.required_calls` (Phase 7.3) is a 1-day add. If no → spawn task **R1.c.i: build minimal AST call-counter**.

**R1.c verdict (verified 2026-04-27)**: AST machinery is **ready**. `sandbox/ast_policy.py` already has `ast.parse`, `ast.walk`, `ast.Call` detection, `_dotted_name` for dotted attribute resolution, and the `FORBIDDEN_CALLS` deny-list pattern that we invert for `required_calls`. Estimated effort: ~2 hours, not 1 day.

### Acceptance R1
- Baseline doc committed; comparison doc patched with corrections (or annotated "verified").
- A one-paragraph note in the plan says "AST machinery: ready / needs walker."

---

## R2 — Re-ground the original plan in real code paths

Edit `RLix-Adoption-Plan.md` in place. Specific corrections:

| Plan section | Wrong reference | Correct reference |
|---|---|---|
| Phase 1.4 (telemetry hook) | "controller/continuous.py" | `controller/engine.py:run_experiment` (the loop body, ~line 180) |
| Phase 2.3 (executor wiring) | "Executor method `execute_with_cancel`" | Add to `controller/executor.py::Executor` Protocol; implement in `TargetExecutor`, `DiffExecutor`, `HybridExecutor` (latter two in `controller/diff_executor.py`) |
| Phase 2.4 (cancelled status) | "RunOutcome / Outcome" | `target/interface.py::RunOutcome` *and* `controller/executor.py::Outcome`; `decision="cancelled"` flows through `controller/types.py::LoopResult` history |
| Phase 3.2 (timeline hooks) | "controller/engine.py" | Correct already; but enumerate the slice points: lines around `executor.execute(proposal, run_dir)`, `_save_version`, `emit(...)` |
| Phase 4.3 (parallel engine) | "new controller/parallel_engine.py mirrors run_experiment" | Confirmed correct shape; explicitly extract the per-iteration body of `run_experiment` (lines ~192–356) into `_run_one_iteration(...)` first so both serial and parallel engines call it. Cuts duplication. |
| Phase 7.3 (diff guardrail) | "sandbox/validator.py gets a new check" | **Correction-of-correction**: `sandbox/validator.py` *is* on the canonical path — `controller/diff_executor.py:110` calls `validate_diff(diff)` from it. Add a new `validate_required_calls(pre_source, post_source, required)` to `sandbox/validator.py` reusing `sandbox/ast_policy.py::_dotted_name`. Hook it from `DiffExecutor.execute` after the existing `validate_diff` call. |
| All phases | Policy reuse | New helpers (`policy/ppo.py`, `policy/gae.py`, `policy/sdpo.py`, `policy/llm_context.py`) already exist — Phase 7.4's batch-diversity ranking can reuse `policy/llm_context.py` for trimming |

### Acceptance R2
- Diff of `RLix-Adoption-Plan.md` shows each correction.
- Every "hooks into" line in the plan resolves to a file that exists in `git ls-files`.

---

## R3 — Close discovered correctness gaps

### R3.a Reward ordering under concurrency (folds into Phase 4)

**Problem**: with k concurrent in-flight iterations, `Learnable.record_reward` may be called for iter 5 before iter 3 — making the policy's reward sequence non-stationary in trial-time.

**Solution**: `controller/parallel_engine.py` maintains an ordered `pending_rewards: dict[int, float]` keyed by `iter_idx`. After each completion, drain the dict in ascending order: while `next_unflushed in pending_rewards: policy.record_reward(pending_rewards.pop(next_unflushed)); next_unflushed += 1`.

**Test**: stub policy whose `record_reward` records call order; submit 4 iters with completion order [3, 1, 4, 2]; assert `record_reward` was called in order 1, 2, 3, 4.

### R3.b Comparability mode for parallelism (folds into Phase 4)

**Problem**: `ComparabilityPolicy` strict mode currently checks `run_budget_s` against wall time. Under `max_concurrency=k`, k trials share wall time; the budget check becomes meaningless.

**Solution**:
- Add `budget_mode: Literal["fixed_wallclock", "parallel_wallclock"]` to `ComparabilityConfig` (`config.py:78`).
- For `parallel_wallclock`: budget is per-trial `elapsed_s` from the Outcome, not loop wall time. Add `max_concurrency_at_submission: int` to the ledger row.
- `check_comparability` rejects mismatched modes by default.

**Test**: a parallel run with `parallel_wallclock` records the right ledger column; a strict campaign mixing both modes fails fast.

### R3.c Threads-vs-asyncio spike (precedes Phase 4)

**One-day spike**, time-boxed:
- Build a fake `BasilicaTarget` whose `run()` sleeps 30 s with periodic 1 s "polls" (mimics today's I/O pattern).
- Run 8 concurrent instances under `ThreadPoolExecutor(max_workers=8)`. Measure wall time, CPU%, scheduling jitter.
- Acceptance: wall time ≤ 1.3× single-iter time. If false, escalate to asyncio + `aiohttp` for `BasilicaTarget` only (other targets stay synchronous).

**Output**: `docs/research/concurrency-spike.md` with verdict + numbers. Decision frozen for Phase 4.

### R3.d Refactor `_BOOTSTRAP` to `string.Template` (precedes Phase 1)

**Problem**: `target/basilica.py:_BOOTSTRAP` uses `.format(port=, cmd=)` and escapes JSON braces with `{{` / `}}`. Adding `/progress` and `/control` handler bodies (Phase 1.3, 2.3) means more dict literals and more brace escaping; high risk of silent bugs.

**Solution**: rewrite `_BOOTSTRAP` to use `string.Template` with `$port` and `$cmd`. Brace escaping disappears entirely.

**Test**: `BasilicaTarget._build_bootstrap_cmd(["python3","train.py"])` produces a script that round-trips through `compile(..., '<bootstrap>', 'exec')` without `SyntaxError`.

### R3.e Positive-presence guardrail (folds into Phase 6)

**Problem**: Phase 7.3's "do not strip `emit_progress`" only fires when calls already exist. A `train.py` that never had them gets a silent zero-progress trial, and intra-iteration cancel can never fire.

**Solution**: in `config_validate.py::validate_runtime` (Phase 6), if `controller.intra_iteration_cancel.enabled` is true, parse the AST of `policy.mutable_file` and require at least one `Call(func=Name("emit_progress"))`. Else `severity=error` with a clear remedy.

**Test**: a config with cancel enabled + a `train.py` without progress calls fails `validate_runtime`; the same config with one call passes.

### Acceptance R3
- R3.c spike report committed.
- R3.d landed as a no-behavior-change refactor with passing test before any Phase 1 work.
- R3.a, R3.b, R3.e marked "to be implemented in Phase 4 / 4 / 6 respectively" in the plan with section pointers.

---

## R4 — Calibration

### R4.a Estimates → t-shirt sizes

Replace day estimates in the plan with `S` (≤1 day), `M` (1–3 days), `L` (3–5 days). Require every phase to ship in PRs each ≤ 1 day. Track velocity in `docs/research/velocity.md` (1 line per merged PR: phase, hours, surprises). After 3 merged PRs, recalibrate L estimates.

### R4.b Prompt eval harness

**Problem**: Phase 7 acceptance leans on "the LLM will do X after we update the system prompt." Hope, not measurement.

**Solution**: tiny harness `tests/eval/prompt_eval.py`:
- Fixed inputs: 5 `(train.py, history snapshot, program.md)` triples committed under `tests/eval/fixtures/`.
- Per prompt change, run the policy against each fixture, snapshot the response, assert structural properties:
  - **emit_progress survival**: post-diff source still contains `emit_progress(`.
  - **batch diversity**: `propose_batch(state, 4)` returns 4 distinct `learning_rate` values.
  - **cancel awareness**: after 3 fixtures with `status="cancelled"`, the assistant message contains "cancel" or "early stop" or "plateau".
- Run as part of CI **only** when `policy/_prompt_fragments.py` or example `program.md` changes. Skip otherwise (cost).
- Use a recorded-response cache (VCR-style) so CI doesn't pay per run.

**Acceptance**: harness passes on a stub LLM that returns canned responses; one real-LLM run is captured to the fixture cache.

### Acceptance R4
- Plan reflects t-shirt sizes.
- Harness lands and runs against stub responses.

---

## Then: continue with the original plan

With R0–R4 complete, execute **Phase 1 → Phase 6** from `RLix-Adoption-Plan.md` with the R2 corrections applied. Phase 7 work is interleaved as already specified (7.1+7.2+7.5 with Phase 1; 7.3 with Phase 2 + R3.e in Phase 6; 7.4 with Phase 4).

The execution order, fully unrolled, becomes:

```
R0  Restore target/                        S, blocker
R1  Baseline + verifications               S
R2  Re-ground plan                         S
R3.d Bootstrap.Template refactor           S, before Phase 1
R3.c Concurrency spike                     S, before Phase 4
R4   Calibration + prompt eval harness     M
─── execution begins ───
Phase 6 Config validation (incl. R3.e)     S
Phase 1 ProgressReport (+ 7.1, 7.2, 7.5)   M
Phase 2 Cancellation (+ 7.3)               M
Phase 3 Timeline export                    S
Phase 4 Parallel engine (+ R3.a, R3.b, 7.4) L
Phase 5 Multi-LoRA target                  M, deferred until needed
```

## Definition of remediation done

- `docs/research/baseline-2026-04-27.md` exists with green or annotated red.
- `docs/research/concurrency-spike.md` exists with a verdict.
- `RLix-Adoption-Plan.md` diff shows R2 corrections.
- All "hooks into" file references in the plan resolve in `git ls-files`.
- `tests/eval/prompt_eval.py` runs with stub responses.
- Tasks R0..R4 in TaskList are `completed`.

Only when all six are true do we open the first PR for Phase 6.
