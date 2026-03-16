# A.1 Unify loop controllers

## Context

Two independent loop systems exist with duplicated logic:
- `controller/continuous.py`: CLI-driven, param-based proposals via `ParamPolicy`, has its own `LoopResult` dataclass, `_current_commit()` helper, stop-guard logic
- `controller/loop.py`: Legacy, diff-based proposals via `ProposalPolicy`/`LearnedDiffPolicy`, has its own `LoopResult` (different fields), `_current_commit_or_local()` helper, stop-guard logic, contract/sandbox integration

Both have: separate `LoopResult` dataclasses, separate commit helpers, separate stop-guard implementations, separate telemetry wiring.

## Your Task

1. **Merge `LoopResult`**: Create a single `LoopResult` dataclass that covers both use cases. It should have at minimum: `best_score: float`, `best_value: float | None`, `iterations: int`. Place it in a shared location (e.g., `controller/types.py` or at the top of the unified controller).

2. **Extract shared helpers**: Create `controller/helpers.py` with:
   - `current_commit(cwd: str | None = None) -> str` -- unified commit lookup (merge of `_current_commit()` and `_current_commit_or_local()`)
   - `check_stop_guards(...)` -- unified stop-guard evaluation (wall time, no-improve streak, failure rate)

3. **Unify the loop**: Create a single `run_loop()` function (or refactor the existing ones) that supports both modes:
   - **Continuous mode** (param-based): uses `TargetAdapter` + `ParamPolicy`, the current CLI path
   - **Legacy mode** (diff-based): uses contract/sandbox + `ProposalPolicy`, the legacy path
   - Configuration determines which mode is active
   - Both modes share: stop guards, telemetry emission, result tracking, commit lookup

4. **Update callers**:
   - `cli.py`: update to use the unified loop
   - `scripts/run_once.py`: update to use the unified loop

5. **Tests**: Run `PYTHONPATH=src pytest -q` to verify all tests pass

6. **Lint**: Run `ruff check src/autoresearch_rl/controller/` to verify

7. Commit the changes

## Important Design Constraints

- The unified loop should NOT be a God function. Extract shared logic into small helpers.
- Keep both modes working -- do not remove legacy mode functionality.
- The continuous mode is the primary path (used by CLI).
- Functions should be < 50 lines each.
- Ruff line length: 100 chars.

## Files to modify

- `src/autoresearch_rl/controller/continuous.py` -- refactor/merge
- `src/autoresearch_rl/controller/loop.py` -- refactor/merge
- `src/autoresearch_rl/controller/__init__.py` -- update exports if needed
- `src/autoresearch_rl/cli.py` -- update to use unified controller
- `scripts/run_once.py` -- update to use unified controller
- New: `src/autoresearch_rl/controller/helpers.py` (shared utilities)
- New: `src/autoresearch_rl/controller/types.py` (shared types)

## Acceptance Criteria

- Single `LoopResult` dataclass used everywhere
- Single `current_commit()` helper
- Stop-guard logic is shared, not duplicated
- Both continuous and legacy modes still work
- CLI still works: `autoresearch-rl --config configs/example.yaml`
- All tests pass
- Lint passes

## Progress Report Format

APPEND to .ralph/A1-unify-loops/progress.md (never replace, always append):

```
## [Date/Time] - A.1

- What was implemented
- Files changed
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
```

## Code Quality

IT IS IMPORTANT TO ADHERE TO THE GOOD SOFTWARE QUALITY PRINCIPLES SUCH AS DRY, SOLID AND KISS.
THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.
Functions must be < 50 lines. Favour modular code.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
