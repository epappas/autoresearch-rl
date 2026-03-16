# A.2 Fix duplicate seed field in ControllerConfig

## Context

In `src/autoresearch_rl/config.py`, the `ControllerConfig` Pydantic model declares the field `seed: int | None = None` twice (lines 37 and 42). This is a bug -- Pydantic silently uses the last declaration but having duplicate fields is incorrect and confusing.

## Your Task

1. Open `src/autoresearch_rl/config.py`
2. Remove the duplicate `seed` field declaration on line 42 of `ControllerConfig`, keeping only the one on line 37
3. Run `PYTHONPATH=src pytest -q` to verify all tests pass
4. Run `ruff check src/autoresearch_rl/config.py` to verify lint passes
5. Run `PYTHONPATH=src mypy src/autoresearch_rl/config.py` to verify type checking passes
6. Commit the fix with a descriptive message

## Files to modify

- `src/autoresearch_rl/config.py` -- remove the duplicate `seed` field

## Acceptance Criteria

- Only ONE `seed: int | None = None` field in `ControllerConfig`
- All tests pass
- Lint and type checks pass

## Progress Report Format

APPEND to .ralph/A2-fix-seed/progress.md (never replace, always append):

```
## [Date/Time] - A.2

- What was implemented
- Files changed
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
```

## Code Quality

IT IS IMPORTANT TO ADHERE TO THE GOOD SOFTWARE QUALITY PRINCIPLES SUCH AS DRY, SOLID AND KISS.
THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.
ALL commits must pass quality checks (typecheck, lint, test).

## Stop Condition

After completing this fix and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
