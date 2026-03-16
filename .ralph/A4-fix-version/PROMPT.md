# A.4 Fix version mismatch

## Context

`pyproject.toml` declares version `0.2.0` but `src/autoresearch_rl/__init__.py` declares `__version__ = "0.1.0"`. These must be consistent.

## Your Task

1. Open `src/autoresearch_rl/__init__.py`
2. Change `__version__ = "0.1.0"` to `__version__ = "0.2.0"` to match `pyproject.toml`
3. Run `PYTHONPATH=src pytest -q` to verify all tests pass
4. Run `ruff check src/autoresearch_rl/__init__.py` to verify lint passes
5. Commit the fix with a descriptive message

## Files to modify

- `src/autoresearch_rl/__init__.py` -- update `__version__` to `"0.2.0"`

## Acceptance Criteria

- `__version__` in `__init__.py` matches `version` in `pyproject.toml` (both `0.2.0`)
- All tests pass

## Progress Report Format

APPEND to .ralph/A4-fix-version/progress.md (never replace, always append):

```
## [Date/Time] - A.4

- What was implemented
- Files changed
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
```

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.
ALL commits must pass quality checks.

## Stop Condition

After completing this fix and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
