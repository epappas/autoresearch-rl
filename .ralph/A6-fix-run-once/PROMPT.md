# A.6 Validate config in run_once.py

## Context

`scripts/run_once.py` calls `yaml.safe_load()` directly, bypassing `RunConfig` validation that the CLI uses. This can silently run with missing/invalid fields.

Current implementation in `scripts/run_once.py`:
```python
cfg = yaml.safe_load(open(config, "r", encoding="utf-8"))
iters = iterations or int(cfg.get("controller", {}).get("max_iterations", 1))
# ... manual extraction of nested fields ...
```

The CLI (`cli.py`) does it properly:
```python
raw = yaml.safe_load(f.read())
run_cfg = RunConfig.model_validate(raw)
```

## Your Task

1. Open `scripts/run_once.py`
2. Replace the manual YAML dict access with proper `RunConfig.model_validate()` usage
3. Import `RunConfig` from `autoresearch_rl.config`
4. Use the validated config object's typed fields instead of raw dict access
5. The script should still accept the same CLI arguments (`--config`, `--iterations`)
6. Run `PYTHONPATH=src pytest -q` to verify all tests pass
7. Run `ruff check scripts/run_once.py` to verify lint passes
8. Commit the fix

## Files to modify

- `scripts/run_once.py` -- use `RunConfig.model_validate()` instead of raw YAML dict access

## Acceptance Criteria

- Config is validated through `RunConfig.model_validate()` before use
- No raw `cfg.get()` chains for values that exist in `RunConfig`
- Script still works with the same CLI interface
- All tests pass

## Progress Report Format

APPEND to .ralph/A6-fix-run-once/progress.md (never replace, always append):

```
## [Date/Time] - A.6

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
