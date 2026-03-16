# A.5 Cache hardware_fingerprint()

## Context

In `src/autoresearch_rl/telemetry/comparability.py`, the `hardware_fingerprint()` function shells out to `nvidia-smi` on every call. This function is called per-iteration in the loop. Hardware does not change during process lifetime, so this is wasteful.

Current implementation:
```python
def hardware_fingerprint() -> str:
    parts = [
        platform.system(),
        platform.release(),
        platform.machine(),
        platform.python_version(),
    ]
    try:
        cp = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False)
        if cp.returncode == 0 and cp.stdout.strip():
            parts.append(cp.stdout.strip().replace("\n", ","))
    except Exception:
        pass
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
```

## Your Task

1. Open `src/autoresearch_rl/telemetry/comparability.py`
2. Add a module-level cache variable (e.g. `_hw_fingerprint_cache: str | None = None`)
3. Modify `hardware_fingerprint()` to compute once on first call and return the cached value on subsequent calls
4. Use `functools.lru_cache` or a simple module-level variable -- keep it simple
5. Run `PYTHONPATH=src pytest -q` to verify all tests pass
6. Run `ruff check src/autoresearch_rl/telemetry/comparability.py` to verify lint passes
7. Commit the fix

## Files to modify

- `src/autoresearch_rl/telemetry/comparability.py` -- cache the result of `hardware_fingerprint()`

## Acceptance Criteria

- `hardware_fingerprint()` only shells out to `nvidia-smi` once per process lifetime
- Subsequent calls return the cached value
- All tests pass
- Lint passes

## Progress Report Format

APPEND to .ralph/A5-cache-hw-fp/progress.md (never replace, always append):

```
## [Date/Time] - A.5

- What was implemented
- Files changed
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
```

## Code Quality

KEEP IT SIMPLE. Use `functools.lru_cache(maxsize=1)` or a module-level variable.
THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.

## Stop Condition

After completing this fix and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
