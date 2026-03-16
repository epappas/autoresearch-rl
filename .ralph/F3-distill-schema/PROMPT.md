# F.3 Add distillation sample schema validation

## Context

**Paper ref:** SDPO Section 4 -- distillation records must include KL-divergence bounds and preference pair data.

`src/autoresearch_rl/telemetry/distill.py` is a bare JSONL appender with no schema validation:

```python
def append_distill_sample(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
```

Any dict can be written, and there's no guarantee the records are well-formed.

## Your Task

1. Open `src/autoresearch_rl/telemetry/distill.py`
2. Define a Pydantic model `DistillSample` with typed fields that capture what the research requires:
   - `episode_id: str` -- episode identifier
   - `iteration: int` -- iteration index
   - `status: str` -- trial status
   - `diff: str` -- the proposed diff
   - `eval_score: float` -- evaluative signal from judge
   - `hint: str` -- directional hint
   - `reward: float | None = None` -- computed reward (optional, may be added later)
   - `kl_divergence: float | None = None` -- KL divergence bound (optional, added during SDPO)
   - `teacher_version: str | None = None` -- teacher checkpoint version (optional, added during SDPO)
   - `timestamp: float` -- unix timestamp of sample creation (auto-set via `Field(default_factory=time.time)`)
3. Update `append_distill_sample()` to accept a `DistillSample` (or validate a dict against it)
4. Update all callers of `append_distill_sample()` in the codebase to pass proper data. The caller is in `controller/loop.py` line 253.
5. Run `PYTHONPATH=src pytest -q` to verify all tests pass
6. Run `ruff check src/autoresearch_rl/telemetry/distill.py` to verify lint passes
7. Commit the changes

## Files to modify

- `src/autoresearch_rl/telemetry/distill.py` -- add Pydantic schema, validate on write
- `src/autoresearch_rl/controller/loop.py` -- update caller to pass proper `DistillSample`

## Acceptance Criteria

- `DistillSample` Pydantic model with typed fields
- `append_distill_sample()` validates input against the schema before writing
- Invalid payloads raise a validation error
- All tests pass
- Lint passes

## Progress Report Format

APPEND to .ralph/F3-distill-schema/progress.md (never replace, always append):

```
## [Date/Time] - F.3

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

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
