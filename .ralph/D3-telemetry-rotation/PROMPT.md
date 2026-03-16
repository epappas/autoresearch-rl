# D.3 Add telemetry rotation

## Context

Append-only JSONL/TSV files grow unbounded in perpetual loops.

**Dependencies:** A.1 (unified loops)

## Your Task

1. Add rotation logic to `src/autoresearch_rl/telemetry/events.py`:
   - Check file size before each write
   - When file exceeds `max_size_bytes`, rotate: rename current to `{name}.1`, shift existing rotations
   - Keep up to `max_rotated` rotated files

2. Add rotation to `src/autoresearch_rl/telemetry/ledger.py` similarly

3. Add config:
   ```python
   class TelemetryConfig(BaseModel):
       # ... existing fields ...
       max_file_size_bytes: int = 50 * 1024 * 1024  # 50MB
       max_rotated_files: int = 5
   ```

4. Write tests in `tests/test_telemetry_rotation.py`
5. Run `PYTHONPATH=src pytest -q` to verify
6. Commit

## Files to modify

- `src/autoresearch_rl/telemetry/events.py` -- add rotation
- `src/autoresearch_rl/telemetry/ledger.py` -- add rotation
- `src/autoresearch_rl/config.py` -- add rotation config
- New: `tests/test_telemetry_rotation.py`

## Acceptance Criteria

- Files rotated when exceeding configured size
- Configurable max rotations
- Default behavior unchanged for small files
- All tests pass

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
