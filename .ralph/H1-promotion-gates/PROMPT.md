# H.1 Implement policy promotion gates

## Context

**Paper ref:** AutoResearch-RL Section 3 -- "commit/revert to best-known config."
**Paper ref:** SDPO Section 5 -- "gradual policy promotion with rollback checkpoints."

No promotion gate exists. Keep/discard decisions exist but only affect artifact versioning, not the active policy.

**Dependencies:** D.1 (checkpoints), F.1 (teacher snapshots), E.2 (forecaster)

## Your Task

1. Implement promotion logic in the loop controller:
   - After N consecutive improvements (configurable), promote candidate policy to active
   - On sustained degradation (detected by forecaster), rollback to last promoted checkpoint
   - Track promotion history

2. Add config:
   ```python
   promotion_threshold: int = 3  # consecutive improvements needed
   degradation_window: int = 10  # window to detect degradation
   ```

3. Write tests
4. Run `PYTHONPATH=src pytest -q` to verify
5. Commit

## Files to modify

- Controller files -- add promotion logic
- `src/autoresearch_rl/checkpoint.py` -- add promotion tracking
- `src/autoresearch_rl/config.py` -- add promotion config

## Acceptance Criteria

- Promotion after N consecutive improvements
- Rollback on sustained degradation
- Promotion history tracked
- All tests pass

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
