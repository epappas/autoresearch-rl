# D.1 Implement checkpoint management for loop state

## Context

**Paper ref:** AutoResearch-RL Section 3 -- "policy checkpointing to enable rollback."

If the process dies mid-run, all progress is lost. `best_value` is only in memory. No save/restore for: episode number, best model state, telemetry refs, convergence history.

**Dependencies:** A.1 (unified loops)

## Your Task

1. Create `src/autoresearch_rl/checkpoint.py` with:

   ```python
   @dataclass
   class LoopCheckpoint:
       episode_id: str
       iteration: int
       best_score: float
       best_value: float | None
       no_improve_streak: int
       history: list[dict]
       recent_statuses: list[str]
       policy_state: dict  # serializable policy state
       elapsed_s: float
       timestamp: float

   def save_checkpoint(path: str, checkpoint: LoopCheckpoint) -> None:
       """Save loop state to JSON file atomically."""

   def load_checkpoint(path: str) -> LoopCheckpoint | None:
       """Load checkpoint from file. Returns None if not found."""
   ```

2. Implement atomic writes (write to temp file, then rename) to prevent corruption from crashes

3. Wire into the unified loop controller:
   - Save checkpoint after each iteration
   - On startup, check for existing checkpoint and resume if found
   - Checkpoint path should be configurable via `ControllerConfig`

4. Add `checkpoint_path: str | None = None` to `ControllerConfig` in `config.py`

5. Write tests in `tests/test_checkpoint.py`
6. Run `PYTHONPATH=src pytest -q` to verify
7. Commit

## Files to create/modify

- New: `src/autoresearch_rl/checkpoint.py` -- checkpoint save/load
- `src/autoresearch_rl/config.py` -- add checkpoint_path config
- Controller files -- wire checkpoint save/load
- New: `tests/test_checkpoint.py` -- unit tests

## Acceptance Criteria

- Atomic checkpoint writes (temp + rename)
- Full loop state serialization/deserialization
- Resume from checkpoint on startup
- Configurable checkpoint path
- All tests pass

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
