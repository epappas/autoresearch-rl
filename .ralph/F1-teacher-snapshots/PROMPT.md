# F.1 Implement policy checkpointing for teacher snapshots

## Context

**Paper ref:** SDPO Section 3 -- "freeze pi_{t-1} as teacher at each iteration."

No mechanism to snapshot a policy version. `LearnedDiffPolicy` writes weights to a single JSON file that is overwritten on each update.

**Dependencies:** C.1 (PPO network), D.1 (checkpoints)

## Your Task

1. After each policy update in PPO, snapshot the current weights as a versioned checkpoint
2. The previous checkpoint serves as teacher for the next SDPO update
3. Implement in the checkpoint module:
   - `save_policy_snapshot(path: str, version: int, weights: dict) -> str` -- returns snapshot path
   - `load_policy_snapshot(path: str, version: int) -> dict | None`
   - `get_latest_snapshot_version(path: str) -> int`

4. Wire into PPO training: after each update, save snapshot

5. Write tests
6. Run `PYTHONPATH=src pytest -q` to verify
7. Commit

## Files to modify

- `src/autoresearch_rl/checkpoint.py` -- add policy snapshot functions
- `src/autoresearch_rl/policy/ppo.py` -- save snapshots after updates

## Acceptance Criteria

- Versioned policy snapshots saved after each update
- Previous version retrievable for teacher role
- All tests pass

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
