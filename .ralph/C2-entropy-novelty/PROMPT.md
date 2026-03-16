# C.2 Add entropy regularization and novelty bonus

## Context

**Paper ref:** AutoResearch-RL Section 5 -- entropy bonus H(pi) + novelty based on distance from previously explored states.

No entropy term in the policy objective. No novelty computation.

**Dependencies:** C.1 (PPO network)

## Your Task

1. Open `src/autoresearch_rl/policy/ppo.py` (from C.1):
   - Ensure entropy bonus is properly integrated into PPO loss: `L = L^CLIP + c1 * L^VF - c2 * H(pi)`
   - The entropy should be computed from the action distribution

2. Add novelty bonus computation:
   - Create a function that computes novelty based on distance from previously explored states
   - Use state feature vectors and compute distance to nearest neighbor in the history
   - Novelty bonus = 1 / (1 + min_distance_to_history)
   - Add novelty bonus to the reward signal before advantage computation

3. Add novelty config to `PPOConfig`:
   ```python
   novelty_coef: float = 0.1
   novelty_k: int = 5  # number of nearest neighbors to consider
   ```

4. Write tests in `tests/test_ppo.py` (extend existing)
5. Run `PYTHONPATH=src pytest -q` to verify
6. Commit

## Files to modify

- `src/autoresearch_rl/policy/ppo.py` -- add entropy and novelty
- `tests/test_ppo.py` -- extend tests

## Acceptance Criteria

- Entropy computed from action distribution
- Novelty bonus based on state history distance
- Both integrated into the PPO training loop
- All tests pass

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
