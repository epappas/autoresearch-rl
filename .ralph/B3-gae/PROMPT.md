# B.3 Implement GAE (Generalized Advantage Estimation)

## Context

**Paper ref:** AutoResearch-RL Section 2 -- GAE for variance-reduced advantage estimation.

Current advantage in `learned.py` is simply `reward - running_mean`. No temporal structure, no value function V(s), no lambda parameter.

GAE formula: `A_t = sum_{l=0}^{T} (gamma * lambda)^l * delta_{t+l}`
where `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`

**Dependencies:** B.1 (MDP primitives), B.2 (trajectory buffer)

## Your Task

1. Create `src/autoresearch_rl/policy/gae.py` with:

   ```python
   def compute_gae(
       rewards: list[float],
       values: list[float],
       next_value: float,
       gamma: float = 0.99,
       lam: float = 0.95,
   ) -> list[float]:
       """Compute Generalized Advantage Estimation.

       Args:
           rewards: rewards at each timestep [r_0, r_1, ..., r_{T-1}]
           values: value estimates at each timestep [V(s_0), V(s_1), ..., V(s_{T-1})]
           next_value: value estimate of the terminal state V(s_T)
           gamma: discount factor
           lam: GAE lambda parameter

       Returns:
           advantages: GAE advantages [A_0, A_1, ..., A_{T-1}]
       """
   ```

2. Also implement `compute_returns`:
   ```python
   def compute_returns(advantages: list[float], values: list[float]) -> list[float]:
       """Compute returns from advantages and values: R_t = A_t + V(s_t)"""
   ```

3. Write thorough tests in `tests/test_gae.py`:
   - Test with known inputs and hand-computed expected outputs
   - Test edge cases (single timestep, zero gamma, zero lambda)
   - Test that gamma=1, lambda=1 reduces to standard returns

4. Run `PYTHONPATH=src pytest -q` to verify
5. Run `ruff check src/autoresearch_rl/policy/gae.py` to verify
6. Commit

## Files to create

- `src/autoresearch_rl/policy/gae.py` -- GAE implementation
- `tests/test_gae.py` -- unit tests

## Acceptance Criteria

- Correct GAE computation matching the formula
- `compute_returns` utility
- All tests pass with hand-verified expected values
- Lint passes

## Progress Report Format

APPEND to .ralph/B3-gae/progress.md (never replace, always append):

```
## [Date/Time] - B.3

- What was implemented
- Files changed
```

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.
Tests must verify with hand-computed expected values, NOT mocks.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
