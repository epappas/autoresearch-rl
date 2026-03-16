# C.1 Implement PPO policy network

## Context

**Paper ref:** AutoResearch-RL Section 2 -- PPO with clipped objective for edit generation.

`LearnedDiffPolicy` is a 5-weight linear scorer over hand-crafted features. This is not a parameterized policy distribution. There is no actor network, no critic network, no action distribution.

**Dependencies:** A.3 (unified policies), B.1 (MDP), B.2 (trajectory), B.3 (GAE)

## Your Task

1. Create `src/autoresearch_rl/policy/ppo.py` with a proper PPO actor-critic:

   **Actor**: maps State -> action distribution (probability over parameter configurations)
   - Input: state features (metric history, budget remaining, iteration, etc.)
   - Output: action probabilities (for discrete param choices) or mean/std (for continuous)
   - Start with a simple MLP (2-3 hidden layers, configurable width)

   **Critic**: maps State -> scalar value estimate V(s)
   - Input: same state features as actor
   - Output: single scalar value

   **PPO Loss**: clipped surrogate + value loss + entropy bonus
   ```
   L = L^CLIP + c1 * L^VF - c2 * H(pi)
   L^CLIP = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
   ```

   **Training loop**:
   - Collect trajectory of N transitions
   - Compute GAE advantages (using gae.py from B.3)
   - Multiple epochs of minibatch updates
   - Clip ratio to [1-epsilon, 1+epsilon]

2. Create `PPOConfig` dataclass:
   ```python
   @dataclass
   class PPOConfig:
       lr: float = 3e-4
       gamma: float = 0.99
       lam: float = 0.95
       epsilon: float = 0.2
       value_coef: float = 0.5
       entropy_coef: float = 0.01
       epochs: int = 4
       batch_size: int = 32
       hidden_dim: int = 64
       n_layers: int = 2
   ```

3. Use pure Python + numpy (no PyTorch/TensorFlow dependency for now -- the policy operates on structured hyperparameter proposals, not raw tensors). If the parameter space is discrete, use softmax policy. If continuous, use Gaussian policy.

4. Write tests in `tests/test_ppo.py`
5. Run `PYTHONPATH=src pytest -q` to verify
6. Commit

## Files to create/modify

- New: `src/autoresearch_rl/policy/ppo.py` -- PPO implementation
- `src/autoresearch_rl/policy/learned.py` -- deprecate or rewire to use PPO
- New: `tests/test_ppo.py` -- unit tests

## Acceptance Criteria

- Actor-Critic architecture with configurable MLP
- PPO clipped surrogate loss
- GAE integration for advantage computation
- Multiple training epochs per update
- Proper log probability computation
- All tests pass

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.
Use numpy for linear algebra. Functions < 50 lines each.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
