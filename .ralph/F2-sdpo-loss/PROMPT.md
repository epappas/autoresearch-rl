# F.2 Implement SDPO loss function

## Context

**Paper ref:** SDPO -- `L_SDPO = L_RL + alpha_t * D_KL(pi_teacher || pi_student)`

No SDPO loss exists. `telemetry/distill.py` appends JSONL records but performs no distillation computation.

**Dependencies:** F.1 (teacher snapshots), C.1 (PPO)

## Your Task

1. Create `src/autoresearch_rl/policy/sdpo.py`:
   - `compute_kl_divergence(teacher_probs: np.ndarray, student_probs: np.ndarray) -> float`
   - `compute_sdpo_loss(ppo_loss: float, kl_div: float, alpha: float) -> float` -- L_SDPO = L_RL + alpha * KL
   - `compute_adaptive_alpha(prev_reward: float, target_reward: float) -> float` -- alpha_t = min(1, R_prev / R_target)

2. Integrate into PPO training loop:
   - Load teacher weights from previous snapshot
   - Compute KL divergence between teacher and student distributions
   - Add SDPO term to PPO loss
   - Use adaptive alpha weighting

3. Write tests
4. Run `PYTHONPATH=src pytest -q` to verify
5. Commit

## Files to create

- `src/autoresearch_rl/policy/sdpo.py`
- `tests/test_sdpo.py`

## Acceptance Criteria

- Correct KL divergence computation
- SDPO loss combines PPO loss with KL term
- Adaptive alpha weighting
- All tests pass

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
