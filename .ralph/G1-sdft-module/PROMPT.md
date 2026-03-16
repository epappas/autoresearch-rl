# G.1 Implement SDFT token-level distillation module

## Context

**Paper ref:** SDFT -- softmax divergence fine-tuning with token-level teacher distribution matching.

SDFT loss: `L_SDFT = sum_t softmax(z_teacher/T) * log(softmax(z_teacher/T) / softmax(z_student/T))`

No SDFT implementation exists. The directional branch (`hint` in judge output) is captured but never used for distillation.

**Dependencies:** F.1 (teacher/student framework)

## Your Task

1. Create `src/autoresearch_rl/distillation/sdft.py`:
   - `softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray`
   - `compute_sdft_loss(teacher_logits: np.ndarray, student_logits: np.ndarray, temperature: float = 2.0) -> float` -- forward KL divergence with temperature
   - `SDFTConfig` dataclass: temperature, top_k, confidence_threshold
   - `apply_top_k_filter(logits: np.ndarray, k: int) -> np.ndarray` -- keep only top-K teacher logits for cost control
   - `should_distill(confidence: float, threshold: float) -> bool` -- confidence gating

2. Create `src/autoresearch_rl/distillation/__init__.py`

3. Write tests in `tests/test_sdft.py` with hand-verified expected values

4. Run `PYTHONPATH=src pytest -q` to verify
5. Commit

## Files to create

- `src/autoresearch_rl/distillation/__init__.py`
- `src/autoresearch_rl/distillation/sdft.py`
- `tests/test_sdft.py`

## Acceptance Criteria

- Correct forward KL computation with temperature scaling
- Top-K logit filtering
- Confidence gating
- All tests pass with numerically verified expected values

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
