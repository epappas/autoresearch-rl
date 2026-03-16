# G.2 Wire directional feedback into distillation sink

## Context

**Paper ref:** SDFT Section 3 -- "convert hints into teacher logits for distillation."

`eval/judge.py` produces `hint` strings that are logged but never consumed by any training component.

**Dependencies:** G.1 (SDFT module), F.2 (SDPO loss)

## Your Task

1. Create `src/autoresearch_rl/distillation/sink.py`:
   - `DistillationSink` class that:
     - Accepts directional hints and teacher signals from judge output
     - Formats them for SDFT-style updates
     - Buffers samples until batch is ready
     - Triggers distillation update when batch is full

2. Wire into the loop controller:
   - After each judge evaluation, pass hint and signals to the sink
   - Sink triggers SDFT update when enough samples are collected

3. Write tests
4. Run `PYTHONPATH=src pytest -q` to verify
5. Commit

## Files to create

- `src/autoresearch_rl/distillation/sink.py`
- `tests/test_distillation_sink.py`

## Acceptance Criteria

- Sink consumes judge hints and teacher signals
- Buffering with configurable batch size
- Integration with SDFT loss computation
- All tests pass

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
