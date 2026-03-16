# D.2 Implement graceful shutdown with signal handling

## Context

The `while True` loop has no signal handling. SIGTERM/SIGINT can corrupt mid-write files.

**Dependencies:** A.1 (unified loops), D.1 (checkpoints)

## Your Task

1. Register SIGTERM/SIGINT handlers that set a `shutdown_requested` flag
2. Loop checks flag between iterations
3. On shutdown: finish current iteration, persist checkpoint, flush telemetry
4. Wire into the unified loop controller and CLI

## Files to modify

- Unified loop controller -- add signal handling
- `src/autoresearch_rl/cli.py` -- register handlers before loop start

## Acceptance Criteria

- SIGTERM and SIGINT are handled gracefully
- Current iteration completes before shutdown
- Checkpoint is saved on shutdown
- No file corruption from mid-write interruption
- All tests pass

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
