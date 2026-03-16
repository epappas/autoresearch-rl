# H.2 Implement experiment tracking interface

## Context

**Paper ref:** AutoResearch-RL Section 5 -- "replayability and auditability."

All experiment data goes to local files with no structured query, comparison, or visualization capability.

**Dependencies:** A.1 (unified loops), D.3 (telemetry rotation)

## Your Task

1. Create `src/autoresearch_rl/tracking.py`:
   - `ExperimentTracker` Protocol:
     ```python
     class ExperimentTracker(Protocol):
         def log_params(self, params: dict[str, object]) -> None: ...
         def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...
         def log_artifact(self, path: str, name: str) -> None: ...
         def set_status(self, status: str) -> None: ...
     ```
   - `LocalFileTracker` implementation (writes to structured directory)
   - Hook into episode lifecycle: parameter logging, metric tracking, artifact storage

2. Wire into the loop controller

3. Write tests
4. Run `PYTHONPATH=src pytest -q` to verify
5. Commit

## Files to create

- `src/autoresearch_rl/tracking.py`
- `tests/test_tracking.py`

## Acceptance Criteria

- `ExperimentTracker` protocol defined
- `LocalFileTracker` implementation
- Integration with loop controller
- All tests pass

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
