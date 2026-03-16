# E.2 Integrate early-stop forecaster into continuous loop

## Context

**Paper ref:** AutoResearch-RL Section 4 -- "power-law forecasting for early abort."

The power-law forecaster exists in `sandbox/runner.py` (`_fit_power_law`, `_forecast_value`) and works in the legacy loop's trial runner. But the continuous loop (`controller/continuous.py`) has no early-stop forecasting.

**Dependencies:** A.1 (unified loops)

## Your Task

1. Extract forecasting from `src/autoresearch_rl/sandbox/runner.py` into a shared module `src/autoresearch_rl/forecasting.py`:
   - `fit_power_law(series: list[float]) -> tuple[float, float, float]` -- returns (a, b, c) coefficients
   - `forecast_value(series: list[float], target_step: int) -> float` -- predict future value
   - `should_early_stop(series: list[float], target: float, confidence: float = 0.95) -> bool`

2. Wire into the continuous loop's target execution for command targets

3. Update `sandbox/runner.py` to import from the shared module instead of using local functions

4. Write tests
5. Run `PYTHONPATH=src pytest -q` to verify
6. Commit

## Files to create/modify

- New: `src/autoresearch_rl/forecasting.py` -- extracted forecasting utilities
- `src/autoresearch_rl/sandbox/runner.py` -- import from shared module
- Controller files -- wire forecasting into continuous loop
- New: `tests/test_forecasting.py`

## Acceptance Criteria

- Forecasting logic extracted into reusable module
- `sandbox/runner.py` uses the shared module
- Continuous loop can early-stop based on forecasts
- All tests pass

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
