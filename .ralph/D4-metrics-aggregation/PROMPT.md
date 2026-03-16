# D.4 Add metrics aggregation and summary reporting

## Context

No episode-level summaries, no rolling averages, no trend detection across episodes. The research requires "comprehensive telemetry with trend analysis" for the forecasting module.

**Dependencies:** A.1 (unified loops), D.3 (telemetry rotation)

## Your Task

1. Create `src/autoresearch_rl/telemetry/aggregation.py`:
   - `EpisodeStats` dataclass with: mean, median, min, max, stdev, count, trend_slope
   - `compute_episode_stats(values: list[float]) -> EpisodeStats`
   - `compute_rolling_stats(values: list[float], window: int) -> EpisodeStats`
   - `compute_trend_slope(values: list[float]) -> float` -- simple linear regression slope

2. Wire into the loop controller to compute and emit aggregated stats after each iteration

3. Write tests
4. Run `PYTHONPATH=src pytest -q` to verify
5. Commit

## Files to create

- `src/autoresearch_rl/telemetry/aggregation.py`
- `tests/test_aggregation.py`

## Acceptance Criteria

- Correct statistical computations
- Trend slope via linear regression
- All tests pass

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
