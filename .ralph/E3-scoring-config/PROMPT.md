# E.3 Make scoring weights configurable

## Context

`src/autoresearch_rl/eval/scoring.py` has hardcoded magic numbers for penalties and bonuses. The `ScoreWeights` dataclass exists but is never exposed in the YAML config. Per SDPO research, scoring should support adaptive weighting.

Current `ScoreWeights`:
```python
@dataclass
class ScoreWeights:
    val_bpb: float = 1.0
    loss: float = 0.15
    fail_penalty: float = 0.8
    timeout_penalty: float = 1.2
    neutral_penalty: float = 0.05
    directional_bonus: float = 0.2
```

Also in `score_from_signals()` there are additional hardcoded values:
- `elif signals.status == "early_stopped": score += 0.4` (not in ScoreWeights)
- `score -= 0.25 * float(signals.eval_score)` (hardcoded 0.25)

## Your Task

1. Open `src/autoresearch_rl/eval/scoring.py`:
   - Add missing weights to `ScoreWeights`: `early_stop_penalty: float = 0.4`, `eval_score_weight: float = 0.25`
   - Replace all hardcoded values in `score_from_signals()` with references to `ScoreWeights` fields
2. Open `src/autoresearch_rl/config.py`:
   - Add a `scoring: ScoreWeights` field to `RunConfig` (or a new `ScoringConfig` Pydantic model that mirrors `ScoreWeights`)
   - Import or re-create the weights as a Pydantic model so YAML config can set them
3. Wire the scoring weights through the controller so they reach `score_from_signals()`
4. Run `PYTHONPATH=src pytest -q` to verify all tests pass
5. Run `ruff check src/autoresearch_rl/eval/scoring.py src/autoresearch_rl/config.py` to verify lint passes
6. Commit the changes

## Files to modify

- `src/autoresearch_rl/eval/scoring.py` -- add missing weight fields, remove hardcoded values
- `src/autoresearch_rl/config.py` -- expose scoring weights in config

## Acceptance Criteria

- ALL numeric constants in `score_from_signals()` come from `ScoreWeights` fields
- Scoring weights are configurable via YAML config
- Default behavior is unchanged (same default values)
- All tests pass
- Lint passes

## Progress Report Format

APPEND to .ralph/E3-scoring-config/progress.md (never replace, always append):

```
## [Date/Time] - E.3

- What was implemented
- Files changed
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
```

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.
ALL commits must pass quality checks.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
