# E.1 Implement real multi-judge diversity

## Context

**Paper ref:** AutoResearch-RL Section 4 -- "Multi-Judge Evaluation with diverse perspectives."

In `src/autoresearch_rl/eval/judge.py`, the `judge_next_state()` function calls `_heuristic_vote()` N times with identical inputs, producing identical votes every time. This is not majority voting -- it's vote duplication. The function produces the illusion of diversity but always returns the same result.

Current implementation:
```python
def judge_next_state(prev_status, next_status, next_stdout, next_stderr, vote_count=3):
    votes = [
        _heuristic_vote(prev_status=prev_status, next_status=next_status, next_stdout=next_stdout, next_stderr=next_stderr)
        for _ in range(max(1, vote_count))
    ]
    # ... all votes are identical since same function, same inputs
```

## Your Task

1. Open `src/autoresearch_rl/eval/judge.py`
2. Create multiple distinct judge strategies that can genuinely disagree:
   - `_status_judge()`: judges based purely on status transitions (failed->ok = +1, ok->failed = -1, etc.)
   - `_metric_judge()`: judges based on metric keywords and values found in stdout (looks for val_bpb improvements, loss decreases)
   - `_log_quality_judge()`: judges based on log quality signals (error keywords, warning counts, traceback presence, output completeness)
3. Update `judge_next_state()` to use these three distinct strategies instead of repeating the same heuristic
4. Each judge function should have the same signature: `(prev_status, next_status, next_stdout, next_stderr) -> JudgeVote`
5. Keep `majority_vote()` and `JudgeVote`/`JudgeResult` dataclasses unchanged
6. Run `PYTHONPATH=src pytest -q` to verify all tests pass
7. Run `ruff check src/autoresearch_rl/eval/judge.py` to verify lint passes
8. Commit the changes

## Files to modify

- `src/autoresearch_rl/eval/judge.py` -- replace single heuristic with three distinct judge strategies

## Design Constraints

- Each judge must be a pure function (no external state, no LLM calls)
- Each judge must be deterministic given the same inputs
- Judges CAN disagree -- that's the whole point of diversity
- Keep the public API (`judge_next_state`, `JudgeResult`, `JudgeVote`, `majority_vote`) stable
- Lines must not exceed 100 chars (ruff config)

## Acceptance Criteria

- At least 3 distinct judge strategies that can produce different votes for the same inputs
- `judge_next_state()` aggregates votes from all three judges
- All tests pass
- Lint passes

## Progress Report Format

APPEND to .ralph/E1-diverse-judges/progress.md (never replace, always append):

```
## [Date/Time] - E.1

- What was implemented
- Files changed
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
```

## Code Quality

IT IS IMPORTANT TO ADHERE TO THE GOOD SOFTWARE QUALITY PRINCIPLES SUCH AS DRY, SOLID AND KISS.
THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.
Each judge must contain REAL evaluation logic, not stubs.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
