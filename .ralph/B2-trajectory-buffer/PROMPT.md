# B.2 Implement trajectory buffer

## Context

**Paper ref:** AutoResearch-RL Section 2 -- "trajectories tau = (s_0, a_0, r_0, ..., s_T)"

No episode storage exists. Policy updates process one sample at a time. No batching for multi-epoch PPO updates.

**Dependencies:** B.1 (MDP primitives)

## Your Task

1. Create `src/autoresearch_rl/trajectory.py` with:

   ```python
   @dataclass
   class Transition:
       state: State
       action: Action
       reward: Reward
       next_state: State
       log_prob: float
       value_estimate: float

   class TrajectoryBuffer:
       """Typed trajectory buffer for episode storage."""

       def __init__(self, max_size: int = 1024):
           ...

       def add(self, transition: Transition) -> None:
           """Add a transition to the buffer."""

       def get_batch(self, batch_size: int) -> list[Transition]:
           """Get a batch of transitions for training."""

       def get_episode(self, start: int = 0, end: int | None = None) -> list[Transition]:
           """Get a contiguous episode window."""

       def clear(self) -> None:
           """Clear the buffer."""

       def __len__(self) -> int:
           ...
   ```

2. The buffer should support:
   - Fixed-size circular storage (oldest transitions dropped when full)
   - Batch retrieval for PPO training
   - Episode windowing for advantage computation
   - All entries are typed `Transition` objects using MDP types from B.1

3. Run `PYTHONPATH=src pytest -q` to verify
4. Run `ruff check src/autoresearch_rl/trajectory.py` to verify
5. Write tests in `tests/test_trajectory.py`
6. Commit

## Files to create

- `src/autoresearch_rl/trajectory.py` -- trajectory buffer implementation
- `tests/test_trajectory.py` -- unit tests

## Acceptance Criteria

- `Transition` dataclass using MDP types
- `TrajectoryBuffer` with add, batch, episode, clear, len operations
- Circular buffer behavior (fixed max size)
- All tests pass
- Lint passes

## Progress Report Format

APPEND to .ralph/B2-trajectory-buffer/progress.md (never replace, always append):

```
## [Date/Time] - B.2

- What was implemented
- Files changed
```

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.
Tests must test REAL functionality. ALL commits must pass quality checks.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
