# B.1 Define typed Research MDP primitives

## Context

**Paper ref:** AutoResearch-RL Section 2 -- Research MDP with State, Action, Reward, Transition.

States are implicit dicts throughout the codebase. Actions are untyped. Rewards are computed ad-hoc in multiple places. There is no formal MDP structure as described in the research paper.

**Dependencies:** A.1 (unified loops), A.3 (unified policies)

## Your Task

1. Create `src/autoresearch_rl/mdp.py` with typed MDP primitives:

   ```python
   @dataclass(frozen=True)
   class State:
       code_hash: str                    # hash of current code snapshot
       history: tuple[dict, ...]         # recent experiment history window
       metrics: dict[str, float]         # current metric values
       resource_budget: float            # remaining budget (seconds)
       iteration: int                    # current iteration index

   @dataclass(frozen=True)
   class Action:
       params: dict[str, object] | None  # param proposal (continuous mode)
       diff: str | None                  # diff proposal (legacy mode)
       rationale: str = ""

   @dataclass(frozen=True)
   class Reward:
       value: float                      # scalar reward
       components: dict[str, float]      # breakdown (val_bpb_delta, status_penalty, etc.)

   TransitionFn = Callable[[State, Action], tuple[State, Reward]]
   ```

2. Create helper functions:
   - `build_state(...)` -- construct State from loop context
   - `compute_reward(prev_score, curr_score, status, ...) -> Reward` -- compute structured reward

3. Refactor `controller/` to use these types where states and rewards are currently implicit dicts

4. Run `PYTHONPATH=src pytest -q` to verify
5. Run `ruff check src/autoresearch_rl/mdp.py` to verify
6. Commit

## Files to create/modify

- New: `src/autoresearch_rl/mdp.py` -- MDP type definitions
- `src/autoresearch_rl/controller/` -- refactor to use MDP types

## Acceptance Criteria

- `State`, `Action`, `Reward` dataclasses with typed fields
- `TransitionFn` type alias
- Helper functions for construction
- Controller uses these types
- All tests pass

## Progress Report Format

APPEND to .ralph/B1-mdp-primitives/progress.md (never replace, always append):

```
## [Date/Time] - B.1

- What was implemented
- Files changed
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
```

## Code Quality

THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER.
ALL commits must pass quality checks. Functions < 50 lines.

## Stop Condition

After completing and verifying all checks pass, reply with:
<promise>COMPLETE</promise>
