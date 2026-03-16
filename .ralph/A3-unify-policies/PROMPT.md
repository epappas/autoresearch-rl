# A.3 Unify policy hierarchies

## Context

Two unrelated policy abstractions coexist:
- `policy/search.py`: `ParamPolicy` base class with `GridPolicy`, `RandomPolicy`, `StaticPolicy` (used by continuous loop for hyperparameter proposals)
- `policy/interface.py`: `ProposalPolicy` protocol with `propose()` / `propose_diff()` (used by legacy loop for code-diff proposals)

Both are called "policies" but have incompatible interfaces and serve the same conceptual role (propose the next action given history).

**Dependency:** This task requires A.1 (unified loops) to be completed first.

## Your Task

1. Define a single `Policy` protocol with a `propose(state: dict) -> Proposal` method in `policy/interface.py`
2. Create a unified `Proposal` type that can represent both param-based and diff-based proposals:
   - `ParamProposal(params: dict, rationale: str)` -- for hyperparameter proposals
   - `DiffProposal(diff: str, rationale: str)` -- for code-diff proposals
   - Both inherit from or implement a common `Proposal` base
3. Update `ParamPolicy` subclasses (`GridPolicy`, `RandomPolicy`, `StaticPolicy`) to implement the unified `Policy` protocol
4. Update `ProposalPolicy` implementations (`GreedyLLMPolicy`, `RandomPolicy` in baselines, `LearnedDiffPolicy`) to implement the unified `Policy` protocol
5. Remove the old `ProposalPolicy` protocol from `policy/interface.py` once migrated
6. Update all callers in the controller
7. Run `PYTHONPATH=src pytest -q` to verify
8. Run `ruff check src/autoresearch_rl/policy/` to verify
9. Commit

## Files to modify

- `src/autoresearch_rl/policy/interface.py` -- unified protocol and proposal types
- `src/autoresearch_rl/policy/search.py` -- adapt ParamPolicy subclasses
- `src/autoresearch_rl/policy/baselines.py` -- adapt diff-based policies
- `src/autoresearch_rl/policy/learned.py` -- adapt LearnedDiffPolicy
- Controller files -- update to use unified policy interface

## Acceptance Criteria

- Single `Policy` protocol used by both loop modes
- `ParamProposal` and `DiffProposal` as typed proposal variants
- All policy classes implement the unified protocol
- All tests pass
- Lint passes

## Progress Report Format

APPEND to .ralph/A3-unify-policies/progress.md (never replace, always append):

```
## [Date/Time] - A.3

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
