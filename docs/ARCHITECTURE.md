# Architecture (v0.2 scaffold)

## Control-plane
- Proposal policy (LLM or RL)
- Loop state (best score, history, sample-type counters)
- Async coordinator with bounded queues

## Data-plane
- Proposal queue (action generation)
- Sandboxed trial runner queue (execution)
- Next-state judge (evaluative + directional)
- Composite scorer (scalar + hint-aware)

## Observability
- Structured event stream (JSONL)
- Run artifacts with immutable IDs
- Replayable manifest with episode_id + run_id + sample_type

## Reward/Signal model
- **Evaluative signal** (`eval_score`): majority-voted {-1, 0, +1} from next-state heuristics
- **Directional signal** (`hint`): concise improvement hint extracted from next-state context
- **Combined scoring**: val metric + penalties + next-state score + hint bonus

## Comparability policy
- Default budget mode: `fixed_wallclock`
- Runs are tagged with `budget_mode`, `budget_s`, and `hardware_fingerprint`
- Strict mode blocks non-comparable runs (budget/hardware mismatches)
- Ledger persists comparability flags for auditability

## Safety posture
- Diff validation before execution
- AST policy checks for forbidden imports/calls
- Optional git-backed patch apply + rollback in trial runner
- Optional early-stop gate from online metric thresholds
- Bounded subprocess timeout in trial runner

## Next steps
1. Replace heuristic judge with model-based multi-vote judge
2. Add true asynchronous trainer sink (online updates)
3. Enforce path-level patch allowlists and full-file AST reparse
4. Add policy versioning + gated promotion/rollback
5. Integrate llmtrace spans/events
