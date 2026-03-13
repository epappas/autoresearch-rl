# Architecture (current scaffold)

## Control-plane
- Proposal policy (currently baseline policies)
- Loop state (best score + telemetry-backed history)
- Async coordinator with bounded queues
- Strict three-file contract enforcement (frozen/mutable/program)

## Data-plane
- Proposal queue (action generation)
- Sandboxed trial runner queue (execution)
- Next-state judge (evaluative + directional)
- Composite scorer (scalar + hint-aware)

## Observability
- Structured event stream (JSONL)
- Run artifacts with immutable IDs
- Replayable manifest with `episode_id` + `run_id` + `sample_type`
- Canonical results ledger (`results.tsv`) with comparability metadata

## Reward/Signal model
- **Evaluative signal** (`eval_score`): majority-voted {-1, 0, +1} from next-state heuristics
- **Directional signal** (`hint`): concise improvement hint extracted from next-state context
- **Combined scoring**: val metric + penalties + next-state score + hint bonus

## Comparability policy
- Default budget mode: `fixed_wallclock`
- Runs are tagged with `budget_mode`, `budget_s`, and `hardware_fingerprint`
- Strict mode blocks non-comparable runs (budget/hardware mismatches)
- CI runs verification-matrix checks via `scripts/verify_matrix.sh`

## Safety posture
- Diff validation before execution
- Best-effort AST guard on added Python lines (not full-file semantic proof)
- Optional git-backed patch apply + rollback in trial runner
- Optional threshold-based early-stop in trial runner
- Bounded subprocess timeout in trial runner

## Next steps
1. Replace heuristic judge with model-based multi-vote judge
2. Add true asynchronous trainer sink (online updates)
3. Strengthen validation to full-file AST reparse after patch apply
4. Add policy versioning + gated promotion/rollback
5. Integrate llmtrace spans/events
