# Reproduction Study — security-judge probe 6

Phase A.2 of issue [#31](https://github.com/epappas/autoresearch-rl/issues/31).
Goal: replicate the probe-6 result on Basilica A100s with the current code
and confirm the published security-judge reward landscape still holds.

**Pass condition:** best `eval_score` within ±0.10 of probe 6's
**0.62**, OR a documented reason why not.

**Verdict: PASS, on the edge.** Best `eval_score = 0.521818` —
`0.62 − 0.522 = 0.098`, just inside the 0.10 band. The reproduction
hit the band, but on the lower side. Two non-trivial divergences from
probe 6 are documented below.

---

## Pinned context

| Field | Probe 6 (reference) | This reproduction |
|---|---|---|
| Date | 2026-04-29 | 2026-04-30 |
| Branch / commit | post-`ec680ba` | `702f43d` (`main`) |
| Target | `examples/security-judge` | same |
| Hardware | A100, K=4 parallel | A100, K=4 parallel (fingerprint `83ead5833612ae3e`) |
| Iterations | 4 | 4 |
| Concurrency | K=4 | K=4 |
| Policy | (LLM-proposed params, parallel) | `policy.type=llm`, K=4 parallel |
| LLM proposer | reached the chat API (Kimi / Chutes) | **429'd out — fell back to seeded random; see "Divergences"** |
| Wall time | ~17 min | ~22 min (19:27:14 → 19:50:02 local) |
| GPU spend | ~$8 (per `velocity.md`) | ~$8 estimated (4× A100 × 22 min) |

`run_id`: `40b8d50b1730`. Run manifest:
[`docs/research/data/repro-2026-04/run-manifest.json`](data/repro-2026-04/run-manifest.json).
The live `artifacts/` and `traces/` directories are gitignored;
this doc's evidence bundle lives under `docs/research/data/repro-2026-04/`.

## Command

The campaign was launched via the canonical
`examples/security-judge/deploy.py` wrapper, which builds the
file-injection setup_cmd and invokes `autoresearch-rl run` with
`--override` flags:

```bash
python3 examples/security-judge/deploy.py --policy llm \
  controller.parallel.enabled=true \
  controller.parallel.max_concurrency=4 \
  'controller.parallel.resources={"gpu":4}' \
  controller.max_iterations=4 \
  controller.max_wall_time_s=3600 \
  target.basilica.ready_timeout_s=1500 \
  target.timeout_s=2400 \
  telemetry.trace_path=traces/repro-2026-04/events.jsonl \
  telemetry.ledger_path=docs/research/data/repro-2026-04/results.tsv \
  telemetry.artifacts_dir=artifacts/repro-2026-04/runs \
  telemetry.versions_dir=artifacts/repro-2026-04/versions \
  telemetry.model_output_dir=/data/models/repro-2026-04 \
  controller.checkpoint_path=artifacts/repro-2026-04/checkpoint.json \
  telemetry.timeline_path=traces/repro-2026-04/timeline.json
```

Env vars (`BASILICA_API_TOKEN`, `CHUTES_API_KEY`, `HF_TOKEN`) sourced
from `~/.env` via `set -a && . .env && set +a` before launching, the
same flow `config_validate` exercises. Full launch log retained at
`/tmp/repro_run.log` on the build host (not committed; contains
upstream API responses).

## Per-iteration results

Source: [`docs/research/data/repro-2026-04/results.tsv`](data/repro-2026-04/results.tsv)
(committed alongside this doc), and `traces/repro-2026-04/events.jsonl`.

| iter | status | decision | eval_score | params | trial elapsed |
|---|---|---|---|---|---|
| 0 | ok | keep | 0.423636 | lr=1e-4, steps=30, gen=3, temp=0.7, rank=4 | 569 s |
| 1 | failed | discard | (no metrics) | lr=3e-4, steps=30, gen=3, temp=0.7, rank=16 | 823 s |
| 2 | ok | keep (best) | **0.521818** | lr=5e-5, steps=30, gen=2, temp=0.9, rank=8 | 1034 s |
| 3 | ok | discard | 0.515455 | lr=5e-5, steps=30, gen=2, temp=0.9, rank=4 | 949 s |

`description` column on every row: `continuous-parallel|conc=4`.

Side-channel metrics on the best iter (iter 2):
`decision_accuracy=0.209091`, `json_compliance=0.963636`,
`loss=0.001292`, `training_seconds=79.8`. The model produced valid
JSON 96% of the time but got the right pass/block/warning verdict
only 21% of the time — same shape as probe 6 (high JSON compliance,
low decision accuracy on a 0.5B model with 30 GRPO steps).

Comparability cell on every row: `comparable=0,
non_comparable_reason=budget_mismatch:3600!=28800`. Expected — the
probe used `expected_budget_s=28800` (8 h) and we cut wall time to
1 h. Strict mode is off in this config, so the runs proceed and the
mismatch is flagged in the ledger rather than rejecting the
campaign. Phase A.3 will keep the same shorter budget for ablation
to keep arms comparable.

## Reference comparison

Probe 6 ledger (from `docs/research/RLix-Adoption-Outcomes.md`):
`eval_score = [0.41, 0.11, 0.55, 0.62]`, best `0.62`.

This run: `eval_score = [0.424, failed, 0.522, 0.515]`, best `0.522`.

| Metric | Probe 6 | Repro | Delta |
|---|---|---|---|
| Best `eval_score` | 0.62 | 0.522 | -0.098 |
| Successful trials | 4 / 4 | 3 / 4 | -1 |
| Mean `eval_score` of successes | 0.4225 | 0.487 | +0.064 |
| Median `eval_score` of successes | 0.48 | 0.515 | +0.035 |
| Wall time | ~17 min | ~22 min | +5 min |

The best-trial gap is **0.098**, just inside the ±0.10 pass band.
Mean and median across successful trials are slightly higher in this
run, so the *typical* outcome reproduces; the *peak* outcome does
not, and that's the soft signal worth surfacing.

## Model checkpoints

LoRA adapters retained in
[`docs/research/data/repro-2026-04/runs/run-XXXX/`](data/repro-2026-04/runs/)
(committed copy of the live `artifacts/` directory; tokenizer and
vocab files were dropped from the bundle since they are stock
Qwen2.5-0.5B-Instruct files). SHA-256 of `adapter_model.safetensors`
for each iter that produced a model:

| iter | rank | size (B) | sha256 |
|---|---|---|---|
| 0 | 4 | 1,093,728 | `5ca75ec15634ffa7e7adbd51439527eacc368c1895b6bdd6b139ef1751abe282` |
| 2 (best) | 8 | 2,175,168 | `f87fe8340c7dfb6106b802b50383cf73b320a8ac093c975d6c2e3348134e6ab0` |
| 3 | 4 | 1,093,728 | `672ab1019e28f12ec3b108aa176c69a22c67683fd029462246d4ea89a14d1f3c` |

Reproducible via:

```bash
sha256sum docs/research/data/repro-2026-04/runs/run-0002/adapter_model.safetensors
```

The 17–20 MB figure quoted in `RLix-Adoption-Outcomes.md` was the
file-listing total including the tokenizer and merges. The adapter
itself is 1–2 MB depending on `lora_rank`. The probe-6 number was
correct, the framing was about per-iter download size in total
bytes, not adapter weights specifically.

## Divergences from probe 6 (honest list)

### 1. LLM proposer fell back to random

Probe 6 used real LLM-proposed parameters via the Chutes / DeepSeek
endpoint configured in `examples/security-judge/config.yaml:40-42`.
This reproduction's first `propose_batch(state, k=4)` call hit the
chat endpoint and got HTTP 429 ("Infrastructure is at maximum
capacity, try again later") on every attempt across the 5-retry
exponential-backoff window. The fallback path
([`policy/llm_search.py:419`](../../src/autoresearch_rl/policy/llm_search.py#L419))
returned 4 seeded-random proposals, which is what every iteration
ran on. From the launch log:

```
LLM API error 429, retry 1/5 in 12s
LLM API error 429, retry 2/5 in 22s
LLM API error 429, retry 3/5 in 42s
LLM API error 429, retry 4/5 in 87s
LLM API error 429, retry 5/5 in 94s
LLM API error 429 (no retry): {"detail":"Infrastructure is at maximum capacity, try again later"}
LLM batch policy failed, falling back to 4 random
```

Implication: the eval-score landscape this reproduction sampled is
the **seeded-random landscape**, not the LLM-guided landscape.
"Best `eval_score` within 0.10 of 0.62" is therefore a statement
about the *target* (security-judge GRPO at K=4 on A100 with this
data), not about the *proposer*. A separate LLM-vs-random comparison
is exactly Phase A.3's job.

### 2. iter 1 failed during training

`status=failed`, no metrics returned, container ran for 823 s before
exiting. Params were aggressive: `lr=3e-4` (the max in the search
space), `lora_rank=16` (max), `num_generations=3`. Most likely cause
is loss divergence or OOM under that combination; without container
logs I am not asserting a specific root cause. Probe 6 had 4/4
successful trials with one weak (eval_score=0.11); this run had
3/4. The framework handled the failure correctly: marked
`status=failed`, kept the surviving siblings' results, did not stop
the campaign.

### 3. Random fallback distribution differs from LLM proposals

Random fallback uses `RandomPolicy` with the engine seed, which is
deterministic given the same seed but does not match the
distribution Kimi/DeepSeek would have proposed. Probe 6 picked
LR ∈ {5e-5, 1e-4, 3e-4} broadly; this run, by chance, drew two
trials with `lr=5e-5, num_generations=2, temperature=0.9` and only
varied `lora_rank` between them (iters 2 and 3). That low-LR
clustering is what produced the 0.515–0.522 cluster at the top of
the leaderboard. A re-run with seed change is likely to land
differently.

## What this reproduction does and does not establish

**Does establish:**
- Probe-6 GRPO of Qwen2.5-0.5B-Instruct on the deepset/prompt-injections
  data still trains end-to-end on Basilica A100s at commit `702f43d`.
- The K=4 parallel engine still cleanly handles 1/4 trial failures
  without blowing up the campaign or corrupting the ledger.
- Model artifacts (LoRA adapters) still download to local disk at
  end of trial.
- The reward landscape's *typical* eval_score (mean ~0.48 of
  successful trials) is in the same neighborhood as probe 6 (mean
  ~0.42). Best is on the lower edge of the pass band.

**Does NOT establish:**
- Whether the LLM proposer adds value — both probe 6 and this run
  are within sampling noise of each other on a 4-trial budget, and
  this run did not actually use the LLM proposer.
- Whether `progress_series` in the prompt steers proposals (the
  proposer never got a chance to be steered — that's A.3).
- Whether the ±0.10 pass band is the right tolerance. With n=4 per
  arm, the standard error on best-of-4 is large; ±0.10 is
  generous. A.3 will report effect sizes with CIs over n=5 paired
  arms instead of single-shot best-of-4.

## Repro recipe (for future readers)

Five steps:

1. Stash / redirect any tracked telemetry paths so a re-run does not
   overwrite committed data:

   ```bash
   make validate CONFIG=examples/security-judge/config.yaml  # warns
   ```

2. Source `~/.env`:

   ```bash
   set -a && . .env && set +a
   ```

3. Launch the campaign with the override block above (substitute a
   fresh output directory, e.g. `repro-2026-05`).

4. Wait ~22 min (Basilica deployment + setup + 4 GRPO trials of
   30 steps each on A100, K=4 parallel).

5. Compare best `eval_score` to 0.62 ±0.10. Report all four trials,
   not only the best. Disclose any 429 fallbacks.

## Files committed

Everything under `docs/research/data/repro-2026-04/`:

- `results.tsv` — full ledger (4 rows).
- `events.jsonl` — full trace.
- `timeline.json` — Chrome-trace timeline (Basilica deployment phases,
  policy.propose, executor.execute spans).
- `run-manifest.json` — git commit, platform, hardware fingerprint,
  full resolved config.
- `versions/v0000.json`, `versions/v0002.json` — kept-iteration
  metadata (params + metrics + run_dir + model_dir).
- `runs/run-{0000,0002,0003}/adapter_model.safetensors` and
  `adapter_config.json` — three LoRA adapters with SHA-256s in the
  table above. Total bundle size ~4.3 MB (tokenizer / vocab files
  dropped — they are the stock Qwen2.5-0.5B-Instruct files and
  recoverable from `huggingface.co/Qwen/Qwen2.5-0.5B-Instruct`).
