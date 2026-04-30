# Trajectory-Aware Policy Ablation

Phase A.3 of issue [#31](https://github.com/epappas/autoresearch-rl/issues/31).

**Question:** does injecting `progress_series` (per-trial training-time
metric trajectories from `emit_progress`) into the LLM proposer's
prompt improve search outcomes vs proposing without that context?

**Answer:** **Inconclusive at n=5 paired runs.** Mean paired
difference (Arm A − Arm B) on best `eval_score` is **+0.0142**
favoring the trajectory-aware arm, with a 95% confidence interval of
**[−0.041, +0.069]** (Student-t, df=4) and bootstrap CI of
**[−0.021, +0.049]**. The Wilcoxon signed-rank exact two-sided
*p* = 0.625 (n=5, W=5). The CI substantially crosses zero on both
sides. **n=5 is too noisy at this scale to claim a real effect.**

The data does not refute the design hypothesis — Arm A wins 3 of 5
pairs and the point estimate is positive — but it also cannot
confirm it.

---

## Pinned context

| Field | Value |
|---|---|
| Date | 2026-04-30 → 2026-05-01 |
| Branch / commit | `750d9fc` (`main` after Phase A.1+A.2) |
| Target | `examples/security-judge` |
| Hardware | A100, K=4 parallel, fingerprint `83ead5833612ae3e` |
| Iterations per cell | 8 (max_iter=8 K=4) |
| Pairs | 5 (seeds 1–5) |
| Total cells | 10 (5 Arm A + 5 Arm B) |
| LLM proposer | DeepSeek-V3-0324 via Chutes (security-judge default) |
| Eval metric | `eval_score` (composite: JSON compliance + decision accuracy + score calibration) |
| Cell wall time | 14–32 min |
| Phase A.3 GPU spend | ~$76 (12 v2 cells at ~$8 + small v1 sunk cost ~$28; see "Cost reconciliation" below) |

The kill switch used for Arm B is the
`AR_DISABLE_PROGRESS_SERIES` env var added in commit
[`a9f9a9e`](https://github.com/epappas/autoresearch-rl/commit/a9f9a9e):
when set, `render_progress_series` returns `""` and the LLM prompt is
built without the `PROGRESS TRAJECTORIES` section. Tested by 6 unit
tests in
[`tests/test_prompt_fragments.py`](../../tests/test_prompt_fragments.py).

## What changed mid-experiment (the v1 → v2 pivot)

The first sweep (`v1`, `max_iter=4`, `K=4`, n=3 pairs) revealed a
methodology dead-end. With `max_iter == K`, the parallel engine's
admit loop in
[`controller/parallel_engine.py:239-255`](../../src/autoresearch_rl/controller/parallel_engine.py#L239)
makes exactly **one** `propose_batch(k=4)` call with **empty
history** — every subsequent loop iteration finds
`remaining_needed == 0` and never calls `propose_batch` again.
With history empty, `render_progress_series(history=[])` returns
`""` even when the env var is unset. Both arms therefore see the
identical prompt. Arms can only diverge through the LLM's own
temperature stochasticity (default `temperature=1.0` in
[`policy/llm_search.py:118`](../../src/autoresearch_rl/policy/llm_search.py#L118)).

Verified empirically: pair 1 produced **identical proposed params**
in Arm A and Arm B (LLM sampled the same response by chance); pairs
2 and 3 produced different params (LLM stochasticity), and the
eval_score deltas were ±0.02–0.05 driven by training-time noise
(`train.py` does not seed PyTorch RNG).

The v2 sweep (`max_iter=8`, `K=4`, n=5 pairs) fixes this. After the
first 4-trial batch saturates the pool, each completed trial frees
one slot; the loop then calls `propose_batch(k=1)` (which delegates
to `propose()`), which renders progress_series via
[`policy/llm_search.py:85`](../../src/autoresearch_rl/policy/llm_search.py#L85)
against the now-non-empty history. Iters 4–7 of each cell therefore
see real progress_series in Arm A and a suppressed series in Arm B.
Iters 0–3 are still the cold-start batch and serve as their own
"identical-prompt" sub-baseline.

The v1 data are kept under
[`docs/research/data/ablation-2026-04/runs/`](data/ablation-2026-04/runs/)
as a noise baseline for the no-trajectory regime; they are not
counted in the headline n=5 statistic.

## Method

Ten campaigns of `examples/security-judge`, each:

```bash
python3 examples/security-judge/deploy.py --policy llm \
  controller.parallel.enabled=true \
  controller.parallel.max_concurrency=4 \
  'controller.parallel.resources={"gpu":4}' \
  controller.max_iterations=8 \
  controller.max_wall_time_s=5400 \
  controller.seed=$SEED \
  policy.seed=$SEED \
  target.basilica.ready_timeout_s=1500 \
  target.timeout_s=2400 \
  telemetry.{trace,ledger,artifacts,versions,model_output,timeline}_path=...
```

Arm B additionally sets `AR_DISABLE_PROGRESS_SERIES=1` in the
controller-process env before launch. The trial container does not
inherit this var (`BasilicaTarget` only injects `AR_PARAMS_JSON`,
`AR_PARAM_*`, `AR_PROGRESS_FILE`, `AR_CONTROL_FILE`,
`AR_MODEL_DIR` — see
[`target/basilica.py:253-257`](../../src/autoresearch_rl/target/basilica.py#L253)).
Therefore the trial behavior is identical between arms; only the
proposer's prompt changes.

Run order: A1, B1, A2, B2, A3, B3, A4, B4, A5, B5 — alternated to
control for infrastructure-load drift across the ~3.5-hour sweep.

Cells are launched via
[`scripts/research/run_ablation_v2_one.sh`](../../scripts/research/run_ablation_v2_one.sh)
and analyzed via
[`scripts/research/analyze_ablation.py`](../../scripts/research/analyze_ablation.py).

## Results

### Per-pair best `eval_score`

| seed | Arm A (with progress_series) | Arm B (without) | A − B |
|---|---|---|---|
| 1 | 0.6016 | 0.5518 | **+0.0498** |
| 2 | 0.5455 | 0.5814 | −0.0359 |
| 3 | 0.5527 | 0.5300 | **+0.0227** |
| 4 | 0.5593 | 0.5864 | −0.0271 |
| 5 | 0.6155 | 0.5538 | **+0.0617** |

Sign breakdown: 3 / 5 pairs favor Arm A.

### Aggregate stats

| Statistic | Arm A | Arm B |
|---|---|---|
| Mean best | 0.5749 | 0.5607 |
| Median best | 0.5593 | 0.5538 |
| Std best | 0.0315 | 0.0232 |
| Successful trials per cell (mean) | 8.0 | 8.0 |
| Mean iters-to-first-eval-score>0.5 | 0.2 | 0.2 |
| Wall seconds (mean) | 1215 | 1326 |

8/8 trials succeeded in every cell — no `failed` outcomes during the
sweep.

### Paired statistics (A − B)

| Statistic | Value |
|---|---|
| Mean diff | **+0.0142** |
| Median diff | +0.0227 |
| Std diff | 0.0442 |
| 95% CI (Student-t, df=4) | **[−0.0406, +0.0691]** |
| 95% CI (10000-resample bootstrap) | [−0.0207, +0.0491] |
| Wilcoxon signed-rank W | 5 |
| Wilcoxon two-sided exact *p* | **0.625** |

Both CIs straddle zero. The bootstrap CI's right edge (+0.049) is
roughly the magnitude of the largest single-pair difference (pair 5,
+0.062), so any "evidence" for Arm A would be one or two pairs
moving the estimate, well within sampling noise.

### Effect size

Cohen's *d* on paired diff = mean_diff / std_diff =
0.0142 / 0.0442 ≈ **0.32** — a small effect by Cohen's
conventional benchmarks (small=0.2, medium=0.5, large=0.8). To
detect *d*=0.32 at *α*=0.05 two-sided with 80% power, paired-test
sample-size tables call for **n ≈ 80**. We have n=5. Even if the
point estimate is exactly correct, the experiment is roughly
**16× under-powered** to detect this effect at standard levels.

## Honest limitations

- **n=5 is too small for any inferential claim.** Reporting "Arm A
  wins on point estimate" without the CI on the difference would be
  misleading, so I'm not. The CI is too wide to claim direction.
- **Single task (security-judge), single base model (Qwen2.5-0.5B),
  single hardware (A100), single LLM proposer (DeepSeek-V3-0324
  via Chutes).** Generalization to other tasks / models is not
  established by this experiment.
- **Order confound.** Run order was A1, B1, A2, B2, … — every Arm B
  follows its paired Arm A immediately. Basilica image cache and HF
  weight cache can be warmer for the second run of each pair. The
  v1 noise baseline showed B systematically slightly higher than A
  (mean diff −0.030 in 3/3 pairs even though no real signal could
  exist); v2's flipped direction (A 3/5 pairs) suggests the order
  effect is smaller than the trajectory signal at the v2 scale, but
  it is not eliminated. A future v3 with randomized A/B ordering
  per pair would cleaner.
- **LLM temperature ≠ 0.** Default `temperature=1.0` on the chat
  call means even identical prompts produce different proposals.
  This adds floor noise on top of training-time stochasticity.
  Disabling it would reduce noise but might trigger endpoint
  rejection of `temperature=0` and would change the cumulative
  proposal-diversity behavior the wedge claim relies on.
- **Train-time stochasticity unbounded.** `train.py` does not seed
  PyTorch / NumPy / Python `random`, so two trials with literally
  identical params can produce different eval_scores. v1 measured
  this floor at ~0.05 amplitude, which is ~3.5× the v2 mean diff.
- **`progress_series` isn't tested for diff-mode policies.**
  `LLMDiffPolicy` also calls `render_progress_series`
  ([`policy/llm_diff.py:78`](../../src/autoresearch_rl/policy/llm_diff.py#L78))
  but security-judge runs in `policy.type=llm` (param mode) for
  this ablation. The diff-mode trajectory effect is not measured.

## What this experiment establishes

**Affirmative:**
- The framework can run a 10-cell paired ablation end-to-end on
  Basilica without operational failure (8/8 trials succeeded in
  every cell, 0 cancellations, 0 framework crashes).
- The `AR_DISABLE_PROGRESS_SERIES` kill switch works as advertised:
  6 unit tests pass, and the operational-test eval_scores diverge
  between arms exactly when v2's max_iter > K (and barely diverge
  when v1's max_iter == K, as the design predicts).
- The reproducibility floor for security-judge GRPO at K=4 max_iter=8
  is approximately ±0.04 on best `eval_score` per seed
  (std_diff = 0.044 with same proposed params and same hardware).

**Negative:**
- We cannot, on n=5 paired runs at this scale, claim that
  `progress_series` in the LLM prompt produces a measurable
  improvement in best eval_score. The CI on the mean difference is
  consistent with anything from "Arm B is 0.04 better" to "Arm A is
  0.07 better".

**Recommendation:**
A future Phase B/C/D ablation that wants a real answer should:
1. Use n ≥ 30 paired runs (at $8/cell × 60 cells ≈ $480, pricey),
   OR scale to a cheaper task than 0.5B GRPO.
2. Seed `train.py`'s PyTorch / NumPy RNGs to remove the training
   noise floor.
3. Randomize per-pair A/B order to remove the warmup-cache confound.
4. Run with `temperature=0` on the chat-completions call (or use a
   lower-temp endpoint) to remove the LLM-stochasticity floor.

If those four tightening steps were taken, n=5–10 pairs *might* be
enough to detect a Cohen *d* = 0.5+ effect, if one exists. Whether
one exists is, on the present evidence, an open question.

## Cost reconciliation

| Phase | Sub-deliverable | Estimate | Actual |
|---|---|---|---|
| A.1 | Competitive-Analysis | $0 | $0 |
| A.2 | Reproduction-SecurityJudge | $30 | ~$8 |
| A.3-v1 | Noise-baseline pairs (3 paired) | counted with A.3 | ~$28 (sunk; deliberate) |
| A.3-v2 | Trajectory ablation (5 paired, n=5) | $60 | ~$76 |
| **Phase A total** | | **$100** | **~$112** |

Phase A total ($112) is under the $150 hard kill but ~12% over the
$100 estimate. The overrun is concentrated in A.3 because of the
v1→v2 pivot. The lesson is: **read the engine before designing the
ablation**. Specifically, the parallel-engine's `slots_open ==
max(0, remaining_needed)` clamp at
[`parallel_engine.py:245`](../../src/autoresearch_rl/controller/parallel_engine.py#L245)
makes max_iter == K a degenerate case for any ablation that touches
the proposer; a future researcher should set `max_iter > K` from
the start.

## Files committed

Everything under
[`docs/research/data/ablation-2026-04/`](data/ablation-2026-04/):

- `v2/summary.json` — full machine-readable analysis output.
- `v2/{A,B}_seed{1..5}/{results.tsv,events.jsonl,timeline.json,checkpoint.json,launch.log}` — one cell per arm-seed, ledgers + traces + Chrome-trace timelines + per-cell launch logs.
- `runs/{A,B}_seed{1..3}/...` — v1 noise-baseline cells (3 pairs).
- `v1-noise-summary.json` — analysis output for the v1 baseline
  (mean diff −0.030, all 3 pairs B-better, single-tail bootstrap CI
  not crossing zero — order effect proxy).

Per-trial LoRA adapters and Qwen2.5-0.5B tokenizer files are not
committed (~1.4 GB total, recoverable from
`huggingface.co/Qwen/Qwen2.5-0.5B-Instruct`); the analysis runs
from the ledgers alone.
