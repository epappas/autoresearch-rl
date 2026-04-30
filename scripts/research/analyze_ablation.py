"""Analyze the Phase A.3 trajectory-aware ablation.

Reads results.tsv from each (arm, seed) cell under
docs/research/data/ablation-2026-04/runs/{A,B}_seed{N}/, computes paired
statistics, and writes:
  - docs/research/data/ablation-2026-04/summary.json
  - prints a markdown-friendly stats block to stdout

Stats reported per arm:
  - mean(best_eval_score), median, n_successful_trials
  - mean iters-to-first-eval-score-greater-than-0.5
  - mean GPU-hours

Paired stats (Arm A best - Arm B best per seed):
  - mean diff, std diff
  - 95% CI via t-distribution (small n correction)
  - 95% CI via paired bootstrap (10000 resamples)
  - Wilcoxon signed-rank statistic + exact p-value

Single dependency: numpy (already in pyproject). No scipy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DATA_DIR = Path("docs/research/data/ablation-2026-04")
THRESHOLD = 0.5  # eval_score > THRESHOLD counts as "good"


def parse_cell(arm: str, seed: int, data_dir: Path) -> dict[str, Any] | None:
    """Parse one (arm, seed) cell. Returns None if results.tsv is missing."""
    cell_dir = data_dir / f"{arm}_seed{seed}"
    tsv = cell_dir / "results.tsv"
    if not tsv.exists():
        return None

    rows: list[dict] = []
    with tsv.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                row["metric_value_f"] = float(row["metric_value"])
                row["iter_i"] = int(row["iter"])
            except (ValueError, KeyError):
                continue
            rows.append(row)

    if not rows:
        return {
            "arm": arm,
            "seed": seed,
            "rows": [],
            "best": None,
            "successes": 0,
            "first_above_threshold_iter": None,
            "wall_seconds": None,
        }

    successful = [r for r in rows if r["status"] != "failed"]
    eval_scores = [r["metric_value_f"] for r in successful if r["metric_value_f"] > 0]
    best = max(eval_scores) if eval_scores else None

    # First iter index whose eval_score crosses threshold (if any).
    first_above = next(
        (r["iter_i"] for r in sorted(rows, key=lambda r: r["iter_i"])
         if r["status"] != "failed" and r["metric_value_f"] > THRESHOLD),
        None,
    )

    # Wall clock from the events.jsonl (first/last ts).
    events_path = cell_dir / "events.jsonl"
    wall_seconds = None
    if events_path.exists():
        ts_values: list[float] = []
        for line in events_path.read_text().splitlines():
            try:
                ts_values.append(json.loads(line).get("ts"))
            except json.JSONDecodeError:
                continue
        ts_values = [t for t in ts_values if t is not None]
        if ts_values:
            wall_seconds = max(ts_values) - min(ts_values)

    return {
        "arm": arm,
        "seed": seed,
        "n_rows": len(rows),
        "n_successful": len(successful),
        "best_eval_score": best,
        "first_above_threshold_iter": first_above,
        "wall_seconds": wall_seconds,
        "per_iter": [
            {
                "iter": r["iter_i"],
                "status": r["status"],
                "decision": r["status"],  # ledger conflates; status is canonical
                "eval_score": r["metric_value_f"],
            }
            for r in sorted(rows, key=lambda r: r["iter_i"])
        ],
    }


def t_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided (1-alpha) confidence interval for mean via Student t.

    Hard-coded t critical for two-sided 0.05 tail at small df because we
    are not depending on scipy. Table values (df, t_0.025): 1: 12.706,
    2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 14: 2.145, 19: 2.093.
    """
    t_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        14: 2.145, 19: 2.093, 29: 2.045,
    }
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(n))
    df = n - 1
    t_crit = t_table.get(df, 2.0)  # asymptotic z=1.96 for large df
    return (mean - t_crit * se, mean + t_crit * se)


def bootstrap_ci(
    values: np.ndarray, n_resamples: int = 10000, alpha: float = 0.05, seed: int = 1234
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of `values`."""
    if len(values) < 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def wilcoxon_signed_rank(diffs: np.ndarray) -> dict[str, Any] | None:
    """Wilcoxon signed-rank for paired data without scipy.

    Returns the test statistic W (sum of positive signed ranks) and an
    exact two-sided p-value computed by exhaustive enumeration of sign
    assignments (works up to n~20). Zero differences are dropped; ties
    are mid-ranked.
    """
    nonzero = diffs[np.abs(diffs) > 1e-12]
    n = len(nonzero)
    if n < 1:
        return None
    abs_d = np.abs(nonzero)
    ranks = _midranks(abs_d)
    W_pos = float(np.sum(ranks[nonzero > 0]))
    W_neg = float(np.sum(ranks[nonzero < 0]))
    W = min(W_pos, W_neg)

    # Exact p-value: enumerate all 2^n sign patterns over the magnitudes.
    if n > 20:
        return {
            "n": n, "W_pos": W_pos, "W_neg": W_neg, "W": W,
            "p_value": None, "p_method": "exact-skipped-n>20",
        }
    rank_array = ranks
    total = 1 << n
    extreme_count = 0
    target = W
    for mask in range(total):
        s = 0.0
        for i in range(n):
            if mask & (1 << i):
                s += rank_array[i]
        # Two-sided: count assignments at least as extreme on either tail
        if s <= target or (np.sum(rank_array) - s) <= target:
            extreme_count += 1
    return {
        "n": n,
        "W_pos": W_pos,
        "W_neg": W_neg,
        "W": W,
        "p_value": extreme_count / total,
        "p_method": "exact-enumeration",
    }


def _midranks(values: np.ndarray) -> np.ndarray:
    """Mid-rank assignment with tie correction. Ranks 1..n inclusive."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(values, dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        midrank = (i + j + 2) / 2.0  # 1-indexed midrank
        for k in range(i, j + 1):
            ranks[order[k]] = midrank
        i = j + 1
    return ranks


def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR / "runs",
        help="Directory containing {A,B}_seed{N}/ subdirs with results.tsv",
    )
    parser.add_argument(
        "--out", type=str, default=None,
    )
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else (args.data_dir.parent / "summary.json")

    cells_a = []
    cells_b = []
    for s in args.seeds:
        a = parse_cell("A", s, args.data_dir)
        b = parse_cell("B", s, args.data_dir)
        if a is None or b is None:
            print(f"WARN: seed {s} missing cell (A={a is not None} B={b is not None})", file=sys.stderr)
            continue
        cells_a.append(a)
        cells_b.append(b)

    if not cells_a:
        print("ERROR: no complete pairs found; nothing to analyze.", file=sys.stderr)
        return 1

    def best(c: dict) -> float:
        return c["best_eval_score"] if c["best_eval_score"] is not None else 0.0

    best_a = np.array([best(c) for c in cells_a])
    best_b = np.array([best(c) for c in cells_b])
    diff = best_a - best_b

    summary: dict[str, Any] = {
        "schema": "ablation-summary-v1",
        "data_dir": str(args.data_dir),
        "n_pairs": len(cells_a),
        "seeds": [c["seed"] for c in cells_a],
        "arm_A": {
            "label": "with progress_series in LLM prompt",
            "best_per_seed": best_a.tolist(),
            "mean_best": float(best_a.mean()),
            "median_best": float(np.median(best_a)),
            "std_best": float(np.std(best_a, ddof=1)) if len(best_a) > 1 else 0.0,
            "n_successful_trials_per_seed": [c["n_successful"] for c in cells_a],
            "mean_successful_trials": float(np.mean([c["n_successful"] for c in cells_a])),
            "first_above_threshold": [c["first_above_threshold_iter"] for c in cells_a],
            "wall_seconds_per_seed": [c["wall_seconds"] for c in cells_a],
        },
        "arm_B": {
            "label": "without progress_series in LLM prompt (AR_DISABLE_PROGRESS_SERIES=1)",
            "best_per_seed": best_b.tolist(),
            "mean_best": float(best_b.mean()),
            "median_best": float(np.median(best_b)),
            "std_best": float(np.std(best_b, ddof=1)) if len(best_b) > 1 else 0.0,
            "n_successful_trials_per_seed": [c["n_successful"] for c in cells_b],
            "mean_successful_trials": float(np.mean([c["n_successful"] for c in cells_b])),
            "first_above_threshold": [c["first_above_threshold_iter"] for c in cells_b],
            "wall_seconds_per_seed": [c["wall_seconds"] for c in cells_b],
        },
        "paired_difference": {
            "diff_per_seed": diff.tolist(),
            "mean_diff": float(diff.mean()),
            "median_diff": float(np.median(diff)),
            "std_diff": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
            "ci_95_t": list(t_ci(diff)),
            "ci_95_bootstrap": list(bootstrap_ci(diff)),
            "wilcoxon": wilcoxon_signed_rank(diff),
        },
        "raw_cells": {"A": cells_a, "B": cells_b},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    # Pretty print for the doc.
    print(f"\nA.3 ablation summary — {summary['n_pairs']} paired runs (seeds {summary['seeds']})\n")
    print(f"Arm A (with progress_series):    best = {best_a}")
    print(f"  mean={best_a.mean():.4f}  median={np.median(best_a):.4f}  "
          f"std={summary['arm_A']['std_best']:.4f}")
    print(f"Arm B (without progress_series): best = {best_b}")
    print(f"  mean={best_b.mean():.4f}  median={np.median(best_b):.4f}  "
          f"std={summary['arm_B']['std_best']:.4f}")
    print(f"\nPaired diff (A - B): {diff}")
    pd = summary["paired_difference"]
    print(f"  mean diff = {pd['mean_diff']:+.4f}  median = {pd['median_diff']:+.4f}")
    print(f"  std diff  = {pd['std_diff']:.4f}")
    print(f"  95% CI (t-dist, df={len(diff)-1}): "
          f"[{pd['ci_95_t'][0]:+.4f}, {pd['ci_95_t'][1]:+.4f}]")
    print(f"  95% CI (bootstrap):                "
          f"[{pd['ci_95_bootstrap'][0]:+.4f}, {pd['ci_95_bootstrap'][1]:+.4f}]")
    if pd["wilcoxon"]:
        w = pd["wilcoxon"]
        print(f"  Wilcoxon W={w['W']}  p={w['p_value']}  ({w['p_method']})")
    return 0


if __name__ == "__main__":
    sys.exit(run())
