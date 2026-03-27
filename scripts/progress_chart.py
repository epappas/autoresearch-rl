#!/usr/bin/env python3
"""Generate a progress chart from a results.tsv ledger.

Produces a scatter plot showing experiment-over-experiment improvement:
  - Gray dots: discarded experiments
  - Green dots: kept experiments (improvements)
  - Red x: failed experiments
  - Step function: running best score

Uses Covenant Labs brand palette.

Usage:
    python scripts/progress_chart.py artifacts/basilica-grpo/results.tsv -o progress.png
    python scripts/progress_chart.py artifacts/deberta/results.tsv --episode fea2ec74e1cf
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402

LOGO_PNG = Path(__file__).parent / "basilica_logo.png"

# -- Covenant Labs Color Palette --
RED = "#FF3A3A"
BLK1000 = "#101010"
BLK800 = "#2F2F2F"
BLK500 = "#828282"
WHT0 = "#F4F4F4"
WHT100ALT = "#DDDDDD"
GREEN = "#2ECC71"


@dataclass
class Row:
    iter: int
    metric_name: str
    metric_value: float
    status: str
    description: str
    episode_id: str
    score: float


def load_results(path: str, episode: str | None = None) -> list[Row]:
    """Load results.tsv and return parsed rows."""
    rows: list[Row] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if episode and r.get("episode_id", "") != episode:
                continue
            rows.append(Row(
                iter=int(r["iter"]),
                metric_name=r["metric_name"],
                metric_value=float(r["metric_value"]),
                status=r["status"],
                description=r.get("description", ""),
                episode_id=r.get("episode_id", ""),
                score=float(r.get("score", 0.0)),
            ))
    return rows


def detect_direction(rows: list[Row]) -> str:
    """Detect if metric is maximized or minimized from keep/discard pattern."""
    kept_values = [r.metric_value for r in rows if r.status == "keep" and r.metric_value > 0]
    if len(kept_values) < 2:
        return "max"
    return "max" if kept_values[-1] >= kept_values[0] else "min"


def plot_progress(
    rows: list[Row],
    output: str,
    title: str | None = None,
    direction: str | None = None,
) -> None:
    """Generate and save the progress chart."""
    if not rows:
        print("No data to plot.", file=sys.stderr)
        return

    direction = direction or detect_direction(rows)

    # Separate by status
    xs_discard, ys_discard = [], []
    xs_keep, ys_keep = [], []
    xs_fail, ys_fail = [], []

    for i, r in enumerate(rows):
        if r.status == "keep":
            xs_keep.append(i)
            ys_keep.append(r.metric_value)
        elif r.metric_value == 0.0 or r.status in ("failed", "crash"):
            xs_fail.append(i)
            ys_fail.append(0.0)
        else:
            xs_discard.append(i)
            ys_discard.append(r.metric_value)

    # Compute running best (step function)
    running_best: list[float] = []
    best = float("-inf") if direction == "max" else float("inf")
    for r in rows:
        if r.metric_value > 0:
            if direction == "max":
                best = max(best, r.metric_value)
            else:
                best = min(best, r.metric_value)
        running_best.append(best if best != float("inf") and best != float("-inf") else 0.0)

    # -- Covenant Labs dark theme --
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.set_facecolor(BLK1000)
    ax.set_facecolor(BLK1000)

    # Grid: subtle against dark background
    ax.grid(True, color=BLK800, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_color(BLK800)
        spine.set_linewidth(0.8)

    # Tick styling: light text on dark
    ax.tick_params(colors=BLK500, labelsize=16)

    # Discarded: muted gray
    if xs_discard:
        ax.scatter(
            xs_discard, ys_discard,
            c=BLK500, s=70, alpha=0.4, zorder=2, label="Discarded",
        )

    # Failed: red x
    if xs_fail:
        ax.scatter(
            xs_fail, [max(ys_keep or ys_discard or [0]) * 0.01] * len(xs_fail),
            c=RED, s=50, alpha=0.6, marker="x", linewidths=2,
            zorder=2, label="Failed",
        )

    # Kept: green (semantic positive/success)
    if xs_keep:
        ax.scatter(
            xs_keep, ys_keep,
            c=GREEN, s=130, edgecolors=WHT0, linewidths=1.0,
            zorder=3, label="Kept (improvement)",
        )

    # Step function: red (primary accent)
    valid_running = [(i, v) for i, v in enumerate(running_best) if v > 0]
    if valid_running:
        rx, ry = zip(*valid_running)
        ax.step(rx, ry, where="post", color=RED, alpha=0.85, linewidth=3, zorder=4)

    # Annotate kept experiments
    for xi, yi in zip(xs_keep, ys_keep):
        row = rows[xi]
        label = f"iter {row.iter}"
        ax.annotate(
            label, (xi, yi),
            textcoords="offset points", xytext=(6, 14),
            fontsize=13, color=WHT100ALT, fontweight="bold",
        )

    n_total = len(rows)
    n_kept = len(xs_keep)
    n_failed = len(xs_fail)
    metric_name = rows[0].metric_name if rows else "metric"
    dir_label = "higher is better" if direction == "max" else "lower is better"

    chart_title = title or (
        f"autoresearch-rl: {n_total} experiments, "
        f"{n_kept} kept, {n_failed} failed"
    )
    ax.set_title(chart_title, fontsize=22, fontweight="bold", color=WHT0, pad=20)
    ax.set_xlabel("Experiment #", fontsize=18, color=BLK500)
    ax.set_ylabel(f"{metric_name} ({dir_label})", fontsize=18, color=BLK500)

    legend = ax.legend(
        loc="upper right" if direction == "min" else "lower right",
        fontsize=15, framealpha=0.9, edgecolor=BLK800,
        facecolor=BLK1000,
    )
    for text in legend.get_texts():
        text.set_color(WHT100ALT)

    # Y-axis range with margin
    all_values = [v for v in ys_keep + ys_discard if v > 0]
    if all_values:
        ymin, ymax = min(all_values), max(all_values)
        margin = (ymax - ymin) * 0.15 if ymax > ymin else 0.05
        ax.set_ylim(ymin - margin, ymax + margin)

    # Basilica logo (bottom right)
    if LOGO_PNG.exists():
        logo_img = mpimg.imread(str(LOGO_PNG))
        imagebox = OffsetImage(logo_img, zoom=0.7, alpha=0.7)
        ab = AnnotationBbox(
            imagebox, (0.98, 0.04),
            xycoords="figure fraction",
            box_alignment=(1.0, 0.0),
            frameon=False,
        )
        fig.add_artist(ab)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output, dpi=150, facecolor=BLK1000, edgecolor="none")
    plt.close(fig)

    best_val = max(ys_keep) if direction == "max" and ys_keep else (
        min(ys_keep) if ys_keep else 0.0
    )
    print(f"Saved {output}: {n_total} experiments, {n_kept} kept, best {metric_name}={best_val:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate progress chart from results.tsv")
    parser.add_argument("results_tsv", help="Path to results.tsv ledger")
    parser.add_argument("-o", "--output", default="progress.png", help="Output image path")
    parser.add_argument("--episode", default=None, help="Filter to specific episode ID")
    parser.add_argument("--title", default=None, help="Custom chart title")
    parser.add_argument(
        "--direction", choices=["min", "max"], default=None,
        help="Metric direction (auto-detected if not set)",
    )
    args = parser.parse_args()

    if not Path(args.results_tsv).exists():
        print(f"File not found: {args.results_tsv}", file=sys.stderr)
        sys.exit(1)

    rows = load_results(args.results_tsv, episode=args.episode)
    if not rows:
        print("No matching rows found.", file=sys.stderr)
        sys.exit(1)

    plot_progress(rows, args.output, title=args.title, direction=args.direction)


if __name__ == "__main__":
    main()
