from __future__ import annotations

import csv
from pathlib import Path

from autoresearch_rl.telemetry.rotation import rotate_if_needed

HEADER = [
    "commit",
    "metric_name",
    "metric_value",
    "memory_gb",
    "status",
    "description",
    "episode_id",
    "iter",
    "score",
    "budget_mode",
    "budget_s",
    "hardware_fingerprint",
    "comparable",
    "non_comparable_reason",
]


def ensure_results_tsv(path: str = "results.tsv") -> Path:
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(HEADER)
    return p


def append_result_row(
    *,
    path: str = "results.tsv",
    commit: str,
    metric_name: str,
    metric_value: float,
    memory_gb: float,
    status: str,
    description: str,
    episode_id: str,
    iter_idx: int,
    score: float,
    budget_mode: str,
    budget_s: int,
    hardware_fingerprint: str,
    comparable: bool,
    non_comparable_reason: str,
    max_file_size_bytes: int | None = None,
    max_rotated_files: int = 5,
) -> None:
    p = ensure_results_tsv(path)

    if max_file_size_bytes is not None:
        rotate_if_needed(path, max_file_size_bytes, max_rotated_files)
        # Re-ensure header exists after potential rotation
        p = ensure_results_tsv(path)

    with p.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                commit,
                metric_name,
                f"{metric_value:.6f}",
                f"{memory_gb:.1f}",
                status,
                description,
                episode_id,
                str(iter_idx),
                f"{score:.6f}",
                budget_mode,
                str(budget_s),
                hardware_fingerprint,
                "1" if comparable else "0",
                non_comparable_reason,
            ]
        )
