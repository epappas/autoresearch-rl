#!/usr/bin/env python3
"""Prepare DeBERTa training data from llmtrace benchmark datasets.

Reads all JSON dataset files from the llmtrace benchmarks directory,
converts string labels ("malicious"/"benign") to integers (1/0),
and writes stratified train/val JSONL splits.

Usage:
    python scripts/prepare_data.py --src /path/to/llmtrace/benchmarks/datasets
    python scripts/prepare_data.py --src /path/to/datasets --split 0.8 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

LABEL_MAP: dict[str, int] = {"malicious": 1, "benign": 0}


def load_datasets(src_dir: Path) -> list[dict[str, str | int]]:
    """Load all JSON dataset files, extract text + label."""
    samples: list[dict[str, str | int]] = []
    seen_texts: set[str] = set()

    json_files = sorted(src_dir.glob("*.json")) + sorted(
        (src_dir / "external").glob("*.json")
    )
    if not json_files:
        print(f"ERROR: no JSON files found in {src_dir}", file=sys.stderr)
        sys.exit(1)

    for path in json_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, list), f"{path.name}: expected JSON array, got {type(data)}"

        skipped = 0
        added = 0
        for record in data:
            text = record.get("text", "").strip()
            label_str = record.get("label", "")

            if not text:
                skipped += 1
                continue
            if label_str not in LABEL_MAP:
                skipped += 1
                continue
            if text in seen_texts:
                skipped += 1
                continue

            seen_texts.add(text)
            samples.append({"text": text, "label": LABEL_MAP[label_str]})
            added += 1

        print(f"  {path.name}: {added} added, {skipped} skipped (empty/dup/unknown label)")

    return samples


def stratified_split(
    samples: list[dict[str, str | int]],
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]]]:
    """Split samples into train/val preserving label ratio."""
    rng = random.Random(seed)

    malicious = [s for s in samples if s["label"] == 1]
    benign = [s for s in samples if s["label"] == 0]

    rng.shuffle(malicious)
    rng.shuffle(benign)

    def split_list(lst: list[dict[str, str | int]]) -> tuple[
        list[dict[str, str | int]], list[dict[str, str | int]]
    ]:
        n = int(len(lst) * train_ratio)
        return lst[:n], lst[n:]

    mal_train, mal_val = split_list(malicious)
    ben_train, ben_val = split_list(benign)

    train = mal_train + ben_train
    val = mal_val + ben_val

    rng.shuffle(train)
    rng.shuffle(val)

    return train, val


def write_jsonl(samples: list[dict[str, str | int]], path: Path) -> None:
    """Write samples as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare DeBERTa data from llmtrace datasets")
    p.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to llmtrace/benchmarks/datasets/",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Output directory (default: ../data/)",
    )
    p.add_argument("--split", type=float, default=0.8, help="Train ratio (default: 0.8)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = p.parse_args()

    assert args.src.is_dir(), f"Source directory not found: {args.src}"
    assert 0.5 < args.split < 1.0, f"Split ratio must be in (0.5, 1.0), got {args.split}"

    print(f"Loading datasets from {args.src} ...")
    samples = load_datasets(args.src)

    n_malicious = sum(1 for s in samples if s["label"] == 1)
    n_benign = len(samples) - n_malicious
    print(f"\nTotal: {len(samples)} unique samples ({n_malicious} malicious, {n_benign} benign)")

    train, val = stratified_split(samples, args.split, args.seed)

    train_mal = sum(1 for s in train if s["label"] == 1)
    val_mal = sum(1 for s in val if s["label"] == 1)
    print(f"Train: {len(train)} ({train_mal} malicious, {len(train) - train_mal} benign)")
    print(f"Val:   {len(val)} ({val_mal} malicious, {len(val) - val_mal} benign)")

    train_path = args.out / "train.jsonl"
    val_path = args.out / "val.jsonl"
    write_jsonl(train, train_path)
    write_jsonl(val, val_path)

    print(f"\nWritten: {train_path} ({len(train)} lines)")
    print(f"Written: {val_path} ({len(val)} lines)")


if __name__ == "__main__":
    main()
