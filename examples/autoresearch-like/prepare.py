"""Frozen data preparation script (not modified by the RL loop).

This file represents the immutable infrastructure that the research
loop must not alter. The contract system enforces that diffs only
touch train.py (the mutable file), never this file.
"""
from __future__ import annotations


def prepare_data() -> dict[str, str]:
    """Return paths to prepared data splits."""
    return {
        "train": "data/train.jsonl",
        "val": "data/val.jsonl",
        "test": "data/test.jsonl",
    }


if __name__ == "__main__":
    paths = prepare_data()
    for split, path in paths.items():
        print(f"{split}: {path}")
