"""Frozen data preparation for deberta-prompt-injection.

Handles dataset loading and tokenization. The RL loop modifies train.py
but must not modify this file.
"""
from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def data_paths() -> dict[str, str]:
    """Return canonical paths to the data splits."""
    return {
        "train": str(DATA_DIR / "train.jsonl"),
        "val": str(DATA_DIR / "val.jsonl"),
    }


def load_splits(train_file: str | None = None, val_file: str | None = None):
    """Load and return train/val dataset splits."""
    from datasets import load_dataset

    paths = data_paths()
    return load_dataset(
        "json",
        data_files={
            "train": train_file or paths["train"],
            "validation": val_file or paths["val"],
        },
    )


def tokenize(ds, tokenizer, max_length: int = 256):
    """Tokenize dataset splits with the given tokenizer."""
    return ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
    )


if __name__ == "__main__":
    paths = data_paths()
    for split, path in paths.items():
        exists = "ok" if Path(path).exists() else "MISSING"
        print(f"{split}: {path} ({exists})")
