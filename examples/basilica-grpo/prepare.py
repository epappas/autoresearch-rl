"""Frozen dataset utilities for basilica-grpo.

Handles GSM8K loading and prompt formatting. The RL loop modifies train.py
but must not modify this file.
"""
from __future__ import annotations

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def format_prompt(question: str) -> str:
    """Format a GSM8K question as a prompt."""
    return (
        "Solve the following math problem step by step.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def load_train_dataset(max_samples: int = 500):
    """Load and format the GSM8K training split."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds.map(
        lambda x: {"prompt": format_prompt(x["question"])},
        remove_columns=ds.column_names,
    )


def load_eval_dataset(max_samples: int = 200):
    """Load the GSM8K test split for evaluation."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


def build_answer_map() -> dict[str, str]:
    """Build a prompt -> ground-truth answer map for the training split."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    return {format_prompt(q): a for q, a in zip(ds["question"], ds["answer"])}


if __name__ == "__main__":
    train = load_train_dataset(max_samples=5)
    print(f"Train samples: {len(train)}")
    print(f"Sample:\n{train[0]['prompt']}")
