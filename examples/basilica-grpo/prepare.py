"""Frozen data preparation for basilica-grpo.

This script runs ONCE before the training loop as a pipeline step
driven by config.yaml's prepare_cmd. It produces data files that
train.py reads at each iteration.

Outputs:
    /app/data/train.jsonl    - formatted training prompts with answers
    /app/data/eval.jsonl     - formatted eval prompts with answers
    /app/data/config.json    - model name, prompt format metadata

The RL loop modifies train.py but must not modify this file.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_DIR = os.environ.get("AR_DATA_DIR", "/app/data")


def format_prompt(question: str) -> str:
    """Format a GSM8K question with explicit answer format instruction."""
    return (
        "Solve the following math problem step by step. "
        "Show your reasoning, then give the final numeric answer "
        "after '####'.\n\n"
        f"Question: {question}"
    )


def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer from a model response.

    Handles multiple formats:
    - GSM8K standard: #### 42
    - Natural language: The answer is 42
    - Boxed LaTeX: \\boxed{42}
    - Bare trailing number: ...therefore 42
    """
    patterns = [
        r"####\s*([\d,]+(?:\.\d+)?)",
        r"\\boxed\{([\d,]+(?:\.\d+)?)\}",
        r"(?:answer|result|total)\s*(?:is|=|:)\s*\$?([\d,]+(?:\.\d+)?)",
        r"([\d,]+(?:\.\d+)?)\s*$",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).replace(",", "").replace("$", "").strip()
    return None


def main() -> None:
    from datasets import load_dataset

    out = Path(DATA_DIR)
    out.mkdir(parents=True, exist_ok=True)

    max_train = int(os.environ.get("AR_MAX_TRAIN", "500"))
    max_eval = int(os.environ.get("AR_MAX_EVAL", "200"))

    print(f"[prepare] loading GSM8K (train={max_train}, eval={max_eval})", flush=True)

    train_ds = load_dataset("openai/gsm8k", "main", split="train")
    test_ds = load_dataset("openai/gsm8k", "main", split="test")

    # Write training data
    train_path = out / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(train_ds):
            if i >= max_train:
                break
            json.dump({
                "prompt": format_prompt(item["question"]),
                "question": item["question"],
                "answer": item["answer"],
                "expected": extract_answer(item["answer"]),
            }, f)
            f.write("\n")

    # Write eval data
    eval_path = out / "eval.jsonl"
    with open(eval_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(test_ds):
            if i >= max_eval:
                break
            json.dump({
                "prompt": format_prompt(item["question"]),
                "question": item["question"],
                "answer": item["answer"],
                "expected": extract_answer(item["answer"]),
            }, f)
            f.write("\n")

    # Write config for train.py to read
    config_path = out / "config.json"
    config = {
        "model_name": MODEL_NAME,
        "train_samples": min(max_train, len(train_ds)),
        "eval_samples": min(max_eval, len(test_ds)),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"[prepare] wrote {train_path} ({config['train_samples']} samples)", flush=True)
    print(f"[prepare] wrote {eval_path} ({config['eval_samples']} samples)", flush=True)
    print(f"[prepare] wrote {config_path}", flush=True)


if __name__ == "__main__":
    main()
