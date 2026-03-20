"""Frozen evaluation and data utilities for basilica-grpo.

This file defines the immutable boundary: model name, prompt formatting,
answer extraction, reward computation, evaluation, and data loading.
The RL loop modifies train.py but must not modify this file.

train.py imports from this module:
    from prepare import (
        MODEL_NAME, format_prompt, extract_answer, compute_reward,
        evaluate_model, load_train_data, load_eval_data, build_answer_map,
    )
"""
from __future__ import annotations

import re

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def format_prompt(question: str, tokenizer=None) -> str:
    """Format a GSM8K question. Uses chat template if tokenizer provided."""
    content = (
        "Solve the following math problem step by step. "
        "Show your reasoning, then give the final numeric answer "
        "after '####'.\n\n"
        f"Question: {question}"
    )
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return content + "\n\nAnswer:"


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


def compute_reward(completion: str, ground_truth: str) -> float:
    """Binary reward: 1.0 if extracted answer matches, 0.0 otherwise."""
    pred = extract_answer(completion)
    expected = extract_answer(ground_truth)
    if pred and expected and pred.strip() == expected.strip():
        return 1.0
    return 0.0


def evaluate_model(model, tokenizer, eval_ds, max_samples: int = 100) -> float:
    """Evaluate pass@1 accuracy on GSM8K test set with greedy decoding."""
    import torch

    model.eval()
    correct = 0
    total = min(len(eval_ds), max_samples)

    for i in range(total):
        prompt = format_prompt(eval_ds[i]["question"], tokenizer=tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        if compute_reward(response, eval_ds[i]["answer"]) > 0:
            correct += 1

    return correct / max(total, 1)


def load_train_data(max_samples: int = 500):
    """Load the raw GSM8K training split (question + answer columns)."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


def load_eval_data(max_samples: int = 200):
    """Load the GSM8K test split for evaluation."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


def build_answer_map(tokenizer=None) -> dict[str, str]:
    """Build a prompt -> ground-truth answer map for the training split."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    return {
        format_prompt(q, tokenizer=tokenizer): a
        for q, a in zip(ds["question"], ds["answer"])
    }


if __name__ == "__main__":
    train = load_train_data(max_samples=5)
    print(f"Train samples: {len(train)}")
    print(f"Sample prompt:\n{format_prompt(train[0]['question'])}")
    print(f"\nSample answer extraction: {extract_answer(train[0]['answer'])}")
