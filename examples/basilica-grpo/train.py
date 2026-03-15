"""GRPO fine-tuning of Qwen2.5-0.5B-Instruct on GSM8K via TRL.

This script runs inside a Basilica deployment. It reads hyperparameters
from AR_PARAMS_JSON env var, trains the model, evaluates, and prints
metrics to stdout for the autoresearch-rl controller to parse.

Metrics output format (parsed by autoresearch-rl):
    eval_score=0.5500
    val_bpb=0.4500
    loss=0.3200
    training_seconds=580.0
"""
from __future__ import annotations

import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_params() -> dict[str, object]:
    """Load hyperparameters from AR_PARAMS_JSON env var."""
    raw = os.environ.get("AR_PARAMS_JSON", "{}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def build_dataset(max_samples: int = 500):
    """Load GSM8K training split."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds.map(
        lambda x: {"prompt": format_prompt(x["question"])},
        remove_columns=ds.column_names,
    )


def build_eval_dataset(max_samples: int = 200):
    """Load GSM8K test split for evaluation."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


def format_prompt(question: str) -> str:
    return (
        f"Solve the following math problem step by step.\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer from a GSM8K-style response."""
    import re
    patterns = [
        r"####\s*([\d,.-]+)",
        r"(?:answer|result)\s*(?:is|=)\s*([\d,.-]+)",
        r"([\d,]+(?:\.\d+)?)\s*$",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "").strip()
    return None


def gsm8k_reward(completions: list[str], ground_truths: list[str]) -> list[float]:
    """Compute reward: 1.0 if extracted answer matches ground truth, else 0.0."""
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        pred = extract_answer(completion)
        expected = extract_answer(gt)
        if pred is not None and expected is not None:
            rewards.append(1.0 if pred.strip() == expected.strip() else 0.0)
        else:
            rewards.append(0.0)
    return rewards


def evaluate_model(model, tokenizer, eval_ds, max_samples: int = 100) -> float:
    """Evaluate pass@1 on GSM8K test set."""
    model.eval()
    correct = 0
    total = min(len(eval_ds), max_samples)

    for i in range(total):
        question = eval_ds[i]["question"]
        ground_truth = eval_ds[i]["answer"]

        prompt = format_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        pred = extract_answer(response)
        expected = extract_answer(ground_truth)

        if pred is not None and expected is not None:
            if pred.strip() == expected.strip():
                correct += 1

    return correct / max(total, 1)


def main() -> None:
    t0 = time.time()
    params = load_params()

    lr = float(params.get("learning_rate", 5e-6))
    batch_size = int(params.get("batch_size", 4))
    max_steps = int(params.get("max_steps", 20))
    grad_clip = float(params.get("grad_clip", 1.0))
    num_generations = int(params.get("num_generations", 4))
    temperature = float(params.get("temperature", 1.0))

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading GSM8K dataset...")
    train_ds = build_dataset(max_samples=500)
    eval_ds = build_eval_dataset(max_samples=100)

    # Evaluate baseline before training
    print("Evaluating baseline...")
    baseline_score = evaluate_model(model, tokenizer, eval_ds, max_samples=50)
    print(f"baseline_score={baseline_score:.6f}")

    config = GRPOConfig(
        output_dir="/tmp/grpo_output",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=1,
        max_steps=max_steps,
        max_grad_norm=grad_clip,
        num_generations=num_generations,
        temperature=temperature,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        bf16=True,
        remove_unused_columns=False,
    )

    # Define reward function for GRPO
    train_answers = load_dataset("openai/gsm8k", "main", split="train")
    answer_map = {
        format_prompt(q): a
        for q, a in zip(train_answers["question"], train_answers["answer"])
    }

    def reward_fn(completions: list[str], prompts: list[str]) -> list[float]:
        rewards = []
        for comp, prompt in zip(completions, prompts):
            gt = answer_map.get(prompt, "")
            pred = extract_answer(comp)
            expected = extract_answer(gt)
            if pred and expected and pred.strip() == expected.strip():
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    print(f"Starting GRPO training: lr={lr}, batch={batch_size}, "
          f"steps={max_steps}, generations={num_generations}")

    trainer = GRPOTrainer(
        model=model,
        config=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    trainer.train()
    training_seconds = time.time() - t0

    # Evaluate after training
    print("Evaluating trained model...")
    eval_score = evaluate_model(model, tokenizer, eval_ds, max_samples=100)

    # val_bpb = 1 - eval_score (lower is better, for compatibility)
    val_bpb = 1.0 - eval_score
    loss = float(trainer.state.log_history[-1].get("loss", 0.0)) if trainer.state.log_history else 0.0

    # Print metrics in parseable format
    print(f"eval_score={eval_score:.6f}")
    print(f"val_bpb={val_bpb:.6f}")
    print(f"loss={loss:.6f}")
    print(f"training_seconds={training_seconds:.1f}")
    print(f"baseline={baseline_score:.6f}")
    print(f"improvement={eval_score - baseline_score:.6f}")


if __name__ == "__main__":
    main()
