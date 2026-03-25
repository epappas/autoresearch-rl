"""GRPO fine-tuning of Qwen2.5-0.5B-Instruct on GSM8K.

Pure PyTorch implementation -- no TRL/accelerate dependency.
Runs inside a Basilica deployment with params from AR_PARAMS_JSON.
Reads prepared data from /app/data/ (produced by prepare.py pipeline step).

Algorithm (DeepSeek-R1 GRPO):
  1. Sample prompts, generate G completions per prompt
  2. Score each completion with reward function
  3. Compute per-prompt advantage: reward_i - mean(rewards)
  4. Compute clipped policy gradient loss
  5. Update model

Metrics output (parsed by autoresearch-rl):
    eval_score=0.5500
    loss=0.3200
    training_seconds=580.0
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = os.environ.get("AR_DATA_DIR", "/app/data")


def load_params() -> dict[str, object]:
    raw = os.environ.get("AR_PARAMS_JSON", "{}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def load_data_config() -> dict:
    config_path = Path(DATA_DIR) / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {"model_name": "Qwen/Qwen2.5-0.5B-Instruct"}


def load_jsonl(name: str) -> list[dict]:
    path = Path(DATA_DIR) / name
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}. Run prepare.py first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer from a model response."""
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


def compute_reward(completion: str, expected: str) -> float:
    """Binary reward: 1.0 if extracted answer matches expected, 0.0 otherwise."""
    pred = extract_answer(completion)
    if pred and expected and pred.strip() == expected.strip():
        return 1.0
    return 0.0


def evaluate_model(model, tokenizer, eval_data: list[dict], max_samples: int = 100) -> float:
    """Evaluate pass@1 accuracy on prepared eval data."""
    model.eval()
    correct = 0
    total = min(len(eval_data), max_samples)

    for i in range(total):
        prompt = eval_data[i]["prompt"]
        expected = eval_data[i].get("expected", "")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        if compute_reward(response, expected) > 0:
            correct += 1

    return correct / max(total, 1)


def generate_completions(
    model, tokenizer, prompt: str, num_gen: int, max_new: int, temperature: float,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Generate completions and return (text, token_ids, log_probs)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    results = []

    for _ in range(num_gen):
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new,
                do_sample=True, temperature=max(temperature, 0.01),
                return_dict_in_generate=True, output_scores=True,
            )
        gen_ids = out.sequences[0, input_len:]
        log_probs = []
        for t, scores in enumerate(out.scores):
            lp = F.log_softmax(scores[0], dim=-1)
            log_probs.append(lp[gen_ids[t]].item())

        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append((text, gen_ids, torch.tensor(log_probs, device=model.device)))

    return results


def grpo_step(
    model, ref_model, tokenizer, optimizer,
    prompt: str, expected_answer: str,
    num_gen: int, max_new: int, temperature: float,
    clip_eps: float = 0.2, kl_coeff: float = 0.01,
) -> dict[str, float]:
    """One GRPO update step on a single prompt."""
    completions = generate_completions(
        model, tokenizer, prompt, num_gen, max_new, temperature,
    )

    rewards = [compute_reward(text, expected_answer) for text, _, _ in completions]
    mean_reward = sum(rewards) / len(rewards)
    advantages = [r - mean_reward for r in rewards]

    if all(a == 0.0 for a in advantages):
        return {"loss": 0.0, "reward": mean_reward, "skipped": 1.0}

    model.train()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    total_loss = torch.tensor(0.0, device=model.device)
    n_tokens = 0

    for (text, gen_ids, old_log_probs), advantage in zip(completions, advantages):
        if len(gen_ids) == 0:
            continue

        full_ids = torch.cat([inputs["input_ids"][0], gen_ids]).unsqueeze(0)
        outputs = model(full_ids)
        logits = outputs.logits[0, input_len - 1:-1, :]
        new_log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = new_log_probs.gather(1, gen_ids.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            ref_outputs = ref_model(full_ids)
            ref_logits = ref_outputs.logits[0, input_len - 1:-1, :]
            ref_lp = F.log_softmax(ref_logits, dim=-1)
            ref_token_lp = ref_lp.gather(1, gen_ids.unsqueeze(1)).squeeze(1)

        min_len = min(len(token_log_probs), len(old_log_probs))
        ratio = torch.exp(token_log_probs[:min_len] - old_log_probs[:min_len])
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        adv = torch.tensor(advantage, device=model.device)
        pg_loss = -torch.min(ratio * adv, clipped * adv).mean()

        kl = (token_log_probs[:min_len] - ref_token_lp[:min_len]).mean()
        total_loss = total_loss + pg_loss + kl_coeff * kl
        n_tokens += min_len

    if n_tokens > 0:
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return {"loss": total_loss.item(), "reward": mean_reward, "skipped": 0.0}


def main() -> None:
    t0 = time.time()
    params = load_params()
    data_config = load_data_config()

    lr = float(params.get("learning_rate", 5e-6))
    max_steps = int(params.get("max_steps", 30))
    num_generations = int(params.get("num_generations", 2))
    temperature = float(params.get("temperature", 1.0))
    max_completion = 256

    model_name = data_config.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f}GB", flush=True)

    print(f"Loading model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to(device)

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    train_data = load_jsonl("train.jsonl")
    eval_data = load_jsonl("eval.jsonl")

    print("Evaluating baseline...", flush=True)
    baseline_score = evaluate_model(model, tokenizer, eval_data, max_samples=50)
    print(f"[baseline] {baseline_score:.4f} pass@1", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    print(f"[config] lr {lr}, steps {max_steps}, gen {num_generations}, temp {temperature}", flush=True)

    total_loss = 0.0
    total_reward = 0.0
    n_train = len(train_data)

    for step in range(max_steps):
        item = train_data[step % n_train]
        prompt = item["prompt"]
        expected = item.get("expected", "")

        metrics = grpo_step(
            model, ref_model, tokenizer, optimizer,
            prompt, expected,
            num_gen=num_generations,
            max_new=max_completion,
            temperature=temperature,
        )

        total_loss += metrics["loss"]
        total_reward += metrics["reward"]

        if (step + 1) % 5 == 0 or step == 0:
            avg_loss = total_loss / (step + 1)
            avg_reward = total_reward / (step + 1)
            print(
                f"[step {step + 1}/{max_steps}] "
                f"avg_loss {avg_loss:.4f}, avg_reward {avg_reward:.4f}",
                flush=True,
            )

    training_seconds = time.time() - t0
    print("Training complete.", flush=True)

    # Save model to AR_MODEL_DIR if configured (framework injects this)
    model_dir = os.environ.get("AR_MODEL_DIR")
    if model_dir:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"[model] saved to {model_dir}", flush=True)

    print("Evaluating trained model...", flush=True)
    eval_score = evaluate_model(model, tokenizer, eval_data, max_samples=100)
    avg_loss = total_loss / max(max_steps, 1)

    print(f"eval_score={eval_score:.6f}", flush=True)
    print(f"loss={avg_loss:.6f}", flush=True)
    print(f"training_seconds={training_seconds:.1f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"FATAL: {exc}", flush=True)
        raise SystemExit(1)
