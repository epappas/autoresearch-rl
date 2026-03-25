"""LoRA + GRPO training for LLM-as-security-judge.

Trains Qwen2.5-0.5B-Instruct with LoRA adapters to output structured
security verdicts: {"decision": "pass|block|warning", "security_score": 0.0-1.0}

Uses GRPO with a multi-component reward:
  0.3 - valid JSON with correct schema
  0.4 - correct decision
  0.3 - calibrated security_score

Reads prepared data from /app/data/ (produced by prepare.py).
Metrics output (parsed by autoresearch-rl):
    eval_score=0.7500
    decision_accuracy=0.8200
    json_compliance=0.9500
"""
from __future__ import annotations

import json
import os
import re
import sys
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
    path = Path(DATA_DIR) / "config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"model_name": "Qwen/Qwen2.5-0.5B-Instruct"}


def load_jsonl(name: str) -> list[dict]:
    path = Path(DATA_DIR) / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run prepare.py first.")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def parse_verdict(text: str) -> dict | None:
    """Extract structured verdict from model output."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None

    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    decision = obj.get("decision", "")
    score = obj.get("security_score")
    if decision not in ("pass", "block", "warning") or score is None:
        return None

    try:
        score = max(0.0, min(1.0, float(score)))
    except (ValueError, TypeError):
        return None

    return {"decision": decision, "security_score": score}


def compute_reward(completion: str, expected: dict) -> float:
    """Multi-component reward for structured judge output."""
    verdict = parse_verdict(completion)
    if verdict is None:
        return 0.0

    reward = 0.3  # valid structured output
    if verdict["decision"] == expected["decision"]:
        reward += 0.4
    score_diff = abs(verdict["security_score"] - expected["security_score"])
    if score_diff <= 0.3:
        reward += 0.3 * (1.0 - score_diff / 0.3)
    return reward


def format_judge_prompt(text: str, system_prompt: str, tokenizer) -> str:
    """Format as chat with judge system prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def evaluate_model(model, tokenizer, eval_data: list[dict], system_prompt: str,
                   max_samples: int = 200) -> dict:
    """Evaluate judge on eval set."""
    model.eval()
    correct = 0
    valid = 0
    total_reward = 0.0
    total = min(len(eval_data), max_samples)

    for i in range(total):
        prompt = format_judge_prompt(eval_data[i]["prompt"], system_prompt, tokenizer)
        expected = eval_data[i]["expected"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )

        verdict = parse_verdict(response)
        reward = compute_reward(response, expected)
        total_reward += reward

        if verdict is not None:
            valid += 1
            if verdict["decision"] == expected["decision"]:
                correct += 1

    return {
        "decision_accuracy": correct / max(total, 1),
        "json_compliance": valid / max(total, 1),
        "avg_reward": total_reward / max(total, 1),
    }


def generate_completions(
    model, tokenizer, prompt: str, num_gen: int, max_new: int, temperature: float,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Generate completions with log probs."""
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
    prompt: str, expected: dict,
    num_gen: int, max_new: int, temperature: float,
    clip_eps: float = 0.2, kl_coeff: float = 0.01,
) -> dict[str, float]:
    """One GRPO step: generate, reward, update."""
    completions = generate_completions(
        model, tokenizer, prompt, num_gen, max_new, temperature,
    )

    rewards = [compute_reward(text, expected) for text, _, _ in completions]
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
        new_lp = F.log_softmax(logits, dim=-1)
        token_lp = new_lp.gather(1, gen_ids.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            ref_out = ref_model(full_ids)
            ref_logits = ref_out.logits[0, input_len - 1:-1, :]
            ref_lp = F.log_softmax(ref_logits, dim=-1)
            ref_token_lp = ref_lp.gather(1, gen_ids.unsqueeze(1)).squeeze(1)

        min_len = min(len(token_lp), len(old_log_probs))
        ratio = torch.exp(token_lp[:min_len] - old_log_probs[:min_len])
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        adv = torch.tensor(advantage, device=model.device)
        pg_loss = -torch.min(ratio * adv, clipped * adv).mean()
        kl = (token_lp[:min_len] - ref_token_lp[:min_len]).mean()

        total_loss = total_loss + pg_loss + kl_coeff * kl
        n_tokens += min_len

    if n_tokens > 0:
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return {"loss": total_loss.item(), "reward": mean_reward, "skipped": 0.0}


def apply_lora(model, rank: int = 8, alpha: int = 16):
    """Apply LoRA adapters to attention layers."""
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def main() -> None:
    t0 = time.time()
    params = load_params()
    data_config = load_data_config()

    lr = float(params.get("learning_rate", 1e-4))
    max_steps = int(params.get("max_steps", 50))
    num_generations = int(params.get("num_generations", 3))
    temperature = float(params.get("temperature", 0.8))
    lora_rank = int(params.get("lora_rank", 8))

    model_name = data_config.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    system_prompt = data_config.get("system_prompt", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f}GB", flush=True)

    print(f"Loading model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to(device)

    # Apply LoRA
    print(f"Applying LoRA (rank={lora_rank})", flush=True)
    model = apply_lora(base_model, rank=lora_rank)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[params] trainable {trainable:,} / {total:,} ({100*trainable/total:.1f}%)", flush=True)

    # Reference model (frozen, no LoRA)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    train_data = load_jsonl("train.jsonl")
    eval_data = load_jsonl("eval.jsonl")

    print("Evaluating baseline...", flush=True)
    baseline = evaluate_model(model, tokenizer, eval_data, system_prompt, max_samples=100)
    print(
        f"[baseline] decision_acc {baseline['decision_accuracy']:.3f}, "
        f"json {baseline['json_compliance']:.3f}, "
        f"reward {baseline['avg_reward']:.3f}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=0.01,
    )

    print(
        f"[config] lr {lr}, steps {max_steps}, gen {num_generations}, "
        f"temp {temperature}, lora_rank {lora_rank}",
        flush=True,
    )

    total_loss = 0.0
    total_reward = 0.0
    n_train = len(train_data)

    for step in range(max_steps):
        item = train_data[step % n_train]
        prompt = format_judge_prompt(item["prompt"], system_prompt, tokenizer)
        expected = item["expected"]

        metrics = grpo_step(
            model, ref_model, tokenizer, optimizer,
            prompt, expected,
            num_gen=num_generations,
            max_new=64,
            temperature=temperature,
        )

        total_loss += metrics["loss"]
        total_reward += metrics["reward"]

        if (step + 1) % 10 == 0 or step == 0:
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
        print(f"[model] saved LoRA adapter to {model_dir}", flush=True)

    print("Evaluating trained model...", flush=True)
    results = evaluate_model(model, tokenizer, eval_data, system_prompt, max_samples=200)

    # Primary metric for autoresearch-rl keep/discard
    eval_score = results["avg_reward"]

    print(f"eval_score={eval_score:.6f}", flush=True)
    print(f"decision_accuracy={results['decision_accuracy']:.6f}", flush=True)
    print(f"json_compliance={results['json_compliance']:.6f}", flush=True)
    print(f"loss={total_loss / max(max_steps, 1):.6f}", flush=True)
    print(f"training_seconds={training_seconds:.1f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"FATAL: {exc}", flush=True)
        raise SystemExit(1)
