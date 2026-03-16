"""Mutable training script (autoresearch-like example)."""

from __future__ import annotations

import json
import os
import time

TIME_BUDGET_S = 1.5

def _env_params() -> dict[str, object]:
    if "AR_PARAMS_JSON" in os.environ:
        try:
            return json.loads(os.environ["AR_PARAMS_JSON"])
        except json.JSONDecodeError:
            return {}
    params: dict[str, object] = {}
    for k, v in os.environ.items():
        if k.startswith("AR_PARAM_"):
            params[k[len("AR_PARAM_"):].lower()] = v
    return params


def run(learning_rate: float, use_qk_norm: bool, grad_clip: float) -> tuple[float, float, int]:
    # simple deterministic loop bounded by fixed wall-clock budget
    start = time.monotonic()
    step = 0
    val_bpb = 1.35

    lr_penalty = abs(learning_rate - 2.6e-3) * 130.0
    qk_bonus = 0.05 if use_qk_norm else 0.0
    clip_penalty = abs(grad_clip - 0.85) * 0.3

    while time.monotonic() - start < TIME_BUDGET_S:
        step += 1
        # deterministic improvement curve
        val_bpb = max(1.05, val_bpb - 0.0004 - qk_bonus + lr_penalty + clip_penalty)
        time.sleep(0.01)

    loss = 2.0 + (val_bpb - 1.1) * 0.7
    return loss, val_bpb, step


def main() -> None:
    env = _env_params()

    learning_rate = float(env.get("learning_rate", 2.8e-3))
    grad_clip = float(env.get("grad_clip", 0.8))
    use_qk_norm = str(env.get("use_qk_norm", "")).lower() in {"1", "true", "yes"}

    t0 = time.monotonic()
    loss, val_bpb, steps = run(learning_rate, use_qk_norm, grad_clip)
    elapsed = time.monotonic() - t0
    print(f"loss={loss:.6f}")
    print(f"val_bpb={val_bpb:.6f}")
    print(f"training_seconds={elapsed:.2f}")
    print(f"num_steps={steps}")


if __name__ == "__main__":
    main()
