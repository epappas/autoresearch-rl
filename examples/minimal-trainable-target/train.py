from __future__ import annotations

import argparse
import json
import os


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


def synthetic_metrics(learning_rate: float, use_qk_norm: bool, grad_clip: float) -> tuple[float, float]:
    """Deterministic toy objective used for scaffold demos.

    Lower is better for both metrics.
    """
    base_val_bpb = 1.35
    base_loss = 2.10

    # Simple hand-crafted landscape with a sweet spot around lr ~= 2.6e-3
    lr_penalty = abs(learning_rate - 2.6e-3) * 130.0

    qk_bonus = 0.05 if use_qk_norm else 0.0
    clip_penalty = abs(grad_clip - 0.85) * 0.3

    val_bpb = base_val_bpb + lr_penalty + clip_penalty - qk_bonus
    loss = base_loss + (val_bpb - 1.2) * 0.7

    return loss, val_bpb


def main() -> None:
    p = argparse.ArgumentParser(description="Toy train.py for scaffold example")
    p.add_argument("--learning-rate", type=float, default=2.8e-3)
    p.add_argument("--use-qk-norm", action="store_true")
    p.add_argument("--grad-clip", type=float, default=0.8)
    args = p.parse_args()
    env = _env_params()

    learning_rate = float(env.get("learning_rate", args.learning_rate))
    grad_clip = float(env.get("grad_clip", args.grad_clip))
    use_qk_norm = args.use_qk_norm or str(env.get("use_qk_norm", "")).lower() in {"1", "true", "yes"}

    loss, val_bpb = synthetic_metrics(
        learning_rate=learning_rate,
        use_qk_norm=use_qk_norm,
        grad_clip=grad_clip,
    )

    print(f"loss={loss:.4f}")
    print(f"val_bpb={val_bpb:.4f}")


if __name__ == "__main__":
    main()
