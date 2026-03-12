from __future__ import annotations

import argparse


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

    loss, val_bpb = synthetic_metrics(
        learning_rate=args.learning_rate,
        use_qk_norm=args.use_qk_norm,
        grad_clip=args.grad_clip,
    )

    print(f"loss={loss:.4f}")
    print(f"val_bpb={val_bpb:.4f}")


if __name__ == "__main__":
    main()
