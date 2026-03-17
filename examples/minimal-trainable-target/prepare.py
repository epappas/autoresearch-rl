"""Frozen evaluation oracle for minimal-trainable-target.

Defines the canonical objective landscape. The RL loop modifies train.py
but must not modify this file.
"""
from __future__ import annotations

OPTIMAL_LR: float = 2.6e-3
OPTIMAL_GRAD_CLIP: float = 0.85
OPTIMAL_USE_QK_NORM: bool = True


def evaluate(
    learning_rate: float,
    use_qk_norm: bool,
    grad_clip: float,
) -> tuple[float, float]:
    """Canonical objective. Returns (loss, val_bpb). Lower is better."""
    base = 1.35
    lr_penalty = abs(learning_rate - OPTIMAL_LR) * 130.0
    qk_bonus = 0.05 if use_qk_norm else 0.0
    clip_penalty = abs(grad_clip - OPTIMAL_GRAD_CLIP) * 0.3
    val_bpb = base + lr_penalty + clip_penalty - qk_bonus
    loss = 2.10 + (val_bpb - 1.2) * 0.7
    return loss, val_bpb


if __name__ == "__main__":
    loss, val_bpb = evaluate(OPTIMAL_LR, OPTIMAL_USE_QK_NORM, OPTIMAL_GRAD_CLIP)
    print(f"Optimal: loss={loss:.4f} val_bpb={val_bpb:.4f}")
