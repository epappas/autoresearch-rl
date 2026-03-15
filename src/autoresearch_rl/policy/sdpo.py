from __future__ import annotations

import numpy as np


def compute_kl_divergence(
    teacher_probs: np.ndarray, student_probs: np.ndarray
) -> float:
    """KL(teacher || student) = sum(p * log(p/q))."""
    p = np.clip(teacher_probs, 1e-8, None)
    q = np.clip(student_probs, 1e-8, None)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def compute_sdpo_loss(
    ppo_loss: float, kl_div: float, alpha: float
) -> float:
    """L_SDPO = L_RL + alpha * KL."""
    return ppo_loss + alpha * kl_div


def compute_adaptive_alpha(
    prev_reward: float, target_reward: float
) -> float:
    """alpha_t = min(1, R_prev / R_target)."""
    if target_reward <= 0:
        return 1.0
    return min(1.0, prev_reward / target_reward)
