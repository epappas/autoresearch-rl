from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SDFTConfig:
    temperature: float = 2.0
    top_k: int = 10
    confidence_threshold: float = 0.5


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature scaling."""
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def compute_sdft_loss(
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
    temperature: float = 2.0,
) -> float:
    """Forward KL divergence: KL(teacher || student) with temperature."""
    p = softmax(teacher_logits, temperature)
    q = softmax(student_logits, temperature)
    log_p = np.log(p + 1e-8)
    log_q = np.log(q + 1e-8)
    return float(np.sum(p * (log_p - log_q)))


def apply_top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep only top-K logits, set rest to -inf."""
    if k >= len(logits):
        return logits.copy()
    result = np.full_like(logits, -np.inf)
    top_indices = np.argpartition(logits, -k)[-k:]
    result[top_indices] = logits[top_indices]
    return result


def should_distill(confidence: float, threshold: float) -> bool:
    """Gate distillation by confidence threshold."""
    return confidence >= threshold
