from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from autoresearch_rl.distillation.sdft import SDFTConfig, compute_sdft_loss, softmax
from autoresearch_rl.distillation.sink import DistillationSample, DistillationSink


@dataclass
class DistillTrainResult:
    """Result of a distillation training step."""

    loss: float
    num_samples: int
    teacher_entropy: float
    student_entropy: float


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy: -sum(p * log(p))."""
    return float(-np.sum(probs * np.log(probs + 1e-8)))


class DistillationTrainer:
    """Orchestrates SDFT distillation from collected samples.

    Drains the DistillationSink when ready, computes SDFT loss
    between teacher and student logits, and returns training results.
    """

    def __init__(self, config: SDFTConfig | None = None) -> None:
        self._config = config or SDFTConfig()
        self._total_updates = 0
        self._cumulative_loss = 0.0

    def maybe_train(
        self, sink: DistillationSink
    ) -> DistillTrainResult | None:
        """Run distillation if sink has enough samples."""
        if not sink.ready:
            return None
        batch = sink.flush()
        return self._train_on_batch(batch)

    def _train_on_batch(
        self, batch: list[DistillationSample]
    ) -> DistillTrainResult:
        """Compute SDFT loss over a batch of samples."""
        total_loss = 0.0
        total_teacher_ent = 0.0
        total_student_ent = 0.0
        count = 0

        for sample in batch:
            if sample.teacher_logits and sample.student_logits:
                teacher = np.array(sample.teacher_logits)
                student = np.array(sample.student_logits)
                loss = compute_sdft_loss(
                    teacher, student, self._config.temperature
                )
                total_loss += loss
                total_teacher_ent += _entropy(
                    softmax(teacher, self._config.temperature)
                )
                total_student_ent += _entropy(
                    softmax(student, self._config.temperature)
                )
                count += 1
            else:
                total_loss += max(0.0, -sample.eval_score)
                count += 1

        avg_loss = total_loss / max(1, count)
        self._total_updates += 1
        self._cumulative_loss += avg_loss

        return DistillTrainResult(
            loss=avg_loss,
            num_samples=len(batch),
            teacher_entropy=total_teacher_ent / max(1, count),
            student_entropy=total_student_ent / max(1, count),
        )

    @property
    def total_updates(self) -> int:
        return self._total_updates

    @property
    def average_loss(self) -> float:
        if self._total_updates == 0:
            return 0.0
        return self._cumulative_loss / self._total_updates
