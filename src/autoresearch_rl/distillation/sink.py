from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DistillationSample:
    hint: str
    eval_score: float
    teacher_logits: list[float] | None = None
    student_logits: list[float] | None = None


class DistillationSink:
    """Buffers directional hints and teacher signals for batch SDFT updates."""

    def __init__(self, batch_size: int = 16) -> None:
        self._buffer: list[DistillationSample] = []
        self._batch_size = batch_size

    def add(self, sample: DistillationSample) -> None:
        self._buffer.append(sample)

    @property
    def ready(self) -> bool:
        return len(self._buffer) >= self._batch_size

    def flush(self) -> list[DistillationSample]:
        batch = self._buffer[: self._batch_size]
        self._buffer = self._buffer[self._batch_size :]
        return batch

    def __len__(self) -> int:
        return len(self._buffer)
