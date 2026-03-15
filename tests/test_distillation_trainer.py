from __future__ import annotations

import numpy as np

from autoresearch_rl.distillation.sdft import SDFTConfig, compute_sdft_loss
from autoresearch_rl.distillation.sink import DistillationSample, DistillationSink
from autoresearch_rl.distillation.trainer import DistillTrainResult, DistillationTrainer


def _make_sample(
    *,
    eval_score: float = 0.0,
    hint: str = "",
    teacher_logits: list[float] | None = None,
    student_logits: list[float] | None = None,
) -> DistillationSample:
    return DistillationSample(
        hint=hint,
        eval_score=eval_score,
        teacher_logits=teacher_logits,
        student_logits=student_logits,
    )


def test_maybe_train_returns_none_when_not_ready() -> None:
    sink = DistillationSink(batch_size=4)
    sink.add(_make_sample(eval_score=1.0))
    trainer = DistillationTrainer()
    result = trainer.maybe_train(sink)
    assert result is None
    assert len(sink) == 1


def test_maybe_train_returns_result_when_ready() -> None:
    sink = DistillationSink(batch_size=2)
    sink.add(_make_sample(eval_score=-0.5))
    sink.add(_make_sample(eval_score=-1.0))
    trainer = DistillationTrainer()
    result = trainer.maybe_train(sink)
    assert result is not None
    assert isinstance(result, DistillTrainResult)
    assert result.num_samples == 2
    assert result.loss > 0.0
    assert len(sink) == 0


def test_train_with_logits_computes_sdft_loss() -> None:
    teacher = [2.0, 1.0, 0.5]
    student = [1.0, 1.0, 1.0]
    samples = [
        _make_sample(teacher_logits=teacher, student_logits=student),
    ]
    trainer = DistillationTrainer(config=SDFTConfig(temperature=2.0))
    result = trainer._train_on_batch(samples)
    assert result.loss > 0.0
    assert result.teacher_entropy > 0.0
    assert result.student_entropy > 0.0

    expected = compute_sdft_loss(
        np.array(teacher), np.array(student), temperature=2.0
    )
    assert abs(result.loss - expected) < 1e-6


def test_train_with_hints_uses_eval_score_proxy() -> None:
    samples = [
        _make_sample(eval_score=-2.0, hint="try larger lr"),
        _make_sample(eval_score=1.0, hint="good"),
    ]
    trainer = DistillationTrainer()
    result = trainer._train_on_batch(samples)
    # proxy: max(0, -eval_score) -> 2.0 and 0.0 -> avg 1.0
    assert abs(result.loss - 1.0) < 1e-8
    assert result.teacher_entropy == 0.0
    assert result.student_entropy == 0.0


def test_total_updates_increments() -> None:
    trainer = DistillationTrainer()
    assert trainer.total_updates == 0

    sink = DistillationSink(batch_size=1)
    sink.add(_make_sample(eval_score=-1.0))
    trainer.maybe_train(sink)
    assert trainer.total_updates == 1

    sink.add(_make_sample(eval_score=-2.0))
    trainer.maybe_train(sink)
    assert trainer.total_updates == 2


def test_average_loss_tracks() -> None:
    trainer = DistillationTrainer()
    assert trainer.average_loss == 0.0

    sink = DistillationSink(batch_size=1)
    sink.add(_make_sample(eval_score=-3.0))
    trainer.maybe_train(sink)
    assert abs(trainer.average_loss - 3.0) < 1e-8

    sink.add(_make_sample(eval_score=-1.0))
    trainer.maybe_train(sink)
    assert abs(trainer.average_loss - 2.0) < 1e-8
