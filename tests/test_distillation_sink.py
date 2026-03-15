from __future__ import annotations

from autoresearch_rl.distillation.sink import DistillationSample, DistillationSink


def _make_sample(i: int) -> DistillationSample:
    return DistillationSample(hint=f"hint-{i}", eval_score=float(i))


class TestDistillationSinkAdd:
    def test_add_increments_length(self) -> None:
        sink = DistillationSink(batch_size=4)
        assert len(sink) == 0
        sink.add(_make_sample(0))
        assert len(sink) == 1

    def test_add_multiple(self) -> None:
        sink = DistillationSink(batch_size=4)
        for i in range(3):
            sink.add(_make_sample(i))
        assert len(sink) == 3


class TestDistillationSinkReady:
    def test_not_ready_below_batch(self) -> None:
        sink = DistillationSink(batch_size=4)
        for i in range(3):
            sink.add(_make_sample(i))
        assert not sink.ready

    def test_ready_at_batch_size(self) -> None:
        sink = DistillationSink(batch_size=4)
        for i in range(4):
            sink.add(_make_sample(i))
        assert sink.ready

    def test_ready_above_batch_size(self) -> None:
        sink = DistillationSink(batch_size=4)
        for i in range(6):
            sink.add(_make_sample(i))
        assert sink.ready


class TestDistillationSinkFlush:
    def test_flush_returns_batch_size_items(self) -> None:
        sink = DistillationSink(batch_size=4)
        for i in range(6):
            sink.add(_make_sample(i))
        batch = sink.flush()
        assert len(batch) == 4
        assert [s.hint for s in batch] == ["hint-0", "hint-1", "hint-2", "hint-3"]

    def test_flush_removes_returned_items(self) -> None:
        sink = DistillationSink(batch_size=4)
        for i in range(6):
            sink.add(_make_sample(i))
        sink.flush()
        assert len(sink) == 2

    def test_partial_flush_when_under_batch(self) -> None:
        sink = DistillationSink(batch_size=4)
        for i in range(2):
            sink.add(_make_sample(i))
        batch = sink.flush()
        assert len(batch) == 2
        assert len(sink) == 0

    def test_consecutive_flushes(self) -> None:
        sink = DistillationSink(batch_size=3)
        for i in range(7):
            sink.add(_make_sample(i))

        first = sink.flush()
        assert len(first) == 3
        assert first[0].hint == "hint-0"

        second = sink.flush()
        assert len(second) == 3
        assert second[0].hint == "hint-3"

        third = sink.flush()
        assert len(third) == 1
        assert third[0].hint == "hint-6"
        assert len(sink) == 0


class TestDistillationSampleFields:
    def test_default_logits_none(self) -> None:
        s = DistillationSample(hint="h", eval_score=1.0)
        assert s.teacher_logits is None
        assert s.student_logits is None

    def test_logits_stored(self) -> None:
        s = DistillationSample(
            hint="h",
            eval_score=1.0,
            teacher_logits=[0.1, 0.2],
            student_logits=[0.3, 0.4],
        )
        assert s.teacher_logits == [0.1, 0.2]
        assert s.student_logits == [0.3, 0.4]
