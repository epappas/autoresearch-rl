from __future__ import annotations

from autoresearch_rl.promotion import PromotionConfig, PromotionTracker


class TestConsecutiveImprovements:
    def test_no_promotion_below_threshold(self) -> None:
        tracker = PromotionTracker(config=PromotionConfig(promotion_threshold=3))
        tracker.record_result(1.0, improved=True)
        tracker.record_result(0.9, improved=True)
        assert not tracker.should_promote

    def test_promotion_at_threshold(self) -> None:
        tracker = PromotionTracker(config=PromotionConfig(promotion_threshold=3))
        tracker.record_result(1.0, improved=True)
        tracker.record_result(0.9, improved=True)
        tracker.record_result(0.8, improved=True)
        assert tracker.should_promote

    def test_reset_on_failure(self) -> None:
        tracker = PromotionTracker(config=PromotionConfig(promotion_threshold=3))
        tracker.record_result(1.0, improved=True)
        tracker.record_result(0.9, improved=True)
        tracker.record_result(1.1, improved=False)
        assert not tracker.should_promote

    def test_reset_then_re_accumulate(self) -> None:
        tracker = PromotionTracker(config=PromotionConfig(promotion_threshold=2))
        tracker.record_result(1.0, improved=True)
        tracker.record_result(1.1, improved=False)
        tracker.record_result(0.9, improved=True)
        tracker.record_result(0.8, improved=True)
        assert tracker.should_promote


class TestPromote:
    def test_promote_resets_consecutive(self) -> None:
        tracker = PromotionTracker(config=PromotionConfig(promotion_threshold=2))
        tracker.record_result(1.0, improved=True)
        tracker.record_result(0.9, improved=True)
        assert tracker.should_promote
        tracker.promote(version=1)
        assert not tracker.should_promote

    def test_last_promoted_version_none_initially(self) -> None:
        tracker = PromotionTracker()
        assert tracker.last_promoted_version is None

    def test_last_promoted_version_tracks(self) -> None:
        tracker = PromotionTracker()
        tracker.promote(version=5)
        assert tracker.last_promoted_version == 5
        tracker.promote(version=12)
        assert tracker.last_promoted_version == 12


class TestShouldRollback:
    def test_no_rollback_with_insufficient_history(self) -> None:
        tracker = PromotionTracker(config=PromotionConfig(degradation_window=5))
        for _ in range(4):
            tracker.record_result(2.0, improved=False)
        assert not tracker.should_rollback

    def test_no_rollback_when_recent_includes_best(self) -> None:
        tracker = PromotionTracker(config=PromotionConfig(degradation_window=3))
        tracker.record_result(1.0, improved=True)
        tracker.record_result(0.5, improved=True)
        tracker.record_result(0.5, improved=False)
        assert not tracker.should_rollback

    def test_rollback_on_sustained_degradation(self) -> None:
        cfg = PromotionConfig(degradation_window=3)
        tracker = PromotionTracker(config=cfg)
        tracker.record_result(1.0, improved=True)
        tracker.record_result(1.5, improved=False)
        tracker.record_result(1.5, improved=False)
        tracker.record_result(1.5, improved=False)
        assert tracker.should_rollback

    def test_no_rollback_if_one_score_close_to_best(self) -> None:
        cfg = PromotionConfig(degradation_window=3)
        tracker = PromotionTracker(config=cfg)
        tracker.record_result(1.0, improved=True)
        tracker.record_result(1.05, improved=False)  # within 10% of best
        tracker.record_result(1.5, improved=False)
        tracker.record_result(1.5, improved=False)
        assert not tracker.should_rollback
