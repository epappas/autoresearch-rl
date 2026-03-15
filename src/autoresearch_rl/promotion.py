from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PromotionConfig:
    promotion_threshold: int = 3  # consecutive improvements needed
    degradation_window: int = 10


@dataclass
class PromotionTracker:
    """Track consecutive improvements and detect degradation for policy promotion."""

    config: PromotionConfig = field(default_factory=PromotionConfig)
    _consecutive_improvements: int = 0
    _history: list[float] = field(default_factory=list)
    _promoted_versions: list[int] = field(default_factory=list)

    def record_result(self, score: float, improved: bool) -> None:
        self._history.append(score)
        if improved:
            self._consecutive_improvements += 1
        else:
            self._consecutive_improvements = 0

    @property
    def should_promote(self) -> bool:
        return self._consecutive_improvements >= self.config.promotion_threshold

    @property
    def should_rollback(self) -> bool:
        w = self.config.degradation_window
        if len(self._history) < w:
            return False
        recent = self._history[-w:]
        # Degradation = all recent scores worse than the best ever
        best = min(self._history)  # lower is better
        return all(s > best * 1.1 for s in recent)

    def promote(self, version: int) -> None:
        self._consecutive_improvements = 0
        self._promoted_versions.append(version)

    @property
    def last_promoted_version(self) -> int | None:
        return self._promoted_versions[-1] if self._promoted_versions else None
