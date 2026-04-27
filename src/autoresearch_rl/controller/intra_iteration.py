"""IntraIterationGuard — cooperative cancellation of in-flight trials.

The guard wraps the existing power-law forecaster (forecasting.should_early_stop)
and applies it to the live progress series of a single trial. When the forecast
says the trial cannot beat the current best, the guard signals cancel.

Cancel propagates via target/progress.py: the controller writes
$AR_CONTROL_FILE; emit_progress() reads it on its next call and sys.exit(42).
The trial owns the exit timing — no signals, no kill -9.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from autoresearch_rl.controller.shutdown import ShutdownHandler
from autoresearch_rl.forecasting import should_early_stop
from autoresearch_rl.target.progress_reader import ProgressReader

logger = logging.getLogger(__name__)


GuardDecision = Literal["continue", "cancel"]


@dataclass
class GuardConfig:
    enabled: bool = True
    min_steps: int = 5
    poll_interval_s: float = 5.0
    min_reports_before_decide: int = 5


class IntraIterationGuard:
    """Watches a ProgressReader and decides when to cancel.

    Stateless evaluation: `evaluate(metric, best_value, direction)` returns
    ('continue' | 'cancel', reason). The watcher thread (start/stop) drives
    evaluation periodically and writes the cancel control file.

    Cancellation cost is bounded: the guard waits for `min_reports_before_decide`
    reports AND `min_steps` worth of trial steps before it can decide cancel.
    """

    def __init__(
        self,
        *,
        reader: ProgressReader,
        control_path: str,
        metric: str,
        direction: str,
        best_value: float | None,
        config: GuardConfig,
    ) -> None:
        self._reader = reader
        self._control_path = Path(control_path)
        self._metric = metric
        self._direction = direction
        self._best_value = best_value
        self._cfg = config
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cancelled = False
        self._cancel_reason = ""
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def cancel_reason(self) -> str:
        return self._cancel_reason

    def start(self, *, shutdown: ShutdownHandler | None = None) -> None:
        if not self._cfg.enabled:
            return
        if self._best_value is None:
            logger.debug("IntraIterationGuard: no best_value yet; running unguarded")
            return
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, args=(shutdown,), daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self._cfg.poll_interval_s + 1.0)
            self._thread = None

    def evaluate(self, series: list[float]) -> tuple[GuardDecision, str]:
        """Decide given the metric series so far."""
        if self._best_value is None:
            return ("continue", "no_best_yet")
        if len(series) < max(self._cfg.min_reports_before_decide, 5):
            return ("continue", "insufficient_reports")
        if self._direction == "max":
            # power-law forecaster assumes minimization. Negate.
            negated_best = -float(self._best_value)
            negated_series = [-v for v in series]
            if should_early_stop(negated_series, negated_best):
                return ("cancel", "forecast_below_best")
            return ("continue", "forecast_above_best")
        # min direction
        if should_early_stop(series, float(self._best_value)):
            return ("cancel", "forecast_above_best")
        return ("continue", "forecast_below_best")

    def _loop(self, shutdown: ShutdownHandler | None) -> None:
        while not self._stop.is_set():
            time.sleep(self._cfg.poll_interval_s)
            if shutdown is not None and shutdown.requested:
                return
            with self._lock:
                if self._cancelled:
                    return
                reports = self._reader.drain()
                if not reports:
                    continue
                latest_step = reports[-1].step
                if latest_step < self._cfg.min_steps:
                    continue
                series = [
                    float(r.metrics[self._metric])
                    for r in reports
                    if self._metric in r.metrics
                ]
                if len(series) < self._cfg.min_reports_before_decide:
                    continue
                decision, reason = self.evaluate(series)
                if decision == "cancel":
                    self._write_cancel(reason)
                    self._cancelled = True
                    self._cancel_reason = reason
                    return

    def _write_cancel(self, reason: str) -> None:
        try:
            self._control_path.parent.mkdir(parents=True, exist_ok=True)
            self._control_path.write_text(
                json.dumps({"action": "cancel", "reason": reason}),
                encoding="utf-8",
            )
            logger.info("IntraIterationGuard: cancel signaled (%s)", reason)
        except OSError as exc:
            logger.warning("IntraIterationGuard: failed to write cancel: %s", exc)
