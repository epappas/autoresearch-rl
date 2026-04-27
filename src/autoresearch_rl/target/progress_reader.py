"""Controller-side reader for emit_progress() reports.

A ProgressReader tails a JSONL file written by the trial. It runs in a
background thread alongside the trial subprocess. The controller drains
reports on demand or when the trial completes.

Used by:
- CommandTarget (Phase 1.2) — file-based.
- BasilicaTarget (Phase 1.3) — proxies the bootstrap server's /progress
  endpoint and writes lines into the same JSONL file shape.
- IntraIterationGuard (Phase 2) — feeds the metric series into the
  forecaster for cooperative cancellation.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from autoresearch_rl.target.progress import ProgressReport

logger = logging.getLogger(__name__)


class ProgressReader:
    """Tails a JSONL progress file. Thread-safe drain()."""

    def __init__(self, path: str, *, poll_interval_s: float = 1.0) -> None:
        self._path = Path(path)
        self._poll_interval_s = poll_interval_s
        self._buffer: list[ProgressReport] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        # Ensure file exists so our tail loop has something to open.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, *, drain_final: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self._poll_interval_s + 1.0)
            self._thread = None
        if drain_final:
            self._read_new_lines()

    def drain(self) -> list[ProgressReport]:
        """Return all buffered reports and clear the buffer."""
        with self._lock:
            out = list(self._buffer)
            self._buffer.clear()
        return out

    def latest(self) -> ProgressReport | None:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._read_new_lines()
            time.sleep(self._poll_interval_s)

    def _read_new_lines(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as f:
                f.seek(self._offset)
                chunk = f.read()
                self._offset = f.tell()
        except OSError:
            return
        if not chunk:
            return
        new_reports: list[ProgressReport] = []
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            report = _parse_line(line)
            if report is not None:
                new_reports.append(report)
        if new_reports:
            with self._lock:
                self._buffer.extend(new_reports)


def _parse_line(line: str) -> ProgressReport | None:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        logger.debug("skipping malformed progress line: %s", line[:120])
        return None
    try:
        return ProgressReport(
            iter=int(data.get("iter", 0)),
            step=int(data.get("step", 0)),
            step_target=int(data.get("step_target", 0)),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
            metrics=dict(data.get("metrics", {})),
            should_continue=bool(data.get("should_continue", True)),
            timestamp=float(data.get("timestamp", time.time())),
        )
    except (TypeError, ValueError):
        return None
