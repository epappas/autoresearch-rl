"""Chrome trace / Perfetto timeline export for autoresearch-rl.

Writes incrementally to a single JSON array on disk that can be opened in
chrome://tracing or ui.perfetto.dev. Each "slice" is a Chrome trace
"complete event" (ph='X') with explicit duration, so nesting/overlap
visualization works without the begin/end (B/E) bookkeeping.

The recorder is no-op when constructed with path=None — every wiring site
checks recorder.enabled before timing, so disabling timeline costs nothing
beyond a couple of attribute reads per phase.

Format reference: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class TimelineRecorder:
    """Append-only writer of Chrome trace events to a JSON array file.

    Incremental write strategy: open as '[' on first write, append
    ',\\n{...}' for each subsequent event, close with '\\n]' on shutdown.
    Chrome trace parser tolerates a missing trailing ']' so the file is
    readable even mid-run (you just lose the last unflushed event).
    """

    def __init__(self, path: str | None, *, pid: int | None = None) -> None:
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._first = True
        self._closed = False
        self._pid = pid if pid is not None else os.getpid()

    @property
    def enabled(self) -> bool:
        return self._path is not None and not self._closed

    def slice(
        self,
        name: str,
        *,
        category: str,
        start_ts_us: int,
        duration_us: int,
        args: dict | None = None,
        tid: int | None = None,
    ) -> None:
        """Record one complete event. Times in microseconds."""
        if not self.enabled:
            return
        event = {
            "name": name,
            "cat": category,
            "ph": "X",
            "ts": int(start_ts_us),
            "dur": int(max(0, duration_us)),
            "pid": self._pid,
            "tid": tid if tid is not None else threading.get_ident(),
        }
        if args:
            event["args"] = args
        self._append(event)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        category: str,
        args: dict | None = None,
        tid: int | None = None,
    ) -> Iterator[dict]:
        """Context manager wrapping slice() with automatic timing.

        Yields a mutable dict the caller can stuff with extra args at
        end-of-span (e.g. result status). The dict is merged into the
        event's args field on exit.
        """
        if not self.enabled:
            yield {}
            return
        start_us = self._now_us()
        end_args: dict = dict(args or {})
        try:
            yield end_args
        finally:
            end_us = self._now_us()
            self.slice(
                name,
                category=category,
                start_ts_us=start_us,
                duration_us=end_us - start_us,
                args=end_args or None,
                tid=tid,
            )

    def close(self) -> None:
        if not self.enabled:
            self._closed = True
            return
        with self._lock:
            assert self._path is not None
            try:
                with open(self._path, "a", encoding="utf-8") as f:
                    if self._first:
                        # No events were ever written — produce a valid empty array.
                        f.write("[]")
                    else:
                        f.write("\n]")
            except OSError:
                pass
            self._closed = True

    def _append(self, event: dict) -> None:
        assert self._path is not None
        line = json.dumps(event, separators=(",", ":"))
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "a", encoding="utf-8") as f:
                    if self._first:
                        f.write("[\n" + line)
                        self._first = False
                    else:
                        f.write(",\n" + line)
            except OSError:
                pass

    @staticmethod
    def _now_us() -> int:
        return int(time.monotonic() * 1_000_000)


_GLOBAL: TimelineRecorder | None = None


def set_global(recorder: TimelineRecorder | None) -> None:
    """Install the recorder used by free-function callers (LLM policies, etc.).

    The engine sets this at run start so policies don't need a recorder
    threaded through their constructors.
    """
    global _GLOBAL
    _GLOBAL = recorder


def get_global() -> TimelineRecorder | None:
    return _GLOBAL


@contextmanager
def global_span(name: str, *, category: str, args: dict | None = None) -> Iterator[dict]:
    """Convenience for callers that just want to time a block via the global recorder."""
    recorder = _GLOBAL
    if recorder is None or not recorder.enabled:
        yield {}
        return
    with recorder.span(name, category=category, args=args) as end_args:
        yield end_args
