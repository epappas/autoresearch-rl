from __future__ import annotations

import logging
import signal
import threading
from types import FrameType

logger = logging.getLogger(__name__)


class ShutdownHandler:
    """Thread-safe graceful shutdown handler.

    Captures SIGTERM and SIGINT so the continuous loop can finish
    its current iteration and exit cleanly instead of corrupting
    mid-write files.
    """

    def __init__(self) -> None:
        self._requested = threading.Event()

    @property
    def requested(self) -> bool:
        return self._requested.is_set()

    def request_shutdown(
        self, signum: int = 0, frame: FrameType | None = None
    ) -> None:
        sig_name = signal.Signals(signum).name if signum else "manual"
        logger.info("Shutdown requested via %s; finishing current iteration", sig_name)
        self._requested.set()

    def register(self) -> None:
        """Register SIGTERM and SIGINT handlers."""
        signal.signal(signal.SIGTERM, self.request_shutdown)
        signal.signal(signal.SIGINT, self.request_shutdown)
