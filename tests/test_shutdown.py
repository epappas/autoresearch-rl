from __future__ import annotations

import signal

from autoresearch_rl.controller.shutdown import ShutdownHandler


def test_handler_starts_not_requested() -> None:
    handler = ShutdownHandler()
    assert not handler.requested


def test_request_shutdown_sets_flag() -> None:
    handler = ShutdownHandler()
    handler.request_shutdown()
    assert handler.requested


def test_request_shutdown_with_signum() -> None:
    handler = ShutdownHandler()
    handler.request_shutdown(signum=signal.SIGTERM.value, frame=None)
    assert handler.requested


def test_register_does_not_crash() -> None:
    handler = ShutdownHandler()
    handler.register()
    # Restore default handlers so other tests are not affected
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def test_requested_property_reflects_internal_event() -> None:
    handler = ShutdownHandler()
    assert not handler.requested
    handler._requested.set()
    assert handler.requested


def test_register_installs_signal_handlers() -> None:
    handler = ShutdownHandler()
    handler.register()
    assert signal.getsignal(signal.SIGTERM) == handler.request_shutdown
    assert signal.getsignal(signal.SIGINT) == handler.request_shutdown
    # Restore defaults
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def test_multiple_shutdown_requests_are_idempotent() -> None:
    handler = ShutdownHandler()
    handler.request_shutdown()
    handler.request_shutdown()
    assert handler.requested
