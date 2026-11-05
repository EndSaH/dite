"""Cooperative cancellation for extraction worker threads.

The extraction stage runs files on worker threads, but only the main thread
receives Ctrl+C. When it does, the stage sets a ``threading.Event``; work
running inside ``cancellation_scope`` with that event checks it before
starting expensive steps (a Docling subprocess, VLM requests) and stops.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class ExtractionCancelledError(BaseException):
    """Raised on a worker thread once the run has been interrupted.

    It derives from ``BaseException``, like ``KeyboardInterrupt``, so the
    broad ``except Exception`` handlers in the extractors let it through.
    """


_scope = threading.local()


@contextmanager
def cancellation_scope(event: threading.Event) -> Iterator[None]:
    """Make ``event`` the cancel signal for work on the current thread."""
    previous = getattr(_scope, "event", None)
    _scope.event = event
    try:
        yield
    finally:
        _scope.event = previous


def current_cancel_event() -> threading.Event | None:
    """Return the current thread's cancel event, or ``None`` outside a scope.

    For handing the signal to work that runs on another thread, such as the
    request runtime's event loop.
    """
    return getattr(_scope, "event", None)


def cancellation_requested() -> bool:
    """Return whether the current thread's scope has been cancelled."""
    event = current_cancel_event()
    return event is not None and event.is_set()


def raise_if_cancelled() -> None:
    """Raise ``ExtractionCancelledError`` if the current scope is cancelled."""
    if cancellation_requested():
        raise ExtractionCancelledError
