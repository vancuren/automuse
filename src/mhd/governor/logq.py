"""Make every `mhd.governor.*` log call non-blocking for the RC thread.

The governor logs rare events (safety landing, wrong-way, exceptions) from inside the RC tick.
A plain StreamHandler writes to stderr synchronously, and a stalled pipe would block the thread
that sends RC packets. install() puts a QueueHandler on the "mhd.governor" logger (no
propagation) and a listener thread hands each record to the root logger's handlers.
"""
from __future__ import annotations

import logging
import logging.handlers
import queue
import threading

_lock = threading.Lock()
_listener = None


class _ToRoot(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        root = logging.getLogger()
        for h in root.handlers:
            if record.levelno >= h.level:
                try:
                    h.handle(record)
                except Exception:  # noqa: BLE001
                    pass


def install() -> None:
    global _listener
    with _lock:
        if _listener is not None:
            return
        q: queue.SimpleQueue = queue.SimpleQueue()
        lg = logging.getLogger("mhd.governor")
        lg.addHandler(logging.handlers.QueueHandler(q))
        lg.propagate = False
        _listener = logging.handlers.QueueListener(q, _ToRoot())
        _listener.start()
