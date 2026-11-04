"""Per-tick control records -> $MHD_LOGS/control-<wallclock>.jsonl, written off the RC thread.

The RC thread calls `TickLog.record(dict)`, which goes through a logging.QueueHandler (a queue put,
no formatting, no file I/O). A QueueListener thread serializes each record to one JSON line.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import queue
import time
from pathlib import Path
from typing import Optional


class _RawQueueHandler(logging.handlers.QueueHandler):
    """QueueHandler that skips formatting on the producer (RC) thread."""

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        return record


class _JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = record.msg
        if isinstance(msg, dict):
            try:
                return json.dumps(msg, separators=(",", ":"), default=str)
            except Exception as exc:  # never lose the line entirely
                return json.dumps({"log_error": str(exc)})
        return json.dumps({"msg": str(msg)})


class TickLog:
    def __init__(self, directory: Optional[Path], enabled: bool = True, name: str = "control"):
        self.enabled = bool(enabled and directory)
        self.path: Optional[Path] = None
        self._listener: Optional[logging.handlers.QueueListener] = None
        self._logger = logging.getLogger(f"mhd.{name}.{id(self)}")
        self._logger.propagate = False
        self._logger.setLevel(logging.INFO)
        self.dropped = 0
        if not self.enabled:
            return
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = directory / f"{name}-{stamp}-{id(self) % 10000:04d}.jsonl"
        self._q: queue.Queue = queue.Queue(maxsize=20000)
        fh = logging.FileHandler(self.path, encoding="utf-8")
        fh.setFormatter(_JsonLineFormatter())
        self._handler = _RawQueueHandler(self._q)
        self._logger.addHandler(self._handler)
        self._listener = logging.handlers.QueueListener(self._q, fh, respect_handler_level=False)
        self._listener.start()

    def record(self, rec: dict) -> None:
        """Called on the RC thread. Never blocks: a full queue drops the record and counts it."""
        if not self.enabled:
            return
        try:
            if self._q.full():
                self.dropped += 1
                return
            self._logger.info(rec)
        except Exception:
            self.dropped += 1

    def close(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()  # flushes the queue
            except Exception:
                pass
            for h in list(self._listener.handlers):
                try:
                    h.close()
                except Exception:
                    pass
            self._listener = None
        if self.enabled:
            self._logger.removeHandler(self._handler)
