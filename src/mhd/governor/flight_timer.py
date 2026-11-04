"""Flight timer per battery pack, persisted to $MHD_LOGS/flight_timer.json.

The RC thread only calls `add_flight_time` (pure arithmetic under a lock). Saving to disk happens
from the governor's housekeeping thread via `save_if_dirty`, never on the RC thread.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FlightTimer:
    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._packs: dict[str, float] = {}
        self._current: str = "1"
        self._dirty = False
        self._load()

    # -- persistence (housekeeping thread only) ------------------------------------------------
    def _load(self) -> None:
        if not self.path or not self.path.exists():
            self._packs = {self._current: 0.0}
            return
        try:
            data = json.loads(self.path.read_text())
            self._packs = {str(k): float(v.get("flight_s", 0.0)) for k, v in data.get("packs", {}).items()}
            self._current = str(data.get("current_pack", "1"))
            self._packs.setdefault(self._current, 0.0)
        except Exception as exc:  # a corrupt file must not stop the backend
            logger.warning("[flight_timer] could not read %s (%s); starting fresh", self.path, exc)
            self._packs = {self._current: 0.0}

    def save_if_dirty(self) -> None:
        if not self.path:
            return
        with self._lock:
            if not self._dirty:
                return
            data = {
                "current_pack": self._current,
                "packs": {k: {"flight_s": round(v, 2)} for k, v in self._packs.items()},
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._dirty = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=1))
            os.replace(tmp, self.path)
        except Exception as exc:
            logger.warning("[flight_timer] could not save %s: %s", self.path, exc)

    # -- RC-thread-safe API --------------------------------------------------------------------
    def add_flight_time(self, dt_s: float) -> None:
        if dt_s <= 0:
            return
        with self._lock:
            self._packs[self._current] = self._packs.get(self._current, 0.0) + dt_s
            self._dirty = True

    def swap_pack(self, pack_id: Optional[str]) -> str:
        """A fresh (charged) pack goes in: its timer restarts at zero."""
        with self._lock:
            pid = str(pack_id) if pack_id else str(len(self._packs) + 1)
            self._current = pid
            self._packs[pid] = 0.0
            self._dirty = True
            return pid

    @property
    def current_pack(self) -> str:
        return self._current

    def flight_s(self) -> float:
        with self._lock:
            return self._packs.get(self._current, 0.0)
