"""Spend accounting for paid API calls, shared by every perception process.

- Every request that reaches the server counts as billable (conservative: a timed-out request may
  still be processed and billed).
- One JSON line per paid batch is appended to $MHD_HOME/logs/spend.jsonl:
  {"ts": ..., "ws": "percep", "service": "...", "units": N, "est_usd": X}
- The cap covers the whole workstream: at start-up the ledger's earlier "percep" lines are summed,
  so restarting the service does not reset the budget.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

WS = "percep"
DEFAULT_CAP_USD = 8.0


def ledger_path() -> Path:
    home = os.environ.get("MHD_HOME") or os.getcwd()
    return Path(home) / "logs" / "spend.jsonl"


def spent_so_far(path: Optional[Path] = None, ws: str = WS) -> float:
    """Sum of est_usd for this workstream in the ledger (0 if missing)."""
    path = path or ledger_path()
    total = 0.0
    try:
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("ws") == ws:
                    total += float(row.get("est_usd", 0.0))
    except FileNotFoundError:
        pass
    return total


class SpendMeter:
    """Counts paid units, enforces a cap, and flushes batches to the ledger."""

    def __init__(self, service: str, usd_per_unit: float, cap_usd: float = DEFAULT_CAP_USD,
                 path: Optional[Path] = None, flush_every: int = 20):
        self.service = service
        self.usd_per_unit = usd_per_unit
        self.cap_usd = cap_usd
        self.path = path or ledger_path()
        self.flush_every = flush_every
        self._lock = threading.Lock()
        self._base = spent_so_far(self.path)     # spend from earlier runs
        self._pending = 0                        # units not yet written to the ledger
        self._session_units = 0

    @property
    def total_usd(self) -> float:
        return self._base + self._session_units * self.usd_per_unit

    def can_spend(self, units: int = 1) -> bool:
        return self.total_usd + units * self.usd_per_unit <= self.cap_usd + 1e-9

    def charge(self, units: int = 1) -> None:
        with self._lock:
            self._session_units += units
            self._pending += units
            if self._pending >= self.flush_every:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if self._pending <= 0:
            return
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ws": WS, "service": self.service,
               "units": self._pending, "est_usd": round(self._pending * self.usd_per_unit, 6)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        self._pending = 0
