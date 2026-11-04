"""Tiny O(1)-amortized statistics used inside the RC tick (no numpy, no allocation storms)."""
from __future__ import annotations

from collections import deque
from typing import Optional


class SlidingMin:
    """Minimum of values seen in the last `window_ms` (monotonic deque, amortized O(1)).

    Used for latency floors: floor = min over 10 s of (receive_ms - capture_ms). When no sample is
    inside the window, the last minimum is kept (a stalled stream must not reset its own floor).
    """

    def __init__(self, window_ms: float):
        self.window_ms = float(window_ms)
        self._dq: deque[tuple[float, float]] = deque()  # (t_ms, value), values increasing
        self._last: Optional[float] = None

    def add(self, t_ms: float, value: float) -> None:
        while self._dq and self._dq[-1][1] >= value:
            self._dq.pop()
        self._dq.append((t_ms, value))
        self._expire(t_ms)
        self._last = self._dq[0][1]

    def _expire(self, now_ms: float) -> None:
        while len(self._dq) > 1 and self._dq[0][0] < now_ms - self.window_ms:
            self._dq.popleft()

    def value(self, now_ms: Optional[float] = None) -> Optional[float]:
        if now_ms is not None and self._dq:
            self._expire(now_ms)
            self._last = self._dq[0][1]
        return self._last

    def reset(self) -> None:
        self._dq.clear()
        self._last = None


class SessionMin:
    """Minimum of every value since start (or reset). Used for the governor's latency floors.

    A sliding minimum re-baselines: a latency that creeps up, or rises and stays high for longer
    than the window, becomes the new "best case" and disappears from the lag (review finding 1).
    A session minimum only ever goes down, so any rise above the best case stays visible.
    """

    def __init__(self) -> None:
        self._min: Optional[float] = None

    def add(self, t_ms: float, value: float) -> None:  # noqa: ARG002 - same signature as SlidingMin
        if self._min is None or value < self._min:
            self._min = value

    def value(self, now_ms: Optional[float] = None) -> Optional[float]:  # noqa: ARG002
        return self._min

    def reset(self) -> None:
        self._min = None


class PeriodEstimator:

    """Exponential moving average of the interval between events (ms). Ignores huge gaps."""

    def __init__(self, alpha: float = 0.1, max_gap_ms: float = 2000.0):
        self.alpha = alpha
        self.max_gap_ms = max_gap_ms
        self._last_t: Optional[float] = None
        self.period_ms: Optional[float] = None

    def add(self, t_ms: float) -> None:
        if self._last_t is not None:
            dt = t_ms - self._last_t
            if 0 < dt <= self.max_gap_ms:
                self.period_ms = dt if self.period_ms is None else (
                    (1 - self.alpha) * self.period_ms + self.alpha * dt)
        self._last_t = t_ms
