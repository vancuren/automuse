"""Flight recorder and replay API: every flight is saved so the UI can play it back.

One directory per flight under $MHD_FLIGHTS_DIR (default $MHD_LOGS/flights):

    meta.json     {"id", "started", "flight_mode", "duration_ms", "frames", "commands": [...], "complete"}
    events.jsonl  one JSON object per line, "t" = ms since the recording started:
                    {"t", "k": "tlm",     "d": Telemetry}            10 Hz, exactly what the UI saw
                    {"t", "k": "agent",   "d": AgentEvent|Candidates} voice transcript, replies, governor events
                    {"t", "k": "cmd",     "d": {"command", "source", "ok", "reason"}}
                    {"t", "k": "mission", "d": {...Mission, "ok"}}
                    {"t", "k": "frame",   "i": n}                    camera frame n is current from here on
    frames.bin    the camera JPEGs, back to back
    frames.idx    one "offset length" line per frame

A recording starts when the flight state leaves the ground (with the last few seconds before it, so the
takeoff command is in it) and ends once the drone has been back on the ground for a few seconds.

Nothing here can reach the drone: the hub calls sample() and note() on the event loop, both only put
items on a bounded queue (a full queue drops them), and one daemon thread does all the file I/O. The
replay routes are GET-only and read files.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

FLYING_STATES = frozenset({"TAKING_OFF", "AIRBORNE", "LANDING"})
PREROLL_S = 4.0            # telemetry and events kept from before takeoff
TAIL_S = 3.0               # keep recording this long after touchdown
MAX_FLIGHT_S = 15 * 60.0   # hard stop: a recording never grows past this
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")


def flights_dir() -> Path:
    env = os.environ.get("MHD_FLIGHTS_DIR")
    if env:
        return Path(env)
    logs = os.environ.get("MHD_LOGS")
    return (Path(logs) if logs else Path(__file__).resolve().parents[3] / "logs") / "flights"


def recording_enabled() -> bool:
    return os.environ.get("MHD_RECORD", "").strip().lower() in ("1", "true", "yes", "on")


class _Writer(threading.Thread):
    """Owns the files of one flight. Items: ("line", str) | ("frame", t_ms, bytes) | ("close", meta)."""

    def __init__(self, directory: Path, meta: dict):
        super().__init__(name="flight-recorder", daemon=True)
        self.dir = directory
        self.meta = meta
        self.q: queue.Queue = queue.Queue(maxsize=4000)
        self.dropped = 0
        self.frames = 0

    def put(self, item: tuple) -> None:
        try:
            self.q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def _write_meta(self) -> None:
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=1), encoding="utf-8")
        tmp.replace(self.dir / "meta.json")

    def run(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._write_meta()
            offset = 0
            with open(self.dir / "events.jsonl", "w", encoding="utf-8") as ev, \
                    open(self.dir / "frames.bin", "wb") as fb, \
                    open(self.dir / "frames.idx", "w", encoding="ascii") as fi:
                while True:
                    item = self.q.get()
                    kind = item[0]
                    if kind == "line":
                        ev.write(item[1])
                        ev.write("\n")
                    elif kind == "frame":
                        _, t_ms, jpeg = item
                        fb.write(jpeg)
                        fi.write(f"{offset} {len(jpeg)}\n")
                        ev.write(f'{{"t":{t_ms:.1f},"k":"frame","i":{self.frames}}}\n')
                        offset += len(jpeg)
                        self.frames += 1
                    elif kind == "close":
                        self.meta.update(item[1])
                        break
                    if self.q.empty():
                        ev.flush()
                        fb.flush()
                        fi.flush()
            self.meta.update({"frames": self.frames, "dropped": self.dropped, "complete": True})
            self._write_meta()
        except Exception:  # noqa: BLE001 - a full disk must never take the backend down
            logger.exception("[recorder] writer failed for %s", self.dir)


class FlightRecorder:
    def __init__(self, root: Optional[Path] = None, flight_mode: str = "twin", fps: float = 10.0,
                 enabled: bool = True):
        self.root = Path(root) if root else flights_dir()
        self.flight_mode = flight_mode
        self.enabled = enabled
        self.frame_gap_s = 0.8 / max(1.0, fps)     # 0.8: the 10 Hz sampler's ticks jitter around 100 ms
        self._pre: deque = deque()                 # (monotonic_s, line_without_t) before takeoff
        self._writer: Optional[_Writer] = None
        self._t0 = 0.0
        self._ground_since: Optional[float] = None
        self._last_frame_id = -1
        self._last_frame_t = 0.0
        self._commands: list[dict] = []
        self.current_id: Optional[str] = None

    # ---- event-loop API -------------------------------------------------------------------------

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def note(self, kind: str, payload: Any) -> None:
        """kind: "agent" | "cmd" | "mission". payload: a dict, or JSON text of one."""
        if not self.enabled:
            return
        try:
            body = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"), default=str)
            now = time.monotonic()
            self._emit(now, f'"k":"{kind}","d":{body}')
            if self._writer is not None and kind in ("cmd", "mission"):
                self._remember(now, kind, payload)
        except Exception:  # noqa: BLE001
            logger.debug("[recorder] note failed", exc_info=True)

    def sample(self, telemetry_json: str, flight_state: str, frame: Optional[tuple] = None) -> None:
        """Called at 10 Hz with the telemetry the UI gets, the flight state, and the newest frame."""
        if not self.enabled:
            return
        try:
            now = time.monotonic()
            flying = flight_state in FLYING_STATES
            if self._writer is None and flying:
                self._start(now)
            self._emit(now, f'"k":"tlm","d":{telemetry_json}')
            if self._writer is None:
                return
            if frame is not None:
                meta, jpeg = frame
                fid = getattr(meta, "frame_id", None)
                if fid != self._last_frame_id and now - self._last_frame_t >= self.frame_gap_s:
                    self._last_frame_id, self._last_frame_t = fid, now
                    self._writer.put(("frame", (now - self._t0) * 1000.0, jpeg))
            if flying:
                self._ground_since = None
            elif self._ground_since is None:
                self._ground_since = now
            if (self._ground_since is not None and now - self._ground_since >= TAIL_S) \
                    or now - self._t0 >= MAX_FLIGHT_S + PREROLL_S:
                self._stop(now)
        except Exception:  # noqa: BLE001
            logger.debug("[recorder] sample failed", exc_info=True)

    def close(self) -> None:
        if self._writer is not None:
            self._stop(time.monotonic())

    # ---- internals ------------------------------------------------------------------------------

    def _emit(self, now: float, body: str) -> None:
        if self._writer is not None:
            self._writer.put(("line", f'{{"t":{(now - self._t0) * 1000.0:.1f},{body}}}'))
            return
        self._pre.append((now, body))
        while self._pre and now - self._pre[0][0] > PREROLL_S:
            self._pre.popleft()

    def _remember(self, now: float, kind: str, payload: Any) -> None:
        if isinstance(payload, str):
            payload = json.loads(payload)
        entry = {"t": round((now - self._t0) * 1000.0), "kind": kind}
        if kind == "cmd":
            entry.update(command=payload.get("command"), source=payload.get("source"), ok=payload.get("ok"))
        else:
            intent = payload.get("intent") or {}
            entry.update(transcript=payload.get("transcript"), action=intent.get("action"),
                         subject=intent.get("subject_phrase"), model=payload.get("model"))
        if len(self._commands) < 200:
            self._commands.append(entry)

    def _start(self, now: float) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.current_id = f"{stamp}-{self.flight_mode}"
        self._t0 = self._pre[0][0] if self._pre else now
        self._commands = []
        self._ground_since = None
        self._last_frame_id, self._last_frame_t = -1, 0.0
        meta = {"id": self.current_id, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "started_unix": round(time.time() - (now - self._t0), 3), "flight_mode": self.flight_mode,
                "complete": False}
        w = _Writer(self.root / self.current_id, meta)
        pre, self._pre = list(self._pre), deque()
        self._writer = w
        for t, body in pre:
            w.put(("line", f'{{"t":{(t - self._t0) * 1000.0:.1f},{body}}}'))
            if '"k":"cmd"' in body[:12] or '"k":"mission"' in body[:16]:
                try:
                    self._remember(t, body[5:body.index('"', 5)], body[body.index('"d":') + 4:])
                except Exception:  # noqa: BLE001
                    pass
        w.start()
        logger.warning("[recorder] recording flight %s", self.current_id)

    def _stop(self, now: float) -> None:
        w, self._writer = self._writer, None
        if w is None:
            return
        w.put(("close", {"duration_ms": round((now - self._t0) * 1000.0), "commands": self._commands}))
        logger.warning("[recorder] flight %s saved (%.0f s)", self.current_id, now - self._t0)
        self.current_id = None


# ==================================================================================================
# Replay: read-only access to the saved flights


def _flight_dir(root: Path, flight_id: str) -> Optional[Path]:
    if not _ID_RE.match(flight_id) or ".." in flight_id:
        return None
    d = root / flight_id
    return d if (d / "events.jsonl").is_file() else None


def list_flights(root: Optional[Path] = None) -> list[dict]:
    root = Path(root) if root else flights_dir()
    out = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir(), reverse=True):
        if not (d / "events.jsonl").is_file():
            continue
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            meta = {}
        meta.setdefault("id", d.name)
        meta["id"] = d.name
        meta.setdefault("complete", False)
        meta.setdefault("commands", [])
        out.append(meta)
    return out


def read_frame(root: Path, flight_id: str, index: int) -> Optional[bytes]:
    d = _flight_dir(root, flight_id)
    if d is None or index < 0:
        return None
    try:
        with open(d / "frames.idx", "r", encoding="ascii") as fi:
            for n, line in enumerate(fi):
                if n == index:
                    off, length = (int(x) for x in line.split())
                    break
            else:
                return None
        with open(d / "frames.bin", "rb") as fb:
            fb.seek(off)
            data = fb.read(length)
        return data if len(data) == length else None
    except (OSError, ValueError):
        return None


def install_replay_routes(app, root: Optional[Path] = None) -> None:
    """GET /flights, /flights/{id}/events, /flights/{id}/frame/{n}. Read-only."""
    from fastapi import HTTPException
    from fastapi.responses import FileResponse, Response

    def base() -> Path:
        return Path(root) if root else flights_dir()

    @app.get("/flights")
    async def get_flights():
        return {"flights": list_flights(base())}

    @app.get("/flights/{flight_id}/events")
    async def get_flight_events(flight_id: str):
        d = _flight_dir(base(), flight_id)
        if d is None:
            raise HTTPException(404, "no such flight")
        return FileResponse(d / "events.jsonl", media_type="application/x-ndjson")

    @app.get("/flights/{flight_id}/frame/{index}")
    async def get_flight_frame(flight_id: str, index: int):
        data = read_frame(base(), flight_id, index)
        if data is None:
            raise HTTPException(404, "no such frame")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})
