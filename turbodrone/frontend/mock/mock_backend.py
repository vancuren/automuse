#!/usr/bin/env python3
"""Mock drone-backend for building, demoing and testing the operator UI (core venv, FastAPI).

This is NOT the real backend: there is no RC link and no governor. It serves the same endpoints
as drone-backend (docs/interfaces.md) and speaks the frozen contracts in src/mhd/contracts, so the
UI can be developed and tested with no drone, twin, perception or agent running.

Run from the repo root (after `source env.sh`):
    $MHD_PY turbodrone/frontend/mock/mock_backend.py --port 8600 [--scenario cycle|manual] [--mode twin|real]

Endpoints, same as drone-backend:
    GET  /mjpeg              1280x720 test pattern: a moving "person in blue shirt" and a look-alike
    WS   /ws                 control: axes, hb, cmd (validated against the contracts)
    POST /command            OperatorCommand over HTTP (the LAND / E-STOP fallback)
    WS   /ws/telemetry       Telemetry at 10 Hz
    WS   /ws/agent?role=ui   AgentEvent and Candidates out; ptt_down/ptt_up and Select in
    GET  /state              latest Telemetry
    GET  /flights ...        saved flights for the UI's replay (--flights DIR; --record saves the mock's own)
    GET  /capabilities       upstream shape for x69_lg
    GET  /chase              2D stand-in for the twin chase view (the real one is the twin's :8766/chase)
    GET  /ui/                a production build of the UI, with --static-ui DIR (build it with --base /ui/)
Test-only endpoints:
    GET  /mock/log           every message received (also appended to --log, one JSON object per line)
    POST /mock/reset         clear the log and reset the simulation
    POST /mock/control       {"flight_mode", "scenario", "ws_mode": "normal"|"down"|"deaf", "extra_flags",
                              "extra_latency": {"fc_battery_v": 3.9}, ...}
    POST /mock/candidates    push a Candidates message to the UI agent sockets
    GET  /mock/stats         connection counts

Scenarios:
    cycle   (demo) the mock takes off, follows, doubts, loses, searches and lands by itself;
            commands still work and E-STOP latches until reset.
    manual  (tests) state changes only on commands; airborne autopilot goes FIND -> CONFIRM -> FOLLOW.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:  # works even without `source env.sh`
    sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from mhd.backend.recorder import FlightRecorder, install_replay_routes  # noqa: E402
from mhd.contracts import (  # noqa: E402
    FRAME_H,
    FRAME_W,
    AgentEvent,
    AutopilotState,
    Box,
    Candidate,
    Candidates,
    FlightState,
    Heartbeat,
    Intent,
    Mission,
    ObsSource,
    OperatorCommand,
    OperatorInput,
    Select,
    SubjectObs,
    SubjectState,
    Telemetry,
    now_ms,
)
from mhd.intent_spec import build_reply  # noqa: E402

FS = FlightState
AP = AutopilotState

TELEMETRY_HZ = 10
MJPEG_FPS = 15
TAKEOFF_S = 3.0          # the real governor holds TAKING_OFF for >= 4 s; shorter keeps tests quick
HUMAN_STALE_S = 0.25     # same rule as the governor: human input is zero after 0.25 s of silence
PACK_WARN_S = 300.0
PACK_LAND_S = 390.0
MAX_LOG_ENTRIES = 200_000  # in memory; the JSONL file keeps everything

# Autopilot script while airborne with a mission: (state, seconds). After the last step the
# script loops back to LOOP_FROM (cycle) or stays in FOLLOW (manual).
AP_SCRIPT = [(AP.find, 3.0), (AP.confirm, 2.5), (AP.follow, 14.0), (AP.doubt, 2.0),
             (AP.follow, 8.0), (AP.lost, 2.0), (AP.search, 3.0), (AP.follow, 10.0)]
LOOP_FROM = 2

# The real agent re-sends this health report every 5 s as a status AgentEvent.
AGENT_HEALTH = {"asr": "meta", "intent": "auto", "glimmer": "glimmer-30b+lora", "spark": "muse-spark-1.3",
                "safety_listener": True, "backend_connected": True, "mic_rate": 24000, "mic_silent": False}

# What the fake agent "hears" on each push-to-talk, in turn, and the intent it returns.
UTTERANCES: list[tuple[str, Intent]] = [
    ("follow the person in the blue shirt",
     Intent(action="follow", subject_phrase="person in blue shirt", category="person", color="blue",
            distance="medium", reason_code="none")),
    ("stay close to the person in blue",
     Intent(action="follow", subject_phrase="person in blue shirt", category="person", color="blue",
            distance="close", reason_code="none")),
    ("follow my friend dave",
     Intent(action="refuse", subject_phrase="", category="none", color="none", distance="medium",
            reason_code="identity_request")),
    ("ok hover there",
     Intent(action="hover", subject_phrase="", category="none", color="none", distance="medium",
            reason_code="none")),
]


# ── Message log ────────────────────────────────────────────────────────────────────────────

class MessageLog:
    """Every message the UI sends, with the server's receive time (mono ms), validity and channel.

    Kept in memory for /mock/log and appended to a JSONL file for tests.
    """

    def __init__(self, path: Path):
        self.path = path
        self.entries: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w", buffering=1, encoding="utf-8")

    def add(self, channel: str, conn: int, msg: Any, valid: bool, error: Optional[str] = None,
            note: Optional[str] = None) -> dict[str, Any]:
        entry: dict[str, Any] = {"t_ms": round(now_ms(), 1), "channel": channel, "conn": conn,
                                 "msg": msg, "valid": valid}
        if error:
            entry["error"] = error
        if note:
            entry["note"] = note
        with self.lock:
            self.entries.append(entry)
            if len(self.entries) > MAX_LOG_ENTRIES:  # long demo runs: keep memory bounded
                del self.entries[: len(self.entries) - MAX_LOG_ENTRIES]
            self._fh.write(json.dumps(entry) + "\n")
        return entry

    def reset(self) -> None:
        with self.lock:
            self.entries.clear()
            self._fh.close()
            self._fh = open(self.path, "w", buffering=1, encoding="utf-8")


def validate_control(msg: Any) -> tuple[bool, Optional[str]]:
    """Validate a /ws message against the contracts. All-zero axes are a contract violation."""
    if not isinstance(msg, dict):
        return False, "not a JSON object"
    kind = msg.get("type")
    model = {"axes": OperatorInput, "hb": Heartbeat, "cmd": OperatorCommand}.get(kind)
    if model is None:
        return False, f"unknown type {kind!r}"
    try:
        parsed = model.model_validate(msg)
    except ValidationError as e:
        return False, e.errors()[0]["msg"]
    if isinstance(parsed, OperatorInput) and not any(
            (parsed.throttle, parsed.yaw, parsed.pitch, parsed.roll)):
        return False, "all-zero axes message (contract: sent only while nonzero)"
    return True, None


def validate_agent(msg: Any) -> tuple[bool, Optional[str]]:
    if not isinstance(msg, dict):
        return False, "not a JSON object"
    model = {"agent_event": AgentEvent, "select": Select}.get(msg.get("type"))
    if model is None:
        return False, f"unknown type {msg.get('type')!r}"
    try:
        model.model_validate(msg)
    except ValidationError as e:
        return False, e.errors()[0]["msg"]
    return True, None


# ── Synthetic world ────────────────────────────────────────────────────────────────────────

def subject_box(t: float) -> Box:
    """The subject ("person in blue shirt") wanders left and right; boxes are 1280x720 pixels."""
    h = 330 + 45 * math.sin(0.23 * t)
    w = 0.42 * h
    cx = 640 + 300 * math.sin(0.31 * t)
    cy = 420 + 18 * math.sin(0.53 * t)
    return Box(x1=cx - w / 2, y1=cy - h / 2, x2=cx + w / 2, y2=cy + h / 2)


def lookalike_box(t: float) -> Box:
    """A look-alike a little farther away (smaller), drifting the other way."""
    h = 250 + 20 * math.sin(0.19 * t + 1.0)
    w = 0.42 * h
    cx = 640 - 340 * math.sin(0.21 * t + 0.8)
    cy = 400 + 10 * math.sin(0.4 * t)
    return Box(x1=cx - w / 2, y1=cy - h / 2, x2=cx + w / 2, y2=cy + h / 2)


def person_polygon(b: Box) -> list[tuple[float, float]]:
    """Outline of the stick-figure person drawn in the test pattern: the SAM mask stand-in."""
    cx, w, h, y1, y2 = b.cx, b.w, b.h, b.y1, b.y2
    head_r = 0.1 * h
    pts = []
    # head: top half-circle, clockwise from the left of the neck
    for k in range(9):
        a = math.pi + k * math.pi / 8
        pts.append((cx + head_r * math.cos(a), y1 + head_r + head_r * math.sin(a)))
    pts += [
        (cx + 0.55 * head_r, y1 + 2.0 * head_r),   # neck right
        (b.x2, y1 + 0.25 * h),                       # right shoulder
        (b.x2, y1 + 0.58 * h),                       # right hip
        (cx + 0.24 * w, y1 + 0.58 * h),
        (cx + 0.24 * w, y2),                         # right foot
        (cx + 0.02 * w, y2),
        (cx + 0.02 * w, y1 + 0.66 * h),              # crotch
        (cx - 0.02 * w, y1 + 0.66 * h),
        (cx - 0.02 * w, y2),
        (cx - 0.24 * w, y2),                         # left foot
        (cx - 0.24 * w, y1 + 0.58 * h),
        (b.x1, y1 + 0.58 * h),                       # left hip
        (b.x1, y1 + 0.25 * h),                       # left shoulder
        (cx - 0.55 * head_r, y1 + 2.0 * head_r),   # neck left
    ]
    return [(round(x, 1), round(y, 1)) for x, y in pts]


def _background() -> np.ndarray:
    """Static sky, ground and perspective grid (BGR), drawn once."""
    img = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    horizon = 300
    for y in range(horizon):  # sky gradient
        f = y / horizon
        img[y, :] = (int(120 + 60 * f), int(80 + 50 * f), int(40 + 30 * f))
    for y in range(horizon, FRAME_H):  # ground gradient
        f = (y - horizon) / (FRAME_H - horizon)
        img[y, :] = (int(45 + 25 * f), int(60 + 35 * f), int(50 + 25 * f))
    vp = (FRAME_W // 2, horizon)
    for x in range(-1600, FRAME_W + 1600, 160):  # lines to the vanishing point
        cv2.line(img, vp, (x, FRAME_H), (90, 110, 95), 1, cv2.LINE_AA)
    y, step = horizon + 8, 8
    while y < FRAME_H:
        cv2.line(img, (0, y), (FRAME_W, y), (90, 110, 95), 1, cv2.LINE_AA)
        step *= 1.35
        y += int(step)
    cv2.putText(img, "MOCK CAMERA (mock_backend.py, not a drone)", (690, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
    return img


def _draw_person(img: np.ndarray, b: Box, shirt_bgr: tuple[int, int, int]) -> None:
    poly = np.array(person_polygon(b), np.int32)
    cv2.fillPoly(img, [poly], shirt_bgr, cv2.LINE_AA)
    legs_top = int(b.y1 + 0.58 * b.h)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    mask[:legs_top] = 0
    img[mask > 0] = (60, 50, 45)  # dark trousers
    cv2.circle(img, (int(b.cx), int(b.y1 + 0.1 * b.h)), int(0.1 * b.h), (120, 160, 210), -1, cv2.LINE_AA)


def render_jpeg(background: np.ndarray, t: float, frame_id: int) -> bytes:
    img = background.copy()
    _draw_person(img, lookalike_box(t), (90, 170, 60))    # look-alike: green shirt
    _draw_person(img, subject_box(t), (200, 110, 30))     # subject: blue shirt
    cv2.putText(img, f"1280x720  frame {frame_id}  t={t:6.1f}s", (880, FRAME_H - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


# ── Simulation state ───────────────────────────────────────────────────────────────────────

@dataclass
class Sim:
    flight_mode: str = "twin"
    scenario: str = "cycle"
    flight_state: FlightState = FS.on_ground
    autopilot_state: AutopilotState = AP.idle
    armed: bool = False
    paused: bool = False
    freeze_autopilot: bool = False
    pack_id: str = "1"
    pack_time_s: dict[str, float] = field(default_factory=lambda: {"1": 0.0, "2": 0.0})
    state_since: float = field(default_factory=time.monotonic)
    ap_step: int = 0
    ap_since: float = field(default_factory=time.monotonic)
    human: dict[str, float] = field(default_factory=lambda: {"throttle": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0})
    human_at: float = -1e9
    last_hb: float = -1e9
    tilt: int = 0
    mission: Optional[Mission] = None
    mission_seq: int = 0
    utterance_idx: int = 0
    ws_mode: str = "normal"          # normal | down (refuse /ws) | deaf (accept, ignore)
    drop_hb: bool = False            # ignore heartbeats (the governor would flag heartbeat_missing)
    land_cause: str = "operator"
    extra_flags: list[str] = field(default_factory=list)
    extra_latency: dict[str, float] = field(default_factory=dict)   # e.g. the Lewei board's fc_battery_v
    frame_age_ms: Optional[float] = None   # override for screenshots/tests
    t0: float = field(default_factory=time.monotonic)
    last_tick: float = field(default_factory=time.monotonic)

    # -- helpers --
    def t(self) -> float:
        return time.monotonic() - self.t0

    def set_flight(self, fs: FlightState) -> None:
        if fs != self.flight_state:
            self.flight_state = fs
            self.state_since = time.monotonic()
        if fs in (FS.on_ground, FS.estopped, FS.landing):
            self.set_ap(AP.idle)  # the autopilot only flies while AIRBORNE

    def set_ap(self, ap: AutopilotState, step: Optional[int] = None) -> None:
        self.autopilot_state = ap
        self.ap_since = time.monotonic()
        if step is not None:
            self.ap_step = step

    def restart_autopilot(self) -> None:
        """New mission while airborne: back to FIND."""
        self.set_ap(AP.find, step=0)

    def new_mission(self, transcript: str, intent: Intent, model: str, latency_ms: float) -> Mission:
        self.mission_seq += 1
        self.mission = Mission(mission_id=f"m{self.mission_seq:03d}", intent=intent, transcript=transcript,
                               model=model, latency_ms=round(latency_ms, 1), reply=build_reply(intent))
        if self.flight_state == FS.airborne and intent.action == "follow":
            self.restart_autopilot()
        elif intent.action in ("hover", "refuse", "clarify"):
            self.set_ap(AP.idle)
        elif intent.action == "land" and self.flight_state in (FS.airborne, FS.taking_off):
            self.set_flight(FS.landing)
        return self.mission

    def default_mission(self) -> Mission:
        text, intent = UTTERANCES[0]
        return self.new_mission(text, intent, "glimmer-30b+lora", 1840.0)

    # -- commands (the governor's job in the real backend) --
    def apply_command(self, cmd: OperatorCommand) -> tuple[bool, str]:
        c = cmd.command
        fs = self.flight_state
        if c == "estop":
            self.set_flight(FS.estopped)
            return True, "estop latched"
        if c == "estop_reset":
            if fs == FS.estopped:
                self.set_flight(FS.on_ground)
                return True, "estop reset"
            return False, "not estopped"
        if c == "land":
            if fs in (FS.taking_off, FS.airborne):
                self.land_cause = "operator"
                self.set_flight(FS.landing)
                return True, "landing"
            return True, f"land ignored in {fs.value}"
        if c == "takeoff":
            if fs != FS.on_ground:
                return False, f"takeoff rejected in {fs.value}"
            if self.flight_mode == "real" and not self.armed:
                return False, "takeoff rejected: not armed (type ARM)"
            self.set_flight(FS.taking_off)
            return True, "taking off"
        if c == "confirm_touchdown":
            if fs == FS.landing:
                self.set_flight(FS.on_ground)
                return True, "touchdown confirmed"
            return False, "not landing"
        if c == "arm":
            if cmd.arm_phrase == "ARM":
                self.armed = True
                return True, "armed"
            return False, "arm rejected: phrase must be ARM"
        if c == "disarm":
            self.armed = False
            return True, "disarmed"
        if c == "pause":
            self.paused = True
            return True, "autopilot paused"
        if c == "resume":
            self.paused = False
            return True, "autopilot resumed"
        if c in ("tilt_up", "tilt_down", "tilt_stop"):
            self.tilt = {"tilt_up": 1, "tilt_down": -1, "tilt_stop": 0}[c]
            return True, c
        if c == "pack_swap":
            if fs != FS.on_ground:
                return False, "pack swap only on the ground"
            if not cmd.pack_id:
                return False, "pack_id missing"
            self.pack_id = cmd.pack_id
            self.pack_time_s.setdefault(cmd.pack_id, 0.0)
            return True, f"pack {cmd.pack_id}"
        return False, f"unhandled {c}"

    # -- 10 Hz step --
    def step(self) -> None:
        now = time.monotonic()
        dt, self.last_tick = now - self.last_tick, now
        fs = self.flight_state
        age = now - self.state_since
        if fs in (FS.taking_off, FS.airborne, FS.landing):
            self.pack_time_s[self.pack_id] = self.pack_time_s.get(self.pack_id, 0.0) + dt

        if fs == FS.taking_off and age >= TAKEOFF_S:
            self.set_flight(FS.airborne)
            if self.mission and self.mission.intent.action == "follow":
                self.restart_autopilot()
        elif fs == FS.airborne and self.pack_time_s[self.pack_id] >= PACK_LAND_S:
            self.land_cause = "flight_timer"
            self.set_flight(FS.landing)  # the governor's per-pack auto-land
        elif self.scenario == "cycle":
            if fs == FS.on_ground and age >= 4.0 and (self.flight_mode == "twin" or self.armed):
                self.set_flight(FS.taking_off)
            elif fs == FS.airborne and age >= 55.0:
                self.land_cause = "demo_loop"
                self.set_flight(FS.landing)
            elif fs == FS.landing and age >= 4.0:
                self.set_flight(FS.on_ground)  # auto "confirm touchdown" in the demo loop
            if self.flight_state == FS.airborne and self.mission is None and age >= 1.5:
                self.default_mission()

        if self.flight_state == FS.airborne and not self.freeze_autopilot and self.autopilot_state != AP.idle:
            state, dur = AP_SCRIPT[self.ap_step]
            if now - self.ap_since >= dur:
                nxt = self.ap_step + 1
                if nxt >= len(AP_SCRIPT):
                    nxt = LOOP_FROM if self.scenario == "cycle" else self.ap_step
                if self.scenario == "manual" and AP_SCRIPT[self.ap_step][0] == AP.follow:
                    nxt = self.ap_step  # manual: once following, keep following
                if nxt != self.ap_step:
                    self.set_ap(AP_SCRIPT[nxt][0], step=nxt)
                else:
                    self.ap_since = now

    # -- telemetry --
    def subject_obs(self, t: float) -> Optional[SubjectObs]:
        ap = self.autopilot_state
        if self.flight_state != FS.airborne or ap == AP.idle:
            return None
        mid = self.mission.mission_id if self.mission else None
        base = dict(mission_id=mid, frame_id=int(t * MJPEG_FPS), capture_ms=now_ms() - 140.0,
                    detector_latency_ms=round(480 + 90 * math.sin(t * 1.7), 1))
        if ap in (AP.find, AP.search):
            return SubjectObs(state=SubjectState.searching, source=ObsSource.sam_mlx, **base)
        if ap in (AP.lost, AP.give_up):
            return SubjectObs(state=SubjectState.lost, source=ObsSource.tracker, confirm_age_ms=2400.0, **base)
        b = subject_box(t)
        poly = person_polygon(b)
        if ap == AP.confirm:
            return SubjectObs(state=SubjectState.candidate, box=b, mask_poly=poly, confidence=0.71,
                              tracker_score=1.0, confirm_age_ms=120.0, source=ObsSource.sam_mlx, track_id=7, **base)
        doubt = ap == AP.doubt
        return SubjectObs(state=SubjectState.confirmed, box=b, mask_poly=poly,
                          confidence=0.62 if doubt else 0.91, tracker_score=0.41 if doubt else 0.93,
                          confirm_age_ms=900.0 if doubt else 260.0,
                          source=ObsSource.tracker if int(t * 2) % 3 else ObsSource.sam_mlx, track_id=7, **base)

    def telemetry(self) -> Telemetry:
        now = time.monotonic()
        t = self.t()
        fs, ap = self.flight_state, self.autopilot_state
        subject = self.subject_obs(t)

        human_fresh = now - self.human_at < HUMAN_STALE_S
        human = dict(self.human) if human_fresh else {k: 0.0 for k in self.human}
        auto = {"throttle": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        if fs == FS.airborne and not self.paused:
            if ap in (AP.follow, AP.confirm) and subject and subject.box:
                auto["yaw"] = round(max(-0.6, min(0.6, 0.9 * (subject.box.cx - 640) / 640)), 3)
                auto["pitch"] = round(max(-0.35, min(0.35, (330 - subject.box.h) / 300)), 3)
            elif ap == AP.search:
                auto["yaw"] = 0.25
        if fs == FS.estopped:
            active, sticks = "estop", {k: 0.0 for k in auto}
        elif human_fresh and any(human.values()):
            active, sticks = "human", human
        elif fs == FS.landing:
            active, sticks = "land", {k: 0.0 for k in auto}
        elif fs == FS.airborne and not self.paused and ap not in (AP.idle, AP.doubt, AP.lost, AP.give_up):
            active, sticks = "autopilot", auto
        else:
            active, sticks = "neutral", {k: 0.0 for k in auto}
        wire = {k: int(max(0, min(255, round(128 + v * 127)))) for k, v in sticks.items()}

        # Flag names follow the real governor (ws/gov src/mhd/governor/core.py).
        flags = list(self.extra_flags)
        if now - self.last_hb > 1.5:
            flags.append("heartbeat_missing")
        if ap == AP.doubt:
            flags.append("obs_stale")
        if self.paused:
            flags.append("autopilot_paused")
        pt = self.pack_time_s.get(self.pack_id, 0.0)
        if pt >= PACK_WARN_S:
            flags.append("flight_timer_warn")
        if fs == FS.landing:
            flags.append(f"landing:{self.land_cause}")

        frame_age = self.frame_age_ms if self.frame_age_ms is not None else round(125 + 35 * math.sin(t * 2.3) + random.uniform(0, 15), 1)
        obs_age = None
        if subject is not None:
            obs_age = round((440 if ap == AP.doubt else 190) + 50 * math.sin(t * 1.3), 1)
        latency = {"capture_to_backend": frame_age, "tick": 0.31, **self.extra_latency}
        if subject is not None and subject.detector_latency_ms is not None:
            latency["detector"] = subject.detector_latency_ms
        if self.mission is not None:
            latency["intent"] = self.mission.latency_ms

        return Telemetry(
            ts_ms=now_ms(), flight_mode=self.flight_mode, flight_state=fs, autopilot_state=ap,
            armed=self.armed, sticks_wire=wire,
            sticks_by_source={"human": human, "autopilot": auto, "safety": {k: 0.0 for k in auto}},
            active_source=active, frame_age_ms=frame_age, obs_age_ms=obs_age, latency_ms=latency,
            flight_time_s=round(pt, 1), pack_id=self.pack_id, safety_flags=flags,
            mission=self.mission, subject=subject,
        )


# ── App ────────────────────────────────────────────────────────────────────────────────────

class Hub:
    """Connections and background tasks, shared by the endpoints."""

    def __init__(self, sim: Sim, log: MessageLog):
        self.sim = sim
        self.log = log
        self.control: set[WebSocket] = set()
        self.telemetry: set[WebSocket] = set()
        self.agent_ui: set[WebSocket] = set()
        self.conn_seq = 0
        self.ptt_task: Optional[asyncio.Task] = None
        self.ptt_down_at: Optional[float] = None
        self.last_telemetry: Optional[Telemetry] = None
        self.frame: bytes = b""
        self.frame_id = 0
        self.frame_event = asyncio.Event()
        self.mjpeg_clients = 0
        self.background_tasks: set[asyncio.Task] = set()
        self.recorder = None  # --record: mhd.backend.recorder.FlightRecorder, like the real hub

    def spawn(self, coro) -> asyncio.Task:
        """create_task, keeping a reference so the task cannot be garbage-collected mid-flight."""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    def record_command(self, command: str, source: str, ok: bool) -> None:
        if self.recorder is not None:
            self.recorder.note("cmd", {"command": command, "source": source, "ok": bool(ok), "reason": None})

    def next_conn(self) -> int:
        self.conn_seq += 1
        return self.conn_seq

    async def send_agent(self, msg: Any) -> None:
        text = msg.model_dump_json() if hasattr(msg, "model_dump_json") else json.dumps(msg)
        if self.recorder is not None and getattr(msg, "kind", None) != "partial":
            self.recorder.note("agent", text)
        for ws in list(self.agent_ui):
            try:
                await ws.send_text(text)
            except Exception:
                self.agent_ui.discard(ws)

    async def push_candidates(self) -> Candidates:
        t = self.sim.t()
        mid = self.sim.mission.mission_id if self.sim.mission else "m-demo"
        msg = Candidates(mission_id=mid, frame_id=int(t * MJPEG_FPS), items=[
            Candidate(index=1, box=subject_box(t), confidence=0.83, color_score=0.91),
            Candidate(index=2, box=lookalike_box(t), confidence=0.81, color_score=0.34),
        ])
        await self.send_agent(msg)
        return msg

    # push-to-talk: stream partial words while held, then final + intent + reply
    async def _stream_partials(self, words: list[str]) -> None:
        for i in range(1, len(words) + 1):
            await self.send_agent(AgentEvent(kind="partial", text=" ".join(words[:i])))
            await asyncio.sleep(0.25)

    async def on_ptt_down(self) -> None:
        self.ptt_down_at = time.monotonic()
        text, _ = UTTERANCES[self.sim.utterance_idx % len(UTTERANCES)]
        await self.send_agent(AgentEvent(kind="status", text="listening"))
        if self.ptt_task:
            self.ptt_task.cancel()
        self.ptt_task = self.spawn(self._stream_partials(text.split()))

    async def on_ptt_up(self) -> None:
        if self.ptt_task:
            self.ptt_task.cancel()
            self.ptt_task = None
        if self.ptt_down_at is None:
            return
        self.ptt_down_at = None
        idx = self.sim.utterance_idx % len(UTTERANCES)
        self.sim.utterance_idx += 1
        text, intent = UTTERANCES[idx]
        released = time.monotonic()
        await self.send_agent(AgentEvent(kind="final", text=text,
                                         data={"model": "muse-voice-transcribe-1.0", "asr_ms": 610.0}))
        await self.send_agent(AgentEvent(kind="status", text="thinking: Glimmer-LoRA and Spark in parallel"))
        self.spawn(self._finish_mission(text, intent, idx, released))

    async def _finish_mission(self, text: str, intent: Intent, idx: int, released: float) -> None:
        await asyncio.sleep(random.uniform(0.9, 1.4))
        model = "muse-spark-1.3" if idx % 3 == 2 else "glimmer-30b+lora"
        latency_ms = round((time.monotonic() - released) * 1000, 1)
        reason = "glimmer valid within 2.5 s" if model.startswith("glimmer") else "spark answered first"
        # Same shape as the real agent: no mission_id here; the Mission goes to the backend.
        await self.send_agent(AgentEvent(kind="intent", text=intent.action, data={
            "intent": intent.model_dump(), "model": model, "latency_ms": latency_ms, "reason": reason,
            "attempts": [], "transcript": text}))
        mission = self.sim.new_mission(text, intent, model, latency_ms) if intent.action in ("follow", "hover", "land") else None
        if mission is not None and self.recorder is not None:
            self.recorder.note("mission", {**json.loads(mission.model_dump_json()), "ok": True})
        await self.send_agent(AgentEvent(kind="reply", text=mission.reply if mission else build_reply(intent)))

    async def sim_loop(self) -> None:
        """10 Hz: step the simulation, relay transitions like the backend hub does, broadcast Telemetry."""
        prev_ap = self.sim.autopilot_state
        prev_fs = self.sim.flight_state
        next_health = 0.0
        while True:
            self.sim.step()
            fs, ap = self.sim.flight_state, self.sim.autopilot_state
            if fs != prev_fs:
                await self.send_agent(AgentEvent(kind="status", text=f"flight {prev_fs.value} -> {fs.value}",
                                                 data={"kind": "flight_state", "from": prev_fs.value, "to": fs.value}))
            if ap != prev_ap:
                await self.send_agent(AgentEvent(kind="status", text=f"autopilot {prev_ap.value} -> {ap.value} (mock)",
                                                 data={"kind": "state", "from": prev_ap.value, "to": ap.value,
                                                       "reason": "mock"}))
            if self.sim.scenario == "cycle" and ap == AP.confirm and prev_ap != AP.confirm:
                await self.push_candidates()
            prev_ap, prev_fs = ap, fs
            if time.monotonic() >= next_health:  # the agent's 5 s "agent alive" report
                next_health = time.monotonic() + 5.0
                await self.send_agent(AgentEvent(kind="status", text="agent alive", data=AGENT_HEALTH))
            tm = self.sim.telemetry()
            self.last_telemetry = tm
            text = tm.model_dump_json()
            if self.recorder is not None:
                frame = (SimpleNamespace(frame_id=self.frame_id), self.frame) if self.frame else None
                self.recorder.sample(text, self.sim.flight_state.value, frame)
            for ws in list(self.telemetry):
                try:
                    await asyncio.wait_for(ws.send_text(text), timeout=0.5)
                except Exception:
                    self.telemetry.discard(ws)
            await asyncio.sleep(1 / TELEMETRY_HZ)

    async def frame_loop(self) -> None:
        """15 fps test pattern, rendered only while someone watches /mjpeg."""
        background = await asyncio.to_thread(_background)
        while True:
            if self.mjpeg_clients > 0 or self.recorder is not None:
                self.frame_id += 1
                self.frame = await asyncio.to_thread(render_jpeg, background, self.sim.t(), self.frame_id)
                self.frame_event.set()
                self.frame_event = asyncio.Event()
            await asyncio.sleep(1 / MJPEG_FPS)


def build_app(sim: Sim, log: MessageLog, static_ui: Optional[Path] = None, flights: Optional[Path] = None,
              record: bool = False) -> FastAPI:
    hub = Hub(sim, log)
    flights = flights or (REPO / "logs" / "flights")
    if record:
        hub.recorder = FlightRecorder(flights, flight_mode="mock")
    tasks: list[asyncio.Task] = []

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        tasks.append(asyncio.create_task(hub.sim_loop()))
        tasks.append(asyncio.create_task(hub.frame_loop()))
        yield
        for task in tasks:
            task.cancel()

    app = FastAPI(title="mock drone-backend (UI tests)", lifespan=lifespan)
    app.state.hub = hub
    # Same permissive CORS as upstream turbodrone, so POST /command works from :5173/:5273.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    install_replay_routes(app, flights)  # GET /flights...: the same replay routes as drone-backend

    @app.get("/")
    async def index():
        return {"mock": True, "endpoints": ["/mjpeg", "/ws", "/command", "/ws/telemetry", "/ws/agent?role=ui",
                                            "/state", "/capabilities", "/chase", "/mock/log", "/mock/reset",
                                            "/mock/control", "/mock/candidates", "/mock/stats"]}

    @app.get("/capabilities")
    async def capabilities():
        return {"drone_type": "x69_lg", "commands": {"takeoff": True, "land": True, "estop": True,
                                                       "camera_tilt": True, "camera_switch": False,
                                                       "speed_control": True}}

    @app.get("/state")
    async def state():
        tm = hub.last_telemetry or sim.telemetry()
        return JSONResponse(json.loads(tm.model_dump_json()))

    # -- control socket --
    @app.websocket("/ws")
    async def ws_control(ws: WebSocket):
        if sim.ws_mode == "down":
            await ws.close(code=1013)  # refused before accept: the browser sees a failed handshake
            return
        await ws.accept()
        conn = hub.next_conn()
        hub.control.add(ws)
        prev_axes_nonzero = False  # one all-zero "release" right after nonzero input is allowed
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    log.add("ws", conn, raw, False, "invalid JSON")
                    continue
                valid, err = validate_control(msg)
                release = False
                if isinstance(msg, dict) and msg.get("type") == "axes":
                    if not valid and err and err.startswith("all-zero") and prev_axes_nonzero:
                        valid, err, release = True, None, True  # VITE_AXES_RELEASE_MESSAGE=1
                    prev_axes_nonzero = valid and not release
                if sim.ws_mode == "deaf":
                    log.add("ws", conn, msg, valid, err, note="ignored (deaf)")
                    continue
                note = "release" if release else None
                if valid:
                    kind = msg["type"]
                    if kind == "hb" and not sim.drop_hb:
                        sim.last_hb = time.monotonic()
                    elif kind == "axes":
                        sim.human = {k: float(msg.get(k, 0.0)) for k in ("throttle", "yaw", "pitch", "roll")}
                        sim.human_at = time.monotonic()
                    elif kind == "cmd":
                        _ok, note = sim.apply_command(OperatorCommand.model_validate(msg))
                        hub.record_command(msg["command"], "ui", _ok)
                log.add("ws", conn, msg, valid, err, note=note)
        except WebSocketDisconnect:
            pass
        finally:
            hub.control.discard(ws)

    @app.post("/command")
    async def http_command(request: Request):
        raw = await request.body()
        try:
            msg = json.loads(raw or b"{}")
            cmd = OperatorCommand.model_validate(msg)
        except (json.JSONDecodeError, ValidationError) as e:
            log.add("http", 0, raw.decode("utf-8", "replace"), False, str(e)[:200])
            return JSONResponse({"ok": False, "error": "invalid OperatorCommand"}, status_code=422)
        ok, note = sim.apply_command(cmd)
        hub.record_command(cmd.command, "http", ok)
        log.add("http", 0, msg, True, note=note)
        return {"ok": ok, "flight_state": sim.flight_state.value, "note": note}

    # -- telemetry --
    @app.websocket("/ws/telemetry")
    async def ws_telemetry(ws: WebSocket):
        await ws.accept()
        hub.telemetry.add(ws)
        try:
            while True:
                await ws.receive_text()  # keep the socket open; clients send nothing
        except WebSocketDisconnect:
            pass
        finally:
            hub.telemetry.discard(ws)

    # -- agent relay (the mock plays the agent and perception) --
    @app.websocket("/ws/agent")
    async def ws_agent(ws: WebSocket, role: str = "ui"):
        await ws.accept()
        conn = hub.next_conn()
        if role == "ui":
            hub.agent_ui.add(ws)
            await ws.send_text(AgentEvent(kind="status", text="agent ready", data=AGENT_HEALTH).model_dump_json())
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    log.add("agent", conn, raw, False, "invalid JSON")
                    continue
                valid, err = validate_agent(msg)
                log.add("agent", conn, msg, valid, err)
                if not valid:
                    continue
                if msg["type"] == "agent_event" and msg["kind"] == "ptt_down":
                    await hub.on_ptt_down()
                elif msg["type"] == "agent_event" and msg["kind"] == "ptt_up":
                    await hub.on_ptt_up()
                elif msg["type"] == "select":
                    await hub.send_agent(AgentEvent(kind="status", text=f"following candidate {msg['index']}"))
                    if sim.flight_state == FS.airborne:
                        sim.set_ap(AP.follow, step=LOOP_FROM)
        except WebSocketDisconnect:
            pass
        finally:
            hub.agent_ui.discard(ws)

    # -- video --
    @app.get("/mjpeg")
    async def mjpeg():
        async def frames():
            hub.mjpeg_clients += 1
            try:
                while True:
                    event = hub.frame_event
                    await event.wait()
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + hub.frame + b"\r\n"
            finally:
                hub.mjpeg_clients -= 1
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/chase", response_class=HTMLResponse)
    async def chase():
        return CHASE_HTML

    if static_ui is not None:  # serve a production build (tests of the real bundle)
        from fastapi.staticfiles import StaticFiles
        app.mount("/ui", StaticFiles(directory=str(static_ui), html=True), name="ui")

    # -- test controls --
    @app.get("/mock/log")
    async def mock_log(channel: Optional[str] = None, since_ms: float = 0.0):
        with log.lock:
            entries = [e for e in log.entries if (channel is None or e["channel"] == channel) and e["t_ms"] >= since_ms]
        return {"now_ms": now_ms(), "entries": entries}

    @app.post("/mock/reset")
    async def mock_reset(request: Request):
        body = await _json(request)
        # A push-to-talk still finishing from the previous test must not leak into the next one.
        for task in list(hub.background_tasks):
            task.cancel()
        hub.ptt_task = None
        hub.ptt_down_at = None
        log.reset()
        mode = body.get("flight_mode", sim.flight_mode)
        scenario = body.get("scenario", sim.scenario)
        fresh = Sim(flight_mode=mode, scenario=scenario)
        for k, v in fresh.__dict__.items():
            setattr(sim, k, v)
        return {"ok": True}

    @app.post("/mock/control")
    async def mock_control(request: Request):
        body = await _json(request)
        if "flight_mode" in body:
            sim.flight_mode = body["flight_mode"]
            sim.armed = False if body["flight_mode"] == "real" else sim.armed
        if "scenario" in body:
            sim.scenario = body["scenario"]
        if "armed" in body:
            sim.armed = bool(body["armed"])
        if "flight_state" in body:
            sim.set_flight(FlightState(body["flight_state"]))
        if body.get("mission") == "default":  # before autopilot_state: a new mission restarts FIND
            sim.default_mission()
        if "autopilot_state" in body:
            ap = AutopilotState(body["autopilot_state"])
            sim.set_ap(ap, step=next((i for i, (s, _) in enumerate(AP_SCRIPT) if s == ap), 0))
        if "drop_hb" in body:
            sim.drop_hb = bool(body["drop_hb"])
        if "freeze_autopilot" in body:
            sim.freeze_autopilot = bool(body["freeze_autopilot"])
        if "pack_time_s" in body:
            sim.pack_time_s[sim.pack_id] = float(body["pack_time_s"])
        if "extra_flags" in body:
            sim.extra_flags = list(body["extra_flags"])
        if "extra_latency" in body:
            sim.extra_latency = {str(k): float(v) for k, v in dict(body["extra_latency"]).items()}
        if "frame_age_ms" in body:
            sim.frame_age_ms = body["frame_age_ms"]
        if "time_offset_s" in body:  # move the synthetic subject to a chosen spot
            sim.t0 = time.monotonic() - float(body["time_offset_s"])
        if "ws_mode" in body:
            sim.ws_mode = body["ws_mode"]
            if sim.ws_mode == "down":
                for ws in list(hub.control):
                    try:
                        await ws.close(code=1012)
                    except Exception:
                        pass
                hub.control.clear()
        if "agent_event" in body:
            await hub.send_agent(AgentEvent.model_validate(body["agent_event"]))
        return {"ok": True, "flight_state": sim.flight_state.value, "ws_mode": sim.ws_mode}

    @app.post("/mock/candidates")
    async def mock_candidates():
        msg = await hub.push_candidates()
        return json.loads(msg.model_dump_json())

    @app.get("/mock/stats")
    async def mock_stats():
        return {"control": len(hub.control), "telemetry": len(hub.telemetry), "agent_ui": len(hub.agent_ui),
                "mjpeg": hub.mjpeg_clients, "flight_state": sim.flight_state.value,
                "autopilot_state": sim.autopilot_state.value, "ws_mode": sim.ws_mode}

    return app


async def _json(request: Request) -> dict:
    raw = await request.body()
    return json.loads(raw) if raw else {}


CHASE_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>mock chase view</title>
<style>html,body{margin:0;height:100%;background:#0b1220;overflow:hidden}canvas{width:100%;height:100%;display:block}</style>
</head><body><canvas id="c"></canvas><script>
// 2D stand-in for the twin chase view: top-down, drone (triangle) trailing the subject (blue).
const c = document.getElementById('c'), g = c.getContext('2d');
function resize(){ c.width = innerWidth * devicePixelRatio; c.height = innerHeight * devicePixelRatio; }
addEventListener('resize', resize); resize();
const trail = [];
function frame(){
  const t = performance.now() / 1000, W = c.width, H = c.height, s = Math.min(W, H) / 10;
  g.fillStyle = '#102016'; g.fillRect(0, 0, W, H);
  g.strokeStyle = '#1f3a2a'; g.lineWidth = 1;
  for (let x = (t * 20) % s; x < W; x += s) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); }
  for (let y = 0; y < H; y += s) { g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke(); }
  const sx = W / 2 + Math.sin(0.31 * t) * W * 0.3, sy = H * 0.35 + Math.sin(0.17 * t) * H * 0.08;
  const lx = W / 2 - Math.sin(0.21 * t + 0.8) * W * 0.32, ly = H * 0.28;
  const dx = sx * 0.6 + W * 0.2, dy = H * 0.72;
  trail.push([dx, dy]); if (trail.length > 90) trail.shift();
  g.strokeStyle = 'rgba(56,189,248,.35)'; g.lineWidth = 3; g.beginPath();
  trail.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.stroke();
  g.fillStyle = '#16a34a'; g.beginPath(); g.arc(lx, ly, s * 0.18, 0, 7); g.fill();
  g.fillStyle = '#2563eb'; g.beginPath(); g.arc(sx, sy, s * 0.22, 0, 7); g.fill();
  const a = Math.atan2(sy - dy, sx - dx);
  g.save(); g.translate(dx, dy); g.rotate(a); g.fillStyle = '#f8fafc';
  g.beginPath(); g.moveTo(s * 0.35, 0); g.lineTo(-s * 0.25, s * 0.2); g.lineTo(-s * 0.25, -s * 0.2); g.fill(); g.restore();
  g.strokeStyle = 'rgba(248,250,252,.5)'; g.setLineDash([8, 8]); g.beginPath(); g.moveTo(dx, dy); g.lineTo(sx, sy); g.stroke(); g.setLineDash([]);
  g.fillStyle = '#cbd5e1'; g.font = `${Math.round(s * 0.28)}px -apple-system, sans-serif`;
  g.fillText('MOCK CHASE VIEW (stand-in for twin :8766/chase)', s * 0.3, s * 0.5);
  requestAnimationFrame(frame);
}
frame();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="loopback only by default")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--scenario", choices=["cycle", "manual"], default="cycle")
    ap.add_argument("--mode", choices=["twin", "real"], default="twin", help="flight_mode reported in telemetry")
    ap.add_argument("--log", type=Path, default=None, help="JSONL message log (default logs/mock_backend_<port>.jsonl)")
    ap.add_argument("--pack-start-s", type=float, default=0.0, help="initial flight time on pack 1")
    ap.add_argument("--static-ui", type=Path, default=None, help="serve this production build at /ui/")
    ap.add_argument("--flights", type=Path, default=None, help="saved flights for replay (default logs/flights)")
    ap.add_argument("--record", action="store_true", help="save the mock's own flights into --flights")
    args = ap.parse_args()
    log_path = args.log or (REPO / "logs" / f"mock_backend_{args.port}.jsonl")
    sim = Sim(flight_mode=args.mode, scenario=args.scenario)
    sim.pack_time_s["1"] = args.pack_start_s
    app = build_app(sim, MessageLog(log_path), static_ui=args.static_ui, flights=args.flights, record=args.record)
    print(f"mock backend on http://{args.host}:{args.port} scenario={args.scenario} mode={args.mode} log={log_path}",
          flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
