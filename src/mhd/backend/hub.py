"""The backend hub: owns the governor and every new endpoint (PLAN.md 2, docs/interfaces.md).

web_server.py (patched turbodrone) creates one Hub, builds the X69 model through it, and hands
every UI, perception and agent socket to it. Nothing here writes sticks: all input becomes a
governor call (submit_human, heartbeat, command, set_mission, submit_obs, note_frame).

Endpoints installed by install_routes(app, hub):
  WS /ws (via serve_ui)  POST /command  WS /ws/telemetry  GET /state  WS /ws/frames
  WS /ws/subject  WS /ws/agent?role=agent|ui  POST /mission
  GET /flights  /flights/{id}/events  /flights/{id}/frame/{n}   (recorder.py: saved flights, read-only)
  DEBUG: POST /debug/rc_pause {"seconds": N}  POST /debug/rc_resume  GET /debug/rc_pause
         (Session A link-loss test 3; real mode only while ON_GROUND, at most 15 s; on the Lewei
         board ALL packets stop: it has no keepalive)
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import signal
import threading
import time
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

from pydantic import ValidationError
from starlette.websockets import WebSocketState

from mhd.contracts import (AgentEvent, Candidates, FindRequest, FlightState, Heartbeat, Mission, NextFrame,
                           OperatorCommand, OperatorInput, Release, Select, SubjectObs, encode_frame, now_ms)
from mhd.governor.config import AXES, load_autopilot_config, load_governor_config, replace
from mhd.governor.core import AGENT_COMMANDS, Governor
from mhd.governor.telemetry import build_telemetry

from .frames import FrameStore
from .recorder import FlightRecorder, install_replay_routes, recording_enabled
from .modes import CAMERA_ONLY_FLAG, BackendEnv, camera_only_requested

logger = logging.getLogger(__name__)

LEGACY_TILT_HOLD_S = 0.3        # legacy UI repeats camera_tilt_direction at 30 Hz inside 'axes'
# POST /command is the UI's HTTP fallback (PLAN 4.8: LAND and E-STOP) plus what the agent's safety
# words need (pause). Everything that starts motion (takeoff, arm, resume, ...) needs /ws. With
# ?role=agent the agent whitelist applies (pause, land). Review finding: HTTP used to have full rights.
HTTP_COMMANDS = frozenset({"land", "estop", "estop_reset", "pause"})
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

# POST /debug/rc_pause (Session A link-loss test 3: RC control packets stop, the 1 Hz keepalive
# continues; on the Lewei board there is no keepalive, so ALL packets to the drone stop). Real mode:
# only while the governor is in one of these states (props-off test on the bench), at most
# RC_PAUSE_MAX_S["real"]. Twin: any state. The pause also ends at once when the flight state changes
# (LAND, E-STOP, TAKEOFF, a safety landing) or the backend shuts down, so those packets always reach
# the drone.
REAL_MODE_RC_PAUSE_STATES = frozenset({FlightState.on_ground})
RC_PAUSE_MAX_S = {"real": 15.0, "twin": 60.0}

# RC link flags (telemetry for the operator; the governor never reads them).
# rc_send_errors: an RC send failed within the last RC_SEND_ERRORS_RECENT_MS. The cumulative count stays
# in the extras (rc_send_errors), but one failure while the drone was powered off no longer lights the
# flag for the rest of the session.
RC_SEND_ERRORS_RECENT_MS = 10_000.0
# Lewei board (no keepalive): the drone's own replies are the link. drone_link_stale when no valid
# downlink packet arrived for DRONE_LINK_STALE_MS (DRONE_LINK_STALE_LANDING_MS while LANDING: the drone
# stops replying for ~5 s during LAND) or the reader thread died. A successful send proves nothing there:
# the iPhone USB tether holds the default route, so datagrams to a vanished drone still "succeed". The UI
# disables TAKEOFF on it (lib/gates.ts); LAND and E-STOP are never gated.
DRONE_LINK_STALE_FLAG = "drone_link_stale"
DRONE_LINK_STALE_MS = 3000.0
DRONE_LINK_STALE_LANDING_MS = 8000.0


def _no_constants(name: str):
    raise ValueError(f"{name} is not valid JSON here")


def loads_strict(text: str) -> Any:
    """json.loads that refuses NaN / Infinity / -Infinity (Python's json accepts them by default,
    and pydantic floats accept NaN: a NaN capture_ms used to defeat the staleness checks)."""
    return json.loads(text, parse_constant=_no_constants)


def allowed_origin_hosts() -> set[str]:
    """Loopback, plus hosts listed in MHD_EXTRA_ORIGIN_HOSTS (comma-separated), e.g. when the UI is
    opened from another address on purpose."""
    extra = {h.strip().lower() for h in os.environ.get("MHD_EXTRA_ORIGIN_HOSTS", "").split(",") if h.strip()}
    return _LOOPBACK_HOSTS | extra


def cors_origin_regex() -> str:
    """CORS for the same pages that may open /ws (the UI's HTTP fallback for LAND / E-STOP)."""
    import re
    hosts = sorted(allowed_origin_hosts() - {"::1"})
    alts = "|".join(re.escape(h) for h in hosts) + r"|\[::1\]"
    return rf"https?://({alts})(:\d+)?"


def origin_allowed(origin: Optional[str]) -> bool:
    """No Origin (a local process) or an allowed page. Any other web page open in the operator's
    browser must not be able to command the drone (WebSockets are not covered by CORS)."""
    if not origin:
        return True
    try:
        host = (urlparse(origin).hostname or "").lower()
    except ValueError:
        return False
    return host in allowed_origin_hosts()



class Hub:
    def __init__(self) -> None:
        self.env: Optional[BackendEnv] = None
        self.gov: Optional[Governor] = None
        self.model = None
        self.frames = FrameStore()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.ui_sockets: set[WebSocket] = set()
        self.perception: set[WebSocket] = set()
        self.agents: set[WebSocket] = set()
        self.agent_uis: set[WebSocket] = set()
        self.find_request: Optional[FindRequest] = None
        self._relay_task: Optional[asyncio.Task] = None
        self._record_task: Optional[asyncio.Task] = None
        self.recorder: Optional[FlightRecorder] = None   # flight replay (MHD_RECORD=1): files only
        self.counters = {"obs": 0, "obs_rejected": 0, "ui_msgs": 0, "bad_msgs": 0, "legacy_axes_refused": 0,
                         "origin_refused": 0}
        self.rc_adapter = None                     # GovernedX69LgRcAdapter (link stats for telemetry)
        self.shutting_down = False
        self._shutdown_t: Optional[float] = None
        self._shutdown_callbacks: list[Callable[[], None]] = []
        self._conn_ids = itertools.count(1)
        self._legacy_warned = False

    # ==========================================================================================
    # Lifecycle (called from web_server.lifespan)

    def setup(self, env: BackendEnv, governor: Optional[Governor] = None):
        """Build the governor and the governed X69 model. Returns the model for FlightController."""
        from mhd.governor.turbodrone_glue import install_governor
        self.__init__()                      # fresh per-run state (no stale FindRequest or sockets)
        self.env = env
        if governor is None:
            cfg = load_governor_config()
            if os.environ.get("GOVERNOR_TWIN_HEADLESS", "").strip().lower() in ("1", "true", "yes", "on"):
                # Twin-only bypass of the UI-heartbeat checks for headless runs. The Governor itself
                # raises GovernorError in real mode, so the backend refuses to start.
                cfg = replace(cfg, twin_headless_bypass=True)
            governor = Governor(cfg, load_autopilot_config(), env.flight_mode)
        self.gov = governor
        self.frames = FrameStore(env.flight_mode, on_frame=self.gov.note_frame)
        self.model = install_governor(self.gov)
        logger.warning("[hub] governor installed: FLIGHT_MODE=%s DRONE_IP=%s board=%s (armed=%s, "
                       "headless_bypass=%s, camera_only=%s)", env.flight_mode, env.drone_ip, env.board,
                       self.gov.armed, self.gov.cfg.twin_headless_bypass, self.camera_only)
        return self.model

    @property
    def camera_only(self) -> bool:
        """MHD_CAMERA_ONLY (real camera, no flight). Either source is enough: the validated env, or
        the raw variable (fail safe, should the RC adapter ever be built before setup())."""
        return bool(self.env is not None and self.env.camera_only) or camera_only_requested()

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.frames.bind_loop(self.loop)
        self._relay_task = asyncio.create_task(self._relay_events())
        if recording_enabled() and self.gov is not None:
            self.recorder = FlightRecorder(flight_mode="camera" if self.camera_only else self.gov.flight_mode)
            self._record_task = asyncio.create_task(self._record_flights())

    def make_rc_adapter(self, drone_ip: str, control_port: int, local_port: int = 0):
        """The X69 RC adapter with keepalive retry and link counters (see turbodrone_glue).

        Camera-only mode: an adapter with no socket at all. The governor still builds every RC
        packet (twin behavior), and nothing is ever sent, to any address, not even loopback.
        MHD_BOARD=lewei (real mode only): the raw 20-byte X69 payload to 192.168.0.1:50000, no wrapper,
        no keepalive, plus a reader for the drone's downlink (telemetry and drone_link_stale).
        """
        from mhd.governor.turbodrone_glue import CameraOnlyRcAdapter, GovernedX69LgRcAdapter, LeweiRcAdapter
        board = self.env.board if self.env is not None else "x69_lg"
        if self.camera_only:
            self.rc_adapter = CameraOnlyRcAdapter(drone_ip, control_port, local_port=local_port)
            logger.warning("[hub] CAMERA ONLY (real camera, NO FLIGHT): RC adapter without a socket; "
                           "no RC packet or keepalive is sent to any address")
        elif board == "lewei":
            self.rc_adapter = LeweiRcAdapter(drone_ip, control_port, local_port=local_port)
            logger.warning("[hub] Lewei board: raw 20-byte X69 payload to %s:%s (no wrapper, no keepalive)",
                           drone_ip, control_port)
        else:
            self.rc_adapter = GovernedX69LgRcAdapter(drone_ip, control_port, local_port=local_port)
        return self.rc_adapter

    def on_shutdown(self, callback: Callable[[], None]) -> None:
        """Run `callback` (no arguments, must not block) when the shutdown begins, e.g. end MJPEG."""
        self._shutdown_callbacks.append(callback)

    def begin_shutdown(self, why: str = "shutdown") -> None:
        """Start landing NOW: every following RC packet is neutral sticks plus land (a latched
        E-stop stays E-stop). Idempotent; safe to call from a signal handler."""
        if self.shutting_down:
            return
        self.shutting_down = True
        self._shutdown_t = time.monotonic()
        if self.gov is not None:
            self.gov.begin_shutdown()
        logger.warning("[hub] %s: neutral + land from now on", why)
        for cb in list(self._shutdown_callbacks):
            try:
                cb()
            except Exception:  # noqa: BLE001
                logger.exception("[hub] shutdown callback failed")

    def install_signal_hooks(self, min_land_s: float = 0.5) -> bool:
        """Begin the landing on the FIRST SIGINT/SIGTERM, before uvicorn waits for open responses.

        uvicorn runs the lifespan shutdown (our 0.5 s land) only after every open HTTP response has
        finished, and skips it after a second Ctrl-C. The UI keeps /mjpeg open, so Ctrl-C used to
        keep flying the last decision and never landed (review finding). This wraps uvicorn's
        handlers: the first signal starts the landing at once; a force-quit (second Ctrl-C) first
        lets the RC thread send land for at least min_land_s. Only possible in the main thread
        (not under TestClient); returns whether the hooks were installed.
        """
        if threading.current_thread() is not threading.main_thread():
            return False
        for sig in (signal.SIGINT, signal.SIGTERM):
            prev = signal.getsignal(sig)

            def handler(signum, frame, _prev=prev):
                first = not self.shutting_down
                self.begin_shutdown(f"signal {signal.Signals(signum).name}")
                chains = callable(_prev)          # uvicorn's handle_exit: it shuts down gracefully
                if (not first or not chains) and self._shutdown_t is not None:
                    left = min_land_s - (time.monotonic() - self._shutdown_t)
                    if left > 0:
                        time.sleep(left)          # the RC thread keeps sending land meanwhile
                if chains:
                    _prev(signum, frame)
                elif _prev == signal.SIG_DFL:     # not under uvicorn: default action, after the land
                    signal.signal(signum, signal.SIG_DFL)
                    signal.raise_signal(signum)
            signal.signal(sig, handler)

        return True

    async def shutdown(self) -> None:
        """Graceful stop: neutral sticks plus land for 0.5 s, before the RC thread is stopped."""
        if self.gov is None:
            return
        self.begin_shutdown()
        left = 0.5 - (time.monotonic() - (self._shutdown_t or time.monotonic()))
        logger.warning("[hub] shutdown: neutral + land for %.1f s more", max(0.0, left))
        if left > 0:
            await asyncio.sleep(left)
        if self._relay_task:
            self._relay_task.cancel()
        if self._record_task:
            self._record_task.cancel()
        if self.recorder is not None:
            self.recorder.close()


    def close(self) -> None:
        if self.gov is not None:
            self.gov.close()

    def strategy_alarm(self) -> bool:
        st = getattr(self.model, "strategy", None)
        return bool(getattr(st, "alarm", False))

    # ==========================================================================================
    # Frames (frame pump thread)

    def on_frame(self, frame) -> None:
        try:
            self.frames.put(frame)
        except Exception:  # noqa: BLE001 - never break the MJPEG pump
            logger.exception("[hub] frame store failed")

    # ==========================================================================================
    # UI (/ws)

    async def serve_ui(self, ws: WebSocket) -> None:
        """Patched /ws: every message goes to the governor. A disconnect zeroes human input
        that this socket owned."""
        if not origin_allowed(ws.headers.get("origin")):
            self.counters["origin_refused"] += 1
            logger.warning("[hub] refused /ws from origin %r", ws.headers.get("origin"))
            await ws.close(code=1008)
            return
        await ws.accept()
        conn = next(self._conn_ids)
        self.ui_sockets.add(ws)
        try:
            while True:
                text = await ws.receive_text()
                try:
                    data = loads_strict(text)
                except ValueError:
                    self.counters["bad_msgs"] += 1
                    continue
                if isinstance(data, dict):
                    self.handle_ui_message(data, conn=conn)
        except WebSocketDisconnect:
            logger.info("[hub] UI /ws disconnected")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[hub] UI /ws error: %s", exc)
        finally:
            self.ui_sockets.discard(ws)
            if self.gov is not None:
                self.gov.ui_disconnected(owner=conn)

    def _counts_as_heartbeat(self, data: dict) -> bool:
        """Real mode: only the UI worker's {"type":"hb","client":"ui"} proves the operator's UI is
        alive (a tool such as sysid streaming axes must not mask a dead UI). Twin mode: any /ws
        message counts, so the stock upstream UI still works against the twin."""
        if self.gov is not None and self.gov.flight_mode == "twin":
            return True
        if data.get("type") != "hb":
            return False
        try:
            return Heartbeat.model_validate(data).client == "ui"
        except ValidationError:
            return False

    def handle_ui_message(self, data: dict, conn=None) -> Optional[dict]:
        gov = self.gov
        if gov is None:
            return None
        self.counters["ui_msgs"] += 1
        if self._counts_as_heartbeat(data):
            gov.heartbeat()
        t = data.get("type")
        if t == "hb":
            return None
        if t == "axes":
            if gov.flight_mode == "real":
                # Real mode: only the contract's OperatorInput (new UI: sent while nonzero, zeroed on
                # blur). The legacy upstream UI streams held keys at 30 Hz with camera_tilt_direction
                # and has no blur handler, so a key stuck after focus loss never went stale.
                try:
                    inp = OperatorInput.model_validate(data)
                except ValidationError:
                    self.counters["legacy_axes_refused"] += 1
                    if not self._legacy_warned:
                        self._legacy_warned = True
                        logger.error("[hub] real mode: refusing non-contract 'axes' (legacy UI?): %s",
                                     {k: data[k] for k in list(data)[:8]})
                    return {"ok": False, "reason": "axes_not_operator_input"}
                gov.submit_human({a: getattr(inp, a) for a in AXES}, inp.mode, owner=conn)
                return None
            # Twin: new UI (OperatorInput) and legacy upstream UI (30 Hz, mode "inc" | "abs" |
            # "mouse", camera_tilt_direction). Both go to the governor.
            gov.submit_human({a: data.get(a, 0.0) for a in AXES}, str(data.get("mode", "abs")), owner=conn)
            if "camera_tilt_direction" in data:
                d = _num(data.get("camera_tilt_direction"))
                gov.set_tilt(int(_sign(d)), hold_s=LEGACY_TILT_HOLD_S)
            return None
        if t == "cmd":
            try:
                cmd = OperatorCommand.model_validate(data)
            except ValidationError as exc:
                self.counters["bad_msgs"] += 1
                return {"ok": False, "reason": f"invalid command: {exc.errors()[0].get('msg')}"}
            return self.command(cmd, source="ui")
        if t in ("takeoff", "land"):
            return self.command(OperatorCommand(command=t), source="ui")
        if t in ("estop", "emergency_stop"):
            return self.command(OperatorCommand(command="estop"), source="ui")
        if t in ("set_camera_tilt", "camera_tilt"):
            if gov.flight_mode == "real":
                return None                       # real mode: tilt only via cmd tilt_up / tilt_down
            if "camera_tilt_direction" in data:

                d = _sign(_num(data.get("camera_tilt_direction")))
            else:
                d = {2: 1, 1: -1}.get(int(_num(data.get("camera_tilt", 0))), 0)
            gov.set_tilt(int(d))
            return None
        if t in ("set_profile", "set_speed_index", "speed_index"):
            return None                          # ignored: the governor pins tier 2 and expo 0
        return None

    # ==========================================================================================
    # Commands and missions

    def command(self, cmd: OperatorCommand, source: str) -> dict:
        if self.gov is None:
            raise HTTPException(503, "governor not running")
        if source == "http" and cmd.command not in HTTP_COMMANDS:
            # Logged by the governor's event list like any refused command.
            return {"ok": False, "flight_state": self.gov.state.value, "reason": "not_allowed_over_http"}
        res = self.gov.command(cmd.command, arm_phrase=cmd.arm_phrase, pack_id=cmd.pack_id, source=source)
        self._record("cmd", {"command": cmd.command, "source": source, "ok": bool(res.ok),
                             "reason": getattr(res, "reason", None)})
        if res.ok and cmd.command == "estop" and self.find_request is not None:
            # E-stop ends the mission (the governor dropped it): perception lets go, and a
            # perception client that reconnects later gets no stale FindRequest.
            self._send_perception(Release(mission_id=self.find_request.mission_id))
            self.find_request = None
        return res.as_dict()

    def apply_mission(self, mission: Mission) -> dict:
        if self.gov is None:
            raise HTTPException(503, "governor not running")
        res = self.gov.set_mission(mission)
        if mission.intent.action == "follow" and res.get("ok"):
            if self.find_request is not None and self.find_request.mission_id != mission.mission_id:
                self._send_perception(Release(mission_id=self.find_request.mission_id))
            i = mission.intent
            self.find_request = FindRequest(mission_id=mission.mission_id, subject_phrase=i.subject_phrase,
                                            category=i.category, color=i.color)
            self._send_perception(self.find_request)
        res["flight_state"] = self.gov.state.value
        self._record("mission", {**mission.model_dump(mode="json"), "ok": bool(res.get("ok")),
                                 "reason": res.get("reason")})
        return res

    # ==========================================================================================
    # Perception (/ws/subject)

    async def serve_perception(self, ws: WebSocket) -> None:
        self.perception.add(ws)
        try:
            if self.find_request is not None:
                await ws.send_text(self.find_request.model_dump_json())
            while True:
                try:
                    data = loads_strict(await ws.receive_text())
                except ValueError:                 # bad JSON, or NaN / Infinity: drop the message
                    self.counters["bad_msgs"] += 1
                    continue
                self.handle_perception_message(data)
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("[hub] perception socket error: %s", exc)
        finally:
            self.perception.discard(ws)

    def handle_perception_message(self, data: Any) -> None:
        if not isinstance(data, dict) or self.gov is None:
            return
        t = data.get("type")
        try:
            if t == "subject_obs":
                obs = SubjectObs.model_validate(data)
                if self.gov.submit_obs(obs):
                    self.counters["obs"] += 1
                else:
                    self.counters["obs_rejected"] += 1
            elif t == "candidates":
                cands = Candidates.model_validate(data)
                self._broadcast_agent_roles(cands.model_dump_json())
        except ValidationError:
            self.counters["bad_msgs"] += 1

    def _send_perception(self, msg) -> None:
        self._schedule(self._broadcast(self.perception, msg.model_dump_json()))

    # ==========================================================================================
    # Agent and agent-UI (/ws/agent?role=agent|ui)

    async def serve_agent(self, ws: WebSocket, role: str) -> None:
        group = self.agents if role == "agent" else self.agent_uis
        group.add(ws)
        try:
            while True:
                try:
                    data = loads_strict(await ws.receive_text())
                except ValueError:
                    self.counters["bad_msgs"] += 1
                    continue
                if isinstance(data, dict):
                    await self.handle_agent_message(data, role, ws)
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("[hub] agent socket (%s) error: %s", role, exc)
        finally:
            group.discard(ws)

    async def handle_agent_message(self, data: dict, role: str, ws: Optional[WebSocket] = None) -> None:
        t = data.get("type")
        try:
            if t == "agent_event":
                ev = AgentEvent.model_validate(data)
                targets = self.agent_uis if role == "agent" else self.agents
                text = ev.model_dump_json()
                if ev.kind != "partial":
                    self._record("agent", text)
                await self._broadcast(targets, text)
            elif t == "mission" and role == "agent":
                res = self.apply_mission(Mission.model_validate(data))
                if ws is not None:
                    await ws.send_text(json.dumps({"type": "mission_ack", **res}))
            elif t == "cmd" and role == "agent":
                res = self.command(OperatorCommand.model_validate(data), source="agent")
                if ws is not None:
                    await ws.send_text(json.dumps({"type": "cmd_ack", **res}))
            elif t == "select" and role == "ui":
                await self._broadcast(self.perception, Select.model_validate(data).model_dump_json())
        except ValidationError:
            self.counters["bad_msgs"] += 1

    def _broadcast_agent_roles(self, text: str) -> None:
        self._record("agent", text)
        self._schedule(self._broadcast(self.agents | self.agent_uis, text))

    # ==========================================================================================
    # Debug: RC pause (Session A link-loss test 3)

    def rc_pause_status(self) -> dict:
        ad = self.rc_adapter
        if ad is None or not hasattr(ad, "rc_pause_status"):
            return {"active": False, "available": False}
        return {"available": True, **ad.rc_pause_status()}

    def rc_pause(self, seconds: Any) -> dict:
        """Stop RC control packets for `seconds`. DEBUG ONLY. X69: the D1 keepalive continues.
        Lewei board: it has no keepalive, so ALL packets to the drone stop.

        Real mode: only while the governor is ON_GROUND (the props-off bench test), at most 15 s.
        Twin mode: any state, at most 60 s. Ends early (and that packet goes out) when the flight
        state changes or the backend shuts down. Logged loudly, shown in telemetry as rc_paused.
        """
        gov, ad = self.gov, self.rc_adapter
        mode = gov.flight_mode if gov is not None else (self.env.flight_mode if self.env else "unknown")
        st = gov.state if gov is not None else None
        base = {"flight_mode": mode, "flight_state": st.value if st is not None else "unknown"}

        def no(reason: str, **kw) -> dict:
            logger.warning("[hub] /debug/rc_pause refused: %s (%s)", reason, {**base, **kw})
            return {"ok": False, "reason": reason, **base, **kw}

        if gov is None:
            return no("governor_not_running")
        if ad is None or not hasattr(ad, "pause_rc"):
            return no("no_governed_rc_adapter")
        try:
            secs = float(seconds)
        except (TypeError, ValueError):
            return no("seconds_not_a_number")
        limit = RC_PAUSE_MAX_S.get(mode)
        if limit is None:
            return no("unknown_flight_mode")
        if not (secs == secs and 0.0 < secs <= limit):   # NaN fails secs == secs
            return no("seconds_out_of_range", max_s=limit)
        if self.shutting_down:
            return no("shutting_down")
        if self.rc_pause_status().get("active"):
            return no("already_paused")
        if mode == "real" and st not in REAL_MODE_RC_PAUSE_STATES:
            return no("real_mode_needs_on_ground",
                      allowed_states=sorted(x.value for x in REAL_MODE_RC_PAUSE_STATES))
        start_state = st

        def keep_paused() -> bool:          # RC thread, every dropped packet: lock-free reads only
            return (not self.shutting_down) and gov.state == start_state

        status = ad.pause_rc(secs, keep_paused, note=f"{mode}/{start_state.value}")
        banner = "=" * 24
        if _no_keepalive(ad):
            what_log = "ALL packets to the drone STOPPED for %.1f s (Lewei board: there is no keepalive)"
            what_ui = f"all packets to the drone stopped for {secs:.0f} s (no keepalive on this board)"
        else:
            what_log = "RC control packets STOPPED for %.1f s, D1 keepalive continues"
            what_ui = f"control packets stopped for {secs:.0f} s, keepalive continues"
        logger.warning("%s RC PAUSE (DEBUG): " + what_log + " (FLIGHT_MODE=%s, flight state %s). "
                       "LAND / E-STOP / TAKEOFF end it at once. %s", banner, secs, mode, start_state.value, banner)
        self._status_event(f"RC PAUSE (debug): {what_ui}",
                           {"kind": "rc_pause", "seconds": secs, "flight_state": start_state.value})
        self._schedule_rc_pause_report(secs)
        return {"ok": True, "seconds": secs, "max_s": limit, **base, "status": status}

    def rc_resume(self) -> dict:
        ad = self.rc_adapter
        if ad is None or not hasattr(ad, "resume_rc"):
            return {"ok": False, "reason": "no_governed_rc_adapter"}
        was = self.rc_pause_status().get("active", False)
        status = ad.resume_rc("debug_resume")
        if was:
            self._status_event("RC PAUSE ended by /debug/rc_resume", {"kind": "rc_pause_end",
                                                                      "reason": "debug_resume"})
        return {"ok": True, "was_active": was, "status": status}

    def _schedule_rc_pause_report(self, secs: float) -> None:
        """After the pause, tell the UI (and the log, already done by the adapter) how it ended."""
        async def report():
            await asyncio.sleep(min(secs, 60.0) + 0.2)
            for _ in range(20):                       # the RC thread ends it on its next packet
                st = self.rc_pause_status()
                if not st.get("active"):
                    break
                await asyncio.sleep(0.1)
            st = self.rc_pause_status()
            sent = ("no keepalive on this board: nothing was sent" if _no_keepalive(self.rc_adapter)
                    else f"{st.get('keepalives_during')} keepalives sent")
            await self._broadcast(self.agents | self.agent_uis, AgentEvent(
                kind="status", ts_ms=now_ms(),
                text=f"RC PAUSE ended ({st.get('end_reason')}): {st.get('dropped')} control packets dropped, {sent}",
                data={"kind": "rc_pause_end", **{k: st.get(k) for k in
                                                 ("end_reason", "dropped", "keepalives_during", "elapsed_s")}},
            ).model_dump_json())
        self._schedule(report())

    def _status_event(self, text: str, data: dict) -> None:
        msg = AgentEvent(kind="status", text=text, data=data, ts_ms=now_ms())
        self._broadcast_agent_roles(msg.model_dump_json())

    # ==========================================================================================
    # Telemetry and event relay

    def link_flags_and_stats(self) -> tuple[list, dict]:
        """RC link health from the governed adapter (review finding: send errors were silent).

        rc_send_errors: an RC send failed within the last 10 s (the extras keep the cumulative count).
        X69: keepalive_stale when the D1 keepalive thread died or sent nothing for 3 s. Lewei board:
        drone_link_stale when the drone's downlink is silent for 3 s (8 s while LANDING) or its reader
        thread died. Telemetry only: no governor decision reads these flags.
        """
        flags: list[str] = []
        extra: dict = {}
        ad = self.rc_adapter
        if ad is not None and hasattr(ad, "link_stats"):
            s = ad.link_stats()
            extra.update({"rc_send_errors": s["rc_errors"], "keepalive_age": s["keepalive_age_ms"],
                          "keepalive_errors": s["keepalive_errors"]})
            err_age = s.get("rc_error_age_ms", 0.0)      # an adapter without it: any error counts (old rule)
            if s["rc_errors"] and err_age is not None and err_age <= RC_SEND_ERRORS_RECENT_MS:
                flags.append("rc_send_errors")
            if getattr(ad, "board", None) == "lewei":
                # keepalive_* are the drone's downlink here (no keepalive on this board): see LeweiRcAdapter.
                if not s["keepalive_alive"] or s["keepalive_age_ms"] > self.drone_link_limit_ms():
                    flags.append(DRONE_LINK_STALE_FLAG)
            elif not s["keepalive_alive"] or s["keepalive_age_ms"] > 3000.0:
                flags.append("keepalive_stale")
        if ad is not None and hasattr(ad, "fc_telemetry"):
            # Lewei board downlink (battery volts, status byte, age): informational only.
            extra.update({k: v for k, v in ad.fc_telemetry().items() if v is not None})
        return flags, extra

    def drone_link_limit_ms(self) -> float:
        """Lewei board: how old the drone's last reply may be before drone_link_stale (lock-free read)."""
        gov = self.gov
        landing = gov is not None and getattr(gov, "state", None) == FlightState.landing
        return DRONE_LINK_STALE_LANDING_MS if landing else DRONE_LINK_STALE_MS

    def telemetry_json(self) -> str:
        if self.gov is None:
            raise HTTPException(503, "governor not running")
        extra = {"frames_received": self.frames.count, "obs_accepted": self.counters["obs"],
                 "frames_rejected": self.gov.frames_rejected,
                 "obs_invalid": sum(self.gov.obs_rejected.values())}
        if self.frames.capture_rise_ms is not None:
            extra["capture_rise"] = round(self.frames.capture_rise_ms, 1)
        link_flags, link_extra = self.link_flags_and_stats()
        extra.update(link_extra)
        flags = list(link_flags)
        if self.camera_only:
            flags.insert(0, CAMERA_ONLY_FLAG)         # the UI's REAL CAMERA - NO FLIGHT banner
        pause = self.rc_pause_status()
        if pause.get("active"):
            flags.append("rc_paused")                 # debug RC pause: no control packets now
            extra["rc_pause_left"] = round(1000.0 * float(pause.get("left_s") or 0.0), 1)
            extra["rc_pause_dropped"] = pause.get("dropped", 0)
        if self.counters["legacy_axes_refused"]:
            flags.append("legacy_axes_refused")
        return build_telemetry(self.gov, alarm=self.strategy_alarm(), extra_flags=flags,
                               extra_latency=extra).model_dump_json()


    async def serve_telemetry(self, ws: WebSocket) -> None:
        try:
            while True:
                await ws.send_text(self.telemetry_json())
                await asyncio.sleep(0.1)          # 10 Hz
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("[hub] telemetry socket closed: %s", exc)

    async def _relay_events(self) -> None:
        """Governor/autopilot events -> AgentEvent(kind='status') to the agent and agent-UI roles."""
        while True:
            await asyncio.sleep(0.1)
            if self.gov is None:
                continue
            for ev in self.gov.drain_events():
                kind = ev.get("kind")
                if kind == "announce":
                    text = ev.get("text")
                elif kind == "safety":
                    text = ev.get("text")
                elif kind == "state":
                    text = f"autopilot {ev.get('from')} -> {ev.get('to')} ({ev.get('reason')})"
                elif kind == "flight_state":
                    text = f"flight {ev.get('from')} -> {ev.get('to')}" + (f" ({ev['cause']})" if ev.get("cause") else "")
                else:
                    continue
                msg = AgentEvent(kind="status", text=text, data=ev, ts_ms=ev.get("ts_ms") or now_ms())
                self._record("agent", msg.model_dump_json())
                await self._broadcast(self.agents | self.agent_uis, msg.model_dump_json())

    # ==========================================================================================
    # Flight recorder (replay in the UI). Files only: nothing here reaches the governor or the drone.

    def _record(self, kind: str, payload) -> None:
        rec = self.recorder
        if rec is not None:
            rec.note(kind, payload)

    async def _record_flights(self) -> None:
        warned = False
        while True:
            await asyncio.sleep(0.1)
            rec, gov = self.recorder, self.gov
            if rec is None or gov is None:
                continue
            try:
                rec.sample(self.telemetry_json(), gov.state.value, self.frames.latest())
            except Exception:  # noqa: BLE001 - recording must never disturb the backend
                if not warned:
                    warned = True
                    logger.exception("[hub] flight recorder sample failed")

    # ==========================================================================================
    # Helpers

    def _schedule(self, coro) -> None:
        loop = self.loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None:
            running.create_task(coro)
        elif loop is not None and not loop.is_closed():
            asyncio.run_coroutine_threadsafe(coro, loop)
        else:
            coro.close()

    async def _broadcast(self, sockets: set, text: str) -> None:
        for s in list(sockets):
            if s.client_state != WebSocketState.CONNECTED:
                sockets.discard(s)
                continue
            try:
                await s.send_text(text)
            except Exception:  # noqa: BLE001
                sockets.discard(s)


def _no_keepalive(ad) -> bool:
    """The Lewei board's adapter sends nothing but RC packets: an RC pause stops ALL packets."""
    return getattr(ad, "board", None) == "lewei"


def _num(v: Any) -> float:
    try:
        f = float(v)
        return f if f == f else 0.0
    except (TypeError, ValueError):
        return 0.0


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


# ==============================================================================================
# Routes


def _check_origin(request: Request, hub: Hub) -> None:
    if not origin_allowed(request.headers.get("origin")):
        hub.counters["origin_refused"] += 1
        raise HTTPException(403, "origin not allowed")


async def _accept_if_origin_ok(ws: WebSocket, hub: Hub) -> bool:
    if not origin_allowed(ws.headers.get("origin")):
        hub.counters["origin_refused"] += 1
        logger.warning("[hub] refused %s from origin %r", ws.url.path, ws.headers.get("origin"))
        await ws.close(code=1008)
        return False
    await ws.accept()
    return True


def install_routes(app: FastAPI, hub: Hub) -> None:
    install_replay_routes(app)     # GET /flights...: saved flights for the UI's replay (read-only)

    @app.post("/command")
    async def post_command(cmd: OperatorCommand, request: Request, role: Optional[str] = None):
        # The UI's HTTP fallback (LAND, E-STOP; also pause and estop_reset): HTTP_COMMANDS only.
        # ?role=agent: the agent whitelist (pause, land). Does not refresh the UI heartbeat.
        _check_origin(request, hub)
        if role not in (None, "", "ui", "agent"):
            raise HTTPException(422, "role must be ui or agent")
        if role == "agent" and cmd.command not in AGENT_COMMANDS:
            return {"ok": False, "flight_state": hub.gov.state.value if hub.gov else "unknown",
                    "reason": "not_allowed_from_agent"}
        return hub.command(cmd, source="agent" if role == "agent" else "http")

    @app.post("/mission")
    async def post_mission(mission: Mission, request: Request):
        _check_origin(request, hub)
        return hub.apply_mission(mission)

    @app.post("/debug/rc_pause")
    async def post_rc_pause(request: Request):
        # Session A link-loss test 3 (DEBUG): {"seconds": 10}. Real mode: ON_GROUND only, <= 15 s.
        _check_origin(request, hub)
        try:
            body = loads_strict((await request.body()).decode() or "{}")
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(422, "body must be JSON like {\"seconds\": 10}")
        if not isinstance(body, dict) or "seconds" not in body:
            raise HTTPException(422, "body must be JSON like {\"seconds\": 10}")
        return hub.rc_pause(body["seconds"])

    @app.post("/debug/rc_resume")
    async def post_rc_resume(request: Request):
        _check_origin(request, hub)
        return hub.rc_resume()

    @app.get("/debug/rc_pause")
    async def get_rc_pause():
        return hub.rc_pause_status()

    @app.get("/state")
    async def get_state():
        from fastapi.responses import Response
        return Response(hub.telemetry_json(), media_type="application/json")

    @app.websocket("/ws/telemetry")
    async def ws_telemetry(ws: WebSocket):
        await ws.accept()
        await hub.serve_telemetry(ws)

    @app.websocket("/ws/frames")
    async def ws_frames(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                req = NextFrame.model_validate(json.loads(await ws.receive_text()))
                item = None
                while item is None:
                    item = await hub.frames.wait_newer(req.after_frame_id, timeout=1.0)
                    if ws.client_state != WebSocketState.CONNECTED:
                        return
                meta, jpeg = item
                await ws.send_bytes(encode_frame(meta, jpeg))
        except (WebSocketDisconnect, RuntimeError, ValidationError, ValueError):
            pass

    @app.websocket("/ws/subject")
    async def ws_subject(ws: WebSocket):
        if await _accept_if_origin_ok(ws, hub):
            await hub.serve_perception(ws)

    @app.websocket("/ws/agent")
    async def ws_agent(ws: WebSocket, role: str = "agent"):
        if role not in ("agent", "ui"):
            await ws.close(code=1008)
            return
        if await _accept_if_origin_ok(ws, hub):
            await hub.serve_agent(ws, role)
