"""The backend hub: owns the governor and every new endpoint (PLAN.md 2, docs/interfaces.md).

web_server.py (patched turbodrone) creates one Hub, builds the X69 model through it, and hands
every UI, perception and agent socket to it. Nothing here writes sticks: all input becomes a
governor call (submit_human, heartbeat, command, set_mission, submit_obs, note_frame).

Endpoints installed by install_routes(app, hub):
  WS /ws (via serve_ui)  POST /command  WS /ws/telemetry  GET /state  WS /ws/frames
  WS /ws/subject  WS /ws/agent?role=agent|ui  POST /mission
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from mhd.contracts import (AgentEvent, Candidates, FindRequest, Mission, NextFrame, OperatorCommand,
                           Release, Select, SubjectObs, encode_frame, now_ms)
from mhd.governor.config import AXES, load_autopilot_config, load_governor_config
from mhd.governor.core import Governor
from mhd.governor.telemetry import build_telemetry

from .frames import FrameStore
from .modes import BackendEnv

logger = logging.getLogger(__name__)

LEGACY_TILT_HOLD_S = 0.3        # legacy UI repeats camera_tilt_direction at 30 Hz inside 'axes'


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
        self.counters = {"obs": 0, "obs_rejected": 0, "ui_msgs": 0, "bad_msgs": 0}

    # ==========================================================================================
    # Lifecycle (called from web_server.lifespan)

    def setup(self, env: BackendEnv, governor: Optional[Governor] = None):
        """Build the governor and the governed X69 model. Returns the model for FlightController."""
        from mhd.governor.turbodrone_glue import install_governor
        self.__init__()                      # fresh per-run state (no stale FindRequest or sockets)
        self.env = env
        self.gov = governor or Governor(load_governor_config(), load_autopilot_config(), env.flight_mode)
        self.frames = FrameStore(env.flight_mode, on_frame=self.gov.note_frame)
        self.model = install_governor(self.gov)
        logger.warning("[hub] governor installed: FLIGHT_MODE=%s DRONE_IP=%s (armed=%s, headless_bypass=%s)",
                       env.flight_mode, env.drone_ip, self.gov.armed, self.gov.cfg.twin_headless_bypass)
        return self.model

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.frames.bind_loop(self.loop)
        self._relay_task = asyncio.create_task(self._relay_events())

    async def shutdown(self) -> None:
        """Graceful stop: neutral sticks plus land for 0.5 s, before the RC thread is stopped."""
        if self.gov is None:
            return
        logger.warning("[hub] shutdown: neutral + land for 0.5 s")
        self.gov.begin_shutdown()
        await asyncio.sleep(0.5)
        if self._relay_task:
            self._relay_task.cancel()

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
        """Patched /ws: every message goes to the governor. A disconnect zeroes human input."""
        self.ui_sockets.add(ws)
        try:
            while True:
                text = await ws.receive_text()
                try:
                    data = json.loads(text)
                except ValueError:
                    self.counters["bad_msgs"] += 1
                    continue
                if isinstance(data, dict):
                    self.handle_ui_message(data)
        except WebSocketDisconnect:
            logger.info("[hub] UI /ws disconnected")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[hub] UI /ws error: %s", exc)
        finally:
            self.ui_sockets.discard(ws)
            if self.gov is not None:
                self.gov.ui_disconnected()

    def handle_ui_message(self, data: dict) -> Optional[dict]:
        gov = self.gov
        if gov is None:
            return None
        self.counters["ui_msgs"] += 1
        gov.heartbeat()                          # any message on /ws proves the UI is alive
        t = data.get("type")
        if t == "hb":
            return None
        if t == "axes":
            # New UI: OperatorInput, sent only while nonzero. Legacy upstream UI: 30 Hz with mode
            # "inc" | "abs" | "mouse" and camera_tilt_direction. Both go to the governor.
            gov.submit_human({a: data.get(a, 0.0) for a in AXES}, str(data.get("mode", "abs")))
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
        res = self.gov.command(cmd.command, arm_phrase=cmd.arm_phrase, pack_id=cmd.pack_id, source=source)
        if res.ok and cmd.command == "estop" and self.find_request is not None:
            self._send_perception(Release(mission_id=self.find_request.mission_id))
        return res.as_dict()

    def apply_mission(self, mission: Mission) -> dict:
        if self.gov is None:
            raise HTTPException(503, "governor not running")
        res = self.gov.set_mission(mission)
        if mission.intent.action == "follow":
            if self.find_request is not None and self.find_request.mission_id != mission.mission_id:
                self._send_perception(Release(mission_id=self.find_request.mission_id))
            i = mission.intent
            self.find_request = FindRequest(mission_id=mission.mission_id, subject_phrase=i.subject_phrase,
                                            category=i.category, color=i.color)
            self._send_perception(self.find_request)
        res["flight_state"] = self.gov.state.value
        return res

    # ==========================================================================================
    # Perception (/ws/subject)

    async def serve_perception(self, ws: WebSocket) -> None:
        self.perception.add(ws)
        try:
            if self.find_request is not None:
                await ws.send_text(self.find_request.model_dump_json())
            while True:
                data = json.loads(await ws.receive_text())
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
                data = json.loads(await ws.receive_text())
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
                await self._broadcast(targets, ev.model_dump_json())
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
        self._schedule(self._broadcast(self.agents | self.agent_uis, text))

    # ==========================================================================================
    # Telemetry and event relay

    def telemetry_json(self) -> str:
        if self.gov is None:
            raise HTTPException(503, "governor not running")
        extra = {"frames_received": self.frames.count, "obs_accepted": self.counters["obs"]}
        return build_telemetry(self.gov, alarm=self.strategy_alarm(), extra_latency=extra).model_dump_json()

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
                await self._broadcast(self.agents | self.agent_uis, msg.model_dump_json())

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


def install_routes(app: FastAPI, hub: Hub) -> None:
    @app.post("/command")
    async def post_command(cmd: OperatorCommand):
        # HTTP fallback for LAND and E-STOP. Does not refresh the UI heartbeat on purpose.
        return hub.command(cmd, source="http")

    @app.post("/mission")
    async def post_mission(mission: Mission):
        return hub.apply_mission(mission)

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
        await ws.accept()
        await hub.serve_perception(ws)

    @app.websocket("/ws/agent")
    async def ws_agent(ws: WebSocket, role: str = "agent"):
        if role not in ("agent", "ui"):
            await ws.close(code=1008)
            return
        await ws.accept()
        await hub.serve_agent(ws, role)
