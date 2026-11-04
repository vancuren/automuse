"""Twin server: the X69 emulator process (PLAN.md 4.2, docs/interfaces.md "twin").

    $MHD_PY -m mhd.twin.server --http-port 8766 --rc-port 23458 --rtsp-port 8554 --scenario s1

- UDP RC server on 127.0.0.1:<rc-port>: X69 control datagrams and keepalives (protocol.py),
  with injectable loss, burst loss, and jitter; inter-arrival p50/p99 with a 60 ms alarm.
- Dynamics loop at 100 Hz (dynamics.py).
- WS /ws/render: the Three.js page connects with ?role=camera (drone camera, frames pulled:
  newest only, one in flight), ?role=chase (passive viewer), or ?role=chase_rec (chase frames
  pulled for the demo recording). The server sends {"type":"scene"} then {"type":"pose"} per
  frame; the page answers with binary [uint32 LE render_id][JPEG].
- Frame path (video.py): JPEG -> decode -> blur/noise -> counter -> latency buffer -> ffmpeg
  hevc_videotoolbox -> mediamtx (RTSP only, 127.0.0.1:<rtsp-port>, path live/ch00_1).
- WS /ws/truth: mhd.contracts.Truth for every rendered frame. GET /truth/latest.
- POST /control: TwinControl (plus the local extensions in TwinControlExt). GET /status.
- GET / (drone camera page), GET /chase (chase view), GET /frame.jpg (latest stamped frame).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from mhd.contracts import FRAME_H, FRAME_W, Truth, TwinControl, now_ms

from . import media
from . import protocol as P
from .dynamics import RcFaultInjector, RcLinkStats, X69Dynamics, load_params
from .scenarios import Scenario, list_scenarios, load_scenario
from .video import FileRecorder, VideoPipeline

log = logging.getLogger("mhd.twin")
TREE = Path(__file__).resolve().parents[3]
WEB_DIR = TREE / "twin" / "web"
LOOPBACK = "127.0.0.1"


class TwinControlExt(TwinControl):
    """Local extension of the frozen TwinControl contract (all fields optional)."""
    rc_burst_rate: Optional[float] = None       # probability per packet of entering a loss burst
    rc_burst_len: Optional[int] = None          # mean burst length (packets)
    rc_jitter_ms: Optional[float] = None        # extra RC delay, uniform 0..jitter, order kept
    headless: Optional[bool] = None
    link_loss_s: Optional[float] = None
    link_loss_action: Optional[Literal["land", "hover", "drop"]] = None
    battery_s: Optional[float] = None
    fps: Optional[float] = None
    noise: Optional[float] = None               # sensor noise sigma (8-bit levels)
    blur: Optional[float] = None                # Gaussian blur sigma (px)
    hfov_deg: Optional[float] = None
    cam_tilt_deg: Optional[float] = None
    pause_actors: Optional[bool] = None
    record_chase: Optional[bool] = None         # start/stop the chase MP4 (needs a chase_rec page)
    drone_pose: Optional[dict] = None           # {"x":..,"y":..,"yaw_deg":..} on reset


class ScenarioClock:
    """Scenario time in seconds, pausable."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.paused_at: Optional[float] = None
        self.paused_total = 0.0

    def now(self) -> float:
        ref = self.paused_at if self.paused_at is not None else time.monotonic()
        return ref - self.t0 - self.paused_total

    def pause(self, on: bool) -> None:
        if on and self.paused_at is None:
            self.paused_at = time.monotonic()
        elif not on and self.paused_at is not None:
            self.paused_total += time.monotonic() - self.paused_at
            self.paused_at = None


class JsonlWriter:
    """Background JSONL writer (never blocks the event loop on file I/O)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._q: queue.Queue = queue.Queue(maxsize=10000)
        self._th = threading.Thread(target=self._loop, name=f"jsonl-{path.stem}", daemon=True)
        self._th.start()

    def write(self, line: str) -> None:
        try:
            self._q.put_nowait(line)
        except queue.Full:
            pass

    def _loop(self) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            while True:
                item = self._q.get()
                if item is None:
                    break
                f.write(item + "\n")
                if self._q.empty():
                    f.flush()

    def close(self) -> None:
        self._q.put(None)
        self._th.join(timeout=3.0)


class RcServer(asyncio.DatagramProtocol):
    def __init__(self, twin: "Twin") -> None:
        self.twin = twin

    def datagram_received(self, data: bytes, addr) -> None:
        self.twin.on_datagram(data, addr)


class RenderClient:
    def __init__(self, ws: WebSocket, role: str) -> None:
        self.ws = ws
        self.role = role
        self.scene_version = -1
        self.info: dict = {}
        self.frames = 0


class Twin:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.params = load_params(args.params)
        if args.hfov_deg:
            self.params["camera"]["hfov_deg"] = float(args.hfov_deg)
        self.dyn = X69Dynamics(self.params, seed=args.seed)
        self.injector = RcFaultInjector(seed=args.seed + 1)
        rs = self.params.get("rc_stats", {})
        self.rc_stats = RcLinkStats(window=int(rs.get("window", 500)), alarm_ms=float(rs.get("p99_alarm_ms", 60)))
        self.applied_stats = RcLinkStats(window=int(rs.get("window", 500)), alarm_ms=float(rs.get("p99_alarm_ms", 60)))
        self.t0_ms = now_ms()
        self.clock = ScenarioClock()
        self.scenario: Scenario = load_scenario(args.scenario)
        self.scene_version = 0
        self.render_id = 0
        self.fps = float(args.fps)
        self.rc_senders: dict[str, int] = {}
        self.renderers: set[RenderClient] = set()
        self.camera_client: Optional[RenderClient] = None
        self.truth_subs: set[asyncio.Queue] = set()
        self.latest_truth: Optional[Truth] = None
        self.frame_times: deque = deque(maxlen=200)
        self.render_timeouts = 0
        self.stale_frames = 0
        self.last_camera_frame_ms = 0.0
        self.placeholder_frames = 0
        self.chase_recorder: Optional[FileRecorder] = None
        self.chase_rec_path: Optional[Path] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.mediamtx = None
        self.video: Optional[VideoPipeline] = None
        self.rtsp_url = f"rtsp://{LOOPBACK}:{args.rtsp_port}/{media.RTSP_PATH}"
        run = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = Path(args.log_dir) / f"twin-{run}-{args.http_port}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.truth_log = JsonlWriter(self.log_dir / "truth.jsonl")
        self.event_log = JsonlWriter(self.log_dir / "events.jsonl")
        self._apply_scenario_start()
        if args.latency_ms:
            self._pending_target = float(args.latency_ms)
        else:
            self._pending_target = None

    # ------------------------------------------------------------------ time
    def sim_t(self) -> float:
        return (now_ms() - self.t0_ms) / 1000.0

    def _apply_scenario_start(self, pose: Optional[dict] = None) -> None:
        d = dict(self.scenario.drone)
        if pose:
            d.update(pose)
        t = self.sim_t()
        self.dyn.reset(float(d.get("x", 0.0)), float(d.get("y", 0.0)), float(d.get("yaw_deg", 0.0)))
        self.dyn.t = t
        if "cam_tilt_deg" in d:
            self.dyn.s.cam_tilt = math.radians(float(d["cam_tilt_deg"]))
        self.dyn.bounds = self.scenario.bounds
        self.dyn.texture = self.scenario.texture
        self.dyn.drift_scale = float(self.scenario.world.get("drift", 1.0))   # POST /control drift overrides
        self.clock = ScenarioClock()

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        await self.loop.create_datagram_endpoint(lambda: RcServer(self), local_addr=(LOOPBACK, self.args.rc_port))
        log.info("RC UDP listening on %s:%s", LOOPBACK, self.args.rc_port)
        if not self.args.no_video:
            if not self.args.external_mediamtx:
                self.mediamtx = await asyncio.to_thread(media.start_mediamtx, self.args.rtsp_port, self.log_dir)
                log.info("mediamtx RTSP on %s", self.rtsp_url)
            self.video = VideoPipeline(self.rtsp_url, fps=int(round(self.fps)), bitrate=self.args.bitrate,
                                       gop=self.args.gop, noise_sigma=self.args.noise, blur_sigma=self.args.blur,
                                       probe=not self.args.no_probe)
            self.video.set_target(self._pending_target)
            self.video.start()
        self._tasks = [asyncio.create_task(self._dynamics_loop(), name="dynamics"),
                       asyncio.create_task(self._placeholder_loop(), name="placeholder")]

    async def stop(self) -> None:
        for t in getattr(self, "_tasks", []):
            t.cancel()
        if self.chase_recorder:
            await asyncio.to_thread(self.chase_recorder.close)
            self.chase_recorder = None
        if self.video:
            self.video.begin_stop()
        media.stop_process(self.mediamtx)          # RTSP readers (probe, backend) disconnect now
        if self.video:
            await asyncio.to_thread(self.video.join)
        self.truth_log.close()
        self.event_log.close()

    async def _dynamics_loop(self) -> None:
        period = self.dyn.dt
        while True:
            self.dyn.advance_to(self.sim_t())
            await asyncio.sleep(period)

    # ------------------------------------------------------------------ RC
    def on_datagram(self, data: bytes, addr) -> None:
        t = time.monotonic()
        key = f"{addr[0]}:{addr[1]}"
        self.rc_senders[key] = self.rc_senders.get(key, 0) + 1
        d = P.decode_datagram(data)
        if isinstance(d, P.Keepalive):
            self.rc_stats.keepalive(t)
            return
        if isinstance(d, P.DecodeError):
            self.rc_stats.error(d.reason)
            return
        self.rc_stats.packet(t)
        release = self.injector.admit(t)
        if release is None:
            return
        delay = release - t
        if delay <= 0.0005:
            self._apply_packet(d)
        else:
            self.loop.call_later(delay, self._apply_packet, d)

    def _apply_packet(self, pkt: P.RcPacket) -> None:
        self.applied_stats.packet(time.monotonic())
        self.dyn.advance_to(self.sim_t())
        before = self.dyn.s.phase
        self.dyn.on_packet(pkt, self.dyn.t)
        if pkt.flags1 or self.dyn.s.phase != before:
            self.event_log.write(json.dumps({"ts_ms": now_ms(), "flags": pkt.flag_names(), "flags1": pkt.flags1,
                                             "phase": self.dyn.s.phase, "sticks": pkt.sticks()}))

    # ------------------------------------------------------------------ rendering
    def snapshot(self) -> tuple[int, float, dict, Truth]:
        """Freeze the state for one frame: pose message for the renderer and its ground truth."""
        self.dyn.advance_to(self.sim_t())
        self.render_id += 1
        rid, rms = self.render_id, now_ms()
        t_scn = self.clock.now()
        cam = self.dyn.camera_pose()
        actors = self.scenario.actor_states(t_scn)
        drone = self.dyn.truth_dict()
        truth = Truth(render_id=rid, render_ms=rms, drone=drone, actors=self.scenario.truth_actors(t_scn, cam))
        msg = {"type": "pose", "render_id": rid, "render_ms": rms, "t": round(t_scn, 4), "camera": cam,
               "drone": {k: drone[k] for k in ("x", "y", "z", "yaw", "pitch", "roll", "motors_on", "cam_tilt")},
               "phase": self.dyn.s.phase, "actors": actors}
        return rid, rms, msg, truth

    def scene_msg(self, role: str) -> str:
        return json.dumps({"type": "scene", "role": role, "version": self.scene_version,
                           "camera": {"width": FRAME_W, "height": FRAME_H,
                                      "hfov_deg": float(self.params["camera"]["hfov_deg"])},
                           **self.scenario.scene_json()})

    async def _ensure_scene(self, c: RenderClient) -> None:
        if c.scene_version != self.scene_version:
            await c.ws.send_text(self.scene_msg(c.role))
            c.scene_version = self.scene_version

    async def _await_frame(self, c: RenderClient, rid: int, timeout: float) -> Optional[bytes]:
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            try:
                m = await asyncio.wait_for(c.ws.receive(), timeout=left)
            except asyncio.TimeoutError:
                return None
            if m.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()
            data = m.get("bytes")
            if data:
                got = int.from_bytes(data[:4], "little")
                if got == rid:
                    return data[4:]
                self.stale_frames += 1
                continue
            text = m.get("text")
            if text:
                self._on_renderer_text(c, text)

    def _on_renderer_text(self, c: RenderClient, text: str) -> None:
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            return
        if msg.get("type") in ("hello", "stats"):
            c.info.update(msg)
            if msg.get("type") == "hello":
                log.info("renderer %s: %s", c.role, msg.get("webgl"))

    async def camera_loop(self, c: RenderClient) -> None:
        if self.camera_client is not None:
            await c.ws.send_text(json.dumps({"type": "error", "error": "a camera renderer is already connected"}))
            await c.ws.close()
            return
        self.camera_client = c
        try:
            next_t = time.monotonic()
            while True:
                await self._ensure_scene(c)
                now = time.monotonic()
                if now < next_t:
                    await asyncio.sleep(next_t - now)
                next_t = max(next_t + 1.0 / self.fps, time.monotonic() - 0.5 / self.fps)
                rid, rms, msg, truth = self.snapshot()
                await c.ws.send_text(json.dumps(msg))
                jpeg = await self._await_frame(c, rid, timeout=2.0)
                if jpeg is None:
                    self.render_timeouts += 1
                    continue
                c.frames += 1
                self._on_camera_frame(rid, rms, truth, jpeg=jpeg)
        finally:
            self.camera_client = None

    def _on_camera_frame(self, rid: int, rms: float, truth: Truth, jpeg: Optional[bytes] = None,
                         bgr: Optional[np.ndarray] = None) -> None:
        self.last_camera_frame_ms = now_ms() if jpeg is not None else self.last_camera_frame_ms
        self.frame_times.append(now_ms())
        self.latest_truth = truth
        line = truth.model_dump_json()
        self.truth_log.write(line)
        for q in list(self.truth_subs):
            if q.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            q.put_nowait(line)
        if self.video is not None:
            self.video.submit(rid, rms, jpeg=jpeg, bgr=bgr)

    async def passive_loop(self, c: RenderClient) -> None:
        """Chase viewers in other browsers (e.g. the operator UI): poses at about 30 Hz, no frames back."""
        async def drain():
            while True:
                m = await c.ws.receive()
                if m.get("type") == "websocket.disconnect":
                    return
                if m.get("text"):
                    self._on_renderer_text(c, m["text"])
        drain_task = asyncio.create_task(drain())
        try:
            while not drain_task.done():
                await self._ensure_scene(c)
                msg = self._passive_pose()
                await c.ws.send_text(json.dumps(msg))
                await asyncio.sleep(1 / 30)
        finally:
            drain_task.cancel()

    def _passive_pose(self) -> dict:
        self.dyn.advance_to(self.sim_t())
        t_scn = self.clock.now()
        drone = self.dyn.truth_dict()
        return {"type": "pose", "render_id": self.render_id, "render_ms": now_ms(), "t": round(t_scn, 4),
                "camera": self.dyn.camera_pose(),
                "drone": {k: drone[k] for k in ("x", "y", "z", "yaw", "pitch", "roll", "motors_on", "cam_tilt")},
                "phase": self.dyn.s.phase, "actors": self.scenario.actor_states(t_scn)}

    async def chase_rec_loop(self, c: RenderClient) -> None:
        """Chase view frames pulled one at a time; written to the chase MP4 while recording."""
        seq = 0
        next_t = time.monotonic()
        while True:
            await self._ensure_scene(c)
            now = time.monotonic()
            if now < next_t:
                await asyncio.sleep(next_t - now)
            next_t = max(next_t + 1.0 / 25.0, time.monotonic() - 0.02)
            seq += 1
            msg = self._passive_pose()
            msg["render_id"] = seq
            await c.ws.send_text(json.dumps(msg))
            jpeg = await self._await_frame(c, seq, timeout=2.0)
            if jpeg is None:
                continue
            c.frames += 1
            if self.chase_recorder is not None:
                self.chase_recorder.submit_jpeg(jpeg)

    async def _placeholder_loop(self) -> None:
        """Without a camera renderer the stream keeps flowing: flat-shaded truth boxes at 10 fps."""
        while True:
            await asyncio.sleep(0.1)
            if self.video is None or self.args.no_placeholder:
                continue
            if self.camera_client is not None and now_ms() - self.last_camera_frame_ms < 1500:
                continue
            rid, rms, _msg, truth = self.snapshot()
            img = placeholder_frame(truth, self.scenario, self.dyn.camera_pose())
            self.placeholder_frames += 1
            self._on_camera_frame(rid, rms, truth, bgr=img)

    # ------------------------------------------------------------------ control and status
    def apply_control(self, c: TwinControlExt) -> dict:
        changed = []
        if c.scenario is not None:
            if c.scenario not in list_scenarios() and not Path(c.scenario).exists():
                raise HTTPException(status_code=404, detail=f"unknown scenario {c.scenario}")
            self.scenario = load_scenario(c.scenario)
            self.scene_version += 1
            self._apply_scenario_start(c.drone_pose)
            changed.append("scenario")
        elif c.reset:
            self._apply_scenario_start(c.drone_pose)
            changed.append("reset")
        if c.latency_ms is not None and self.video is not None:
            self.video.set_target(c.latency_ms)
            changed.append("latency_ms")
        if c.drift is not None:
            self.dyn.drift_scale = max(0.0, float(c.drift))
            changed.append("drift")
        if any(v is not None for v in (c.rc_loss, c.rc_burst_rate, c.rc_burst_len, c.rc_jitter_ms)):
            self.injector.configure(loss=c.rc_loss, burst_rate=c.rc_burst_rate, burst_len=c.rc_burst_len,
                                    jitter_ms=c.rc_jitter_ms)
            changed.append("rc_faults")
        if c.stall is not None and self.video is not None:
            self.video.set_stall(c.stall)
            changed.append("stall")
        if c.headless is not None:
            self.dyn.s.headless = bool(c.headless)
            self.dyn.s.headless_ref_yaw = self.dyn.s.yaw
            changed.append("headless")
        if c.link_loss_s is not None:
            self.params["link"]["loss_timeout_s"] = float(c.link_loss_s)
            changed.append("link_loss_s")
        if c.link_loss_action is not None:
            self.params["link"]["loss_action"] = c.link_loss_action
            changed.append("link_loss_action")
        if c.battery_s is not None:
            self.params["battery"]["flight_time_s"] = float(c.battery_s)
            changed.append("battery_s")
        if c.fps is not None:
            self.fps = max(1.0, min(60.0, float(c.fps)))
            changed.append("fps")
        if (c.noise is not None or c.blur is not None) and self.video is not None:
            self.video.set_noise(c.noise, c.blur)
            changed.append("noise")
        if c.hfov_deg is not None:
            self.params["camera"]["hfov_deg"] = float(c.hfov_deg)
            self.scene_version += 1
            changed.append("hfov_deg")
        if c.cam_tilt_deg is not None:
            cam = self.params["camera"]
            self.dyn.s.cam_tilt = math.radians(max(cam["tilt_min_deg"], min(cam["tilt_max_deg"], c.cam_tilt_deg)))
            changed.append("cam_tilt_deg")
        if c.pause_actors is not None:
            self.clock.pause(bool(c.pause_actors))
            changed.append("pause_actors")
        if c.record_chase is not None:
            if c.record_chase and self.chase_recorder is None:
                out = Path(self.args.media_dir) / f"chase-{self.scenario.name}-{time.strftime('%H%M%S')}.mp4"
                out.parent.mkdir(parents=True, exist_ok=True)
                self.chase_recorder = FileRecorder(out, 1280, 720, 25)
                self.chase_rec_path = out
            elif not c.record_chase and self.chase_recorder is not None:
                rec, self.chase_recorder = self.chase_recorder, None
                threading.Thread(target=rec.close, daemon=True).start()
            changed.append("record_chase")
        self.event_log.write(json.dumps({"ts_ms": now_ms(), "control": c.model_dump(exclude_none=True)}))
        return {"ok": True, "changed": changed, "status": self.status()}

    def status(self) -> dict:
        times = [t for t in self.frame_times if now_ms() - t <= 2000]
        fps = (len(times) - 1) / max(1e-3, (times[-1] - times[0]) / 1000) if len(times) > 1 else 0.0
        cam = self.camera_client
        rc = self.rc_stats.summary()
        rc_age = None if not self.rc_stats.arrivals else round(time.monotonic() - self.rc_stats.arrivals[-1], 3)
        out = {
            "ts_ms": now_ms(), "uptime_s": round(self.sim_t(), 2),
            "scenario": self.scenario.name, "scenario_t": round(self.clock.now(), 2),
            "fps": round(fps, 2), "target_fps": self.fps, "render_id": self.render_id,
            "renderer": {"camera_connected": cam is not None, "camera_frames": cam.frames if cam else 0,
                         "webgl": cam.info.get("webgl") if cam else None, "timeouts": self.render_timeouts,
                         "stale_frames": self.stale_frames, "placeholder_frames": self.placeholder_frames,
                         "clients": sorted(r.role for r in self.renderers)},
            "rc": {**rc, "last_packet_age_s": rc_age, "senders": dict(self.rc_senders),
                   "applied": self.applied_stats.summary(),
                   "faults": {"loss": self.injector.loss, "burst_rate": self.injector.burst_rate,
                              "burst_len": self.injector.burst_len, "jitter_ms": self.injector.jitter_ms,
                              "dropped": self.injector.dropped}},
            "link": {"lost": self.dyn.s.link_lost, "timeout_s": self.params["link"]["loss_timeout_s"],
                     "action": self.params["link"]["loss_action"]},
            "drone": self.dyn.status(),
            "video": self.video.status() if self.video else None,
            "rtsp_url": self.rtsp_url if self.video else None,
            "chase_recording": str(self.chase_rec_path) if self.chase_recorder else None,
            "log_dir": str(self.log_dir),
        }
        if self.video:
            v = out["video"]
            out["latency"] = {"target_total_ms": v["target_total_ms"], "baseline_ms_p50": v["baseline_ms_p50"],
                              "total_ms_p50": v["total_ms_p50"], "total_ms_p95": v["total_ms_p95"]}
        return out


# ---------------------------------------------------------------------- placeholder renderer
def _hex_bgr(h: str, default=(128, 128, 128)) -> tuple[int, int, int]:
    try:
        h = h.lstrip("#")
        return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))
    except Exception:
        return default


def placeholder_frame(truth: Truth, scenario: Scenario, cam: dict) -> np.ndarray:
    """Flat 2D stand-in used only when no WebGL renderer is connected (clearly labeled)."""
    img = np.empty((FRAME_H, FRAME_W, 3), np.uint8)
    f = (FRAME_W / 2) / math.tan(math.radians(cam["hfov_deg"]) / 2)
    horizon = int(max(0, min(FRAME_H, FRAME_H / 2 + f * math.tan(cam["pitch"]))))
    img[:horizon] = (235, 206, 160)
    img[horizon:] = (60, 130, 70)
    colors = {a.id: a.colors for a in scenario.actors}
    for ta in sorted(truth.actors, key=lambda a: -a.distance_m):
        if ta.box is None:
            continue
        c = colors.get(ta.actor_id, {})
        bgr = _hex_bgr(c.get("shirt") or c.get("paint") or "#888888")
        b = ta.box
        cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), bgr, -1)
    cv2.putText(img, "NO RENDERER (placeholder)", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    return img


# ---------------------------------------------------------------------- app
def build_app(args: argparse.Namespace) -> FastAPI:
    twin = Twin(args)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        await twin.start()
        yield
        await twin.stop()

    app = FastAPI(title="X69 digital twin", lifespan=lifespan)
    app.state.twin = twin

    @app.middleware("http")
    async def no_cache_for_pages(request, call_next):
        # Chrome must never run a stale renderer after the JS changes (three.js itself may be cached).
        response = await call_next(request)
        if not request.url.path.startswith("/vendor/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204)

    @app.get("/chase")
    async def chase():
        return FileResponse(WEB_DIR / "chase.html")

    @app.get("/status")
    async def status():
        return JSONResponse(twin.status())

    @app.get("/scene")
    async def scene():
        return JSONResponse(json.loads(twin.scene_msg("api")))

    @app.get("/scenarios")
    async def scenarios():
        return {"scenarios": list_scenarios(), "current": twin.scenario.name}

    @app.post("/control")
    async def control(c: TwinControlExt):
        return JSONResponse(twin.apply_control(c))

    @app.get("/truth/latest")
    async def truth_latest():
        if twin.latest_truth is None:
            raise HTTPException(status_code=404, detail="no frame rendered yet")
        return JSONResponse(json.loads(twin.latest_truth.model_dump_json()))

    @app.get("/frame.jpg")
    async def frame_jpg(source: str = "stamped"):
        img = None
        if twin.video is not None:
            img = twin.video.latest_probe_bgr if source == "rtsp" else twin.video.latest_bgr
        if img is None:
            raise HTTPException(status_code=404, detail="no frame yet")
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    @app.get("/frames/log")
    async def frames_log(n: int = 200):
        if twin.video is None:
            return []
        return list(twin.video.frame_log)[-n:]

    @app.websocket("/ws/truth")
    async def ws_truth(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        twin.truth_subs.add(q)
        try:
            while True:
                line = await q.get()
                await ws.send_text(line)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            twin.truth_subs.discard(q)

    @app.websocket("/ws/render")
    async def ws_render(ws: WebSocket):
        await ws.accept()
        role = ws.query_params.get("role", "camera")
        c = RenderClient(ws, role)
        twin.renderers.add(c)
        try:
            if role == "camera":
                await twin.camera_loop(c)
            elif role == "chase_rec":
                await twin.chase_rec_loop(c)
            else:
                await twin.passive_loop(c)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            twin.renderers.discard(c)

    if (WEB_DIR / "js").exists():
        app.mount("/js", StaticFiles(directory=WEB_DIR / "js"), name="js")
    three_dir = WEB_DIR / "node_modules" / "three"
    if three_dir.exists():
        app.mount("/vendor/three", StaticFiles(directory=three_dir), name="three")
    return app


def parse_args(argv=None) -> argparse.Namespace:
    home = os.environ.get("MHD_HOME", str(TREE))
    ap = argparse.ArgumentParser(description="X69 wire-level digital twin")
    ap.add_argument("--http-port", type=int, default=8766)
    ap.add_argument("--rc-port", type=int, default=23458, help="UDP port for X69 RC datagrams")
    ap.add_argument("--rtsp-port", type=int, default=8554)
    ap.add_argument("--scenario", default="s1")
    ap.add_argument("--params", default=None, help="dynamics YAML (default config/twin/x69.yaml)")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--latency-ms", type=float, default=None, help="target TOTAL camera-to-decode latency")
    ap.add_argument("--hfov-deg", type=float, default=None)
    ap.add_argument("--bitrate", default="3M")
    ap.add_argument("--gop", type=int, default=25)
    ap.add_argument("--noise", type=float, default=3.0)
    ap.add_argument("--blur", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-video", action="store_true", help="RC and dynamics only (no mediamtx/ffmpeg)")
    ap.add_argument("--no-probe", action="store_true", help="do not read back the RTSP stream")
    ap.add_argument("--no-placeholder", action="store_true")
    ap.add_argument("--external-mediamtx", action="store_true", help="mediamtx is started by someone else")
    ap.add_argument("--log-dir", default=os.environ.get("MHD_LOGS", str(TREE / "logs")))
    ap.add_argument("--media-dir", default=str(Path(home) / "media" / "shared" / "twin"))
    ap.add_argument("--log-level", default="info")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if os.environ.get("DRONE_IP") and os.environ["DRONE_IP"] not in ("127.0.0.1", "localhost"):
        raise SystemExit("the twin only runs on loopback; unset DRONE_IP or set it to 127.0.0.1")
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = build_app(args)
    uvicorn.run(app, host=LOOPBACK, port=args.http_port, log_level="warning", ws_max_size=16 * 1024 * 1024)


if __name__ == "__main__":
    main()
