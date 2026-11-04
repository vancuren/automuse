"""Record the end-to-end demo in the wire-level twin, with the fine-tuned Glimmer answering live.

Starts the whole stack on the default ports, opens the operator UI in Chrome (Playwright, headless,
GPU), and records it while a scripted operator flies one mission:

  1. the dashboard in TWIN mode            5. the subject walking, the drone following
  2. TAKEOFF clicked in the UI             6. "okay drone, please land now" heard by the local
  3. Space held: "drone, follow the           stop/land listener (open mic, no button, no model)
     person in the blue shirt" (WAV made   7. LANDING, then Confirm touchdown
     with `say` + `afconvert`, fed into the agent's simulated microphone): transcript, Glimmer
     intent, mission card with the model badge glimmer-30b+lora
  4. FIND, CONFIRM, FOLLOW with the SAM 3.1 mask overlay for --follow-s seconds

    source env.sh
    $MHD_PY scripts/demo_twin.py                                  # S1, "person in blue shirt"
    $MHD_PY scripts/demo_twin.py --demo-scenario s2               # S2: the red-shirt walker crosses
    $MHD_PY scripts/demo_twin.py --stills                         # also write the README stills
    $MHD_PY scripts/demo_twin.py --keep-glimmer                   # leave Glimmer up for the next take

Stack (all loopback, FLIGHT_MODE=twin):
  twin        scripts/run_twin.sh --scenario <s> --latency-ms 500 --chase-rec        (8766, UDP 23458, RTSP 8554)
  backend     scripts/run_backend.sh --mode twin + config/twin/{governor,autopilot}.yaml, NOT headless:
              the UI's Web Worker heartbeat keeps it flying                          (8000)
  perception  mhd_ml.perception.service --detector sam_mlx --tracker auto (SAM 3.1 MLX + VitTrack + Kalman)
  Glimmer     scripts/serve_glimmer.sh: mlx_lm.server + LoRA glimmer-intent-v1        (8080)
  agent       scripts/demo_agent.py = mhd_ml.agent.service.AgentService, config/demo/agent.yaml, simulated mic
  UI          production build (VITE_BACKEND_URL=http://localhost:8000), vite preview  (5173)

Outputs: logs/demo-<stamp>-<take>/ (every log, telemetry, agent events, demo_result.json, metrics
report.json + rows.csv), media/demo/demo-<take>-<stamp>-ui.mp4 (1920x1080 H.264, with a sound track
of the spoken commands and the agent's MMS-TTS replies) and -chase.mp4 (the twin's chase view,
1280x720 H.264), both copied to $MHD_HOME/media/demo/. With --stills, PNG stills go to docs/img/.
Everything it starts runs in its own process group and is stopped at the end (also on Ctrl-C).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
import wave
from pathlib import Path

import httpx
import websockets

TREE = Path(os.environ["MHD_TREE"])
HOME = Path(os.environ["MHD_HOME"])
PY, ML_PY = os.environ["MHD_PY"], os.environ["MHD_ML_PY"]
BACKEND, TWIN, RC, RTSP_DEFAULT, UI_PORT, GLIMMER = 8000, 8766, 23458, 8554, 5173, 8080
FRONTEND = TREE / "turbodrone" / "frontend"
UI_URL = f"http://127.0.0.1:{UI_PORT}/"
CHROME_ARGS = ["--use-angle=metal", "--ignore-gpu-blocklist", "--disable-background-timer-throttling",
               "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows",
               "--no-first-run", "--no-default-browser-check", "--use-mock-keychain", "--password-store=basic",
               "--disable-background-networking", "--disable-component-update", "--disable-sync",
               "--autoplay-policy=no-user-gesture-required"]

PRESETS = {
    "s1": dict(scenario="s1", subject="person in blue shirt", category="person", color="blue",
               command="Drone, follow the person in the blue shirt.",
               readme={"1_dashboard_twin": "1_dashboard_twin", "2_voice_mission_follow": "2_voice_mission_follow",
                       "5_landing_voice": "5_landing_voice", "chase_follow": "chase_follow"}),
    # S2: the walkers cross at scenario t = 11, 32, 53 s (both 0.3 m/s on 6 m legs, 1 s pauses).
    "s2": dict(scenario="s2", subject="person in blue shirt", category="person", color="blue",
               command="Drone, follow the person in the blue shirt.", still_scn_s="32,53",
               readme={"scn_53s": "crossing_follow"}),
    "s3": dict(scenario="s3", subject="green car", category="vehicle", color="green",
               command="Drone, follow the green car.",
               readme={"2_voice_mission_follow": "2_voice_mission_follow"}),
}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Proc:
    def __init__(self, name: str, cmd: list[str], logf: Path, env: dict | None = None, cwd: Path = TREE,
                 stdin=None, stdout: Path | None = None):
        self.name, self.log = name, logf
        e = dict(os.environ)
        e.update(env or {})
        self.f = open(logf, "w")
        self.fo = open(stdout, "w") if stdout else None
        self.p = subprocess.Popen(cmd, stdout=self.fo or self.f, stderr=self.f if self.fo else subprocess.STDOUT,
                                  stdin=stdin, env=e, cwd=cwd, start_new_session=True)

    def alive(self) -> bool:
        return self.p.poll() is None

    def stop(self, grace: float = 6.0) -> None:
        if self.alive():
            for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 4), (signal.SIGKILL, 2)):
                try:
                    os.killpg(self.p.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.p.wait(wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        self.sweep()

    def sweep(self) -> None:
        """Stop anything still left in the process group after its leader exited: the twin server's
        watchdog used to restart mediamtx during shutdown, and that copy outlived the twin (orphaned,
        holding the RTSP port). The group id is the leader's pid (start_new_session)."""
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.p.pid, sig)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(0.5)


def tcp_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def udp_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def wait_for(pred, timeout: float, what: str, interval: float = 0.5, procs=()):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        for p in procs:
            if not p.alive():
                raise RuntimeError(f"{p.name} exited while waiting for {what}; see {p.log}")
        try:
            v = pred()
            if v:
                return v
        except Exception:
            pass
        time.sleep(interval)
    raise TimeoutError(f"timed out after {timeout:.0f} s waiting for {what}")


def state() -> dict:
    return httpx.get(f"http://127.0.0.1:{BACKEND}/state", timeout=3).json()


def twin_status() -> dict:
    return httpx.get(f"http://127.0.0.1:{TWIN}/status", timeout=3).json()


def twin_control(**kw) -> dict:
    return httpx.post(f"http://127.0.0.1:{TWIN}/control", json=kw, timeout=10).json()


def glimmer_ready() -> bool:
    r = httpx.get(f"http://127.0.0.1:{GLIMMER}/v1/models", timeout=2)
    return r.status_code == 200 and "Glimmer" in r.text


def glimmer_warm(text: str) -> dict:
    sys.path.insert(0, str(TREE / "src"))
    from mhd.intent_spec import GLIMMER_SYSTEM_PROMPT
    t0 = time.monotonic()
    r = httpx.post(f"http://127.0.0.1:{GLIMMER}/v1/chat/completions", timeout=180, json={
        "model": "default_model", "temperature": 0, "max_tokens": 300,
        "messages": [{"role": "system", "content": GLIMMER_SYSTEM_PROMPT}, {"role": "user", "content": text}]})
    content = r.json()["choices"][0]["message"].get("content") or ""
    return {"s": round(time.monotonic() - t0, 2), "answer": content.rsplit("<|message|>", 1)[-1][:300]}


def make_wav(text: str, voice: str, dest: Path, rate: int = 24000) -> float:
    """`say` -> AIFF -> `afconvert` -> 16-bit mono WAV. Returns the duration in seconds."""
    aiff = dest.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True, timeout=30)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{rate}", "-c", "1", str(aiff), str(dest)],
                   check=True, timeout=30)
    aiff.unlink(missing_ok=True)
    with wave.open(str(dest)) as w:
        return w.getnframes() / float(w.getframerate())


async def record_telemetry(path: Path, stop: asyncio.Event) -> None:
    with open(path, "w") as f:
        while not stop.is_set():
            try:
                async with websockets.connect(f"ws://127.0.0.1:{BACKEND}/ws/telemetry", max_size=None) as ws:
                    while not stop.is_set():
                        try:
                            msg = await asyncio.wait_for(ws.recv(), 1.0)
                        except asyncio.TimeoutError:
                            continue
                        f.write(msg if isinstance(msg, str) else msg.decode())
                        f.write("\n")
            except Exception:
                await asyncio.sleep(0.5)


class Screencast:
    """Chrome's CDP screencast to numbered JPEG files (frames arrive only on repaint)."""

    def __init__(self, cdp, out_dir: Path):
        self.cdp, self.dir = cdp, out_dir
        self.frames: list[tuple[float, str]] = []      # (CDP timestamp s, file)
        self.dir.mkdir(parents=True, exist_ok=True)

    def on_frame(self, ev: dict) -> None:
        p = self.dir / f"f{len(self.frames):06d}.jpg"
        p.write_bytes(base64.b64decode(ev["data"]))
        self.frames.append((float(ev["metadata"]["timestamp"]), str(p)))
        asyncio.ensure_future(self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]}))

    async def start(self) -> None:
        self.cdp.on("Page.screencastFrame", self.on_frame)
        await self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 88,
                                                      "maxWidth": 1920, "maxHeight": 1080})

    async def stop(self) -> None:
        try:
            await self.cdp.send("Page.stopScreencast")
        except Exception:
            pass


def encode_screencast(frames: list[tuple[float, str]], out: Path, fps: int = 25, audio: Path | None = None) -> dict:
    """Resample the repaint-driven frames to a constant fps (each frame held until the next one) and
    encode H.264 1080p with the system ffmpeg (libx264), optionally muxing a sound track."""
    if not frames:
        raise RuntimeError("no screencast frames")
    lst = out.with_suffix(".frames.txt")
    t0 = frames[0][0]
    with open(lst, "w") as f:
        for i, (t, p) in enumerate(frames):
            d = (frames[i + 1][0] - t) if i + 1 < len(frames) else 1.0 / fps
            f.write(f"file '{p}'\nduration {max(d, 0.001):.4f}\n")
        f.write(f"file '{frames[-1][1]}'\n")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if audio:
        cmd += ["-i", str(audio)]
    cmd += ["-vf", f"fps={fps},scale=1920:1080:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-preset", "medium",
            "-crf", "20", "-profile:v", "high", "-movflags", "+faststart"]
    if audio:
        cmd += ["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "160k"]
    cmd += [str(out)]
    subprocess.run(cmd, check=True, timeout=900)
    lst.unlink(missing_ok=True)
    return {"duration_s": round(frames[-1][0] - t0, 2), "screencast_frames": len(frames)}


def build_soundtrack(events: list[tuple[float, Path]], duration_s: float, out: Path) -> Path | None:
    """Mix WAVs placed at the given offsets (s) into one track of duration_s (48 kHz stereo WAV)."""
    events = [(t, p) for t, p in events if p.exists() and 0 <= t < duration_s]
    if not events:
        return None
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-t", f"{duration_s:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
    for _, p in events:
        cmd += ["-i", str(p)]
    parts, labels = [], ["[0:a]"]
    for i, (t, _) in enumerate(events, start=1):
        ms = int(round(t * 1000))
        parts.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,adelay={ms}|{ms}[a{i}]")
        labels.append(f"[a{i}]")
    parts.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:normalize=0,"
                 f"atrim=0:{duration_s:.3f}[out]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[out]", "-c:a", "pcm_s16le", str(out)]
    subprocess.run(cmd, check=True, timeout=300)
    return out


def transcode_chase(src: Path, out: Path, audio: Path | None = None) -> dict:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
    if audio:
        cmd += ["-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "160k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out)]
    subprocess.run(cmd, check=True, timeout=900)
    return probe(out)


def probe(p: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,width,height,"
                        "avg_frame_rate,nb_frames", "-of", "json", str(p)], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
    except Exception:
        return {}
    v = [s for s in d.get("streams", []) if s.get("width")]
    a = [s for s in d.get("streams", []) if not s.get("width")]
    out = {"duration_s": round(float(d.get("format", {}).get("duration", 0)), 2), "bytes": p.stat().st_size}
    if v:
        out.update(codec=v[0].get("codec_name"), width=v[0].get("width"), height=v[0].get("height"),
                   fps=v[0].get("avg_frame_rate"), frames=v[0].get("nb_frames"))
    if a:
        out["audio"] = a[0].get("codec_name")
    return out


def save_still(png_bytes: bytes, dest: Path, width: int = 1280, max_bytes: int = 400_000) -> dict:
    """Downscale a screenshot and keep it under max_bytes (PNG; palette-quantized only if needed)."""
    import io

    from PIL import Image

    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, optimize=True)
    mode = "rgb"
    if dest.stat().st_size > max_bytes:
        im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG).save(
            dest, optimize=True)
        mode = "palette256"
    if dest.stat().st_size > max_bytes:
        w2 = int(width * 0.85)
        im.resize((w2, round(im.height * w2 / im.width)), Image.LANCZOS).quantize(
            colors=192, method=Image.Quantize.MEDIANCUT).save(dest, optimize=True)
        mode = "palette192-smaller"
    return {"path": str(dest), "bytes": dest.stat().st_size, "mode": mode}


class Demo:
    def __init__(self, args):
        self.args = args
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.stamp = stamp
        self.take = args.take or args.preset
        self.out = TREE / "logs" / f"demo-{stamp}-{self.take}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.media = TREE / "media" / "demo"
        self.media.mkdir(parents=True, exist_ok=True)
        self.procs: list[Proc] = []
        self.agent: Proc | None = None
        self.glimmer: Proc | None = None
        self.result: dict = {"run_dir": str(self.out), "take": self.take, "scenario": args.scenario,
                             "subject": args.subject, "command": args.command, "land_phrase": args.land_text,
                             "latency_ms": args.latency_ms, "follow_s": args.follow_s, "marks": {},
                             "stills": {}, "fallbacks": []}
        self.t_mark0 = None
        self.wall_minus_mono = time.time() - time.monotonic()
        self.stills_raw: dict[str, bytes] = {}
        self.cast_frames: list[tuple[float, str]] = []
        self.t_chase0 = self.t_chase1 = None

    # ---- helpers -------------------------------------------------------------------------------
    def mark(self, name: str, **extra) -> None:
        t = time.monotonic()
        rec = {"mono_s": round(t, 3), **extra}
        if self.t_mark0 is not None:
            rec["video_s"] = round(t - self.t_mark0, 2)
        self.result["marks"][name] = rec
        log(f"mark {name} {rec.get('video_s', '')} {extra if extra else ''}")

    def agent_send(self, line: str) -> None:
        self.agent.p.stdin.write((line + "\n").encode())
        self.agent.p.stdin.flush()

    def agent_events(self) -> list[dict]:
        out = []
        p = self.out / "agent_events.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
        return out

    async def still(self, page, name: str) -> None:
        try:
            self.stills_raw[name] = await page.screenshot(type="png")
            self.mark(f"still_{name}")
        except Exception as e:
            log(f"still {name} failed: {e}")

    async def wait_state(self, pred, timeout: float, what: str, interval: float = 0.25):
        return await asyncio.to_thread(wait_for, pred, timeout, what, interval, self.procs)

    # ---- stack ---------------------------------------------------------------------------------
    def preflight(self) -> None:
        if os.environ.get("FLIGHT_MODE") == "real":
            raise SystemExit("demo_twin.py is twin-only")
        a = self.args
        busy = [p for p in (BACKEND, TWIN, UI_PORT) if not tcp_free(p)]
        if not udp_free(RC):
            busy.append(f"udp/{RC}")
        if busy:
            raise SystemExit(f"ports in use: {busy}; stop the other stack first")
        self.rtsp = a.rtsp_port or RTSP_DEFAULT
        if not tcp_free(self.rtsp):
            if a.rtsp_port:
                raise SystemExit(f"RTSP port {self.rtsp} is in use")
            alt = 8555
            log(f"WARNING: RTSP port {self.rtsp} is in use by another process; using {alt} for this run")
            self.result["fallbacks"].append(f"rtsp port {self.rtsp} busy (another process); used {alt}")
            self.rtsp = alt
            if not tcp_free(alt):
                raise SystemExit(f"RTSP ports {RTSP_DEFAULT} and {alt} are both in use")
        self.result["rtsp_port"] = self.rtsp

    def build_ui(self) -> None:
        dist = FRONTEND / "dist" / "index.html"
        if self.args.no_build and dist.exists():
            return
        log("building the UI (production)")
        env = dict(os.environ, VITE_BACKEND_URL=f"http://localhost:{BACKEND}",
                   VITE_TWIN_CHASE_URL=f"http://localhost:{TWIN}/chase")
        r = subprocess.run(["npm", "run", "build"], cwd=FRONTEND, env=env, capture_output=True, text=True,
                           timeout=300)
        (self.out / "ui_build.log").write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            raise RuntimeError(f"UI build failed; see {self.out / 'ui_build.log'}")

    def start_glimmer(self) -> None:
        try:
            if glimmer_ready():
                log("Glimmer already serving on 8080: reusing it")
                self.result["glimmer"] = {"reused": True}
                return
        except Exception:
            pass
        if not tcp_free(GLIMMER):
            raise SystemExit(f"port {GLIMMER} is busy but is not a Glimmer server")
        log("starting Glimmer + LoRA (scripts/serve_glimmer.sh)")
        self.glimmer = Proc("glimmer", ["bash", "scripts/serve_glimmer.sh", "--port", str(GLIMMER)],
                            self.out / "glimmer.log")
        self.result["glimmer"] = {"reused": False}

    def wait_glimmer(self) -> None:
        procs = [self.glimmer] if self.glimmer else []
        t0 = time.monotonic()
        wait_for(glimmer_ready, 300, "Glimmer /v1/models", 1.0, procs)
        warm = glimmer_warm(self.args.command)
        self.result["glimmer"].update(ready_s=round(time.monotonic() - t0, 1), warmup=warm)
        log(f"Glimmer ready; warm-up answer in {warm['s']} s: {warm['answer'][:120]}")

    def start_stack(self) -> None:
        a = self.args
        self.twin = Proc("twin", ["bash", "scripts/run_twin.sh", "--ports", f"{TWIN},{RC},{self.rtsp}",
                                  "--scenario", a.scenario, "--latency-ms", str(a.latency_ms), "--chase-rec",
                                  "--", "--media-dir", str(self.out / "twin_media")], self.out / "twin.log")
        self.procs.append(self.twin)
        st = wait_for(lambda: (lambda s: s if s.get("fps", 0) >= 15 else None)(twin_status()), 120,
                      "twin rendering at 15 fps or more", procs=self.procs)
        self.result["twin_renderer"] = st.get("renderer")
        self.result["twin_log_dir"] = st.get("log_dir")
        log(f"twin up: {st.get('fps')} fps, {st.get('renderer', {}).get('webgl')}")
        self.backend = Proc("backend", ["bash", "scripts/run_backend.sh", "--mode", "twin", "--port", str(BACKEND),
                                        "--control-port", str(RC),
                                        "--rtsp-url", f"rtsp://127.0.0.1:{self.rtsp}/live/ch00_1",
                                        "--logs", str(self.out / "backend"),
                                        "--governor-config", a.governor_config,
                                        "--autopilot-config", a.autopilot_config], self.out / "backend.log")
        self.procs.append(self.backend)
        wait_for(lambda: (lambda s: s if (s.get("frame_age_ms") or 1e9) < 3000 else None)(state()), 60,
                 "backend receiving video", procs=self.procs)
        log("backend up (not headless: the UI's heartbeat keeps it flying)")
        self.perc = Proc("perception", [ML_PY, "-m", "mhd_ml.perception.service", "--backend",
                                        f"ws://127.0.0.1:{BACKEND}", "--detector", "sam_mlx", "--tracker", "auto",
                                        "--hfov-deg", str(a.hfov_deg), "--obs-log", str(self.out / "obs.jsonl"),
                                        "--log-level", "INFO"], self.out / "perception.log")
        self.procs.append(self.perc)
        self.t_perc = time.monotonic()
        self.agent = Proc("agent", [ML_PY, "scripts/demo_agent.py", "--config", a.agent_config, "--backend",
                                    f"ws://127.0.0.1:{BACKEND}", "--out", str(self.out / "agent"), "-v"],
                          self.out / "agent.log", stdin=subprocess.PIPE, stdout=self.out / "agent_events.jsonl")
        self.procs.append(self.agent)
        self.vite = Proc("ui", [str(FRONTEND / "node_modules" / ".bin" / "vite"), "preview", "--port", str(UI_PORT),
                                "--strictPort"], self.out / "ui.log", cwd=FRONTEND)
        self.procs.append(self.vite)
        wait_for(lambda: httpx.get(UI_URL, timeout=2).status_code == 200, 30, "UI on 5173", procs=self.procs)
        wait_for(lambda: any(e.get("type") == "ready" for e in self.agent_events()), 120, "agent ready",
                 1.0, self.procs)
        ready = [e for e in self.agent_events() if e.get("type") == "ready"][0]
        self.result["agent_status"] = ready.get("status")
        log(f"agent ready: {ready.get('status')}")
        left = a.perception_warmup_s - (time.monotonic() - self.t_perc)
        if left > 0:
            log(f"perception warm-up: {left:.0f} s more")
            time.sleep(left)

    def stop_all(self) -> None:
        if self.agent and self.agent.alive():
            try:
                self.agent_send("quit")
                self.agent.p.wait(20)
            except Exception:
                pass
        for p in reversed(self.procs):
            p.stop()
        if self.glimmer and not self.args.keep_glimmer:
            self.glimmer.stop()
        elif self.glimmer:
            self.result["glimmer"]["left_running_pid"] = self.glimmer.p.pid

    # ---- the scripted flight -------------------------------------------------------------------
    async def fly(self) -> None:
        from playwright.async_api import async_playwright

        a = self.args
        audio = self.out / "audio"
        audio.mkdir(exist_ok=True)
        cmd_wav, land_wav = audio / "command.wav", audio / "land.wav"
        self.result["wav"] = {"command": {"file": str(cmd_wav), "s": round(make_wav(a.command, a.voice, cmd_wav), 2)},
                              "land": {"file": str(land_wav), "s": round(make_wav(a.land_text, a.voice, land_wav), 2)}}
        stop = asyncio.Event()
        tele = asyncio.create_task(record_telemetry(self.out / "telemetry.jsonl", stop))
        profile = HOME / ".cache" / "chrome-demo"
        profile.mkdir(parents=True, exist_ok=True)
        frames_dir = self.media / f"tmp-frames-{self.stamp}"
        async with async_playwright() as pw:
            ctx = await pw.chromium.launch_persistent_context(
                str(profile), channel="chrome", headless=not a.headed, args=CHROME_ARGS,
                viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
            page = ctx.pages[0] if ctx.pages else await ctx.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            await page.goto(UI_URL)
            await page.wait_for_selector('[data-testid="takeoff-blocker"] >> text=ready', timeout=60000)
            await page.wait_for_selector('[data-testid="agent-health"] >> text=agent alive', timeout=30000)
            webgl = await page.evaluate("""() => { const c = document.createElement('canvas');
                const gl = c.getContext('webgl2'); const e = gl && gl.getExtension('WEBGL_debug_renderer_info');
                return e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : null; }""")
            self.result["ui_webgl"] = webgl
            # A fresh start for the story: the drone on the ground at the start pose, the subject at
            # the start of its path (it starts walking 1-2 s later), the chase recorder running.
            twin_control(reset=True)
            self.t_reset = time.monotonic()
            await asyncio.sleep(2.5)                     # chase iframe and camera settle after the reset
            cdp = await ctx.new_cdp_session(page)
            cast = Screencast(cdp, frames_dir)
            self.cast_frames = cast.frames                 # the same list: filled while recording
            twin_control(record_chase=True)
            self.t_chase0 = time.monotonic()
            await cast.start()
            self.t_mark0 = time.monotonic()
            self.mark("recording_start")
            try:
                await self.story(page)
            finally:
                await cast.stop()
                self.mark("recording_end")
                try:
                    twin_control(record_chase=False)
                except Exception:
                    pass
                self.t_chase1 = time.monotonic()
                self.result["page_errors"] = errors[:20]
                await ctx.close()
        stop.set()
        try:
            await asyncio.wait_for(tele, 3)
        except Exception:
            pass

    async def story(self, page) -> None:
        a = self.args
        sel = lambda t: f'[data-testid="{t}"]'                       # noqa: E731
        await asyncio.sleep(a.intro_s)                               # 1. the dashboard, TWIN mode, ready
        await self.still(page, "1_dashboard_twin")
        await page.click(sel("takeoff-btn"))                         # 2. TAKEOFF, button only
        self.mark("takeoff_click")
        await self.wait_state(lambda: state()["flight_state"] == "AIRBORNE", 25, "AIRBORNE")
        self.mark("airborne")
        await asyncio.sleep(a.pre_command_s)
        await page.keyboard.down("Space")                            # 3. push-to-talk, held
        self.mark("ptt_down")
        await asyncio.sleep(0.45)
        self.agent_send(f"play {self.result['wav']['command']['file']}")
        await asyncio.sleep(self.result["wav"]["command"]["s"] + 0.35)
        await page.keyboard.up("Space")
        self.mark("ptt_up")
        try:
            await page.wait_for_selector(sel("model-badge"), timeout=15000)
            self.mark("mission_card", badge=(await page.inner_text(sel("model-badge"))).strip())
        except Exception as e:
            self.result["fallbacks"].append(f"no mission card within 15 s of ptt_up: {e}")
            raise
        # The governor holds a mission that may predate the takeoff (it never happens here, but if
        # the ack says paused the operator presses Resume, as in a flight).
        await asyncio.sleep(1.0)
        s = state()
        paused = "autopilot_paused" in (s.get("safety_flags") or [])
        if paused:
            await asyncio.sleep(1.0)
            paused = "autopilot_paused" in (state().get("safety_flags") or [])
        if paused:
            self.result["fallbacks"].append("mission acked paused: pressed Resume")
            await page.click(sel("resume-btn"))
            self.mark("resume_click")
        t_find = time.monotonic()
        try:
            await self.wait_state(lambda: state()["autopilot_state"] == "FOLLOW", a.find_timeout_s, "FOLLOW")
            self.mark("follow", after_mission_card_s=round(time.monotonic() - t_find, 2))
        except TimeoutError:
            self.result["fallbacks"].append("autopilot never reached FOLLOW")
            self.mark("no_follow")
        await asyncio.sleep(1.2)
        await self.still(page, "2_voice_mission_follow")             # 4. FOLLOW with the SAM mask
        states: dict[str, int] = {}
        t_follow0 = time.monotonic()
        took_mid = took_late = False
        scn_stills = sorted(float(x) for x in a.still_scn_s.split(",") if x.strip()) if a.still_scn_s else []
        while time.monotonic() - t_follow0 < a.follow_s:
            s = await asyncio.to_thread(state)
            states[s["autopilot_state"]] = states.get(s["autopilot_state"], 0) + 1
            for p in self.procs:
                if not p.alive():
                    raise RuntimeError(f"{p.name} died during the follow; see {p.log}")
            el = time.monotonic() - t_follow0
            if not took_mid and el > a.follow_s * 0.4:
                took_mid = True
                await self.still(page, "3_follow_walking")           # 5. the subject walking
            if not took_late and el > a.follow_s * 0.8:
                took_late = True
                await self.still(page, "4_follow_late")
            for ts in list(scn_stills):
                if time.monotonic() - self.t_reset >= ts:
                    scn_stills.remove(ts)
                    await self.still(page, f"scn_{ts:g}s")
            await asyncio.sleep(0.5)
        self.result["autopilot_state_samples"] = states
        self.agent_send(f"play {self.result['wav']['land']['file']}")  # 6. "land", open mic
        self.mark("land_spoken")
        try:
            await self.wait_state(lambda: state()["flight_state"] == "LANDING",
                                  self.result["wav"]["land"]["s"] + 6, "LANDING after the spoken land")
            self.mark("landing", via="voice")
        except TimeoutError:
            self.result["fallbacks"].append("spoken land not acted on in time: pressed LAND")
            await page.keyboard.press("KeyL")
            await self.wait_state(lambda: state()["flight_state"] == "LANDING", 5, "LANDING")
            self.mark("landing", via="L key")
        await asyncio.sleep(1.0)
        await self.still(page, "5_landing_voice")                    # 7. LANDING ...
        await self.wait_state(lambda: twin_status()["drone"]["phase"] == "landed", 25, "twin touchdown")
        self.mark("touchdown_twin")
        await asyncio.sleep(1.5)
        await page.click(sel("touchdown-btn"))                       # ... then Confirm touchdown
        self.mark("confirm_touchdown_click")
        await self.wait_state(lambda: state()["flight_state"] == "ON_GROUND", 8, "ON_GROUND")
        self.mark("on_ground")
        await asyncio.sleep(a.outro_s)
        await self.still(page, "6_on_ground")
        # End after the agent's last spoken line, so the sound track never stops mid-sentence.
        t_end = time.monotonic() + 5.0
        while time.monotonic() < t_end:
            tts = [e for e in self.agent_events() if e.get("type") == "tts"]
            if not tts or tts[-1]["t_ms"] / 1000.0 + tts[-1]["duration_s"] + 0.4 < time.monotonic():
                break
            await asyncio.sleep(0.2)

    # ---- after the run -------------------------------------------------------------------------
    def soundtrack(self, t_ref: float, duration_s: float, out: Path) -> Path | None:
        """Operator WAVs at the moments they were fed to the mic, agent TTS lines when they started."""
        ev: list[tuple[float, Path]] = []
        for e in self.agent_events():
            if e.get("type") == "mic" and e.get("event") == "start":
                ev.append((e["t_ms"] / 1000.0 - t_ref, Path(e["file"])))
            elif e.get("type") == "tts":
                ev.append((e["t_ms"] / 1000.0 - t_ref, Path(e["file"])))
        try:
            return build_soundtrack(ev, duration_s, out)
        except Exception as ex:
            log(f"sound track failed: {ex}")
            return None

    def finish_media(self) -> None:
        name = f"demo-{self.take}-{self.stamp}"
        ui_mp4 = self.media / f"{name}-ui.mp4"
        frames = self.cast_frames
        if not frames:
            self.result["ui_video"] = {"error": "nothing was recorded"}
            return
        # CDP timestamps are wall-clock seconds; map them onto the monotonic clock of the marks.
        t_video0_mono = frames[0][0] - self.wall_minus_mono if frames else self.t_mark0
        self.result["video_t0_minus_mark0_s"] = round(t_video0_mono - self.t_mark0, 3) if frames else None
        dur = frames[-1][0] - frames[0][0] if frames else 0
        track = self.soundtrack(t_video0_mono, dur, self.out / "soundtrack_ui.wav") if not self.args.no_audio else None
        log(f"encoding the UI video ({len(frames)} screencast frames, {dur:.1f} s)")
        info = encode_screencast(frames, ui_mp4, audio=track)
        self.result["ui_video"] = {"path": str(ui_mp4), **info, **probe(ui_mp4)}
        if frames and not self.args.keep_frames:
            shutil.rmtree(Path(frames[0][1]).parent, ignore_errors=True)
        raw = sorted((self.out / "twin_media").glob("chase-*.mp4"))
        if raw:
            chase_mp4 = self.media / f"{name}-chase.mp4"
            wall = (self.t_chase1 or time.monotonic()) - self.t_chase0
            ctrack = self.soundtrack(self.t_chase0, wall, self.out / "soundtrack_chase.wav") \
                if not self.args.no_audio else None
            raw_info = probe(raw[-1])
            info = transcode_chase(raw[-1], chase_mp4, audio=ctrack)
            self.result["chase_video"] = {"path": str(chase_mp4), "raw_hevc": str(raw[-1]), "raw": raw_info,
                                          "recorded_wall_s": round(wall, 2), **info}
        else:
            self.result["chase_video"] = {"error": "no chase recording found"}
        dest = HOME / "media" / "demo"
        dest.mkdir(parents=True, exist_ok=True)
        for k in ("ui_video", "chase_video"):
            p = self.result.get(k, {}).get("path")
            if p and Path(p).exists():
                shutil.copy2(p, dest / Path(p).name)
                self.result[k]["shared_copy"] = str(dest / Path(p).name)
        # stills: full size in the run dir; README-sized copies in docs/img/ with --stills
        readme = PRESETS[self.args.preset].get("readme", {})
        for nm, png in self.stills_raw.items():
            (self.out / "stills").mkdir(exist_ok=True)
            (self.out / "stills" / f"{nm}.png").write_bytes(png)
            if self.args.stills and nm in readme:
                self.result["stills"][nm] = save_still(png, TREE / "docs" / "img" / f"demo_{self.take}_{readme[nm]}.png")
        if self.args.stills and "chase_follow" in readme and self.result["chase_video"].get("path"):
            mid = self.result["marks"].get("still_3_follow_walking", {}).get("mono_s")
            t = (mid - self.t_chase0) if mid else 30.0
            tmp = self.out / "stills" / "chase_follow.png"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(t, 0):.2f}", "-i",
                            self.result["chase_video"]["path"], "-frames:v", "1", str(tmp)], check=False)
            if tmp.exists():
                self.result["stills"]["chase_follow"] = save_still(
                    tmp.read_bytes(), TREE / "docs" / "img" / f"demo_{self.take}_chase_follow.png")

    def metrics(self) -> None:
        truth = Path(self.result.get("twin_log_dir") or "") / "truth.jsonl"
        if not truth.exists():
            self.result["metrics_error"] = f"no truth log at {truth}"
            return
        rep = self.out / "report.json"
        cmd = [PY, "-m", "mhd.twin.metrics", "--truth", str(truth), "--obs", str(self.out / "obs.jsonl"),
               "--telemetry", str(self.out / "telemetry.jsonl"), "--subject", self.args.subject,
               "--hfov-deg", str(self.args.hfov_deg), "--out", str(rep), "--csv", str(self.out / "rows.csv")]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=TREE)
        self.result["metrics_rc"] = r.returncode
        if rep.exists():
            self.result["metrics"] = json.loads(rep.read_text())
        else:
            self.result["metrics_error"] = (r.stderr or r.stdout)[-1500:]

    def follow_window(self) -> None:
        """Control quality over the follow segment only: from FOLLOW (+ settle_s) to the spoken land.
        report.json covers everything from the mission on, including the acquisition turn and the landing,
        where the subject keeps walking while the drone no longer turns."""
        import csv
        import math

        m = self.result["marks"]
        if "follow" not in m or "land_spoken" not in m or not (self.out / "rows.csv").exists():
            return
        t0 = (m["follow"]["mono_s"] + self.args.settle_s) * 1000.0
        t1 = m["land_spoken"]["mono_s"] * 1000.0
        rows = [r for r in csv.DictReader(open(self.out / "rows.csv")) if t0 <= float(r["render_ms"]) <= t1]
        if not rows:
            return
        from collections import Counter

        ids = Counter(r["identity"] for r in rows if r["identity"])
        sid = ids.most_common(1)[0][0] if ids else None        # the actor followed most of the time
        vis = [r for r in rows if r["visible"] == "True"]
        b = [float(r["truth_bearing"]) for r in vis if r["truth_bearing"] not in ("", None)]
        idents = [r["identity"] for r in rows if r["identity"]]
        hfov = math.radians(self.args.hfov_deg)
        self.result["follow_window"] = {
            "from_video_s": round(m["follow"].get("video_s", 0) + self.args.settle_s, 2),
            "to_video_s": m["land_spoken"].get("video_s"), "frames": len(rows),
            "duration_s": round((float(rows[-1]["render_ms"]) - float(rows[0]["render_ms"])) / 1000, 1),
            "subject_visible_frac": round(len(vis) / len(rows), 4),
            "subject_in_frame_frac": round(sum(abs(x) < hfov / 2 for x in b) / max(1, len(rows)), 4),
            "bearing_rms_rad": round(math.sqrt(sum(x * x for x in b) / len(b)), 4) if b else None,
            "bearing_max_abs_rad": round(max(abs(x) for x in b), 4) if b else None,
            "locked_frac": round(sum(r["matched"] == "True" for r in rows) / len(rows), 4),
            "subject_id": sid,
            "frames_on_other_actor": sum(1 for i in idents if sid and i != sid),
            "frames_identified": len(idents),
            "autopilot_states": {k: sum(1 for r in rows if r["autopilot_state"] == k)
                                 for k in sorted({r["autopilot_state"] for r in rows})},
        }

    def voice_summary(self) -> None:
        ev = self.agent_events()
        pick = lambda k: [e for e in ev if e.get("type") == "agent_event" and e.get("kind") == k]  # noqa: E731
        finals, intents, safety = pick("final"), pick("intent"), pick("safety_word")
        self.result["voice"] = {
            "finals": [{"text": e["text"], "asr": (e.get("data") or {}).get("model"),
                        "asr_ms": (e.get("data") or {}).get("asr_ms")} for e in finals],
            "intents": [{"action": e["text"], "model": (e.get("data") or {}).get("model"),
                         "latency_ms": (e.get("data") or {}).get("latency_ms"),
                         "reason": (e.get("data") or {}).get("reason"),
                         "intent": (e.get("data") or {}).get("intent")} for e in intents],
            "safety_words": [{"word": e["text"], **{k: (e.get("data") or {}).get(k) for k in
                              ("command", "transcript", "decode_ms")}} for e in safety],
            "partials": len(pick("partial")), "errors": [e["text"] for e in pick("error")],
            "tts": [{"text": e["text"], "duration_s": e["duration_s"]} for e in ev if e.get("type") == "tts"]}
        # Voice Transcribe bills audio seconds: one session per push-to-talk utterance (warm_session off).
        if "wav" not in self.result:
            return
        secs = sum(self.result["wav"][k]["s"] for k in ("command",)) + 1.0
        self.result["spend"] = {"voice_transcribe_audio_s": round(secs, 1), "est_usd": round(secs / 3600 * 0.18, 5)}
        with open(HOME / "logs" / "spend.jsonl", "a") as f:
            f.write(json.dumps({"ts": round(time.time(), 3), "ws": "demo", "service": "meta:muse-voice-transcribe-1.0",
                                "units": round(secs, 1), "est_usd": round(secs / 3600 * 0.18, 6),
                                "note": f"demo take {self.take} {self.stamp}: push-to-talk audio seconds"}) + "\n")

    async def run(self) -> dict:
        self.preflight()
        err: BaseException | None = None
        try:
            self.start_glimmer()
            self.build_ui()
            self.start_stack()
            self.wait_glimmer()
            await self.fly()
        except BaseException as e:                 # record it, still save what was recorded, then re-raise
            err = e
            self.result["error"] = f"{e.__class__.__name__}: {e}"
            log(f"ERROR: {self.result['error']}")
        finally:
            self.stop_all()
        for step in (self.voice_summary, self.metrics, self.follow_window, self.finish_media):
            try:
                step()
            except Exception as e:
                self.result.setdefault("post_errors", []).append(f"{step.__name__}: {e.__class__.__name__}: {e}")
                log(f"{step.__name__} failed: {e}")
        (self.out / "demo_result.json").write_text(json.dumps(self.result, indent=1, default=str))
        if err is not None:
            raise err
        return self.result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo-scenario", dest="preset", choices=sorted(PRESETS), default="s1",
                    help="story preset: scenario, subject and spoken command (s1 default; s2 = the crossing)")
    ap.add_argument("--scenario", help="twin scenario name or YAML path (overrides the preset)")
    ap.add_argument("--subject", help="subject label for the metrics (overrides the preset)")
    ap.add_argument("--command", help="the spoken push-to-talk command (overrides the preset)")
    ap.add_argument("--land-text", default="Okay drone, please land now.")
    ap.add_argument("--voice", default="Samantha", help="macOS `say` voice for the operator")
    ap.add_argument("--take", help="name for the outputs (default: the preset)")
    ap.add_argument("--latency-ms", type=int, default=500)
    ap.add_argument("--hfov-deg", type=float, default=75.0)
    ap.add_argument("--follow-s", type=float, default=40.0)
    ap.add_argument("--find-timeout-s", type=float, default=30.0)
    ap.add_argument("--intro-s", type=float, default=4.0)
    ap.add_argument("--pre-command-s", type=float, default=2.5)
    ap.add_argument("--outro-s", type=float, default=1.5)
    ap.add_argument("--perception-warmup-s", type=float, default=20.0)
    ap.add_argument("--rtsp-port", type=int, default=None,
                    help=f"default {RTSP_DEFAULT}; if that is taken by another process, 8555 is used")
    ap.add_argument("--agent-config", default=str(TREE / "config" / "demo" / "agent.yaml"))
    ap.add_argument("--governor-config", default=str(TREE / "config" / "twin" / "governor.yaml"))
    ap.add_argument("--autopilot-config", default=str(TREE / "config" / "twin" / "autopilot.yaml"))
    ap.add_argument("--stills", action="store_true",
                    help="write the preset's README stills to docs/img/ (all stills always go to the run dir)")
    ap.add_argument("--keep-glimmer", action="store_true", help="leave the Glimmer server running afterwards")
    ap.add_argument("--no-build", action="store_true", help="reuse turbodrone/frontend/dist if present")
    ap.add_argument("--no-audio", action="store_true", help="no sound track in the MP4s")
    ap.add_argument("--headed", action="store_true", help="show the UI's Chrome window (debugging)")
    ap.add_argument("--keep-frames", action="store_true", help="keep the raw screencast JPEGs")
    ap.add_argument("--still-scn-s", default=None,
                    help="extra stills at these scenario times (s after the reset), e.g. S2's crossings 32,53")
    ap.add_argument("--settle-s", type=float, default=5.0,
                    help="follow_window metrics start this long after FOLLOW (acquisition turn excluded)")
    args = ap.parse_args()
    pre = PRESETS[args.preset]
    args.scenario = args.scenario or pre["scenario"]
    args.subject = args.subject or pre["subject"]
    args.command = args.command or pre["command"]
    args.still_scn_s = args.still_scn_s or pre.get("still_scn_s")
    demo = Demo(args)

    def on_signal(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_signal)
    res = asyncio.run(demo.run())
    m = res.get("metrics") or {}
    print(json.dumps({k: res.get(k) for k in ("run_dir", "take", "scenario", "fallbacks", "autopilot_state_samples",
                                              "follow_window", "voice", "ui_video", "chase_video", "stills",
                                              "metrics_error")},
                     indent=1, default=str)[:6000])
    print(json.dumps(m.get("summary", m), indent=1)[:2500])


if __name__ == "__main__":
    main()
