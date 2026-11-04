"""Session A: the guided PROPS-OFF hardware checklist (PLAN.md section 6, Session A).

Every step prints what to do and what to watch, runs what it can (preflight, ffprobe, backend
commands over /ws, the latency measurement), asks for your observation, and records everything to
logs/session_a-<time>.json after each answer. It ends with a GO / NO-GO summary and the values to
carry into Session B (MHD_CAPTURE_OFFSET_MS, HFOV, theta, land fallback, link-loss behavior).

    source env.sh
    $MHD_PY scripts/session_a.py                       # the real X69 (172.16.11.1), interactive
    $MHD_PY scripts/session_a.py --only flags,linkloss_rcpause --resume logs/session_a-....json
    $MHD_PY scripts/session_a.py --list                # step names

Answers: y / n / ? (unknown) / s (skip the step) / q (save and quit). Enter takes the [default].

The motors run at most --motor-s seconds per test (default 6); session_a sends LAND itself.
Keep a hand ready to pull the battery. Props OFF for every step.

Tonight's test (twin only, loopback): --auto-twin answers every question from the twin's /status,
starts and kills its own twin-mode backend, and sends UI heartbeats itself (twin mode only):

    $MHD_PY scripts/session_a.py --auto-twin --backend-port 8150 --twin http://127.0.0.1:8666 \\
        --rc-port 23550 --rtsp-url rtsp://127.0.0.1:8650/live/ch00_1 --counter-port 8157
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

TREE = Path(os.environ.get("MHD_TREE") or Path(__file__).resolve().parents[1])
PY = os.environ.get("MHD_PY", sys.executable)
ML_PY = os.environ.get("MHD_ML_PY", "")
DRONE_IP = "172.16.11.1"


class Quit(Exception):
    pass


class SkipStep(Exception):
    pass


# ============================================================================================== I/O

class IO:
    """Prompts for the operator. In auto-twin mode every question takes its `auto` answer."""

    def __init__(self, auto: bool, color: bool):
        self.auto = auto
        self.c = color

    def _col(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.c else s

    def header(self, n: int, total: int, title: str) -> None:
        print("\n" + self._col("1;36", f"=== Step {n}/{total}: {title} " + "=" * max(0, 60 - len(title))))

    def do(self, *lines: str) -> None:
        for ln in lines:
            print(self._col("1", "  DO:    ") + ln)

    def watch(self, *lines: str) -> None:
        for ln in lines:
            print(self._col("33", "  WATCH: ") + ln)

    def say(self, *lines: str) -> None:
        for ln in lines:
            print("  " + ln)

    def warn(self, *lines: str) -> None:
        for ln in lines:
            print(self._col("1;31", "  !! ") + ln)

    def _input(self, prompt: str) -> str:
        try:
            ans = input(self._col("1;32", "  ?  ") + prompt + " ").strip()
        except EOFError:
            raise Quit("stdin closed")
        if ans.lower() == "q":
            raise Quit("operator quit")
        if ans.lower() == "s":
            raise SkipStep()
        return ans

    def enter(self, prompt: str = "press Enter when ready") -> None:
        if self.auto:
            print(f"  [auto] {prompt}")
            return
        self._input(prompt + " [Enter]")

    def yn(self, prompt: str, auto: Any = None, default: Optional[bool] = None) -> Optional[bool]:
        if self.auto:
            v = auto() if callable(auto) else auto
            print(f"  [auto] {prompt} -> {v}")
            return v
        d = {True: "Y/n/?", False: "y/N/?", None: "y/n/?"}[default]
        while True:
            a = self._input(f"{prompt} ({d})").lower()
            if not a and default is not None:
                return default
            if a in ("y", "yes"):
                return True
            if a in ("n", "no"):
                return False
            if a in ("?", "u", "unknown"):
                return None

    def num(self, prompt: str, unit: str = "", auto: Any = None, default: Optional[float] = None,
            lo: float = -math.inf, hi: float = math.inf) -> Optional[float]:
        if self.auto:
            v = auto() if callable(auto) else auto
            print(f"  [auto] {prompt} -> {v} {unit}")
            return v
        while True:
            dflt = f" [{default}]" if default is not None else ""
            a = self._input(f"{prompt} ({unit}, ? = unknown){dflt}")
            if not a and default is not None:
                return float(default)
            if a in ("?", ""):
                return None
            try:
                v = float(a)
            except ValueError:
                print("    a number, please")
                continue
            if not (lo <= v <= hi):
                print(f"    expected {lo}..{hi}")
                continue
            return v

    def choice(self, prompt: str, options: dict[str, str], auto: Any = None) -> Optional[str]:
        if self.auto:
            v = auto() if callable(auto) else auto
            print(f"  [auto] {prompt} -> {v}")
            return v
        menu = ", ".join(f"{k}={v}" for k, v in options.items())
        while True:
            a = self._input(f"{prompt} [{menu}, ?=unknown]").lower()
            if a in options:
                return options[a]
            if a == "?":
                return None

    def text(self, prompt: str, auto: Any = "") -> str:
        if self.auto:
            v = auto() if callable(auto) else auto
            print(f"  [auto] {prompt} -> {v!r}")
            return v
        return self._input(prompt)

    def countdown(self, seconds: float, label: str) -> None:
        end = time.monotonic() + seconds
        if not sys.stdout.isatty():
            print(f"  ... {label}: {seconds:.1f} s", flush=True)
            time.sleep(seconds)
            return
        while True:
            left = end - time.monotonic()
            if left <= 0:
                break
            print(f"\r  ... {label}: {left:4.1f} s ", end="", flush=True)
            time.sleep(min(0.2, left))
        print(f"\r  ... {label}: done      ")


# ============================================================================================== recording

class Recorder:
    def __init__(self, path: Path, meta: dict, resume: Optional[dict] = None):
        self.path = path
        self.data = resume or {"meta": meta, "steps": {}}
        self.data["meta"].update({k: v for k, v in meta.items() if k not in self.data["meta"]})
        self.data.setdefault("events", [])

    def step(self, name: str) -> dict:
        return self.data["steps"].setdefault(name, {"status": "started", "answers": {}, "measured": {}})

    def answer(self, step: str, key: str, value: Any) -> Any:
        self.step(step)["answers"][key] = value
        self.save()
        return value

    def measure(self, step: str, key: str, value: Any) -> Any:
        self.step(step)["measured"][key] = value
        self.save()
        return value

    def status(self, step: str, status: str, note: str = "") -> None:
        s = self.step(step)
        s["status"] = status
        s["t_end"] = time.strftime("%H:%M:%S")
        if note:
            s["note"] = note
        self.save()

    def event(self, text: str, **kw) -> None:
        self.data["events"].append({"t": time.strftime("%H:%M:%S"), "mono_ms": round(time.monotonic() * 1000),
                                    "text": text, **kw})
        self.save()

    def get(self, step: str, kind: str, key: str, default=None):
        return self.data["steps"].get(step, {}).get(kind, {}).get(key, default)

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str))
        tmp.replace(self.path)


# ============================================================================================== backend / twin

class Backend:
    def __init__(self, port: int):
        self.port = port
        self.http = f"http://127.0.0.1:{port}"
        self.ws_url = f"ws://127.0.0.1:{port}/ws"
        self._ws = None
        self._lock = threading.Lock()

    def up(self) -> bool:
        try:
            return httpx.get(self.http + "/state", timeout=1.5).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def state(self) -> dict:
        return httpx.get(self.http + "/state", timeout=3.0).json()

    def send(self, msg: dict) -> None:
        import websockets.sync.client as wsc
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._ws is None:
                        with warnings.catch_warnings():       # a long-lived socket, closed in close()
                            warnings.simplefilter("ignore", DeprecationWarning)
                            self._ws = wsc.connect(self.ws_url, open_timeout=3)
                    self._ws.send(json.dumps(msg))
                    return
                except Exception:  # noqa: BLE001
                    try:
                        if self._ws is not None:
                            self._ws.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self._ws = None
                    if attempt == 2:
                        raise

    def cmd(self, command: str, **kw) -> None:
        self.send({"type": "cmd", "command": command, **kw})

    def close(self) -> None:
        with self._lock:
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:  # noqa: BLE001
                    pass
                self._ws = None

    def wait(self, pred: Callable[[dict], bool], timeout: float, interval: float = 0.1) -> Optional[dict]:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                s = self.state()
                if pred(s):
                    return s
            except Exception:  # noqa: BLE001
                pass
            time.sleep(interval)
        return None

    def pid(self) -> Optional[int]:
        try:
            out = subprocess.run(["lsof", "-nP", f"-iTCP:{self.port}", "-sTCP:LISTEN", "-t"], capture_output=True,
                                 text=True, timeout=5).stdout.split()
            return int(out[0]) if out else None
        except Exception:  # noqa: BLE001
            return None

    def post(self, path: str, body: Optional[dict] = None) -> dict:
        r = httpx.post(self.http + path, json=body or {}, timeout=5.0)
        try:
            return r.json()
        except ValueError:
            return {"status": r.status_code, "text": r.text[:200]}


class Twin:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def status(self) -> dict:
        return httpx.get(self.url + "/status", timeout=3.0).json()

    def drone(self) -> dict:
        return self.status()["drone"]

    def control(self, **kw) -> dict:
        return httpx.post(self.url + "/control", json=kw, timeout=5.0).json()


class Heartbeat(threading.Thread):
    """TWIN ONLY: stands in for the UI worker ({"type":"hb","client":"ui"} at 5 Hz)."""

    def __init__(self, port: int):
        super().__init__(daemon=True)
        self.port = port
        self.stop_ev = threading.Event()

    def run(self) -> None:
        import websockets.sync.client as wsc
        while not self.stop_ev.is_set():
            try:
                with wsc.connect(f"ws://127.0.0.1:{self.port}/ws", open_timeout=2) as ws:
                    while not self.stop_ev.is_set():
                        ws.send(json.dumps({"type": "hb", "client": "ui"}))
                        self.stop_ev.wait(0.2)
            except Exception:  # noqa: BLE001
                self.stop_ev.wait(0.5)


# ============================================================================================== session

class Session:
    def __init__(self, a: argparse.Namespace, io: IO, rec: Recorder):
        self.a, self.io, self.rec = a, io, rec
        self.auto = a.auto_twin
        self.be = Backend(a.backend_port)
        self.twin = Twin(a.twin) if self.auto else None
        self.children: list[subprocess.Popen] = []
        self.backend_proc: Optional[subprocess.Popen] = None
        self.hb: Optional[Heartbeat] = None
        self.out_dir = rec.path.with_suffix("")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir = Path(a.media_dir) if a.media_dir else TREE / "media" / self.out_dir.name
        self.drone_ip = "127.0.0.1" if self.auto else a.drone_ip
        self.rtsp_url = a.rtsp_url or f"rtsp://{self.drone_ip}/live/ch00_1"

    # ------------------------------------------------------------------ process helpers
    def spawn(self, cmd: list[str], log: Path, env: Optional[dict] = None) -> subprocess.Popen:
        e = dict(os.environ)
        e.update(env or {})
        f = open(log, "w")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=TREE, env=e, start_new_session=True)
        self.children.append(p)
        return p

    @staticmethod
    def stop_proc(p: Optional[subprocess.Popen], sig=signal.SIGINT, wait: float = 6.0) -> None:
        if p is None or p.poll() is not None:
            return
        for s, w in ((sig, wait), (signal.SIGTERM, 3.0), (signal.SIGKILL, 2.0)):
            try:
                os.killpg(p.pid, s)
            except ProcessLookupError:
                return
            try:
                p.wait(w)
                return
            except subprocess.TimeoutExpired:
                continue

    def start_twin_backend(self) -> None:
        """Auto-twin: our own twin-mode backend (so link-loss test 1 can kill -9 and restart it)."""
        if self.be.up():
            raise SystemExit(f"a backend already listens on {self.a.backend_port}; --auto-twin starts its own")
        cmd = ["bash", "scripts/run_backend.sh", "--mode", "twin", "--port", str(self.a.backend_port),
               "--control-port", str(self.a.rc_port), "--rtsp-url", self.rtsp_url,
               "--logs", str(self.out_dir / "backend")]
        n = sum(1 for p in self.out_dir.glob("backend-*.log"))
        self.backend_proc = self.spawn(cmd, self.out_dir / f"backend-{n}.log")
        if not self.be.wait(lambda s: (s.get("latency_ms") or {}).get("frames_received", 0) > 5, 45, 0.5):
            raise SystemExit(f"twin backend did not come up with video; see {self.out_dir}/backend-{n}.log")
        if self.hb is None or not self.hb.is_alive():
            self.hb = Heartbeat(self.a.backend_port)
            self.hb.start()
        time.sleep(1.0)

    def cleanup(self) -> None:
        if self.hb:
            self.hb.stop_ev.set()
        self.be.close()
        for p in reversed(self.children):
            self.stop_proc(p)

    # ------------------------------------------------------------------ backend helpers
    def need_backend(self, step: str) -> dict:
        while not self.be.up():
            if self.auto:
                self.start_twin_backend()
                continue
            self.io.warn(f"no backend on 127.0.0.1:{self.a.backend_port}")
            self.io.do("start it in a second window of the SAME terminal app:",
                       "  scripts/run_stack.sh --mode real --no-perception --no-agent   (backend + UI)")
            self.io.enter("press Enter once it is up (s = skip this step)")
        s = self.be.state()
        if not self.auto and s.get("flight_mode") != "real":
            self.io.warn(f"the backend is in {s.get('flight_mode')} mode, not real")
        return s

    def ensure_ground_and_armed(self, step: str) -> dict:
        s = self.need_backend(step)
        fs = s.get("flight_state")
        if fs == "ESTOPPED":
            self.be.cmd("estop_reset")
            self.be.wait(lambda x: x["flight_state"] == "LANDING", 3)
            fs = "LANDING"
        if fs == "LANDING":
            self.io.say("the governor is LANDING: confirming touchdown (motors must be stopped)")
            if not self.auto:
                self.io.enter("motors stopped? press Enter to confirm touchdown")
            self.be.cmd("confirm_touchdown")
            self.be.wait(lambda x: x["flight_state"] == "ON_GROUND", 3)
        elif fs in ("TAKING_OFF", "AIRBORNE"):
            self.be.cmd("land")
            self.be.wait(lambda x: x["flight_state"] == "LANDING", 3)
            if self.auto:
                self.wait_twin(lambda d: not d["motors_on"], 15)
            else:
                self.io.enter("governor was flying: LAND sent. Motors stopped? press Enter to confirm touchdown")
            self.be.cmd("confirm_touchdown")
            self.be.wait(lambda x: x["flight_state"] == "ON_GROUND", 3)
        s = self.be.state()
        if s.get("flight_mode") == "real" and not s.get("armed"):
            while True:
                phrase = self.io.text("type ARM to arm the governor (once per backend start, and after E-stop reset)")
                self.be.cmd("arm", arm_phrase=phrase.strip().upper())
                s = self.be.wait(lambda x: x.get("armed"), 2) or self.be.state()
                if s.get("armed"):
                    break
                self.io.warn("not armed (the phrase is exactly ARM)")
        return self.be.state()

    def takeoff(self, step: str) -> bool:
        """Send TAKEOFF and wait for the governor to accept it. Explains a refusal."""
        s = self.ensure_ground_and_armed(step)
        lat = s.get("latency_ms") or {}
        hb = lat.get("hb_age")
        if not self.auto and (hb is None or hb > 1000):
            self.io.warn(f"UI heartbeat is {hb} ms old: open the UI (http://127.0.0.1:5173) in its own Chrome window")
            self.io.enter("press Enter when the UI shows HB 5.0 Hz")
        self.be.cmd("takeoff")
        s = self.be.wait(lambda x: x["flight_state"] in ("TAKING_OFF", "AIRBORNE"), 2.0)
        if s is None:
            st = self.be.state()
            why = [f for f in st.get("safety_flags", []) if f in ("video_stale", "heartbeat_missing", "governor_fault",
                                                                    "rc_tick_stale")]
            if st.get("flight_mode") == "real" and not st.get("armed"):
                why.append("not armed")
            self.io.warn(f"TAKEOFF not accepted (state {st.get('flight_state')}; {', '.join(why) or 'see the backend log'})")
            self.rec.event("takeoff refused", step=step, flags=st.get("safety_flags"))
            return False
        self.rec.event("takeoff accepted", step=step)
        return True

    def land_and_confirm(self, step: str, ask: bool = True) -> Optional[bool]:
        """LAND, then ask (or check the twin) whether the motors stopped; confirm touchdown."""
        t0 = time.monotonic()
        self.be.cmd("land")
        self.be.wait(lambda x: x["flight_state"] == "LANDING", 2)
        stopped = None
        if ask:
            if self.auto:
                stopped = self.wait_twin(lambda d: not d["motors_on"], 15) is not None
                print(f"  [auto] motors stopped after LAND -> {stopped} ({time.monotonic() - t0:.1f} s)")
            else:
                stopped = self.io.yn("did the motors STOP after LAND?")
        if stopped is False:
            self.io.warn("LAND did not stop them: sending E-STOP now")
            self.be.cmd("estop")
            self.be.wait(lambda x: x["flight_state"] == "ESTOPPED", 2)
            cut = self.io.yn("did E-STOP cut the motors?", auto=lambda: not self.twin.drone()["motors_on"])
            self.rec.answer(step, "estop_after_failed_land_cut", cut)
            self.be.cmd("estop_reset")
            self.be.wait(lambda x: x["flight_state"] == "LANDING", 2)
        if self.auto:
            self.wait_twin(lambda d: d["phase"] == "landed", 15)
        self.be.cmd("confirm_touchdown")
        self.be.wait(lambda x: x["flight_state"] == "ON_GROUND", 3)
        return stopped

    def wait_twin(self, pred: Callable[[dict], bool], timeout: float) -> Optional[dict]:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                d = self.twin.drone()
                if pred(d):
                    return d
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.1)
        return None

    def motors_on(self) -> Optional[bool]:
        return self.twin.drone()["motors_on"] if self.twin else None

    # ============================================================================== steps
    def step_safety(self, st: str) -> None:
        io = self.io
        io.say("Session A is the hard gate: no props-on flight until it passes. Props OFF for every step.")
        io.do("work on a non-flammable surface; keep a hand ready to pull the battery",
              "motor runs are short: this script sends LAND after %.0f s" % self.a.motor_s,
              "film every motor test and every link-loss test with the SECOND phone (not the tether iPhone)",
              "batteries: charge attended, never leave a pack charging alone, battery out when not testing")
        ok = io.yn("props OFF, drone on a non-flammable surface, battery hand ready?", auto=True)
        self.rec.answer(st, "props_off_ready", ok)
        if not ok:
            raise Quit("not ready: props must be off")

    def step_props(self, st: str) -> None:
        io = self.io
        io.do("mark each prop and its motor arm A or B (paint pen or tape): props are handed",
              "photograph each corner (prop + arm) with the second phone",
              "remove all four props with the prop tool; keep them in a labeled bag")
        io.watch("A and B props differ by blade twist; a swapped prop flips that motor's thrust in Session B")
        self.rec.answer(st, "marked_photographed_removed", io.yn("props marked, photographed, removed?", auto=None))

    def run_preflight(self, st: str, only: str) -> dict:
        out = self.out_dir / f"preflight-{only.replace(',', '_')}.json"
        cmd = [PY, str(TREE / "scripts" / "preflight.py"), "--drone-ip", self.drone_ip, "--only", only,
               "--rtsp-url", self.rtsp_url, "--json", str(out), "--backend-port", str(self.a.backend_port)]
        if self.a.skip_ml:
            cmd.append("--skip-ml")
        while True:
            code = subprocess.run(cmd, cwd=TREE).returncode
            try:
                res = json.loads(out.read_text())
            except Exception:  # noqa: BLE001
                res = {"exit_code": code, "results": []}
            if code == 0 or self.auto:
                return res
            self.io.warn("preflight FAILED: apply the fixes above, then re-run")
            again = self.io.yn("re-run preflight now? (n = record the failure and go on)", default=True)
            if not again:
                return res

    def step_preflight(self, st: str) -> None:
        io = self.io
        io.do("iPhone: Settings > Personal Hotspot > Allow Others to Join; plug it in by USB; Trust",
              "System Settings > Network > (...) > Set Service Order: iPhone USB ABOVE Wi-Fi",
              "power the drone (props off), join its Wi-Fi on the Mac, turn off auto-join for other networks",
              "run this from the SAME terminal app that will launch the backend (Local Network permission)")
        res = self.run_preflight(st, "env,wifi,routes,internet,localnet,system")
        summary = {f"{r['group']}/{r['name']}": r["status"] for r in res.get("results", [])}
        self.rec.measure(st, "results", summary)
        self.rec.measure(st, "exit_code", res.get("exit_code"))
        fails = [k for k, v in summary.items() if v == "FAIL"]
        self.rec.measure(st, "fails", fails)
        io.say("G0 over the tether (optional, ~$0.05): in another window run",
               f"  {ML_PY or '$MHD_ML_PY'} -m mhd_ml.agent.bench spark --n 10 ;  ... bench voice --n 5")
        self.rec.answer(st, "g0_tether_ok", io.yn("G0 over the tether OK (Spark p95 <= 4 s, voice <= 1 s)? "
                                                   "(? = not run)", auto=None))

    def step_ffprobe(self, st: str) -> None:
        io = self.io
        if self.be.up() and not self.auto:
            io.warn("a backend is running: stop it for this step (the X69 may allow only one RTSP client)")
            io.enter("press Enter when the backend is stopped (s = skip)")
        io.say("ffprobe: frame rate, keyframe interval, bitrate, and whether a second RTSP client is allowed")
        res = self.run_preflight(st, "video")
        rows = {r["name"]: r for r in res.get("results", []) if r["group"] == "video"}
        stream = rows.get("RTSP stream", {}).get("data", {})
        stats = rows.get("fps / keyframes / bitrate", {}).get("data", {})
        sc = rows.get("second simultaneous client", {}).get("data", {})
        self.rec.measure(st, "stream", {k: stream.get(k) for k in ("codec", "width", "height", "profile",
                                                                    "r_frame_rate", "transport")})
        self.rec.measure(st, "fps", stats.get("fps"))
        self.rec.measure(st, "keyframe_interval_frames", stats.get("keyframe_interval_frames"))
        self.rec.measure(st, "keyframe_interval_s", stats.get("keyframe_interval_s"))
        self.rec.measure(st, "bitrate_kbps", stats.get("bitrate_kbps"))
        self.rec.measure(st, "second_client", sc.get("verdict"))
        self.rec.measure(st, "ok", rows.get("RTSP stream", {}).get("status") == "PASS")

    def step_recordings(self, st: str) -> None:
        io = self.io
        single = self.rec.get("ffprobe", "measured", "second_client") in ("only_one", "kicks_first")
        io.say(f"recordings go to {self.media_dir} (gitignored; raw footage of people never enters the repo)")
        if self.be.up() and single and not self.auto:
            io.warn("the backend is running and the drone allows one client: stop the backend for these recordings")
            io.enter("press Enter when the backend is stopped (s = skip)")
        self.media_dir.mkdir(parents=True, exist_ok=True)
        secs = 5 if self.auto else 60
        if io.yn(f"record {secs} s of raw RTSP now (ffmpeg -c copy)?", auto=True, default=True):
            f = self.media_dir / "raw_rtsp.mp4"
            self.rec.measure(st, "raw_rtsp", self.ffmpeg_copy(f, secs))
        if not self.auto:
            io.do("hand-carry the drone (props off) and film the subjects for 2 minutes: a person in a blue shirt,",
                  "a person in another color, a green toy car; slow pans, 2-4 m, the way it will fly")
            if io.yn("record the 2-minute subjects clip now?", default=True):
                io.enter("ready to walk? press Enter to start recording")
                self.rec.measure(st, "subjects", self.ffmpeg_copy(self.media_dir / "subjects.mp4", 120))
            io.do(f"mic test (grants the Microphone permission to this app): {ML_PY or '$MHD_ML_PY'} -m "
                  "mhd_ml.agent.mic_check --seconds 5   (say 'stop' and 'land')")
            self.rec.answer(st, "mic_check_ok", io.yn("mic check showed levels and SAFETY lines?"))
            self.rec.answer(st, "webcam_clip", io.yn("webcam clip recorded (optional: Photo Booth or ffmpeg avfoundation)?"))
        else:
            self.rec.measure(st, "subjects", "skipped (twin)")

    def ffmpeg_copy(self, path: Path, seconds: float) -> dict:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-rtsp_transport", "tcp", "-i", self.rtsp_url,
               "-c", "copy", "-t", str(seconds), str(path)]
        t0 = time.monotonic()
        self.io.say(f"recording {seconds:.0f} s -> {path}")
        code = subprocess.run(cmd, cwd=TREE).returncode
        size = path.stat().st_size if path.exists() else 0
        dur = None
        if size:
            p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(path)], capture_output=True, text=True)
            try:
                dur = float(p.stdout.strip())
            except ValueError:
                pass
        ok = code == 0 and size > 0
        (self.io.say if ok else self.io.warn)(f"{'saved' if ok else 'FAILED'}: {size / 1e6:.1f} MB, {dur} s "
                                              f"in {time.monotonic() - t0:.0f} s")
        return {"path": str(path), "ok": ok, "bytes": size, "duration_s": dur}

    def step_backend(self, st: str) -> None:
        io = self.io
        if self.auto:
            if not self.be.up():
                self.start_twin_backend()
        else:
            io.do("in a second window of the SAME terminal app:",
                  "  scripts/run_stack.sh --mode real --no-perception --no-agent",
                  "then open the URL it prints (http://127.0.0.1:5173) in its own Chrome window, never minimized")
            io.enter("press Enter when the stack says it is up")
        s = self.need_backend(st)
        lat = s.get("latency_ms") or {}
        f0 = lat.get("frames_received", 0)
        time.sleep(2.0)
        s = self.be.state()
        lat = s.get("latency_ms") or {}
        fps = (lat.get("frames_received", 0) - f0) / 2.0
        self.rec.measure(st, "flight_mode", s.get("flight_mode"))
        self.rec.measure(st, "backend_frame_rate", round(fps, 1))
        self.rec.measure(st, "frame_lag_ms", lat.get("frame_lag"))
        self.rec.measure(st, "flags", s.get("safety_flags"))
        io.say(f"backend: mode {s.get('flight_mode')}, state {s.get('flight_state')}, video {fps:.1f} fps into the "
               f"backend, frame lag {lat.get('frame_lag')} ms, flags {s.get('safety_flags')}")
        io.watch("the UI banner reads REAL DRONE (red stripes), the header chips CTRL / HB 5.0 Hz / TLM 10 Hz are green",
                 "the camera panel shows live video from the drone")
        self.rec.answer(st, "ui_video_ok", io.yn("UI shows live drone video and the REAL DRONE banner?",
                                                 auto=lambda: fps > 5))

    def step_latency(self, st: str) -> None:
        io = self.io
        port = self.a.counter_port
        srv = self.spawn([PY, "-m", "mhd.twin.screen_counter", "serve", "--port", str(port)],
                         self.out_dir / "screen_counter.log")
        try:
            ok = False
            for _ in range(40):
                try:
                    ok = httpx.get(f"http://127.0.0.1:{port}/counter", timeout=1).status_code == 200
                    if ok:
                        break
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.25)
            self.rec.measure(st, "counter_page_served", ok)
            url = f"http://127.0.0.1:{port}/counter"
            io.do(f"open {url} in Chrome, full screen (Cmd-Ctrl-F), screen brightness up, Night Shift off",
                  "point the drone camera at the screen, 30-60 cm away, the whole code and its 4 markers in view")
            io.watch("the UI camera panel shows the 8x8 code with the four ArUco markers, sharp and not clipped")
            if not self.auto and io.yn("open the page in Chrome for you now?", default=True):
                subprocess.run(["open", "-a", "Google Chrome", url])
            io.enter("press Enter to start a %.0f s measurement" % self.a.latency_s)
            res = self.measure_latency(self.a.latency_s)
            self.rec.measure(st, "result", res)
            if res.get("decoded", 0) < 10:
                io.warn(f"only {res.get('decoded', 0)} frames decoded")
                io.say("fallback: film a phone stopwatch on the screen next to the UI video, read the difference")
                v = io.num("latency from the stopwatch method", "ms", auto=None, lo=0, hi=5000)
                self.rec.answer(st, "manual_latency_ms", v)
            else:
                lm = res["latency_ms"]
                io.say(f"camera -> backend latency: p50 {lm['p50']} ms, p05 {lm['p05']} ms, p95 {lm['p95']} ms, "
                       f"min {lm['min']} ms over {res['decoded']} frames ({res['source']})")
                io.say(f"-> set MHD_CAPTURE_OFFSET_MS={int(lm['p05'])} for Session B (export it before starting the backend)")
        finally:
            self.stop_proc(srv)

    def measure_latency(self, seconds: float) -> dict:
        """Decode the screen counter (or, in the twin, its corner counter) on the backend's /ws/frames.
        latency = backend receive time - render time on the shared monotonic clock."""
        import cv2
        import numpy as np
        import websockets.sync.client as wsc

        from mhd.contracts import decode_frame
        from mhd.twin import counter as corner
        from mhd.twin.screen_counter import decode_screen
        if not self.be.up():
            return self.measure_latency_rtsp(seconds)
        rows, frames, rises = [], 0, []
        src = {"screen": 0, "corner": 0}
        t_end = time.monotonic() + seconds
        last_rise = 0.0
        with wsc.connect(f"ws://127.0.0.1:{self.a.backend_port}/ws/frames", max_size=16 * 1024 * 1024) as ws:
            last = -1
            while time.monotonic() < t_end:
                ws.send(json.dumps({"type": "next", "after_frame_id": last}))
                meta, jpeg = decode_frame(ws.recv(timeout=5))
                last = meta.frame_id
                frames += 1
                img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                d = decode_screen(img, ref_ms=meta.receive_ms)
                kind = "screen"
                if d is None and self.auto:
                    d = corner.decode(img, ref_ms=meta.receive_ms)
                    kind = "corner"
                if d:
                    src[kind] += 1
                    rows.append((meta.receive_ms - d[1], meta.capture_ms - d[1]))
                if time.monotonic() - last_rise > 1.0:
                    last_rise = time.monotonic()
                    try:
                        rises.append((self.be.state().get("latency_ms") or {}).get("capture_rise"))
                    except Exception:  # noqa: BLE001
                        pass
        out = {"source": "backend /ws/frames (" + ", ".join(f"{k} {v}" for k, v in src.items() if v) + ")",
               "frames": frames, "decoded": len(rows), "display_ms_subtracted": self.a.display_ms}
        if rows:
            lat = np.array([r[0] for r in rows]) - self.a.display_ms
            cap = np.array([r[1] for r in rows]) - self.a.display_ms
            q = lambda arr, p: round(float(np.percentile(arr, p)), 1)  # noqa: E731
            out["latency_ms"] = {"p05": q(lat, 5), "p50": q(lat, 50), "p95": q(lat, 95), "min": round(float(lat.min()), 1),
                                 "max": round(float(lat.max()), 1)}
            out["capture_estimate_error_ms"] = {"p50": q(cap, 50), "p05": q(cap, 5), "p95": q(cap, 95)}
            hist, edges = np.histogram(lat, bins=np.arange(0, max(100.0, float(lat.max()) + 50), 50))
            out["histogram_50ms"] = {f"{int(edges[i])}-{int(edges[i + 1])}": int(hist[i]) for i in range(len(hist)) if hist[i]}
            with open(self.out_dir / "latency.csv", "w") as fh:
                fh.write("latency_ms,capture_error_ms\n")
                for r in rows:
                    fh.write(f"{r[0]:.1f},{r[1]:.1f}\n")
        rises = [r for r in rises if r is not None]
        if rises:
            out["capture_rise_ms"] = {"first": rises[0], "last": rises[-1], "max": max(rises)}
        return out

    def measure_latency_rtsp(self, seconds: float) -> dict:
        csv = self.out_dir / "latency_rtsp.csv"
        p = subprocess.run([PY, "-m", "mhd.twin.screen_counter", "measure", "--rtsp", self.rtsp_url, "--seconds",
                            str(seconds), "--csv", str(csv), "--display-ms", str(self.a.display_ms)],
                           capture_output=True, text=True, cwd=TREE)
        try:
            res = json.loads(p.stdout[p.stdout.index("{"):])
        except ValueError:
            return {"source": "rtsp", "decoded": 0, "error": (p.stderr or p.stdout)[-300:]}
        res["source"] = "rtsp (screen_counter measure)"
        return res

    def step_hfov(self, st: str) -> None:
        io = self.io
        io.do("put the drone exactly 1.00 m from a flat wall, camera level and square to the wall",
              "stretch a tape measure horizontally across the camera's whole view, at the image center height")
        snap = self.snapshot(self.out_dir / "hfov.jpg") if self.be.up() else None
        if snap:
            io.say(f"snapshot saved: {snap} (open it to read the tape at the left and right frame edges)")
            if not self.auto and io.yn("open the snapshot in Preview?", default=False):
                subprocess.run(["open", snap])
        io.watch("read the tape where it leaves the image on the LEFT and on the RIGHT edge (full frame, not cropped)")
        twin_hfov = self.a.twin_hfov_deg
        d = io.num("distance camera to wall", "m", auto=1.0, default=1.0, lo=0.2, hi=5)
        w = io.num("visible tape width, left edge to right edge", "m",
                   auto=lambda: round(2 * 1.0 * math.tan(math.radians(twin_hfov) / 2), 3), lo=0.05, hi=20)
        self.rec.answer(st, "distance_m", d)
        self.rec.answer(st, "width_m", w)
        if d and w:
            hfov = math.degrees(2 * math.atan(w / (2 * d)))
            self.rec.measure(st, "hfov_deg", round(hfov, 2))
            io.say(f"HFOV = 2*atan(w/2d) = {hfov:.1f} deg (config default 75; apply with scripts/apply_sysid.py --hfov-deg)")
            if not (40 <= hfov <= 140):
                io.warn("outside 40-140 deg: re-measure")

    def snapshot(self, path: Path) -> Optional[str]:
        try:
            import websockets.sync.client as wsc

            from mhd.contracts import decode_frame
            with wsc.connect(f"ws://127.0.0.1:{self.a.backend_port}/ws/frames", max_size=16 * 1024 * 1024) as ws:
                ws.send(json.dumps({"type": "next", "after_frame_id": -1}))
                _, jpeg = decode_frame(ws.recv(timeout=5))
            path.write_bytes(jpeg)
            return str(path)
        except Exception as exc:  # noqa: BLE001
            self.io.warn(f"snapshot failed: {exc!r}")
            return None

    # ------------------------------------------------------------------ flags
    def step_flag_takeoff_land(self, st: str) -> None:
        io = self.io
        io.do("props OFF. This sends TAKEOFF over /ws, keeps the motors running %.0f s, then sends LAND" % self.a.motor_s)
        io.watch("do the motors spin within ~2 s of TAKEOFF? do they stop after LAND, and how fast?")
        io.enter("ready (hand near the battery)? press Enter to send TAKEOFF")
        if not self.takeoff(st):
            self.rec.measure(st, "takeoff_accepted", False)
            return
        self.rec.measure(st, "takeoff_accepted", True)
        spun_auto = None
        if self.auto:
            spun_auto = self.wait_twin(lambda d: d["motors_on"], 3) is not None
        io.countdown(self.a.motor_s, "motors running (LAND follows)")
        spun = self.rec.answer(st, "motors_spun", io.yn("did the motors SPIN after TAKEOFF?", auto=spun_auto))
        if spun is False:
            self.land_and_confirm(st, ask=False)
            arm = io.text("motors did not spin: does the X69 need an arm/unlock step (remote, sticks)? describe",
                          auto="")
            self.rec.answer(st, "arm_step", arm or "unknown")
            io.warn("NO-GO for backend takeoff until this is understood (try the physical remote's start, note it)")
            return
        self.rec.answer(st, "spin_delay_s", io.num("roughly how long after the press did they spin?", "s",
                                                   auto=None, lo=0, hi=10))
        stopped = self.land_and_confirm(st)
        self.rec.answer(st, "land_stopped_motors", stopped)
        if stopped:
            self.rec.answer(st, "land_stop_time_s", io.num("how long did they take to stop after LAND?", "s",
                                                           auto=None, lo=0, hi=30))
        else:
            io.warn("0x80 did not land: enable the throttle-ramp landing fallback for Session B:",
                    "  $MHD_PY scripts/apply_sysid.py --throttle-ramp on --write   (config/real/governor.yaml)")
        self.rec.answer(st, "arm_step", "none needed" if spun else "unknown")

    def step_flag_estop(self, st: str) -> None:
        io = self.io
        io.do("TAKEOFF, then ONE E-STOP while the motors spin; then TAKEOFF again while E-stopped (must be refused),",
              "then 5 more E-STOPs; then E-STOP reset (-> LANDING), Landed, ARM again")
        io.watch("the motors must CUT at once on the first E-STOP and stay off through everything after it")
        io.enter("press Enter to send TAKEOFF")
        if not self.takeoff(st):
            return
        if self.auto:
            self.wait_twin(lambda d: d["motors_on"], 3)
        io.countdown(min(3.0, self.a.motor_s), "motors running (E-STOP follows)")
        self.be.cmd("estop")
        s = self.be.wait(lambda x: x["flight_state"] == "ESTOPPED", 2)
        self.rec.measure(st, "governor_estopped", s is not None)
        cut = io.yn("did the motors CUT at once on E-STOP?",
                    auto=lambda: self.wait_twin(lambda d: not d["motors_on"], 2) is not None)
        self.rec.answer(st, "estop_cut_motors", cut)
        if cut:
            self.rec.answer(st, "estop_cut_time_s", io.num("how fast (0 = instantly)?", "s", auto=None, lo=0, hi=10))
        self.be.cmd("takeoff")                    # must be refused while E-stopped (latched)
        time.sleep(1.0)
        s = self.be.state()
        self.rec.measure(st, "takeoff_refused_while_estopped", s.get("flight_state") == "ESTOPPED")
        for _ in range(5):
            self.be.cmd("estop")
            time.sleep(0.1)
        time.sleep(1.0)
        latched = io.yn("motors stayed OFF through the TAKEOFF attempt and 5 more E-STOPs?",
                        auto=lambda: not self.twin.drone()["motors_on"])
        self.rec.answer(st, "estop_latched", latched)
        self.rec.measure(st, "still_estopped", self.be.state().get("flight_state") == "ESTOPPED")
        self.be.cmd("estop_reset")
        s = self.be.wait(lambda x: x["flight_state"] == "LANDING", 2)
        self.rec.measure(st, "reset_goes_to_landing", s is not None)
        quiet = io.yn("after the E-STOP reset (governor LANDING, 0x80 re-asserted): did the motors stay off?",
                      auto=lambda: not self.twin.drone()["motors_on"])
        self.rec.answer(st, "reset_motors_stayed_off", quiet)
        self.be.cmd("confirm_touchdown")
        self.be.wait(lambda x: x["flight_state"] == "ON_GROUND", 2)

    def step_flag_land_ground(self, st: str) -> None:
        io = self.io
        io.do("motors OFF, on the ground: this sends LAND (0x80, re-asserted 5x per second) for 3 s, then Landed")
        io.watch("nothing should happen: no motor twitch, no beeps; note any LED change")
        self.ensure_ground_and_armed(st)
        io.enter("press Enter to send LAND on the ground")
        self.be.cmd("land")
        self.be.wait(lambda x: x["flight_state"] == "LANDING", 2)
        twitch = None
        if self.auto:
            seen = self.wait_twin(lambda d: d["motors_on"], 3.0)
            twitch = seen is not None
        else:
            time.sleep(3.0)
        noop = io.yn("was it a no-op (motors stayed still)?", auto=None if twitch is None else not twitch)
        self.rec.answer(st, "land_on_ground_noop", noop)
        self.rec.answer(st, "notes", io.text("anything else (LEDs, beeps)? Enter = nothing", auto=""))
        self.be.cmd("confirm_touchdown")
        self.be.wait(lambda x: x["flight_state"] == "ON_GROUND", 2)

    def step_flag_tilt(self, st: str) -> None:
        io = self.io
        io.do("motors OFF. Watch the camera tilt servo (or the UI video moving up/down)")
        tilt0 = self.twin.drone()["cam_tilt_deg"] if self.twin else None
        for d in ("down", "up"):
            io.enter(f"press Enter for ONE short tilt-{d} press (0.3 s)")
            self.be.cmd(f"tilt_{d}")
            time.sleep(0.3)
            self.be.cmd("tilt_stop")
            time.sleep(0.6)
            if self.twin:
                t1 = self.twin.drone()["cam_tilt_deg"]
                delta = round(t1 - tilt0, 2)
                tilt0 = t1
            v = io.num(f"how many degrees did it move {d} on that press? (0 = none)", "deg",
                       auto=(lambda: abs(delta)) if self.twin else None, lo=0, hi=120)
            self.rec.answer(st, f"per_press_{d}_deg", v)
        io.enter("press Enter to HOLD tilt-down for 3 s (end of travel?)")
        for _ in range(3):
            self.be.cmd("tilt_down")
            time.sleep(1.0)
        self.be.cmd("tilt_stop")
        full = io.num("total travel top to bottom, if you saw the end stop", "deg",
                      auto=(lambda: self.twin.drone()["cam_tilt_deg"]) if self.twin else None, lo=-120, hi=120)
        self.rec.answer(st, "hold_3s_reading_deg" if self.twin else "full_travel_deg", full)
        self.rec.answer(st, "set_tilt_before_takeoff",
                        io.text("leave the camera where you want it for flights (slightly down); note the position",
                                auto="twin"))

    # ------------------------------------------------------------------ link loss
    def _observe_motors(self, st: str, seconds: float, t0: float) -> tuple[Optional[str], Optional[float]]:
        """After a link loss at t0: what the motors did and when (twin: from its status)."""
        if self.auto:
            stop_t = None
            end = t0 + seconds
            link_lost_t = None
            while time.monotonic() < end:
                d = self.twin.drone()
                if d["link_lost"] and link_lost_t is None:
                    link_lost_t = round(time.monotonic() - t0, 2)
                if not d["motors_on"] and stop_t is None:
                    stop_t = round(time.monotonic() - t0, 2)
                    break
                time.sleep(0.1)
            self.rec.measure(st, "twin_link_lost_after_s", link_lost_t)
            behavior = "landed (twin loss_action land)" if stop_t is not None else "kept spinning"
            print(f"  [auto] motors -> {behavior} after {stop_t} s (link lost after {link_lost_t} s)")
            return behavior, stop_t
        self.io.countdown(seconds, "watch the motors")
        behavior = self.io.choice("what did the motors do?", {"c": "kept spinning", "s": "stopped at once",
                                                               "r": "ramped down (landing)", "o": "other"})
        when = self.io.num("after how many seconds?", "s", lo=0, hi=60) if behavior != "kept spinning" else None
        return behavior, when

    def step_linkloss_kill(self, st: str) -> None:
        io = self.io
        io.do("link loss 1 of 3: the backend dies (kill -9, NOT Ctrl-C: Ctrl-C lands first). Motors running, filmed.")
        io.watch("what the motors do with no RC packets and no keepalive, and how fast")
        io.enter("start filming; press Enter to TAKEOFF")
        if not self.takeoff(st):
            return
        if self.auto:
            self.wait_twin(lambda d: d["motors_on"], 3)
        io.countdown(3.0, "motors running")
        pid = self.be.pid()
        if pid is None:
            io.warn("cannot find the backend PID (lsof); kill it yourself: kill -9 $(lsof -t -iTCP:%d -sTCP:LISTEN)"
                    % self.a.backend_port)
            io.enter("press Enter right when you kill it")
        else:
            io.enter(f"press Enter to kill -9 the backend now (pid {pid})")
            os.kill(pid, signal.SIGKILL)
        t0 = time.monotonic()
        self.rec.event("backend killed", pid=pid)
        self.be.close()
        behavior, when = self._observe_motors(st, self.a.link_s, t0)
        self.rec.answer(st, "motors", behavior)
        self.rec.answer(st, "after_s", when)
        if behavior == "kept spinning" and not self.auto:
            io.warn("motors still spinning: restart the backend and press E-STOP (Esc in the UI), or pull the battery")
        if self.auto:
            self.stop_proc(self.backend_proc, signal.SIGKILL)
            self.start_twin_backend()
        else:
            io.do("restart the backend in its window (up-arrow, Enter), or: scripts/run_stack.sh --mode real "
                  "--no-perception --no-agent")
            io.enter("press Enter when it is back up")
        self.need_backend(st)
        if not self.auto and behavior == "kept spinning":
            if io.yn("motors still spinning? (sends E-STOP)", default=True):
                self.be.cmd("estop")
        self.ensure_ground_and_armed(st)

    def step_linkloss_wifi(self, st: str) -> None:
        io = self.io
        io.do("link loss 2 of 3: the Mac's Wi-Fi goes off for %.0f s (RC and keepalive stop). Motors running, filmed."
              % self.a.link_s)
        io.watch("the motors; then, after Wi-Fi is back, the Mac must rejoin the drone's network by itself")
        io.enter("start filming; press Enter to TAKEOFF")
        if not self.takeoff(st):
            return
        if self.auto:
            self.wait_twin(lambda d: d["motors_on"], 3)
        io.countdown(3.0, "motors running")
        if self.auto:
            self.twin.control(rc_loss=1.0)       # the twin drops every control packet: same as no Wi-Fi
            t0 = time.monotonic()
            self.rec.event("twin rc_loss=1.0 (stands in for Wi-Fi off)")
        else:
            io.enter("turn the Mac's Wi-Fi OFF now (menu bar), then press Enter")
            t0 = time.monotonic()
            self.rec.event("wifi off")
        behavior, when = self._observe_motors(st, self.a.link_s, t0)
        self.rec.answer(st, "motors", behavior)
        self.rec.answer(st, "after_s", when)
        if self.auto:
            self.twin.control(rc_loss=0.0)
        else:
            io.enter("turn Wi-Fi back ON, wait for the drone network to rejoin, press Enter")
            self.rec.answer(st, "rejoined_by_itself", io.yn("did the Mac rejoin the drone network by itself?"))
        s = self.be.wait(lambda x: True, 10) or {}
        self.rec.measure(st, "flags_after", s.get("safety_flags"))
        self.rec.measure(st, "flight_state_after", s.get("flight_state"))
        io.say(f"governor after the outage: {s.get('flight_state')} {s.get('safety_flags')}")
        self.ensure_ground_and_armed(st)

    def step_linkloss_rcpause(self, st: str) -> None:
        io = self.io
        secs = float(self.a.link_s)
        if self.auto:
            io.do(f"link loss 3 of 3 (twin): AIRBORNE, then POST /debug/rc_pause {{\"seconds\": {secs:.0f}}}: control "
                  "packets stop, the 1 Hz keepalive continues")
            if not self.takeoff(st):
                return
            self.be.wait(lambda x: x["flight_state"] == "AIRBORNE", 8)
        else:
            io.say("REAL mode allows /debug/rc_pause only while the governor is ON_GROUND (props-off bench test,",
                   "at most 15 s), so the motors are idle here: this shows whether the drone tolerates keepalive-only",
                   "traffic (video, LEDs) and still takes off right after. LAND / E-STOP / TAKEOFF end the pause at once.")
            self.ensure_ground_and_armed(st)
            secs = min(secs, 15.0)
        tw0 = self.twin.status()["rc"] if self.twin else None
        io.enter(f"press Enter to pause RC control packets for {secs:.0f} s")
        r = self.be.post("/debug/rc_pause", {"seconds": secs})
        self.rec.measure(st, "rc_pause_response", {k: r.get(k) for k in ("ok", "reason", "seconds", "flight_state",
                                                                          "flight_mode")})
        if not r.get("ok"):
            io.warn(f"refused: {r.get('reason')}")
            return
        t0 = time.monotonic()
        timeline = []
        while time.monotonic() - t0 < secs + 2.0:
            row = {"t": round(time.monotonic() - t0, 2)}
            try:
                bs = self.be.state()
                row["rc_paused"] = "rc_paused" in bs.get("safety_flags", [])
            except Exception:  # noqa: BLE001
                pass
            if self.twin:
                ts = self.twin.status()
                row.update({"rc_packets": ts["rc"]["packets"], "keepalives": ts["rc"]["keepalives"],
                            "link_lost": ts["drone"]["link_lost"], "phase": ts["drone"]["phase"],
                            "motors_on": ts["drone"]["motors_on"]})
            timeline.append(row)
            time.sleep(0.5)
        status = httpx.get(self.be.http + "/debug/rc_pause", timeout=3).json()
        self.rec.measure(st, "pause_status", {k: status.get(k) for k in ("end_reason", "dropped", "keepalives_during",
                                                                          "elapsed_s", "rc_sent_during")})
        self.rec.measure(st, "timeline", timeline)
        if self.twin:
            during = [r for r in timeline if 0.3 <= r["t"] <= secs - 0.3]
            pk = [r["rc_packets"] for r in during]
            ka = [r["keepalives"] for r in during]
            lost = next((r["t"] for r in timeline if r.get("link_lost")), None)
            landed = next((r["t"] for r in timeline if r.get("phase") in ("landing", "landed")), None)
            after = timeline[-1]
            checks = {"control_packets_stopped": bool(pk) and max(pk) - min(pk) <= 1,
                      "keepalives_continued": bool(ka) and max(ka) - min(ka) >= max(1, int((secs - 0.6) * 0.7)),
                      "twin_link_loss_fired_s": lost, "twin_landing_s": landed,
                      "packets_resumed": after["rc_packets"] > (pk[-1] if pk else 0) + 10}
            self.rec.measure(st, "twin_checks", checks)
            io.say(f"twin: {checks}")
            self.rec.answer(st, "motors", "landed (twin loss_action land)" if landed is not None else "kept spinning")
            self.rec.answer(st, "after_s", landed)
            self.land_and_confirm(st)
        else:
            self.rec.answer(st, "video_kept_streaming", io.yn("did the UI video keep streaming during the pause?"))
            self.rec.answer(st, "led_change", io.text("any LED change on the drone? (Enter = none)"))
            io.enter("pause over: press Enter to TAKEOFF once more (motors should spin; LAND follows)")
            if self.takeoff(st):
                io.countdown(min(4.0, self.a.motor_s), "motors running")
                self.rec.answer(st, "takeoff_after_pause_spun", io.yn("did the motors spin?"))
                self.land_and_confirm(st)

    def step_remote(self, st: str) -> None:
        io = self.io
        if self.auto:
            self.rec.answer(st, "skipped", "no physical remote in the twin")
            io.say("[auto] skipped: no physical remote in the twin")
            return
        io.do("remote OFF to start. TAKEOFF from the backend; while the motors spin, switch the remote ON")
        io.enter("press Enter to TAKEOFF")
        if not self.takeoff(st):
            return
        io.countdown(2.0, "motors running")
        self.rec.answer(st, "remote_on_changed_anything", io.yn("switch the remote ON now: did anything change?"))
        self.rec.answer(st, "remote_stop_cut_motors", io.yn("press the remote's stop / emergency button: did the motors stop?"))
        self.rec.answer(st, "remote_button", io.text("which remote control stops the motors? (Enter = none)"))
        self.land_and_confirm(st)
        io.say("if the remote can stop the motors, keep it in hand during every flight (PLAN 6, solo rules)")

    def step_weight(self, st: str) -> None:
        io = self.io
        if self.auto:
            io.say("[auto] skipped: no scale in the twin")
            return
        self.rec.answer(st, "weight_g", io.num("drone weight WITH battery", "g", lo=50, hi=1000))
        self.rec.answer(st, "low_battery_led", io.text("low-battery LED behavior (Enter = not seen yet)"))
        io.say("the flight timer warns at 5:00 and lands at 6:30 until the real endurance is known")

    # ============================================================================== summary
    def summary(self) -> dict:
        g = lambda step, key, kind="answers": self.rec.get(step, kind, key)  # noqa: E731
        rows = []

        def gate(name, ok, detail, hard=True):
            rows.append({"gate": name, "result": "PASS" if ok is True else ("FAIL" if ok is False else "UNKNOWN"),
                         "detail": detail, "hard": hard})
        pf = self.rec.data["steps"].get("preflight", {}).get("measured", {})
        fails = pf.get("fails")
        gate("preflight (routes, Local Network, internet)", None if fails is None else not fails,
             "fails: " + (", ".join(fails) if fails else "none") if fails is not None else "not run")
        gate("video stream", g("ffprobe", "ok", "measured"), f"{g('ffprobe', 'stream', 'measured')} "
             f"{g('ffprobe', 'fps', 'measured')} fps, GOP {g('ffprobe', 'keyframe_interval_frames', 'measured')}")
        gate("takeoff spins the motors", g("flag_takeoff_land", "motors_spun"), f"arm step: {g('flag_takeoff_land', 'arm_step')}")
        land = g("flag_takeoff_land", "land_stopped_motors")
        gate("land (0x80) stops the motors", land,
             "if FAIL: Session B uses the throttle-ramp landing (apply_sysid --throttle-ramp on)", hard=False)
        gate("E-stop cuts at once and stays latched",
             (g("flag_estop", "estop_cut_motors") and g("flag_estop", "estop_latched")) if g("flag_estop", "estop_cut_motors") is not None else None,
             f"cut {g('flag_estop', 'estop_cut_motors')}, latched {g('flag_estop', 'estop_latched')}")
        gate("land on the ground is a no-op", g("flag_land_ground", "land_on_ground_noop"), "", hard=False)
        stops = []
        for step in ("linkloss_kill", "linkloss_wifi"):
            m = g(step, "motors")
            stops.append(None if m is None else m != "kept spinning")
        remote = g("remote", "remote_stop_cut_motors")
        ll_ok = None if None in stops else all(stops)
        gate("link loss stops the drone (kill -9, Wi-Fi off)", ll_ok,
             f"kill: {g('linkloss_kill', 'motors')} after {g('linkloss_kill', 'after_s')} s; "
             f"wifi: {g('linkloss_wifi', 'motors')} after {g('linkloss_wifi', 'after_s')} s")
        gate("physical remote can stop the motors", remote, g("remote", "remote_button") or "", hard=False)
        lat = g("latency", "result", "measured") or {}
        manual = g("latency", "manual_latency_ms")
        lat_ms = (lat.get("latency_ms") or {}).get("p50", manual)
        gate("latency measured", lat_ms is not None, f"p50 {lat_ms} ms", hard=False)
        hfov = g("hfov", "hfov_deg", "measured")
        gate("HFOV measured", hfov is not None, f"{hfov} deg", hard=False)

        hard_fail = [r for r in rows if r["hard"] and r["result"] == "FAIL"]
        unknown = [r for r in rows if r["hard"] and r["result"] == "UNKNOWN"]
        if any(r["gate"].startswith("link loss") for r in hard_fail) and remote is True and len(hard_fail) == 1:
            verdict = "GO WITH CONDITIONS: link loss does not stop the drone; keep the physical remote in hand"
        elif hard_fail:
            verdict = "NO-GO: " + "; ".join(r["gate"] for r in hard_fail)
        elif unknown:
            verdict = "INCOMPLETE: " + "; ".join(r["gate"] for r in unknown)
        else:
            verdict = "GO for Session B (props on)"
            if land is False:
                verdict += ", with the throttle-ramp landing ON"
        carry = {}
        if lat.get("latency_ms"):
            carry["MHD_CAPTURE_OFFSET_MS"] = int(lat["latency_ms"]["p05"])
            carry["latency_p50_ms"] = lat["latency_ms"]["p50"]
        elif manual is not None:
            carry["latency_p50_ms"] = manual
        if hfov:
            carry["hfov_deg"] = hfov
        if land is False:
            carry["landing.throttle_ramp"] = True
        for step in ("linkloss_kill", "linkloss_wifi", "linkloss_rcpause"):
            if g(step, "motors") is not None:
                carry[f"{step}"] = f"{g(step, 'motors')} after {g(step, 'after_s')} s"
        out = {"verdict": verdict, "gates": rows, "carry_forward": carry}
        self.rec.data["summary"] = out
        self.rec.save()
        return out

    def print_summary(self, s: dict) -> None:
        io = self.io
        print("\n" + io._col("1;36", "=== Session A summary " + "=" * 50))
        for r in s["gates"]:
            col = {"PASS": "32", "FAIL": "1;31", "UNKNOWN": "33"}[r["result"]]
            print(f"  {io._col(col, r['result'].ljust(8))} {'(hard) ' if r['hard'] else '       '}{r['gate']}: {r['detail']}")
        print()
        col = "1;32" if s["verdict"].startswith("GO") else "1;31"
        print("  " + io._col(col, s["verdict"]))
        if s["carry_forward"]:
            print("\n  carry into Session B:")
            for k, v in s["carry_forward"].items():
                print(f"    {k}: {v}")
            if "MHD_CAPTURE_OFFSET_MS" in s["carry_forward"]:
                print(f"\n    export MHD_CAPTURE_OFFSET_MS={s['carry_forward']['MHD_CAPTURE_OFFSET_MS']}   "
                      "# before starting the backend for Session B")
            if "hfov_deg" in s["carry_forward"]:
                lat = s["carry_forward"].get("latency_p50_ms")
                print(f"    $MHD_PY scripts/apply_sysid.py --hfov-deg {s['carry_forward']['hfov_deg']}"
                      + (f" --latency-ms {lat}" if lat is not None else "") + "   # shows the diff; add --write")
        print(f"\n  results: {self.rec.path}")


STEPS = [
    ("safety", "Before you start (props off)", "step_safety"),
    ("props", "Mark, photograph and remove the props", "step_props"),
    ("preflight", "Tether, routes, Local Network permission (preflight)", "step_preflight"),
    ("ffprobe", "RTSP stream format: fps, keyframes, bitrate, one client?", "step_ffprobe"),
    ("recordings", "Recordings: raw RTSP, subjects clip, mic (backend stopped if one client)", "step_recordings"),
    ("backend", "Start the backend and the UI, confirm video", "step_backend"),
    ("latency", "Camera-to-backend latency (screen counter)", "step_latency"),
    ("hfov", "HFOV from the wall measurement", "step_hfov"),
    ("flag_takeoff_land", "Flags: takeoff spins the motors, land stops them, arm step?", "step_flag_takeoff_land"),
    ("flag_estop", "Flags: E-stop once, then latched", "step_flag_estop"),
    ("flag_land_ground", "Flags: land on the ground is a no-op", "step_flag_land_ground"),
    ("flag_tilt", "Flags: camera tilt travel per press", "step_flag_tilt"),
    ("linkloss_kill", "Link loss 1: kill -9 the backend", "step_linkloss_kill"),
    ("linkloss_wifi", "Link loss 2: the Mac's Wi-Fi off", "step_linkloss_wifi"),
    ("linkloss_rcpause", "Link loss 3: RC packets stop, keepalive continues", "step_linkloss_rcpause"),
    ("remote", "The physical remote", "step_remote"),
    ("weight", "Weight and the low-battery LED", "step_weight"),
]
ALIASES = {"flags": ["flag_takeoff_land", "flag_estop", "flag_land_ground", "flag_tilt"],
           "linkloss": ["linkloss_kill", "linkloss_wifi", "linkloss_rcpause"]}


def expand(names: str) -> list[str]:
    out = []
    for n in [x.strip() for x in names.split(",") if x.strip()]:
        out += ALIASES.get(n, [n])
    known = {s[0] for s in STEPS}
    bad = [n for n in out if n not in known]
    if bad:
        raise SystemExit(f"unknown step(s) {bad}; see --list")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Session A props-off checklist", epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drone-ip", default=DRONE_IP)
    ap.add_argument("--rtsp-url", default=None, help="default rtsp://<drone-ip>/live/ch00_1")
    ap.add_argument("--backend-port", type=int, default=8000)
    ap.add_argument("--counter-port", type=int, default=8766, help="screen counter page (the twin is off in real mode)")
    ap.add_argument("--only", default="", help="comma list of steps (aliases: flags, linkloss)")
    ap.add_argument("--skip", default="", help="comma list of steps to skip")
    ap.add_argument("--resume", default=None, help="continue a results file; finished steps are skipped")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--motor-s", type=float, default=6.0, help="motor run per flag test before LAND")
    ap.add_argument("--link-s", type=float, default=10.0, help="link-loss duration")
    ap.add_argument("--latency-s", type=float, default=60.0)
    ap.add_argument("--display-ms", type=float, default=0.0, help="subtract this display latency from the counter")
    ap.add_argument("--skip-ml", action="store_true", help="preflight: skip the ml-venv doctor")
    ap.add_argument("--media-dir", default=None)
    ap.add_argument("--auto-twin", action="store_true", help="non-interactive test against the TWIN (loopback only)")
    ap.add_argument("--twin", default="http://127.0.0.1:8766", help="auto-twin: twin HTTP URL")
    ap.add_argument("--rc-port", type=int, default=23458, help="auto-twin: twin UDP RC port for our backend")
    ap.add_argument("--twin-hfov-deg", type=float, default=75.0, help="auto-twin: the twin camera's HFOV")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for name, title, _ in STEPS:
            print(f"{name:20s} {title}")
        print("aliases: " + "; ".join(f"{k} = {','.join(v)}" for k, v in ALIASES.items()))
        return 0
    if a.auto_twin:
        a.drone_ip = "127.0.0.1"
        if a.rtsp_url and not a.rtsp_url.startswith("rtsp://127.0.0.1"):
            raise SystemExit("--auto-twin is loopback only")
        if not a.twin.startswith("http://127.0.0.1"):
            raise SystemExit("--auto-twin: the twin must be on 127.0.0.1")
        a.latency_s = min(a.latency_s, 10.0)
        a.link_s = min(a.link_s, 6.0)
        a.motor_s = min(a.motor_s, 4.0)
    elif os.environ.get("MHD_OPS_LOOPBACK_ONLY") == "1":
        raise SystemExit("MHD_OPS_LOOPBACK_ONLY=1: only --auto-twin may run")
    io = IO(a.auto_twin, sys.stdout.isatty() and not a.no_color)
    resume = json.loads(Path(a.resume).read_text()) if a.resume else None
    path = Path(a.resume) if a.resume else TREE / "logs" / f"session_a-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = Recorder(path, {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "mode": "auto-twin" if a.auto_twin else "real",
                          "drone_ip": a.drone_ip, "argv": sys.argv[1:]}, resume)
    s = Session(a, io, rec)
    names = expand(a.only) if a.only else [x[0] for x in STEPS]
    skip = set(expand(a.skip)) if a.skip else set()
    todo = [x for x in STEPS if x[0] in names and x[0] not in skip]
    print(f"Session A ({'AUTO-TWIN test, loopback only' if a.auto_twin else 'REAL drone ' + a.drone_ip}): "
          f"{len(todo)} steps; results -> {path}")
    code = 0
    try:
        for i, (name, title, fn) in enumerate(todo, 1):
            prev = rec.data["steps"].get(name, {}).get("status")
            if resume and prev in ("done", "skipped"):
                print(f"\n(step {name}: already {prev} in the resumed file)")
                continue
            io.header(i, len(todo), title)
            rec.step(name)["title"] = title
            rec.step(name)["t_start"] = time.strftime("%H:%M:%S")
            try:
                getattr(s, fn)(name)
                rec.status(name, "done")
            except SkipStep:
                rec.status(name, "skipped")
                print("  (skipped)")
            except (httpx.HTTPError, OSError, TimeoutError) as exc:
                rec.status(name, "error", repr(exc)[:300])
                io.warn(f"step failed: {exc!r}")
                if a.auto_twin:
                    code = 1
    except (Quit, KeyboardInterrupt) as exc:
        print(f"\nstopped: {exc or 'Ctrl-C'}; resume with --resume {path}")
        code = 2
    finally:
        try:
            if s.be.up():
                st = s.be.state()
                if st.get("flight_state") in ("TAKING_OFF", "AIRBORNE"):
                    s.be.cmd("land")
                    print("  (sent LAND on exit: the governor was still flying)")
        except Exception:  # noqa: BLE001
            pass
        summary = s.summary()
        s.print_summary(summary)
        s.cleanup()
    if a.auto_twin and code == 0:
        bad = [n for n, v in rec.data["steps"].items() if v.get("status") == "error"]
        code = 1 if bad else 0
    return code


if __name__ == "__main__":
    sys.exit(main())
