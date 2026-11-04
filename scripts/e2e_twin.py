"""End-to-end follow test in the wire-level twin.

Starts the twin (Chrome renderer, HEVC/RTSP), the governed backend, and perception, then:
takeoff -> Mission "follow <subject>" -> FOLLOW for N seconds -> land -> confirm touchdown,
recording telemetry, and scores the run against the twin's ground truth (mhd.twin.metrics).

    source env.sh
    $MHD_PY scripts/e2e_twin.py --scenario s1 --subject "person in blue shirt" --latency-ms 500
    $MHD_PY scripts/e2e_twin.py --perception oracle          # ground-truth perception, no GPU
    $MHD_PY scripts/e2e_twin.py --perception sam_api         # hosted SAM 3.1 (paid)

Twin mode only: loopback addresses, FLIGHT_MODE=twin, the governor's twin-only headless flag.
Everything it starts runs in its own process group and is stopped at the end (also on Ctrl-C).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import websockets

TREE = Path(os.environ["MHD_TREE"])
PY, ML_PY = os.environ["MHD_PY"], os.environ["MHD_ML_PY"]
BACKEND, TWIN, RC, RTSP = 8000, 8766, 23458, 8554


class Proc:
    def __init__(self, name: str, cmd: list[str], log: Path, env: dict | None = None):
        self.name, self.log = name, log
        e = dict(os.environ)
        e.update(env or {})
        self.f = open(log, "w")
        self.p = subprocess.Popen(cmd, stdout=self.f, stderr=subprocess.STDOUT, env=e, cwd=TREE,
                                  start_new_session=True)

    def alive(self) -> bool:
        return self.p.poll() is None

    def stop(self) -> None:
        if not self.alive():
            return
        for sig, wait in ((signal.SIGINT, 6), (signal.SIGTERM, 4), (signal.SIGKILL, 2)):
            try:
                os.killpg(self.p.pid, sig)
            except ProcessLookupError:
                return
            try:
                self.p.wait(wait)
                return
            except subprocess.TimeoutExpired:
                continue


def port_free(port: int) -> bool:
    import socket
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_for(pred, timeout: float, what: str, interval: float = 0.5, procs: list[Proc] = ()):
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


async def send_cmd(command: str) -> None:
    async with websockets.connect(f"ws://127.0.0.1:{BACKEND}/ws") as ws:
        await ws.send(json.dumps({"type": "cmd", "command": command}))
        await asyncio.sleep(0.3)


def post_mission(subject: str, category: str, color: str) -> dict:
    m = {"type": "mission", "mission_id": f"e2e-{uuid.uuid4().hex[:6]}",
         "intent": {"action": "follow", "subject_phrase": subject, "category": category, "color": color,
                    "distance": "medium", "reason_code": "none"},
         "transcript": f"follow the {subject}", "model": "e2e-script", "latency_ms": 0.0}
    r = httpx.post(f"http://127.0.0.1:{BACKEND}/mission", json=m, timeout=5)
    return {"status": r.status_code, "body": r.text[:300]}


async def run(args) -> dict:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = TREE / "logs" / f"e2e-{stamp}-{args.scenario}-{args.perception}"
    out.mkdir(parents=True, exist_ok=True)
    busy = [p for p in (BACKEND, TWIN, RTSP) if not port_free(p)]
    if busy:
        raise SystemExit(f"ports in use: {busy}; stop the other stack first")
    procs: list[Proc] = []
    result: dict = {"run_dir": str(out), "scenario": args.scenario, "perception": args.perception,
                    "latency_ms": args.latency_ms, "subject": args.subject,
                    "governor_config": args.governor_config, "autopilot_config": args.autopilot_config}
    stop = asyncio.Event()
    tele_task = None
    try:
        twin = Proc("twin", ["bash", "scripts/run_twin.sh", "--scenario", args.scenario,
                             "--latency-ms", str(args.latency_ms)] + (["--chase-rec"] if args.chase_rec else []),
                    out / "twin.log")
        procs.append(twin)
        st = wait_for(lambda: (lambda s: s if s.get("fps", 0) >= 15 else None)(twin_status()), 90,
                      "twin rendering at 15 fps or more", procs=procs)
        result["twin_renderer"] = st.get("renderer")
        backend = Proc("backend", ["bash", "scripts/run_backend.sh", "--mode", "twin", "--port", str(BACKEND),
                                   "--control-port", str(RC), "--rtsp-url", f"rtsp://127.0.0.1:{RTSP}/live/ch00_1",
                                   "--headless", "--logs", str(out / "backend"),
                                   "--governor-config", args.governor_config,
                                   "--autopilot-config", args.autopilot_config], out / "backend.log")
        procs.append(backend)
        wait_for(lambda: (lambda s: s if (s.get("frame_age_ms") or 1e9) < 3000 else None)(state()), 60,
                 "backend receiving video", procs=procs)
        if args.perception == "oracle":
            perc = Proc("perception", [PY, "-m", "mhd.twin.oracle", "--twin", f"http://127.0.0.1:{TWIN}",
                                       "--backend", f"ws://127.0.0.1:{BACKEND}", "--latency-ms", "300",
                                       "--log", str(out / "obs.jsonl")], out / "perception.log")
        else:
            perc = Proc("perception", [ML_PY, "-m", "mhd_ml.perception.service", "--backend",
                                       f"ws://127.0.0.1:{BACKEND}", "--detector", args.perception,
                                       "--tracker", "auto", "--hfov-deg", str(args.hfov_deg),
                                       "--obs-log", str(out / "obs.jsonl"), "--log-level", "INFO"],
                        out / "perception.log")
        procs.append(perc)
        time.sleep(args.perception_warmup_s)
        tele_task = asyncio.create_task(record_telemetry(out / "telemetry.jsonl", stop))

        t_takeoff = time.monotonic()
        await send_cmd("takeoff")
        await asyncio.to_thread(wait_for, lambda: state()["flight_state"] == "AIRBORNE", 20, "AIRBORNE",
                                0.3, procs)
        result["takeoff_to_airborne_s"] = round(time.monotonic() - t_takeoff, 2)
        await asyncio.sleep(0.5)
        t_mission = time.monotonic()
        result["mission_post"] = post_mission(args.subject, args.category, args.color)
        try:
            await asyncio.to_thread(wait_for, lambda: state()["autopilot_state"] == "FOLLOW", args.find_timeout_s,
                                    "autopilot FOLLOW", 0.3, procs)
            result["mission_to_follow_s"] = round(time.monotonic() - t_mission, 2)
        except TimeoutError as e:
            result["mission_to_follow_s"] = None
            result["find_error"] = str(e)
        states: dict[str, int] = {}
        t_end = time.monotonic() + args.follow_s
        while time.monotonic() < t_end:
            s = await asyncio.to_thread(state)
            states[s["autopilot_state"]] = states.get(s["autopilot_state"], 0) + 1
            for p in procs:
                if not p.alive():
                    raise RuntimeError(f"{p.name} died during the follow phase; see {p.log}")
            await asyncio.sleep(0.5)
        result["autopilot_state_samples"] = states
        await send_cmd("land")
        await asyncio.to_thread(wait_for, lambda: twin_status()["drone"]["phase"] == "landed", 20, "landed", 0.3, procs)
        await send_cmd("confirm_touchdown")
        await asyncio.to_thread(wait_for, lambda: state()["flight_state"] == "ON_GROUND", 5, "ON_GROUND", 0.3, procs)
        st = twin_status()
        result["twin_latency"] = st.get("latency")
        result["rc_link"] = {k: st.get("rc", {}).get(k) for k in ("interarrival_p50_ms", "interarrival_p99_ms",
                                                                  "interarrival_max_ms", "p99_alarm", "errors")}
        log_dir = st.get("log_dir") or st.get("logs") or ""
        result["twin_log_dir"] = log_dir
    finally:
        stop.set()
        if tele_task:
            try:
                await asyncio.wait_for(tele_task, 3)
            except Exception:
                pass
        for p in reversed(procs):
            p.stop()
    truth = Path(result.get("twin_log_dir") or "") / "truth.jsonl"
    if truth.exists():
        rep = out / "report.json"
        cmd = [PY, "-m", "mhd.twin.metrics", "--truth", str(truth), "--obs", str(out / "obs.jsonl"),
               "--telemetry", str(out / "telemetry.jsonl"), "--subject", args.subject,
               "--hfov-deg", str(args.hfov_deg), "--out", str(rep), "--csv", str(out / "rows.csv")]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=TREE)
        result["metrics_rc"] = r.returncode
        if rep.exists():
            result["metrics"] = json.loads(rep.read_text())
        else:
            result["metrics_error"] = (r.stderr or r.stdout)[-1500:]
    else:
        result["metrics_error"] = f"no truth log found at {truth}"
    (out / "e2e_result.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="s1")
    ap.add_argument("--subject", default="person in blue shirt")
    ap.add_argument("--category", default="person")
    ap.add_argument("--color", default="blue")
    ap.add_argument("--latency-ms", type=int, default=500)
    ap.add_argument("--perception", choices=["sam_mlx", "sam_api", "oracle"], default="sam_mlx")
    ap.add_argument("--hfov-deg", type=float, default=75.0)
    ap.add_argument("--follow-s", type=float, default=60.0)
    ap.add_argument("--find-timeout-s", type=float, default=40.0)
    ap.add_argument("--perception-warmup-s", type=float, default=20.0)
    ap.add_argument("--chase-rec", action="store_true", help="also record the chase view MP4")
    ap.add_argument("--governor-config", default=str(TREE / "config" / "twin" / "governor.yaml"),
                    help="twin-calibrated by default (the twin's deadbands and caps)")
    ap.add_argument("--autopilot-config", default=str(TREE / "config" / "twin" / "autopilot.yaml"),
                    help="twin-calibrated by default (the twin's yaw rate per count)")
    args = ap.parse_args()
    if os.environ.get("FLIGHT_MODE") == "real":
        raise SystemExit("e2e_twin.py is twin-only")
    res = asyncio.run(run(args))
    m = res.get("metrics") or {}
    print(json.dumps({k: res.get(k) for k in ("run_dir", "scenario", "perception", "latency_ms",
                                              "takeoff_to_airborne_s", "mission_to_follow_s",
                                              "autopilot_state_samples", "rc_link", "find_error",
                                              "metrics_error")}, indent=1))
    print(json.dumps(m.get("summary", m), indent=1)[:3000])


if __name__ == "__main__":
    main()
