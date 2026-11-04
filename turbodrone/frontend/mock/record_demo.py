#!/usr/bin/env python3
"""Record a ~35 s screen video of the operator UI against the mock backend (no drone, no twin).

Starts the mock on :8603 and vite on :5273, drives a scripted story in Google Chrome
(voice mission -> follow -> look-alike pick -> doubt -> E-STOP -> link drop), and writes
media/ui_screens/ui_demo_mock.mp4, copied to $MHD_HOME/media/shared/ui/.

Frames come from Chrome's CDP screencast and are encoded by the system ffmpeg
(h264_videotoolbox), so Playwright's own ffmpeg download is not needed.
Run from the repo root after `source env.sh`:

    $MHD_PY turbodrone/frontend/mock/record_demo.py
"""
from __future__ import annotations

import base64
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

TREE = Path(__file__).resolve().parents[3]
HOME = Path(os.environ.get("MHD_HOME") or TREE)
FRONTEND = TREE / "turbodrone" / "frontend"
MOCK = f"http://127.0.0.1:8603"
UI = "http://127.0.0.1:5273/"
OUT = TREE / "media" / "ui_screens"
SHARED = HOME / "media" / "shared" / "ui"
LOGS = TREE / "logs" / "record_demo"


def wait(url: str, timeout: float = 30) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if httpx.get(url, timeout=1).status_code < 500:
                return
        except Exception:
            time.sleep(0.2)
    raise SystemExit(f"{url} did not come up")


def control(**kw) -> None:
    httpx.post(MOCK + "/mock/control", json=kw, timeout=5)


def sel(t: str) -> str:
    return f'[data-testid="{t}"]'


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    SHARED.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, VITE_BACKEND_URL=MOCK, VITE_TWIN_CHASE_URL=MOCK + "/chase")
    mock = subprocess.Popen([sys.executable, str(FRONTEND / "mock" / "mock_backend.py"), "--port", "8603",
                             "--scenario", "manual", "--log", str(LOGS / "mock.jsonl")],
                            stdout=open(LOGS / "mock.out", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    vite = subprocess.Popen([str(FRONTEND / "node_modules" / ".bin" / "vite"), "--port", "5273", "--strictPort"],
                            cwd=FRONTEND, env=env, stdout=open(LOGS / "vite.out", "w"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        wait(MOCK + "/state")
        wait(UI)
        profile = HOME / ".cache" / "chrome-ui"
        frames: list[tuple[float, bytes]] = []
        with sync_playwright() as p:
            ctx = p.chromium.launch_persistent_context(
                str(profile), channel="chrome", headless=True, viewport={"width": 1920, "height": 1080})
            page = ctx.new_page()
            page.goto(UI)
            page.wait_for_selector(f'{sel("takeoff-blocker")} >> text=ready', timeout=20000)
            cdp = ctx.new_cdp_session(page)

            def on_frame(ev: dict) -> None:
                frames.append((ev["metadata"]["timestamp"], base64.b64decode(ev["data"])))
                cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})

            cdp.on("Page.screencastFrame", on_frame)
            cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 85, "maxWidth": 1920, "maxHeight": 1080})
            pause = page.wait_for_timeout  # keeps Playwright's event loop (and frame acks) running

            control(time_offset_s=2.0)
            pause(2000)
            page.click(sel("takeoff-btn"))                       # 1. take off (button only)
            pause(3500)
            page.keyboard.down("Space")                           # 2. "follow the person in the blue shirt"
            pause(2400)
            page.keyboard.up("Space")
            page.wait_for_selector(sel("model-badge"), timeout=6000)
            pause(3500)                                           # FIND -> CONFIRM -> FOLLOW
            control(autopilot_state="CONFIRM", freeze_autopilot=True)
            httpx.post(MOCK + "/mock/candidates", timeout=5)      # 3. look-alikes tie: operator picks one
            pause(2500)
            page.click(sel("candidate-1"))
            pause(5000)                                           # following, sticks steering
            control(autopilot_state="DOUBT")                      # 4. tracker unsure: hold position
            pause(3000)
            control(autopilot_state="FOLLOW")
            pause(3000)
            page.keyboard.press("Escape")                         # 5. E-STOP (Esc), latched
            pause(3000)
            control(ws_mode="down")                               # 6. control link drops: banner, HTTP fallback
            pause(2000)
            page.keyboard.press("KeyL")
            pause(2000)
            control(ws_mode="normal")
            pause(3000)
            cdp.send("Page.stopScreencast")
            page.close()
            ctx.close()
        if not frames:
            raise SystemExit("no screencast frames")
        # Screencast frames arrive only on repaint: resample to a constant 25 fps.
        fps, t0, t_end = 25, frames[0][0], frames[-1][0]
        mp4 = OUT / "ui_demo_mock.mp4"
        enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg",
                                "-r", str(fps), "-i", "-", "-c:v", "h264_videotoolbox", "-b:v", "8M",
                                "-pix_fmt", "yuv420p", str(mp4)], stdin=subprocess.PIPE)
        i, n_out = 0, int((t_end - t0) * fps) + 1
        for k in range(n_out):
            t = t0 + k / fps
            while i + 1 < len(frames) and frames[i + 1][0] <= t:
                i += 1
            enc.stdin.write(frames[i][1])
        enc.stdin.close()
        enc.wait(timeout=120)
        shutil.copy2(mp4, SHARED / mp4.name)
        print(f"wrote {mp4} ({n_out} frames at {fps} fps from {len(frames)} screencast frames, {t_end - t0:.1f} s)")
    finally:
        for proc in (vite, mock):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception:
                os.killpg(proc.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
