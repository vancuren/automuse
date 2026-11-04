"""Truth-driven demo pilot for twin footage (NOT the autopilot, and never for the real drone).

It reads the twin's ground truth (/ws/truth), and flies the twin over its real UDP RC port with
X69 packets: takeoff, then yaw toward the subject and keep a standoff distance. It exists to
produce sample frames for perception (see capture.py) and the chase-view video for the demo.

    $MHD_PY -m mhd.twin.demo_pilot --twin http://127.0.0.1:8766 --rc-port 23458 \
        --subject "person in blue shirt" --duration 40 [--record-chase] [--samples-out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time
from pathlib import Path
from typing import Optional

import httpx

from . import protocol as P
from .capture import TruthBuffer, capture


class DemoPilot:
    def __init__(self, twin: str, rc_port: int, subject: str, standoff_m: float = 3.5) -> None:
        if not twin.startswith(("http://127.0.0.1", "http://localhost")):
            raise SystemExit("the demo pilot only flies a loopback twin")
        self.twin = twin
        self.addr = ("127.0.0.1", rc_port)
        self.subject = subject
        self.standoff = standoff_m
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.truths = TruthBuffer(twin.replace("http://", "ws://") + "/ws/truth")
        self.flags = 0
        self.running = False
        self.th: Optional[threading.Thread] = None
        self.sticks = {"yaw": 128, "pitch": 128, "roll": 128, "throttle": 128}

    def _latest(self) -> Optional[dict]:
        with self.truths.lock:
            if not self.truths.order:
                return None
            return self.truths.by_id.get(self.truths.order[-1])

    @staticmethod
    def _counts(rate: float, per_count: float, deadband: int = 7, limit: int = 90) -> int:
        if abs(rate) < 1e-6:
            return 128
        n = min(limit, deadband + abs(rate) / per_count)
        return int(round(128 + math.copysign(n, rate)))

    def _control(self) -> None:
        tr = self._latest()
        if tr is None:
            return
        a = next((x for x in tr["actors"] if x["label"] == self.subject or x["actor_id"] == self.subject), None)
        if a is None or a.get("box") is None or a.get("visible_frac", 0) < 0.2:
            self.sticks.update(yaw=128, pitch=128)
            return
        bearing = a["bearing_rad"]
        rate_dps = max(-45.0, min(45.0, math.degrees(bearing) * 1.6)) if abs(bearing) > 0.03 else 0.0
        self.sticks["yaw"] = self._counts(rate_dps, 0.95)
        err = a["distance_m"] - self.standoff
        speed = max(-0.25, min(0.35, 0.5 * err)) if abs(err) > 0.25 and abs(bearing) < 0.35 else 0.0
        self.sticks["pitch"] = self._counts(speed, 0.012)

    def _loop(self) -> None:
        while self.running:
            self._control()
            pkt = P.encode_control(roll=self.sticks["roll"], pitch=self.sticks["pitch"],
                                   throttle=self.sticks["throttle"], yaw=self.sticks["yaw"], flags1=self.flags)
            self.flags = 0
            self.sock.sendto(pkt, self.addr)
            time.sleep(0.04)

    def start(self, takeoff: bool = True) -> None:
        self.running = True
        if takeoff:
            self.flags = P.FLAG_TAKEOFF
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def land(self) -> None:
        self.flags = P.FLAG_LAND

    def stop(self) -> None:
        self.running = False
        if self.th:
            self.th.join(timeout=1)
        self.truths.stop = True
        self.sock.close()


def control(twin: str, **kw) -> dict:
    r = httpx.post(f"{twin}/control", json=kw, timeout=5)
    r.raise_for_status()
    return r.json()


def wait_phase(twin: str, phase: str, timeout: float = 10.0) -> bool:
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if httpx.get(f"{twin}/status", timeout=3).json()["drone"]["phase"] == phase:
            return True
        time.sleep(0.1)
    return False


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Truth-driven demo pilot for twin footage")
    ap.add_argument("--twin", default="http://127.0.0.1:8766")
    ap.add_argument("--rc-port", type=int, default=23458)
    ap.add_argument("--scenario", default=None, help="switch the twin to this scenario first")
    ap.add_argument("--subject", default="person in blue shirt")
    ap.add_argument("--standoff-m", type=float, default=3.5)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--record-chase", action="store_true", help="record the chase view (needs a chase_rec page)")
    ap.add_argument("--samples-out", default=None, help="save HEVC-path frames + truth here while flying")
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--every", type=int, default=8)
    a = ap.parse_args(argv)
    if a.scenario:
        control(a.twin, scenario=a.scenario)
        time.sleep(1.0)
    pilot = DemoPilot(a.twin, a.rc_port, a.subject, a.standoff_m)
    pilot.start(takeoff=True)
    result = {}
    try:
        if not wait_phase(a.twin, "flying", 10):
            raise SystemExit("twin did not take off")
        if a.record_chase:
            control(a.twin, record_chase=True)
        t_end = time.monotonic() + a.duration
        if a.samples_out:
            result["capture"] = capture(a.twin, Path(a.samples_out), a.count, a.every, overlay=True,
                                        timeout_s=a.duration + 60)
        while time.monotonic() < t_end:
            time.sleep(0.2)
        if a.record_chase:
            out = control(a.twin, record_chase=False)
            result["chase"] = out["status"].get("chase_recording")
        pilot.land()
        wait_phase(a.twin, "landed", 8)
    finally:
        pilot.stop()
    st = httpx.get(f"{a.twin}/status", timeout=3).json()
    result["status"] = {"scenario": st["scenario"], "fps": st["fps"], "latency": st.get("latency")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
