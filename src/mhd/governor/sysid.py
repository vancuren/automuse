"""Session B system identification (PLAN.md 6): step one axis at several amplitudes per sign,
measure the response on an ArUco marker, and fit K (rate per wire count), firmware deadband and lag.

    # drone AIRBORNE and hovering, ArUco DICT_4X4_50 marker on the wall in view, UI open (heartbeat)
    $MHD_PY -m mhd.governor.sysid --backend http://127.0.0.1:8000 --axis yaw --amps 0.10 0.20 0.35
    $MHD_PY -m mhd.governor.sysid --backend http://127.0.0.1:8000 --axis pitch --amps 0.10 0.20 0.25 --hold 1.5

Safety:
- Steps are ordinary human stick input on /ws ("abs" mode), so the governor applies the human caps,
  and LAND / E-STOP from the UI always win (they have priority over human sticks).
- Refuses to start unless the backend reports AIRBORNE. Aborts the moment the backend is not
  AIRBORNE, or (0.3 s into a step) the governor is not flying our sticks (active_source is not
  "human"), or, in real mode, the UI heartbeat is older than 1 s. Ctrl-C releases the sticks.
  Amplitudes above 0.5 are refused.
- Sends no heartbeat of its own, and in real mode its axes do not count as one: only the UI
  worker's {"type":"hb","client":"ui"} does. If the UI dies, the governor goes neutral at 1.5 s and
  lands at 5 s as in normal flight. (In twin mode any /ws message counts, for the stock UI.)

- Each amplitude is flown + then - so the drone ends near where it started.

Output: $MHD_LOGS/sysid-<axis>-<time>.json with the step schedule (monotonic ms, aligned with
control-*.jsonl), every marker sample (capture_ms, bearing, range) and the fit.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from mhd.contracts import now_ms

from .config import load_autopilot_config, logs_dir

FULL_SCALE = 127


@dataclass
class Step:
    axis: str
    value: float          # normalized stick
    hold_s: float
    rest_s: float
    t_start_ms: float = 0.0
    t_end_ms: float = 0.0
    wire_count: Optional[int] = None


def build_schedule(axis: str, amps: list[float], hold_s: float, rest_s: float) -> list[Step]:
    if axis not in ("yaw", "pitch", "roll", "throttle"):
        raise ValueError("axis must be yaw, pitch, roll or throttle")
    steps = []
    for a in amps:
        if not 0.0 < a <= 0.5:
            raise ValueError(f"amplitude {a} outside (0, 0.5]")
        steps.append(Step(axis, +a, hold_s, rest_s))
        steps.append(Step(axis, -a, hold_s, rest_s))
    return steps


# ----------------------------------------------------------------------------------------------
# ArUco measurement


def aruco_detector():
    import cv2
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())


def aruco_measure(img_bgr, hfov_deg: float, marker_size_m: float, detector=None,
                  marker_id: Optional[int] = None) -> Optional[tuple[float, float]]:
    """(bearing_rad, range_m) of the marker center, or None. Pinhole model with the given HFOV."""
    import cv2
    det = detector or aruco_detector()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    idx = 0
    if marker_id is not None:
        hits = [i for i, v in enumerate(ids.flatten()) if int(v) == marker_id]
        if not hits:
            return None
        idx = hits[0]
    c = corners[idx].reshape(4, 2)
    h, w = gray.shape[:2]
    f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    cx = float(c[:, 0].mean())
    side_px = float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))
    bearing = math.atan((cx - w / 2.0) / f)
    rng = f * marker_size_m / side_px if side_px > 1 else float("nan")
    return bearing, rng


# ----------------------------------------------------------------------------------------------
# Fit


def fit_steps(steps: list[Step], samples: list[tuple[float, float, float]], axis: str,
              settle_s: float = 0.5) -> dict:
    """samples: (capture_ms, bearing_rad, range_m). Yaw rate = -d(bearing)/dt (turning right moves
    a fixed marker left); forward speed = -d(range)/dt. Rate is fitted over the steady part of each
    hold (after settle_s); onset lag = first time the measure leaves its pre-step noise band."""
    out = []
    t = np.array([s[0] for s in samples]) if samples else np.array([])
    for st in steps:
        sel = (t >= st.t_start_ms + settle_s * 1000.0) & (t <= st.t_end_ms)
        rec = {"value": st.value, "wire_count": st.wire_count, "rate": None, "lag_s": None, "n": int(sel.sum())}
        if sel.sum() >= 3:
            tt = t[sel] / 1000.0
            yy = np.array([samples[i][1] if axis == "yaw" else samples[i][2] for i in np.flatnonzero(sel)])
            ok = np.isfinite(yy)
            if ok.sum() >= 3:
                slope = float(np.polyfit(tt[ok], yy[ok], 1)[0])
                rec["rate"] = -slope
            pre = (t >= st.t_start_ms - 500.0) & (t < st.t_start_ms)
            during = (t >= st.t_start_ms) & (t <= st.t_end_ms)
            if pre.sum() >= 2 and during.sum() >= 2:
                col = 1 if axis == "yaw" else 2
                base = np.array([samples[i][col] for i in np.flatnonzero(pre)])
                band = max(3 * float(np.std(base)), 0.01 if axis == "yaw" else 0.02)
                ref = float(np.mean(base))
                for i in np.flatnonzero(during):
                    if abs(samples[i][col] - ref) > band:
                        rec["lag_s"] = (samples[i][0] - st.t_start_ms) / 1000.0
                        break
        out.append(rec)
    # |rate| = K * (|counts| - deadband) for |counts| > deadband: a line through the step points.
    pts = [(abs(r["wire_count"] if r["wire_count"] is not None else round(r["value"] * FULL_SCALE)), abs(r["rate"]))
           for r in out if r["rate"] is not None]
    fit = {"steps": out, "k_per_count": None, "deadband_counts": None, "lag_s_median": None}
    if len(pts) >= 2 and len({p[0] for p in pts}) >= 2:
        x = np.array([p[0] for p in pts], float)
        y = np.array([p[1] for p in pts], float)
        k, b = np.polyfit(x, y, 1)
        fit["k_per_count"] = float(k)
        fit["deadband_counts"] = float(max(0.0, -b / k)) if k > 0 else None
    lags = [r["lag_s"] for r in out if r["lag_s"] is not None]
    if lags:
        fit["lag_s_median"] = float(np.median(lags))
    return fit


# ----------------------------------------------------------------------------------------------
# Live run


def _abort_reason(s: dict, st: Step, axis: str, in_step_s: float) -> Optional[str]:
    """Why the step must stop now (None: carry on). Records the wire count while ours."""
    if s.get("flight_state") != "AIRBORNE":
        return f"flight state {s.get('flight_state')}"
    hb = (s.get("latency_ms") or {}).get("hb_age")
    if s.get("flight_mode") == "real" and (hb is None or hb > 1000.0):
        return f"UI heartbeat {hb} ms old: is the UI open?"
    if s.get("active_source") == "human":
        st.wire_count = int(s["sticks_wire"][axis]) - 128
    elif in_step_s > 0.3:
        return f"governor is not flying our sticks (active_source={s.get('active_source')})"
    return None


def run(backend: str, axis: str, amps: list[float], hold_s: float, rest_s: float, marker_size_m: float,

        hfov_deg: Optional[float] = None, marker_id: Optional[int] = None, measure: bool = True) -> dict:
    import httpx
    import websockets.sync.client as wsc

    from mhd.contracts import decode_frame

    base = backend.rstrip("/")
    ws_base = "ws" + base[4:] if base.startswith("http") else base
    hfov = hfov_deg or load_autopilot_config().camera.hfov_deg
    state = httpx.get(base + "/state", timeout=3.0).json()
    if state.get("flight_state") != "AIRBORNE":
        raise SystemExit(f"refusing: backend reports {state.get('flight_state')}, need AIRBORNE")
    steps = build_schedule(axis, amps, hold_s, rest_s)
    samples: list[tuple[float, float, float]] = []
    stop = threading.Event()

    def frames():
        det = aruco_detector()
        import cv2
        with wsc.connect(ws_base + "/ws/frames", max_size=16 * 1024 * 1024) as fws:
            last = -1
            while not stop.is_set():
                fws.send(json.dumps({"type": "next", "after_frame_id": last}))
                meta, jpeg = decode_frame(fws.recv(timeout=3))
                last = meta.frame_id
                img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                m = aruco_measure(img, hfov, marker_size_m, det, marker_id) if img is not None else None
                if m is not None:
                    samples.append((meta.capture_ms, m[0], m[1]))

    th = threading.Thread(target=frames, daemon=True) if measure else None
    if th:
        th.start()
    aborted = None
    try:
        with wsc.connect(ws_base + "/ws") as ws:
            time.sleep(1.0)                                    # pre-step baseline
            for st in steps:
                print(f"[sysid] {axis} {st.value:+.2f} for {st.hold_s:.1f} s")
                st.t_start_ms = now_ms()
                began = time.monotonic()
                end = began + st.hold_s
                checked = 0.0
                while time.monotonic() < end:
                    ws.send(json.dumps({"type": "axes", "mode": "abs", axis: st.value}))
                    if time.monotonic() - checked > 0.2:
                        checked = time.monotonic()
                        aborted = _abort_reason(httpx.get(base + "/state", timeout=1.0).json(), st, axis,
                                                in_step_s=checked - began)
                        if aborted:
                            break
                    time.sleep(1 / 30)

                ws.send(json.dumps({"type": "axes", "mode": "abs", axis: 0.0}))   # release now
                st.t_end_ms = now_ms()
                if aborted:
                    break
                time.sleep(st.rest_s)
    except KeyboardInterrupt:
        aborted = "ctrl-c"
    finally:
        stop.set()
    if aborted:
        print(f"[sysid] ABORTED: {aborted}")
    fit = fit_steps([s for s in steps if s.t_end_ms], samples, axis)
    result = {"axis": axis, "amps": amps, "hold_s": hold_s, "rest_s": rest_s, "hfov_deg": hfov,
              "marker_size_m": marker_size_m, "aborted": aborted, "schedule": [asdict(s) for s in steps],
              "samples": samples, "fit": fit}
    path = logs_dir() / f"sysid-{axis}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    print(f"[sysid] fit: K={fit['k_per_count']} per count, deadband={fit['deadband_counts']} counts, "
          f"lag={fit['lag_s_median']} s -> {path}")
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Session B step identification through the governor")
    ap.add_argument("--backend", default="http://127.0.0.1:8000")
    ap.add_argument("--axis", default="yaw", choices=["yaw", "pitch", "roll", "throttle"])
    ap.add_argument("--amps", type=float, nargs="+", default=[0.10, 0.20, 0.35])
    ap.add_argument("--hold", type=float, default=2.0, help="seconds per step")
    ap.add_argument("--rest", type=float, default=3.0, help="seconds of neutral between steps")
    ap.add_argument("--marker-size", type=float, default=0.15, help="ArUco marker side, metres")
    ap.add_argument("--marker-id", type=int, default=None)
    ap.add_argument("--hfov", type=float, default=None, help="default: config/autopilot.yaml")
    ap.add_argument("--no-measure", action="store_true", help="steps only (log counts, no ArUco)")
    a = ap.parse_args(argv)
    run(a.backend, a.axis, a.amps, a.hold, a.rest, a.marker_size, a.hfov, a.marker_id, not a.no_measure)


if __name__ == "__main__":
    main()
