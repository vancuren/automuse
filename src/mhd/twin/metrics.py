"""Twin metrics: join ground truth with perception and telemetry logs by render id.

    $MHD_PY -m mhd.twin.metrics --truth logs/twin-.../truth.jsonl --obs obs.jsonl \
        [--telemetry telemetry.jsonl] --subject "person in blue shirt" [--out report.json] [--csv joined.csv]

Inputs are JSONL. Truth lines are mhd.contracts.Truth; obs lines are SubjectObs (they carry
render_id in the twin); telemetry lines are Telemetry (joined by time to the frame's render_ms).
A single mixed log works too: lines are routed by their "type" field.

Definitions (frame = one Truth record):
- subject: the truth actor whose label (or actor_id) is --subject.
- matched: the observation for that frame has a box with IoU >= --iou with the subject's truth box.
- identity: the truth actor with the highest IoU with the observation box (>= --iou), or None.
- time_to_find_s: from t0 (first observation, or --t0-ms) to the first confirmed, matched observation.
- lock_rate: matched frames / frames where the subject is visible (visible_frac >= --min-visible)
  and an observation exists after time-to-find; also lock_rate_all over every visible frame after t0.
- bearing_rms_rad (control quality): RMS of the subject's truth bearing while locked.
- perception_bearing_err_rms_rad: RMS of atan((cx_obs-640)/f) - atan((cx_truth-640)/f) on matched frames.
- id_switches: changes of identity between consecutive identified frames, plus track_id changes.
- settling_time_s: from the first lock until |truth bearing| <= --settle-tol and stays there >= 2 s.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Optional

from mhd.contracts import FRAME_W, Box


def load_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def split_mixed(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    truth, obs, tel = [], [], []
    for r in records:
        t = r.get("type")
        if t == "truth":
            truth.append(r)
        elif t == "subject_obs":
            obs.append(r)
        elif t == "telemetry":
            tel.append(r)
    return truth, obs, tel


def _box(d: Optional[dict]) -> Optional[Box]:
    if not d:
        return None
    return Box(**{k: d[k] for k in ("x1", "y1", "x2", "y2")})


def _bearing_px(cx: float, hfov_deg: float) -> float:
    f = (FRAME_W / 2) / math.tan(math.radians(hfov_deg) / 2)
    return math.atan((cx - FRAME_W / 2) / f)


def _rms(vals: list[float]) -> Optional[float]:
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else None


def compute(truth: list[dict], obs: list[dict], telemetry: Optional[list[dict]] = None, *,
            subject: str, t0_ms: Optional[float] = None, iou_thr: float = 0.3, min_visible: float = 0.5,
            settle_tol: float = 0.1, settle_hold_s: float = 2.0, hfov_deg: float = 75.0,
            follow_only: bool = False) -> dict:
    truth = sorted(truth, key=lambda r: r["render_id"])
    obs_by_rid: dict[int, dict] = {}
    for o in obs:
        rid = o.get("render_id")
        if rid is not None:
            obs_by_rid[int(rid)] = o           # last observation per frame wins
    tel = sorted(telemetry or [], key=lambda r: r["ts_ms"])
    tel_ts = [r["ts_ms"] for r in tel]

    def tel_at(ms: float) -> Optional[dict]:
        i = bisect_right(tel_ts, ms) - 1
        return tel[i] if i >= 0 else None

    def find_subject(tr: dict) -> Optional[dict]:
        for a in tr["actors"]:
            if a["label"] == subject or a["actor_id"] == subject:
                return a
        return None

    if t0_ms is None:
        cands = [o["capture_ms"] for o in obs if o.get("capture_ms") is not None]
        t0_ms = min(cands) if cands else (truth[0]["render_ms"] if truth else 0.0)

    rows = []
    first_lock_ms = None
    for tr in truth:
        if tr["render_ms"] < t0_ms:
            continue
        sa = find_subject(tr)
        if sa is None:
            continue
        t = tel_at(tr["render_ms"]) if tel else None
        if follow_only and t is not None and t.get("autopilot_state") not in ("FOLLOW", "CONFIRM", "DOUBT"):
            continue
        o = obs_by_rid.get(tr["render_id"])
        sbox = _box(sa.get("box"))
        visible = sbox is not None and sa.get("visible_frac", 0.0) >= min_visible
        obox = _box(o.get("box")) if o else None
        iou = sbox.iou(obox) if (sbox is not None and obox is not None) else 0.0
        ident, best = None, 0.0
        if obox is not None:
            for a in tr["actors"]:
                b = _box(a.get("box"))
                if b is None:
                    continue
                v = b.iou(obox)
                if v > best:
                    ident, best = a["actor_id"], v
            if best < iou_thr:
                ident = None
        state = o.get("state") if o else None
        matched = iou >= iou_thr and state in ("confirmed", "candidate")
        if matched and state == "confirmed" and first_lock_ms is None:
            first_lock_ms = tr["render_ms"]
        rows.append({
            "render_id": tr["render_id"], "render_ms": tr["render_ms"], "t_s": (tr["render_ms"] - t0_ms) / 1000.0,
            "visible": visible, "visible_frac": sa.get("visible_frac", 0.0), "truth_bearing": sa.get("bearing_rad"),
            "truth_cx": sbox.cx if sbox else None, "obs_state": state, "obs_cx": obox.cx if obox else None,
            "iou": round(iou, 3), "matched": matched, "identity": ident,
            "track_id": o.get("track_id") if o else None,
            "obs_source": o.get("source") if o else None,
            "autopilot_state": t.get("autopilot_state") if t else None,
        })

    after = [r for r in rows if first_lock_ms is not None and r["render_ms"] >= first_lock_ms]
    vis_after = [r for r in after if r["visible"]]
    vis_all = [r for r in rows if r["visible"]]
    lock_rate = (sum(r["matched"] for r in vis_after) / len(vis_after)) if vis_after else None
    lock_rate_all = (sum(r["matched"] for r in vis_all) / len(vis_all)) if vis_all else None

    locked = [r for r in after if r["matched"]]
    bearing_rms = _rms([r["truth_bearing"] for r in locked if r["truth_bearing"] is not None])
    perr = [_bearing_px(r["obs_cx"], hfov_deg) - _bearing_px(r["truth_cx"], hfov_deg)
            for r in locked if r["obs_cx"] is not None and r["truth_cx"] is not None]

    switches, prev = 0, None
    track_switches, prev_tid = 0, None
    for r in rows:
        if r["identity"] is not None:
            if prev is not None and r["identity"] != prev:
                switches += 1
            prev = r["identity"]
        if r["track_id"] is not None:
            if prev_tid is not None and r["track_id"] != prev_tid:
                track_switches += 1
            prev_tid = r["track_id"]

    settling = None
    if first_lock_ms is not None:
        ok_since = None
        for r in after:
            b = r["truth_bearing"]
            if b is not None and abs(b) <= settle_tol:
                ok_since = ok_since if ok_since is not None else r["render_ms"]
                if r["render_ms"] - ok_since >= settle_hold_s * 1000:
                    settling = (ok_since - first_lock_ms) / 1000.0
                    break
            else:
                ok_since = None

    duration = (rows[-1]["render_ms"] - rows[0]["render_ms"]) / 1000.0 if len(rows) > 1 else 0.0
    report = {
        "subject": subject, "frames": len(rows), "frames_with_obs": sum(1 for r in rows if r["obs_state"]),
        "visible_frames": len(vis_all), "duration_s": round(duration, 2),
        "time_to_find_s": None if first_lock_ms is None else round((first_lock_ms - t0_ms) / 1000.0, 3),
        "lock_rate": None if lock_rate is None else round(lock_rate, 4),
        "lock_rate_all": None if lock_rate_all is None else round(lock_rate_all, 4),
        "bearing_rms_rad": None if bearing_rms is None else round(bearing_rms, 4),
        "perception_bearing_err_rms_rad": None if not perr else round(_rms(perr), 4),
        "id_switches": switches, "track_id_switches": track_switches,
        "id_switches_per_min": round(switches / (duration / 60.0), 3) if duration > 0 else None,
        "settling_time_s": None if settling is None else round(settling, 3),
        "params": {"iou": iou_thr, "min_visible": min_visible, "settle_tol_rad": settle_tol,
                   "settle_hold_s": settle_hold_s, "hfov_deg": hfov_deg, "follow_only": follow_only},
    }
    return {"report": report, "rows": rows}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Twin metrics: truth x perception x telemetry")
    ap.add_argument("--truth", help="Truth JSONL (from the twin's log dir)")
    ap.add_argument("--obs", help="SubjectObs JSONL")
    ap.add_argument("--telemetry", help="Telemetry JSONL")
    ap.add_argument("--mixed", help="one JSONL with truth, subject_obs and telemetry lines")
    ap.add_argument("--subject", required=True, help='label or actor id, e.g. "person in blue shirt"')
    ap.add_argument("--t0-ms", type=float, default=None)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--min-visible", type=float, default=0.5)
    ap.add_argument("--settle-tol", type=float, default=0.1)
    ap.add_argument("--hfov-deg", type=float, default=75.0)
    ap.add_argument("--follow-only", action="store_true", help="only frames where the autopilot follows")
    ap.add_argument("--out", help="write the report JSON here")
    ap.add_argument("--csv", help="write the joined per-frame table here")
    a = ap.parse_args(argv)
    truth, obs, tel = [], [], []
    if a.mixed:
        truth, obs, tel = split_mixed(load_jsonl(a.mixed))
    if a.truth:
        truth += load_jsonl(a.truth)
    if a.obs:
        obs += load_jsonl(a.obs)
    if a.telemetry:
        tel += load_jsonl(a.telemetry)
    res = compute(truth, obs, tel, subject=a.subject, t0_ms=a.t0_ms, iou_thr=a.iou, min_visible=a.min_visible,
                  settle_tol=a.settle_tol, hfov_deg=a.hfov_deg, follow_only=a.follow_only)
    print(json.dumps(res["report"], indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res["report"], indent=2))
    if a.csv and res["rows"]:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res["rows"][0].keys()))
            w.writeheader()
            w.writerows(res["rows"])


if __name__ == "__main__":
    main()
