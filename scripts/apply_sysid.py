"""Apply Session A/B measurements to the REAL-drone configs: config/real/{autopilot,governor}.yaml.

The files are copies of config/autopilot.yaml and config/governor.yaml (comments kept) with only
the measured values changed:
  autopilot: camera.hfov_deg, plant.yaw_rate_per_count (K), plant.forward_speed_per_count,
             plant.yaw_lag_s, delay.theta_s
  governor:  wire.sign.yaw / wire.sign.pitch, wire.deadband_counts.yaw / .pitch
plus, only when asked, the Session B/C stage switches below. It prints a unified diff and writes
nothing without --write. Both files are validated with the governor's own strict loaders first.

    source env.sh
    # after Session A (HFOV from the wall, latency from the screen counter):
    $MHD_PY scripts/apply_sysid.py --hfov-deg 78.4 --latency-ms 520
    # after Session B (sysid JSONs from logs/):
    $MHD_PY scripts/apply_sysid.py --yaw logs/sysid-yaw-*.json --pitch logs/sysid-pitch-*.json \\
        --hfov-deg 78.4 --latency-ms 520 --write
    # Session C stages (explicit switches; diff shown as always):
    $MHD_PY scripts/apply_sysid.py --yaw-only on --yaw-gain-scale 0.6 --write     # C1
    $MHD_PY scripts/apply_sysid.py --yaw-only off --yaw-gain-scale 1.0 --write    # C2
    $MHD_PY scripts/apply_sysid.py --feedforward on --write                       # after K is measured
    $MHD_PY scripts/apply_sysid.py --throttle-ramp on --write                     # if 0x80 does not land

Then fly with: scripts/run_stack.sh --mode real --config-dir config/real

Base: config/real/*.yaml if they exist (so earlier measurements and switches are kept), else
config/*.yaml; --base config starts over from the defaults.

theta (PLAN 4.4) = camera-to-backend latency (Session A p50) + perception (default 60 ms) + half the
camera frame period + half the RC period (20 ms) + yaw lag.
The sysid lag is measured on capture time: if the backend ran WITHOUT MHD_CAPTURE_OFFSET_MS, the
latency floor is inside it and is subtracted (--sysid-capture-offset-ms says what the backend had).
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

TREE = Path(os.environ.get("MHD_TREE") or Path(__file__).resolve().parents[1])

BOUNDS = {  # (lo, hi): outside these the script refuses unless --force
    "yaw_rate_per_count": (0.002, 0.08),        # rad/s per count (default guess 0.012; twin 0.0166)
    "forward_speed_per_count": (0.001, 0.08),   # m/s per count
    "deadband_counts": (0, 30),
    "hfov_deg": (40.0, 140.0),
    "theta_s": (0.3, 3.0),
    "yaw_lag_s": (0.0, 1.5),
}


# ============================================================================================= YAML text edits

_KEY = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_]+):(?P<rest>.*)$")


def _split_comment(rest: str) -> tuple[str, str]:
    """' 0.012 # K: ...' -> (' 0.012 ', '# K: ...'), ignoring '#' inside quotes."""
    q = None
    for i, ch in enumerate(rest):
        if ch in "\"'":
            q = None if q == ch else (ch if q is None else q)
        elif ch == "#" and q is None:
            return rest[:i], rest[i:]
    return rest, ""


def find_line(lines: list[str], dotted: str) -> int:
    """Index of the line holding `dotted` (e.g. 'wire.sign.yaw'), tracking nesting by indent."""
    stack: list[tuple[int, str]] = []
    want = dotted.split(".")
    for i, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _KEY.match(line)
        if not m:
            continue
        ind = len(m.group("indent"))
        while stack and stack[-1][0] >= ind:
            stack.pop()
        path = [k for _, k in stack] + [m.group("key")]
        if path == want:
            return i
        value, _ = _split_comment(m.group("rest"))
        if not value.strip():
            stack.append((ind, m.group("key")))
    raise KeyError(dotted)


def get_value(text: str, dotted: str) -> str:
    lines = text.splitlines()
    m = _KEY.match(lines[find_line(lines, dotted)])
    return _split_comment(m.group("rest"))[0].strip()


def set_value(text: str, dotted: str, value: Any, note: str) -> tuple[str, str, str]:
    """Replace the value of `dotted`, keep the comment, tag it with `note`. Returns (text, old, new)."""
    lines = text.splitlines(keepends=True)
    i = find_line([ln.rstrip("\n") for ln in lines], dotted)
    line = lines[i].rstrip("\n")
    m = _KEY.match(line)
    old_val, comment = _split_comment(m.group("rest"))
    new_val = fmt(value)
    pad = max(1, len(old_val) - len(new_val) - 1) if comment else 0
    comment = re.sub(r"\s*\[measured[^\]]*\]|\s*\[set[^\]]*\]", "", comment)
    tag = f" [{note}]" if note else ""
    if comment:
        new_line = f"{m.group('indent')}{m.group('key')}: {new_val}{' ' * pad}{comment.rstrip()}{tag}"
    else:
        new_line = f"{m.group('indent')}{m.group('key')}: {new_val}" + (f"   #{tag}" if tag else "")
    lines[i] = new_line + "\n"
    return "".join(lines), old_val.strip(), new_val


def fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = f"{v:.5g}"
        if "e" in s:
            s = f"{v:.7f}".rstrip("0")
        if "." not in s:
            s += ".0"                  # keep it a YAML float
        return s
    return str(v)


# ============================================================================================= sysid results

def load_sysid(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    for k in ("axis", "schedule", "samples", "fit"):
        if k not in d:
            raise SystemExit(f"{path}: not a sysid result (missing {k!r}; see mhd.governor.sysid)")
    return d


def refit(d: dict, hfov_deg: Optional[float], marker_size_m: Optional[float]) -> tuple[dict, list[str]]:
    """Re-run sysid's own fit with the samples corrected to the measured HFOV / marker size."""
    from mhd.governor.sysid import Step, fit_steps
    notes = []
    h0 = float(d.get("hfov_deg") or 75.0)
    s0 = float(d.get("marker_size_m") or 0.15)
    h1 = hfov_deg or h0
    s1 = marker_size_m or s0
    ratio = math.tan(math.radians(h1) / 2) / math.tan(math.radians(h0) / 2)   # f_used / f_true
    samples = []
    same_hfov = abs(h1 - h0) < 1e-6
    for t, b, r in d["samples"]:
        # a marker in view always has |bearing| < HFOV/2 < pi/2, so atan(tan(b)) is exact
        bb = b if same_hfov or b is None or not math.isfinite(b) else math.atan(math.tan(b) * ratio)
        rr = (r / ratio) * (s1 / s0) if r is not None and math.isfinite(r) else r
        samples.append((t, bb, rr))
    if abs(h1 - h0) > 0.05:
        notes.append(f"{d['axis']}: re-fit with HFOV {h1:.1f} deg (the run used {h0:.1f})")
    if abs(s1 - s0) > 1e-4:
        notes.append(f"{d['axis']}: re-fit with marker size {s1:.3f} m (the run used {s0:.3f})")
    steps = []
    for s in d["schedule"]:
        if not s.get("t_end_ms"):
            continue
        steps.append(Step(s["axis"], s["value"], s["hold_s"], s["rest_s"], s["t_start_ms"], s["t_end_ms"],
                          s.get("wire_count")))
    return fit_steps(steps, samples, d["axis"]), notes


def native_sign(fit: dict) -> tuple[Optional[int], str]:
    """+1 if positive wire counts give positive rates (yaw clockwise / pitch forward), else -1."""
    recs = [r for r in fit["steps"] if r.get("rate") is not None]
    if not recs:
        return None, "no steps with a fitted rate"
    big = max(abs(r["rate"]) for r in recs)
    votes, assumed = [], False
    for r in recs:
        if abs(r["rate"]) < 0.2 * big:
            continue                                   # inside the deadband or noise
        c = r.get("wire_count")
        if c is None or c == 0:
            c = r["value"]
            assumed = True
        votes.append(1 if (r["rate"] > 0) == (c > 0) else -1)
    if not votes:
        return None, "no step above the noise"
    s = sum(votes)
    agree = max(votes.count(1), votes.count(-1))
    sign = 1 if s > 0 else (-1 if s < 0 else None)
    why = f"{agree}/{len(votes)} steps agree" + (" (some steps had no wire count: assumed sign +1 during the run)"
                                                  if assumed else "")
    return sign, why


def fit_quality(fit: dict) -> dict:
    pts = [(abs(r["wire_count"] if r.get("wire_count") is not None else round(r["value"] * 127)), abs(r["rate"]))
           for r in fit["steps"] if r.get("rate") is not None]
    k, db = fit.get("k_per_count"), fit.get("deadband_counts")
    if not pts or k is None:
        return {"n": len(pts)}
    pred = [k * max(0.0, c - (db or 0.0)) for c, _ in pts]
    ys = [y for _, y in pts]
    mean = sum(ys) / len(ys)
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    ss_tot = sum((y - mean) ** 2 for y in ys) or 1e-12
    return {"n": len(pts), "r2": round(1 - ss_res / ss_tot, 3)}


# ============================================================================================= main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaw", help="sysid result for yaw (logs/sysid-yaw-*.json)")
    ap.add_argument("--pitch", help="sysid result for pitch (logs/sysid-pitch-*.json)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--hfov-deg", type=float, help="measured HFOV (Session A)")
    g.add_argument("--wall-width-m", type=float, help="visible tape width at --wall-distance-m: HFOV = 2 atan(w/2d)")
    ap.add_argument("--wall-distance-m", type=float, default=1.0)
    ap.add_argument("--marker-size-m", type=float, help="measured printed marker side, if it differs from the run")
    ap.add_argument("--latency-ms", type=float, help="Session A camera-to-backend latency p50 (for theta)")
    ap.add_argument("--latency-floor-ms", type=float, help="Session A minimum latency (default: --latency-ms)")
    ap.add_argument("--sysid-capture-offset-ms", type=float,
                    default=float(os.environ.get("MHD_CAPTURE_OFFSET_MS", "0") or 0),
                    help="MHD_CAPTURE_OFFSET_MS the backend had during sysid (default: this shell's value or 0)")
    ap.add_argument("--perception-ms", type=float, default=60.0, help="perception p50 per observation")
    ap.add_argument("--camera-fps", type=float, default=25.0, help="drone stream fps (Session A ffprobe)")
    ap.add_argument("--yaw-lag-s", type=float, help="override the yaw lag")
    ap.add_argument("--theta-s", type=float, help="override theta entirely")
    ap.add_argument("--feedforward", choices=["on", "off"], help="yaw.feedforward.enabled (after K is measured)")
    ap.add_argument("--yaw-gain-scale", type=float, help="yaw.loop_gain = this x config/autopilot.yaml's gain")
    ap.add_argument("--yaw-only", choices=["on", "off"], help="caps.autopilot.pitch = 0 (on) or the default (off)")
    ap.add_argument("--throttle-ramp", choices=["on", "off"], help="landing.throttle_ramp (if 0x80 does not land)")
    ap.add_argument("--base", default=None, help="config dir to start from (default config/real if present)")
    ap.add_argument("--out-dir", default=str(TREE / "config" / "real"))
    ap.add_argument("--write", action="store_true", help="write the files (default: diff only)")
    ap.add_argument("--force", action="store_true", help="accept values outside the sanity bounds")
    ap.add_argument("--allow-aborted", action="store_true", help="use a sysid run that aborted part-way")
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir)
    if out_dir.resolve() in {(TREE / "config").resolve(), (TREE / "config" / "twin").resolve()}:
        raise SystemExit("--out-dir must not be config/ or config/twin/ (those belong to the integrator)")
    default_dir = TREE / "config"
    base_dir = Path(a.base) if a.base else (out_dir if (out_dir / "autopilot.yaml").exists()
                                             and (out_dir / "governor.yaml").exists() else default_dir)
    if "twin" in base_dir.resolve().parts:
        raise SystemExit("refusing a twin config as the base for the real drone")
    texts = {n: (base_dir / f"{n}.yaml").read_text() for n in ("autopilot", "governor")}
    defaults = {n: (default_dir / f"{n}.yaml").read_text() for n in ("autopilot", "governor")}
    new = dict(texts)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    changes: list[tuple[str, str, str, str, str]] = []     # file, key, old, new, why
    warnings: list[str] = []
    problems: list[str] = []

    def put(file: str, key: str, value: Any, why: str, tag: str = "measured") -> None:
        bound = next((BOUNDS[p] for p in key.split(".") if p in BOUNDS), None)
        if bound and isinstance(value, (int, float)) and not isinstance(value, bool) and \
                not (bound[0] <= value <= bound[1]):
            problems.append(f"{file}:{key} = {value} is outside the sanity range {bound} ({why})")
        new[file], old, nv = set_value(new[file], key, value, f"{tag} {stamp}: {why}")
        changes.append((file, key, old, nv, why))

    # ------------------------------------------------------------------ HFOV
    hfov = a.hfov_deg
    if a.wall_width_m is not None:
        hfov = math.degrees(2 * math.atan(a.wall_width_m / (2 * a.wall_distance_m)))
        print(f"HFOV = 2*atan({a.wall_width_m}/(2*{a.wall_distance_m})) = {hfov:.2f} deg")
    if hfov is not None:
        put("autopilot", "camera.hfov_deg", round(hfov, 1), "Session A wall measurement")

    # ------------------------------------------------------------------ sysid
    yaw_lag = None
    for axis, path in (("yaw", a.yaw), ("pitch", a.pitch)):
        if not path:
            continue
        d = load_sysid(path)
        if d["axis"] != axis:
            raise SystemExit(f"{path} is a {d['axis']} run, given as --{axis}")
        if d.get("aborted"):
            msg = f"{path}: the run ABORTED ({d['aborted']}); its fit uses only the steps flown"
            if not a.allow_aborted:
                raise SystemExit(msg + " (pass --allow-aborted to use it anyway)")
            warnings.append(msg)
        fit, notes = refit(d, hfov, a.marker_size_m)
        warnings += notes
        q = fit_quality(fit)
        print(f"\n{axis} sysid {Path(path).name}: K {fit['k_per_count']} per count, deadband "
              f"{fit['deadband_counts']} counts, lag {fit['lag_s_median']} s, fit R^2 {q.get('r2')} over {q['n']} steps")
        print(f"   {'value':>6} {'count':>6} {'rate':>9} {'lag_s':>6} {'n':>4}")
        for r in fit["steps"]:
            rate = f"{r['rate']:+.4f}" if r.get("rate") is not None else "-"
            print(f"   {r['value']:+6.2f} {str(r.get('wire_count')):>6} {rate:>9} {str(r.get('lag_s')):>6} {r['n']:>4}")
        if q["n"] < 4:
            warnings.append(f"{axis}: only {q['n']} steps with a fitted rate (want 6: 3 amplitudes x 2 signs)")
        if q.get("r2") is not None and q["r2"] < 0.8:
            warnings.append(f"{axis}: poor fit (R^2 {q['r2']}): check the marker stayed in view")
        unit = "rad/s" if axis == "yaw" else "m/s"
        key = "plant.yaw_rate_per_count" if axis == "yaw" else "plant.forward_speed_per_count"
        if fit["k_per_count"] is None:
            problems.append(f"{axis}: no K fitted (need 2+ amplitudes with a measured rate)")
        else:
            put("autopilot", key, round(fit["k_per_count"], 5),
                f"{Path(path).name}, {unit} per count")
        if fit["deadband_counts"] is not None:
            put("governor", f"wire.deadband_counts.{axis}", int(round(fit["deadband_counts"])),
                f"{Path(path).name} firmware deadband")
        sign, why = native_sign(fit)
        if sign is None:
            warnings.append(f"{axis}: sign undetermined ({why}); wire.sign.{axis} unchanged")
        else:
            cur = int(get_value(new["governor"], f"wire.sign.{axis}"))
            if sign != cur:
                warnings.append(f"!!! {axis.upper()} AXIS SIGN FLIPPED: {cur} -> {sign} ({why}). The drone moved the WRONG way "
                                f"for {'+yaw (clockwise)' if axis == 'yaw' else '+pitch (forward)'}; re-check by hand "
                                "before following.")
            put("governor", f"wire.sign.{axis}", sign, f"{Path(path).name}: {why}")
        if axis == "yaw" and fit["lag_s_median"] is not None:
            floor = a.latency_floor_ms if a.latency_floor_ms is not None else a.latency_ms
            sub = max(0.0, (floor or 0.0) - a.sysid_capture_offset_ms) / 1000.0 if floor is not None else 0.0
            yaw_lag = max(0.05, fit["lag_s_median"] - sub)
            if floor is None and a.sysid_capture_offset_ms == 0:
                warnings.append("yaw lag: no --latency-ms given and no MHD_CAPTURE_OFFSET_MS during the run, so the "
                                "measured lag may include the camera latency floor")
            print(f"   yaw lag = measured {fit['lag_s_median']:.3f} s - latency floor not covered by the capture "
                  f"offset {sub:.3f} s = {yaw_lag:.3f} s")
    if a.yaw_lag_s is not None:
        yaw_lag = a.yaw_lag_s
    if yaw_lag is not None:
        put("autopilot", "plant.yaw_lag_s", round(yaw_lag, 3), "sysid onset lag" if a.yaw_lag_s is None else "given")

    # ------------------------------------------------------------------ theta
    theta = a.theta_s
    if theta is None and a.latency_ms is not None:
        lag = yaw_lag if yaw_lag is not None else float(get_value(new["autopilot"], "plant.yaw_lag_s"))
        parts = {"latency": a.latency_ms / 1000, "perception": a.perception_ms / 1000,
                 "half frame": 0.5 / a.camera_fps, "half RC": 0.02, "yaw lag": lag}
        theta = sum(parts.values())
        print("\ntheta = " + " + ".join(f"{k} {v:.3f}" for k, v in parts.items()) + f" = {theta:.3f} s")
    if theta is not None:
        put("autopilot", "delay.theta_s", round(theta, 3), "latency + perception + half periods + yaw lag"
            if a.theta_s is None else "given")

    # ------------------------------------------------------------------ stage switches (explicit only)
    if a.feedforward:
        put("autopilot", "yaw.feedforward.enabled", a.feedforward == "on", "--feedforward", "set")
    if a.yaw_gain_scale is not None:
        if not 0.1 <= a.yaw_gain_scale <= 1.5:
            raise SystemExit("--yaw-gain-scale must be 0.1..1.5")
        g0 = float(get_value(defaults["autopilot"], "yaw.loop_gain"))
        put("autopilot", "yaw.loop_gain", round(g0 * a.yaw_gain_scale, 4),
            f"--yaw-gain-scale {a.yaw_gain_scale} x {g0}", "set")
    if a.yaw_only:
        v = 0 if a.yaw_only == "on" else int(get_value(defaults["governor"], "caps.autopilot.pitch"))
        put("governor", "caps.autopilot.pitch", v, f"--yaw-only {a.yaw_only}", "set")
    if a.throttle_ramp:
        put("governor", "landing.throttle_ramp", a.throttle_ramp == "on", "--throttle-ramp", "set")

    if not changes:
        for p in problems:
            print(f"PROBLEM: {p}")
        if problems:
            return 1
        print("nothing to apply: give --yaw/--pitch, --hfov-deg, --latency-ms or a switch (see --help)")
        return 2

    # ------------------------------------------------------------------ header, validation, diff
    for n in new:
        hdr = (f"# REAL-DRONE calibration of config/{n}.yaml (scripts/apply_sysid.py, last run {stamp}).\n"
               f"# Identical to config/{n}.yaml except the values tagged [measured ...] or [set ...]. Never for the twin.\n")
        body = re.sub(r"\A(# REAL-DRONE calibration[^\n]*\n# Identical[^\n]*\n)", "", new[n])
        new[n] = hdr + body
    from mhd.governor.config import load_autopilot_config, load_governor_config
    tmp = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=tmp) as td:
            pa, pg = Path(td) / "autopilot.yaml", Path(td) / "governor.yaml"
            pa.write_text(new["autopilot"])
            pg.write_text(new["governor"])
            ap_cfg = load_autopilot_config(pa)
            gv_cfg = load_governor_config(pg)
            gv_cfg.validate()
            ap_cfg.validate()
    except Exception as exc:  # noqa: BLE001
        print(f"\nINVALID: the new config does not load: {exc}")
        return 1

    print(f"\nbase: {base_dir}   ->   {out_dir}{'' if a.write else '   (dry run)'}")
    for n in ("autopilot", "governor"):
        target = out_dir / f"{n}.yaml"
        old = target.read_text() if target.exists() else texts[n]
        label = str(target) if target.exists() else f"{base_dir / (n + '.yaml')} (copy)"
        diff = list(difflib.unified_diff(old.splitlines(keepends=True), new[n].splitlines(keepends=True),
                                         fromfile=label, tofile=str(target)))
        sys.stdout.write("".join(diff) if diff else f"(no change to {target})\n")
    print("\nchanges:")
    for f, k, o, nv, why in changes:
        print(f"  {f}.{k}: {o} -> {nv}   ({why})")
    for w in warnings:
        print(f"WARNING: {w}")
    if problems:
        for p in problems:
            print(f"OUT OF RANGE: {p}")
        if not a.force:
            print("refusing (fix the inputs, or --force if you are sure)")
            return 1
    if not a.write:
        print("\ndry run: nothing written. Add --write to write these files.")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in ("autopilot", "governor"):
        (out_dir / f"{n}.yaml").write_text(new[n])
    print(f"\nwrote {out_dir}/autopilot.yaml and governor.yaml. Fly with:\n"
          f"  scripts/run_stack.sh --mode real --config-dir {os.path.relpath(out_dir, TREE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
