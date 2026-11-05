#!/usr/bin/env python3
"""Turn flights flown BEFORE the flight recorder existed into replays for the UI's Flights view.

New flights are saved automatically (src/mhd/backend/recorder.py). This rebuilds older ones from what
was logged at the time: the governor's control-*.jsonl ticks (flight state, autopilot state, the sticks
sent), the perception obs.jsonl (the subject's box), the backend log (commands) and the agent's
ptt_metrics.jsonl (what was said). There is no camera recording for them, so the replay shows the
subject box, the sticks and the commands on an empty picture.

    scripts/import_flights.py                       # every run_stack.sh flight in logs/ of 5 s or more
    scripts/import_flights.py logs/control-20261104-164255-4496.jsonl --min-air-s 20

Read-only on the logs; writes logs/flights/<date>-<time>-<mode>-log/.
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import time
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]
FLYING = {"TAKING_OFF", "AIRBORNE", "LANDING"}
PRE_MS, TAIL_MS, LANDING_MAX_MS = 4000.0, 3000.0, 20000.0
CMD_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+) .*\[governor\] cmd=(\w+) source=(\w+) -> ok=(\w+)")
AXES = ("throttle", "yaw", "pitch", "roll")


def log_start_unix(path: Path) -> float:
    m = re.search(r"control-(\d{8}-\d{6})", path.name)
    if not m:
        raise SystemExit(f"{path}: not a control-YYYYMMDD-HHMMSS log")
    return time.mktime(time.strptime(m.group(1), "%Y%m%d-%H%M%S"))


def stack_mode(start_unix: float) -> str | None:
    """real / twin / camera, from the scripts/run_stack.sh run that started within 15 s before this log."""
    for d in (TREE / "logs").glob("stack-*-*"):
        m = re.match(r"stack-(\d{8}-\d{6})-(\w+)$", d.name)
        if m and 0 <= start_unix - time.mktime(time.strptime(m.group(1), "%Y%m%d-%H%M%S")) <= 15:
            return m.group(2)
    return None


def read_ticks(path: Path) -> list[dict]:
    ticks = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d.get("t"), (int, float)) and "fs" in d:
                ticks.append(d)
    return ticks


def segments(ticks: list[dict], min_air_ms: float) -> list[tuple[int, int]]:
    """(first, last) tick indexes of every flight: takeoff to back on the ground (or 20 s of LANDING)."""
    out, i, n = [], 0, len(ticks)
    while i < n:
        if ticks[i]["fs"] != "TAKING_OFF" or (i and ticks[i - 1]["fs"] == "TAKING_OFF"):
            i += 1
            continue
        start, j, air, landing_since = i, i, 0.0, None
        while j < n and ticks[j]["fs"] in FLYING:
            if ticks[j]["fs"] == "AIRBORNE" and j > start:
                air += ticks[j]["t"] - ticks[j - 1]["t"]
            if ticks[j]["fs"] == "LANDING":
                landing_since = landing_since if landing_since is not None else ticks[j]["t"]
                if ticks[j]["t"] - landing_since > LANDING_MAX_MS:
                    break
            else:
                landing_since = None
            j += 1
        if air >= min_air_ms:
            out.append((start, min(j, n - 1)))
        i = max(j, i + 1)
    return out


def read_commands(logs: list[Path]) -> list[tuple[float, dict]]:
    out = []
    for p in logs:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            m = CMD_RE.match(line)
            if m:
                unix = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")) + int(m.group(2)) / 1000.0
                out.append((unix, {"command": m.group(3), "source": m.group(4), "ok": m.group(5) == "True",
                                   "reason": None}))
    return sorted(out, key=lambda e: e[0])


def read_obs(paths: list[Path], lo: float, hi: float) -> list[dict]:
    out = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    c = d.get("capture_ms")
                    if isinstance(c, (int, float)) and lo <= c <= hi:
                        out.append(d)
        except OSError:
            continue
    return sorted(out, key=lambda d: d["capture_ms"])


def read_transcripts(path: Path, lo_unix: float, hi_unix: float) -> list[dict]:
    """The agent's push-to-talk log has no timestamps: use it only if it was last written during this flight."""
    try:
        if not lo_unix <= path.stat().st_mtime <= hi_unix + 60:
            return []
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        return []
    return [r for r in rows if r.get("transcript") and (r.get("delivered") or {}).get("ok")]


def build(ticks, first, last, t0, start_unix, obs_paths, commands, ptt_path, mode) -> tuple[dict, list[str]]:
    lo, hi = ticks[first]["t"] - PRE_MS, ticks[last]["t"] + TAIL_MS
    a = next(i for i in range(first, -1, -1) if ticks[i]["t"] < lo or i == 0)
    b = next(i for i in range(last, len(ticks)) if ticks[i]["t"] > hi or i == len(ticks) - 1)
    seg = ticks[a:b + 1]
    base = seg[0]["t"]
    seg_unix = start_unix + (base - t0) / 1000.0
    end_unix = start_unix + (seg[-1]["t"] - t0) / 1000.0
    obs = read_obs(obs_paths, base, seg[-1]["t"])
    obs_t = [o["capture_ms"] for o in obs]

    mission_ids: list[str] = []
    for o in obs:
        if o.get("mission_id") and o["mission_id"] not in mission_ids:
            mission_ids.append(o["mission_id"])
    said = read_transcripts(ptt_path, seg_unix, end_unix)[-len(mission_ids):] if mission_ids else []
    said = [None] * (len(mission_ids) - len(said)) + said
    missions = {}
    for mid, row in zip(mission_ids, said):
        transcript = (row or {}).get("transcript") or ""
        phrase = re.sub(r"^(please\s+)?(follow|track|chase|find)\s+(the\s+)?", "", transcript.strip().rstrip(".").lower()) or "subject"
        missions[mid] = {"type": "mission", "mission_id": mid, "transcript": transcript,
                         "intent": {"action": "follow", "subject_phrase": phrase},
                         "model": (row or {}).get("intent_model") or "", "reply": "",
                         "latency_ms": (row or {}).get("final_to_intent_ms") or 0.0}

    events: list[tuple[float, dict]] = []
    summary: list[dict] = []
    seen: set[str] = set()
    prev_fs, prev_ap, last_tlm = seg[0]["fs"], seg[0]["ap"], -1e9
    for tick in seg:
        t = tick["t"] - base
        if tick["fs"] != prev_fs:
            events.append((t, {"k": "agent", "d": {"type": "agent_event", "kind": "status", "text": "", "data": {
                "kind": "flight_state", "from": prev_fs, "to": tick["fs"]}}}))
            prev_fs = tick["fs"]
        if tick["ap"] != prev_ap:
            events.append((t, {"k": "agent", "d": {"type": "agent_event", "kind": "status", "text": "", "data": {
                "kind": "state", "from": prev_ap, "to": tick["ap"], "reason": tick.get("gate") or None}}}))
            prev_ap = tick["ap"]
        if t - last_tlm < 99.0:
            continue
        last_tlm = t
        k = bisect.bisect_right(obs_t, tick["t"]) - 1
        subject = obs[k] if k >= 0 and tick["t"] - obs_t[k] < 1000.0 and tick["fs"] in FLYING else None
        mid = subject.get("mission_id") if subject else None
        if mid and mid not in seen:
            seen.add(mid)
            m = missions[mid]
            if m["transcript"]:
                events.append((t, {"k": "agent", "d": {"type": "agent_event", "kind": "final", "text": m["transcript"]}}))
            events.append((t, {"k": "mission", "d": {**m, "ok": True}}))
            summary.append({"t": round(t), "kind": "mission", "transcript": m["transcript"],
                            "subject": m["intent"]["subject_phrase"]})
        w = tick.get("w") or [128, 128, 128, 128]
        events.append((t, {"k": "tlm", "d": {
            "type": "telemetry", "ts_ms": tick["t"], "flight_mode": "real" if mode == "real" else "twin",
            "flight_state": tick["fs"], "autopilot_state": tick["ap"], "armed": True,
            "sticks_wire": dict(zip(AXES, w)), "sticks_by_source": {}, "active_source": tick.get("src") or "none",
            "frame_age_ms": tick.get("frame_lag"), "obs_age_ms": tick.get("obs_lag"), "latency_ms": {},
            "flight_time_s": max(0.0, (tick["t"] - ticks[first]["t"]) / 1000.0), "pack_id": None,
            "safety_flags": tick.get("fl") or [], "mission": missions.get(mid) if mid else None,
            "subject": subject if subject and subject.get("box") else None}}))
    for unix, cmd in commands:
        if seg_unix <= unix <= end_unix:
            t = (unix - seg_unix) * 1000.0
            events.append((t, {"k": "cmd", "d": cmd}))
            summary.append({"t": round(t), "kind": "cmd", "command": cmd["command"], "source": cmd["source"],
                            "ok": cmd["ok"]})
    events.sort(key=lambda e: e[0])
    summary.sort(key=lambda e: e["t"])
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(seg_unix))
    meta = {"id": f"{stamp}-{mode}-log", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(seg_unix)),
            "started_unix": round(seg_unix, 3), "flight_mode": mode, "duration_ms": round(seg[-1]["t"] - base),
            "frames": 0, "complete": True, "imported": True, "commands": summary}
    return meta, [json.dumps({"t": round(t, 1), **e}, separators=(",", ":")) for t, e in events]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", type=Path, help="control-*.jsonl files (default: all in logs/)")
    ap.add_argument("--out", type=Path, default=TREE / "logs" / "flights")
    ap.add_argument("--min-air-s", type=float, default=5.0, help="skip hops shorter than this in the air")
    ap.add_argument("--mode", default=None, help="label for the flights [from the stack-*-<mode> run that wrote the log]")
    args = ap.parse_args()
    logs = args.logs or sorted((TREE / "logs").glob("control-*.jsonl"))
    obs_paths = sorted((TREE / "logs").glob("stack-*/obs.jsonl"))
    commands = read_commands(sorted((TREE / "logs").glob("stack-*/backend.log")))
    made = 0
    for path in logs:
        ticks = read_ticks(path)
        if not ticks:
            continue
        start_unix, t0 = log_start_unix(path), ticks[0]["t"]
        mode = args.mode or stack_mode(start_unix)
        if mode is None:
            if not args.logs:
                continue                      # a test's log, not a scripts/run_stack.sh run
            mode = "real"
        for first, last in segments(ticks, args.min_air_s * 1000.0):
            meta, lines = build(ticks, first, last, t0, start_unix, obs_paths, commands,
                                TREE / "logs" / "agent" / "ptt_metrics.jsonl", mode)
            d = args.out / meta["id"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            (d / "frames.bin").write_bytes(b"")
            (d / "frames.idx").write_text("", encoding="ascii")
            (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            made += 1
            print(f"{meta['id']}: {meta['duration_ms'] / 1000:.0f} s, {len(lines)} events, "
                  f"commands: {[c.get('command') or c.get('transcript') for c in meta['commands']]}")
    print(f"{made} flight(s) written to {args.out}")


if __name__ == "__main__":
    main()
