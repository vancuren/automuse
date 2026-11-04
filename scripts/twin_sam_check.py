"""Spot-check that SAM 3.1 (Meta Model API) detects the twin's actors in HEVC-path frames (gate G2).

    $MHD_PY scripts/twin_sam_check.py --dir $MHD_HOME/media/shared/twin/s1 --phrase "person in blue shirt" \
        --subject walker_blue --n 10 [--cap-usd 0.5]

Sends n evenly spaced frames (640x360 JPEG, like perception does) with one noun phrase, parses
the <|box;...|> tokens, and reports recall of the subject (IoU >= 0.3) and hits on other actors.
Paid: $0.0025 per image; one line per batch goes to $MHD_HOME/logs/spend.jsonl (ws "twin").
The API key comes from MODEL_API_KEY and is never printed.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mhd.contracts import Box  # noqa: E402
from mhd.twin import counter  # noqa: E402

USD_PER_IMAGE = 0.0025
BOX_RE = re.compile(r"<\|box;([^|]*)\|>")


def ledger() -> Path:
    return Path(os.environ["MHD_HOME"]) / "logs" / "spend.jsonl"


def spent_twin() -> float:
    total = 0.0
    try:
        for line in ledger().read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("ws") == "twin":
                total += float(row.get("est_usd", 0))
    except OSError:
        pass
    return total


def parse_boxes(text: str) -> list[dict]:
    out = []
    for m in BOX_RE.finditer(text or ""):
        f = dict(kv.split("=", 1) for kv in m.group(1).split(";") if "=" in kv)
        try:
            out.append({"x1": int(f["x1"]), "y1": int(f["y1"]), "x2": int(f["x2"]) + 1, "y2": int(f["y2"]) + 1,
                        "w": int(f["w"]), "h": int(f["h"]), "c": float(f.get("c", "nan"))})
        except (KeyError, ValueError):
            continue
    return out


def output_text(body: dict) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    parts = []
    for item in body.get("output") or []:
        for c in item.get("content") or []:
            if c.get("type") == "output_text":
                parts.append(c.get("text", ""))
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--phrase", required=True)
    ap.add_argument("--subject", required=True, help="actor_id of the subject")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--cap-usd", type=float, default=0.5, help="twin workstream cap for these checks")
    ap.add_argument("--min-conf", type=float, default=0.4)
    a = ap.parse_args()
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise SystemExit("MODEL_API_KEY not set (source env.sh)")
    if spent_twin() + a.n * USD_PER_IMAGE > a.cap_usd:
        raise SystemExit(f"would exceed the twin cap of ${a.cap_usd}")
    d = Path(a.dir)
    truths = {json.loads(l)["render_id"]: json.loads(l) for l in (d / "truth.jsonl").read_text().splitlines() if l}
    rids = sorted(truths)
    pick = [rids[int(i * (len(rids) - 1) / max(1, a.n - 1))] for i in range(a.n)]
    client = httpx.Client(timeout=20.0)
    rows, paid = [], 0
    for rid in pick:
        img = cv2.imread(str(d / f"frame_{rid:06d}.jpg"))
        counter.mask_roi(img)
        small = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        body = {"model": "sam-3.1", "input": [{"role": "user", "content": [
            {"type": "input_text", "text": a.phrase},
            {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()},
        ]}], "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"}}
        t0 = time.monotonic()
        try:
            r = client.post("https://api.meta.ai/v1/responses", json=body,
                            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            paid += 1
            status = r.status_code
            text = output_text(r.json()) if status == 200 else ""
        except Exception as e:  # noqa: BLE001
            status, text = f"error {type(e).__name__}", ""
            paid += 1
        dt = time.monotonic() - t0
        dets = [b for b in parse_boxes(text) if not (b["c"] == b["c"]) or b["c"] >= a.min_conf]
        tr = truths[rid]
        best = {}
        for act in tr["actors"]:
            tb = act.get("box")
            if not tb:
                continue
            tbox = Box(x1=tb["x1"] / 2, y1=tb["y1"] / 2, x2=tb["x2"] / 2, y2=tb["y2"] / 2)
            best[act["actor_id"]] = max([tbox.iou(Box(x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"]))
                                         for b in dets] or [0.0])
        subj = next(x for x in tr["actors"] if x["actor_id"] == a.subject)
        rows.append({"render_id": rid, "status": status, "latency_s": round(dt, 2), "detections": len(dets),
                     "conf": [round(b["c"], 2) for b in dets], "subject_iou": round(best.get(a.subject, 0.0), 3),
                     "subject_visible": subj.get("visible_frac", 0), "subject_box_h": (subj["box"] or {}).get("y2", 0) - (subj["box"] or {}).get("y1", 0),
                     "others_hit": {k: round(v, 2) for k, v in best.items() if k != a.subject and v >= 0.3},
                     "sam_boxes_1280": [[2 * b["x1"], 2 * b["y1"], 2 * b["x2"], 2 * b["y2"]] for b in dets],
                     "truth_box_1280": [round(subj["box"][k]) for k in ("x1", "y1", "x2", "y2")] if subj.get("box") else None})
    with open(ledger(), "a") as f:
        f.write(json.dumps({"ts": time.time(), "ws": "twin", "service": "meta-api sam-3.1 image", "units": paid,
                            "est_usd": round(paid * USD_PER_IMAGE, 4),
                            "note": f"G2 spot-check {d.name} '{a.phrase}'"}) + "\n")
    vis = [r for r in rows if r["subject_visible"] >= 0.5]
    hits = [r for r in vis if r["subject_iou"] >= 0.3]
    summary = {"dir": str(d), "phrase": a.phrase, "frames": len(rows), "subject_visible_frames": len(vis),
               "recall": round(len(hits) / len(vis), 3) if vis else None,
               "frames_with_other_actor_hit": sum(1 for r in rows if r["others_hit"]),
               "latency_s_median": sorted(r["latency_s"] for r in rows)[len(rows) // 2], "paid_images": paid}
    print(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
