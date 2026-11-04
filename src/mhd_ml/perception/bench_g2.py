"""G2: SAM 3.1 recall on twin frames (media/shared/twin/<scenario>/frame_<render_id>.jpg + truth.jsonl).

  $MHD_ML_PY -m mhd_ml.perception.bench_g2 --twin $MHD_HOME/media/shared/twin --out logs/bench_g2.json \
      --crops docs/img/percep/g2_examples.jpg

For every frame and every actor label in the scenario (e.g. "person in blue shirt"), SAM is
prompted with the label. A hit is a detection with score >= --min-score and IoU >= 0.5 with
that actor's truth box. Also counted: top-1 correctness, confusions (the best detection lands on a
different actor) and false alarms (detections while the actor is off screen). The twin counter
block is masked first, as the service does. One ViT backbone per frame is shared by all prompts.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .benchutil import load_snapshot
from .fusion import mask_counter
from .types import iou


def _box(b):
    return None if not b else (b["x1"], b["y1"], b["x2"], b["y2"])


def run_scenario(eng, folder: Path, res: int, min_score: float, every: int = 1) -> dict:
    import cv2

    eng.set_resolution(res)
    truth = [json.loads(line) for line in open(folder / "truth.jsonl")]
    labels = sorted({a["label"] for t in truth for a in t["actors"]})
    rows = []
    for k, t in enumerate(truth):
        if k % every:
            continue
        f = folder / f"frame_{t['render_id']:06d}.jpg"
        img = cv2.imread(str(f))
        if img is None:
            continue
        mask_counter(img)
        t0 = time.perf_counter()
        with eng.lock:
            feats = eng.backbone(eng.preprocess(img))
            t_bb = (time.perf_counter() - t0) * 1e3
            per_label = {}
            for lab in labels:
                t1 = time.perf_counter()
                dets = eng.to_detections(eng.detect_features(feats, lab, 0.2), lab)
                per_label[lab] = (dets, (time.perf_counter() - t1) * 1e3)
        actors = {a["label"]: a for a in t["actors"]}
        for lab in labels:
            dets, head_ms = per_label[lab]
            good = [d for d in dets if d.score >= min_score]
            a = actors.get(lab)
            tb = _box(a["box"]) if a else None
            others = [_box(o["box"]) for o in t["actors"] if o["label"] != lab and o["box"]]
            row = {"frame": f.name, "render_id": t["render_id"], "label": lab, "visible": tb is not None,
                   "area": ((tb[2] - tb[0]) * (tb[3] - tb[1]) / (1280 * 720)) if tb else 0.0,
                   "visible_frac": a["visible_frac"] if a else 0.0, "n": len(good),
                   "backbone_ms": round(t_bb, 1), "head_ms": round(head_ms, 1),
                   "top": [round(v, 1) for v in good[0].box] if good else None,
                   "top_score": round(good[0].score, 3) if good else 0.0}
            if tb is not None:
                hits = [d for d in good if iou(d.box, tb) >= 0.5]
                row["hit"] = bool(hits)
                row["hit_score"] = round(max((d.score for d in hits), default=0.0), 3)
                row["top1_ok"] = bool(good) and iou(good[0].box, tb) >= 0.5
                row["confused"] = bool(good) and not row["top1_ok"] and any(iou(good[0].box, o) >= 0.5 for o in others)
            else:
                row["false_alarm"] = bool(good)
                row["confused"] = bool(good) and any(iou(good[0].box, o) >= 0.5 for o in others)
            rows.append(row)
    out = {"labels": labels, "rows": rows}
    for lab in labels:
        vis = [r for r in rows if r["label"] == lab and r["visible"]]
        invis = [r for r in rows if r["label"] == lab and not r["visible"]]
        big = [r for r in vis if r["area"] >= 0.05]
        out[lab] = {
            "visible_frames": len(vis),
            "recall": round(sum(r["hit"] for r in vis) / len(vis), 3) if vis else None,
            "top1": round(sum(r["top1_ok"] for r in vis) / len(vis), 3) if vis else None,
            "confused": sum(r["confused"] for r in vis),
            "recall_area_ge_5pct": (round(sum(r["hit"] for r in big) / len(big), 3) if big else None),
            "frames_area_ge_5pct": len(big),
            "median_hit_score": (round(float(np.median([r["hit_score"] for r in vis if r["hit"]])), 3)
                                 if any(r["hit"] for r in vis) else None),
            "offscreen_frames": len(invis),
            "false_alarms_offscreen": sum(r["false_alarm"] for r in invis),
            "median_area": round(float(np.median([r["area"] for r in vis])), 4) if vis else None,
        }
    out["backbone_ms_p50"] = round(float(np.median([r["backbone_ms"] for r in rows])), 1) if rows else None
    out["head_ms_p50"] = round(float(np.median([r["head_ms"] for r in rows])), 1) if rows else None
    return out


def example_sheet(twin: Path, results: dict, out_path: Path, per_label: int = 3) -> None:
    """Crops around the truth box with SAM's top box: green = hit, red = miss / wrong."""
    import cv2

    tiles = []
    for scen, r in results.items():
        if not isinstance(r, dict) or "rows" not in r:
            continue
        for lab in r["labels"]:
            vis = [x for x in r["rows"] if x["label"] == lab and x["visible"]]
            if not vis:
                continue
            misses = [x for x in vis if not x["hit"]]
            hits = [x for x in vis if x["hit"]]
            pick = (misses[:1] + hits[::max(1, len(hits) // per_label)])[:per_label]
            truth = {json.loads(l)["render_id"]: json.loads(l) for l in open(twin / scen / "truth.jsonl")}
            for x in pick:
                img = cv2.imread(str(twin / scen / x["frame"]))
                a = next(a for a in truth[x["render_id"]]["actors"] if a["label"] == lab)
                tb = _box(a["box"])
                cx, cy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
                half = max(tb[2] - tb[0], tb[3] - tb[1]) * 0.8 + 20
                x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
                x1, y1 = int(min(1280, cx + half)), int(min(720, cy + half))
                vis_img = img.copy()
                cv2.rectangle(vis_img, (int(tb[0]), int(tb[1])), (int(tb[2]), int(tb[3])), (255, 255, 255), 1)
                if x["top"]:
                    b = x["top"]
                    color = (0, 220, 0) if x["top1_ok"] else (0, 0, 255)
                    cv2.rectangle(vis_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
                crop = cv2.resize(vis_img[y0:y1, x0:x1], (240, 240))
                cv2.rectangle(crop, (0, 0), (240, 34), (0, 0, 0), -1)
                cv2.putText(crop, f"{scen} {lab}"[:30], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(crop, f"{'HIT' if x['hit'] else 'MISS'} {x['top_score']:.2f} area {x['area']*100:.1f}%",
                            (4, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0) if x["hit"] else (0, 0, 255), 1)
                tiles.append(crop)
    if not tiles:
        return
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", default=os.path.join(os.environ.get("MHD_HOME", "."), "media", "shared", "twin"))
    ap.add_argument("--scenarios", default="s1,s2,s3")
    ap.add_argument("--res", type=int, default=672)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--min-score", type=float, default=0.4)
    ap.add_argument("--out", default="logs/bench_g2.json")
    ap.add_argument("--crops", default=None)
    args = ap.parse_args()
    from mlx_vlm.generate import wired_limit

    from .detectors.sam_mlx import SamMlxEngine

    twin = Path(args.twin)
    eng = SamMlxEngine(resolution=args.res)
    result = {"when": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "res": args.res, "min_score": args.min_score,
              "load_before": load_snapshot()}
    with wired_limit(eng.model):
        for s in args.scenarios.split(","):
            if (twin / s / "truth.jsonl").exists():
                result[s] = run_scenario(eng, twin / s, args.res, args.min_score, args.every)
                print(s, {k: v for k, v in result[s].items() if k not in ("rows",)})
    result["load_after"] = load_snapshot()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    if args.crops:
        example_sheet(twin, result, Path(args.crops))


if __name__ == "__main__":
    main()
