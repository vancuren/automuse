"""G1 benchmark: SAM 3.1 MLX detection and tracking latency, plus the OpenCV trackers.

Run (ml venv, nothing else heavy on the GPU):
  $MHD_ML_PY -m mhd_ml.perception.bench --images media/bench/coco --out logs/bench_g1.json

Measures, on 1280x720 images:
- text-prompted detection at each --res (square model input), p50/p95 over --runs runs, with a
  breakdown: preprocess, ViT backbone, heads (DETR + masks + NMS + polygon);
- SAM 3.1 multiplex tracker steps at 1008 (fresh backbone every frame, which is what a moving
  camera needs) and the propagate-only part;
- VitTrack, KCF and CSRT update times at 640x360, idle and while SAM detection runs in parallel.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time

import numpy as np

from .benchutil import load_bench_images, load_snapshot, pan_sequence, pan_truth, stats
from .types import WORK_H, WORK_W, iou


def bench_detect(eng, images, res: int, runs: int, warmup: int = 3) -> dict:
    eng.set_resolution(res)
    rows = []
    for i in range(warmup + runs):
        name, img, phrase = images[i % len(images)]
        tm: dict = {}
        dets = eng.detect(img, phrase, timings=tm)
        if i >= warmup:
            rows.append({**tm, "n": len(dets), "top": dets[0].score if dets else 0.0, "name": name})
    out = {k: stats([r[k] for r in rows]) for k in ("total_ms", "pre_ms", "backbone_ms", "head_ms")}
    out["mean_dets"] = round(float(np.mean([r["n"] for r in rows])), 2)
    out["mean_top_score"] = round(float(np.mean([r["top"] for r in rows])), 3)
    return out


def bench_sam_tracker(eng, img, phrase: str, steps: int) -> dict:
    """Detect on frame 0, start the SAM 3.1 tracker, then time steps with a fresh backbone."""
    eng.set_resolution(1008)
    zoom = 1.25
    frames, offsets = pan_sequence(img, n=steps + 4, step_px=20, zoom=zoom)
    src = eng.detect(img, phrase)
    if not src:
        return {"error": f"no detection for {phrase!r}"}
    src_box = src[0].box
    dets0 = eng.detect(frames[0], phrase)
    if not dets0:
        return {"error": "no detection on frame 0"}
    t0 = time.perf_counter()
    state = eng.track_init(eng.last_features, [dets0[0].mask])
    init_ms = (time.perf_counter() - t0) * 1e3
    step_ms, bb_ms, prop_ms, ious, scores = [], [], [], [], []
    for i in range(1, len(frames)):
        t0 = time.perf_counter()
        feats = eng.backbone(eng.preprocess(frames[i]))
        t1 = time.perf_counter()
        r = eng.track_step(state, feats, i, n_objects=1, run_mem_encoder=True)
        t2 = time.perf_counter()
        if i > 2:                                   # skip two warm-up steps
            step_ms.append((t2 - t0) * 1e3)
            bb_ms.append((t1 - t0) * 1e3)
            prop_ms.append((t2 - t1) * 1e3)
        b = r.boxes[0] * (1280 / WORK_W) if len(r.boxes) else np.zeros(4)
        ious.append(iou(tuple(b), pan_truth(src_box, zoom, offsets[i])))
        scores.append(float(r.scores[0]) if len(r.scores) else 0.0)
    return {"init_ms": round(init_ms, 1), "step_ms": stats(step_ms), "backbone_ms": stats(bb_ms),
            "propagate_ms": stats(prop_ms), "iou_vs_truth_mean": round(float(np.mean(ious)), 3),
            "iou_vs_truth_min": round(float(np.min(ious)), 3),
            "score_mean": round(float(np.mean(scores)), 3)}


def make_cv_tracker(kind: str):
    import cv2

    if kind == "vit":
        p = cv2.TrackerVit_Params()
        p.net = os.path.join(os.environ.get("MHD_MODELS", ""), "opencv", "object_tracking_vittrack_2023sep.onnx")
        return cv2.TrackerVit_create(p)
    if kind == "kcf":
        return cv2.TrackerKCF_create()
    if kind == "csrt":
        return cv2.TrackerCSRT_create()
    raise ValueError(kind)


def bench_cv_tracker(kind: str, frames, offsets, src_box, zoom: float) -> dict:
    import cv2

    small = [cv2.resize(f, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA) for f in frames]
    s = WORK_W / 1280
    tb = pan_truth(src_box, zoom, offsets[0])
    tr = make_cv_tracker(kind)
    t0 = time.perf_counter()
    tr.init(small[0], (int(tb[0] * s), int(tb[1] * s), int((tb[2] - tb[0]) * s), int((tb[3] - tb[1]) * s)))
    init_ms = (time.perf_counter() - t0) * 1e3
    times, ious, oks = [], [], 0
    for i in range(1, len(small)):
        t0 = time.perf_counter()
        ok, bb = tr.update(small[i])
        times.append((time.perf_counter() - t0) * 1e3)
        oks += int(bool(ok))
        b = (bb[0] / s, bb[1] / s, (bb[0] + bb[2]) / s, (bb[1] + bb[3]) / s)
        ious.append(iou(b, pan_truth(src_box, zoom, offsets[i])) if ok else 0.0)
    return {"init_ms": round(init_ms, 1), "update_ms": stats(times[2:]),
            "iou_vs_truth_mean": round(float(np.mean(ious)), 3), "ok_frac": round(oks / (len(small) - 1), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="media/bench/coco")
    ap.add_argument("--res", default="1008,672,504,336")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", default="logs/bench_g1.json")
    ap.add_argument("--skip-sam", action="store_true", help="only the OpenCV trackers")
    args = ap.parse_args()

    images = load_bench_images(args.images)
    assert images, f"no images in {args.images}"
    result: dict = {"when": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "load_before": load_snapshot(),
                    "images": [n for n, _, _ in images], "runs": args.runs}
    skier = next(img for n, img, _ in images if n.endswith("0785"))

    # OpenCV trackers alone first (CPU), on the pan sequence of the skier.
    zoom = 1.25
    frames, offsets = pan_sequence(skier, n=args.runs + 4, step_px=20, zoom=zoom)
    src_box = (561.0, 76.0, 985.0, 660.0)    # SAM's box for the skier in the 1280x720 source
    result["cv_idle"] = {k: bench_cv_tracker(k, frames, offsets, src_box, zoom) for k in ("vit", "kcf", "csrt")}
    print(json.dumps(result["cv_idle"], indent=1))

    if not args.skip_sam:
        import mlx.core as mx
        from mlx_vlm.generate import wired_limit

        from .detectors.sam_mlx import SamMlxEngine

        eng = SamMlxEngine()
        result["load_s"] = round(eng.load_s, 2)
        with wired_limit(eng.model):
            result["detect"] = {}
            for res in [int(r) for r in args.res.split(",")]:
                result["detect"][str(res)] = bench_detect(eng, images, res, args.runs)
                print(res, json.dumps(result["detect"][str(res)]))
            result["sam_tracker_1008"] = bench_sam_tracker(eng, skier, "person in red jacket", args.runs)
            print("sam tracker", json.dumps(result["sam_tracker_1008"]))

            # OpenCV trackers while SAM detection runs continuously in another thread (the service's
            # real situation: detector on the GPU at 1-2 Hz, tracker on the CPU at camera rate).
            eng.set_resolution(1008)
            stop = threading.Event()
            n_bg = [0]

            def background():
                while not stop.is_set():
                    eng.detect(images[n_bg[0] % len(images)][1], "person")
                    n_bg[0] += 1

            th = threading.Thread(target=background, daemon=True)
            th.start()
            time.sleep(1.0)
            result["cv_with_sam_detect"] = {k: bench_cv_tracker(k, frames, offsets, src_box, zoom)
                                            for k in ("vit", "kcf", "csrt")}
            stop.set()
            th.join(timeout=10)
            result["cv_with_sam_detect"]["background_detects"] = n_bg[0]
            print(json.dumps(result["cv_with_sam_detect"], indent=1))
        result["mlx_peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    result["load_after"] = load_snapshot()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
