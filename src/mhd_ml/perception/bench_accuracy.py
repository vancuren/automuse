"""G1 accuracy proxy: does SAM 3.1 (and our ranking) pick the right subject among look-alikes?

Session A clips do not exist yet, so this builds look-alike scenes from COCO val2017 images
(CC BY 4.0, kept out of git in media/bench):
- two copies of the same person (the skier in 000000000785), one jacket red and one recolored
  blue (or green), pasted at random positions and scales on people-free backgrounds; queried with
  "person in blue shirt" and "person in red shirt";
- car scenes (000000001532, 000000006723, 000000005037) with one car recolored green, queried
  with "green car".
Masks for cutting out and recoloring come from SAM 3.1 MLX at 1008.

  $MHD_ML_PY -m mhd_ml.perception.bench_accuracy --out logs/bench_g1_accuracy.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .benchutil import load_snapshot, to_1280x720
from .fusion import color_fraction, color_score, size_prior, work_image
from .types import iou

HUES = {"blue": 110, "red": 176, "green": 60}       # OpenCV hue targets for recoloring


def recolor(img: np.ndarray, mask: np.ndarray, color: str) -> np.ndarray:
    """Shift the hue of masked pixels to `color`, keep shading (V), make them saturated."""
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = mask.astype(bool)
    hsv[..., 0][m] = HUES[color]
    hsv[..., 1][m] = np.maximum(hsv[..., 1][m], 170)
    v = hsv[..., 2][m].astype(int)
    hsv[..., 2][m] = np.clip(v * 0.85 + 20, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def full_masks(eng, img, phrase, threshold=0.3):
    """Detections with full-size (1280x720) masks, best first."""
    with eng.lock:
        feats = eng.backbone(eng.preprocess(img))
        r = eng.detect_features(feats, phrase, threshold, size=(1280, 720))
    order = np.argsort(-r.scores)
    return [(r.boxes[i], float(r.scores[i]), r.masks[i].astype(np.uint8)) for i in order]


def paste(canvas, cut, cut_mask, x, y):
    """Alpha-blend a cut-out (feathered mask) onto the canvas with its top-left at (x, y)."""
    import cv2

    h, w = cut.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    a = cv2.GaussianBlur(cut_mask.astype(np.float32), (5, 5), 0)[y0 - y:y1 - y, x0 - x:x1 - x, None]
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (region * (1 - a) + cut[y0 - y:y1 - y, x0 - x:x1 - x] * a).astype(np.uint8)
    ys, xs = np.nonzero(cut_mask[y0 - y:y1 - y, x0 - x:x1 - x])
    return (float(xs.min() + x0), float(ys.min() + y0), float(xs.max() + x0 + 1), float(ys.max() + y0 + 1))


def build_people(eng, bench: Path, out_dir: Path, n: int, rng) -> list[dict]:
    import cv2

    src = to_1280x720(cv2.imread(str(bench / "coco" / "000000000785.jpg")))
    person = full_masks(eng, src, "person")[0]
    jacket = full_masks(eng, src, "red jacket")
    pm = person[2]
    jm = jacket[0][2] & pm if jacket else None
    if jm is None or jm.sum() < 500:
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        jm = (((hsv[..., 0] <= 8) | (hsv[..., 0] >= 165)) & (hsv[..., 1] > 80)).astype(np.uint8) & pm
    x1, y1, x2, y2 = (int(v) for v in person[0])
    variants = {"red": src, "blue": recolor(src, jm, "blue"), "green": recolor(src, jm, "green")}
    cuts = {c: (im[y1:y2, x1:x2].copy(), pm[y1:y2, x1:x2].copy()) for c, im in variants.items()}
    bgs = [to_1280x720(cv2.imread(str(bench / "unused" / f"00000000{k}.jpg"))) for k in ("3255", "2923")]
    scenes = []
    for k in range(n):
        bg = bgs[k % len(bgs)].copy()
        pair = ("red", "blue") if k % 3 else ("red", "green") if k % 2 else ("blue", "green")
        order = list(pair) if rng.random() < 0.5 else list(pair)[::-1]
        truth = {}
        xs = [rng.integers(60, 420), rng.integers(700, 1000)]
        for slot, color in enumerate(order):
            cut, m = cuts[color]
            s = rng.uniform(0.55, 0.85)
            cw, ch = int(cut.shape[1] * s), int(cut.shape[0] * s)
            c2 = cv2.resize(cut, (cw, ch), interpolation=cv2.INTER_AREA)
            m2 = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_NEAREST)
            if rng.random() < 0.5:
                c2, m2 = c2[:, ::-1].copy(), m2[:, ::-1].copy()
            y = int(rng.integers(40, max(41, 720 - ch - 10)))
            truth[color] = paste(bg, c2, m2, int(xs[slot]), y)
        path = out_dir / f"people_{k:02d}.jpg"
        cv2.imwrite(str(path), bg, [cv2.IMWRITE_JPEG_QUALITY, 90])
        scenes.append({"file": str(path), "kind": "people", "truth": truth})
    return scenes


def build_cars(eng, bench: Path, out_dir: Path, per_image: int) -> list[dict]:
    import cv2

    scenes = []
    for name in ("000000001532", "000000006723", "000000005037"):
        img = to_1280x720(cv2.imread(str(bench / "coco" / f"{name}.jpg")))
        cars = [c for c in full_masks(eng, img, "car", 0.4) if (c[0][2] - c[0][0]) * (c[0][3] - c[0][1]) > 2500]
        for j, (box, score, m) in enumerate(cars[:per_image]):
            out = recolor(img, m, "green")
            path = out_dir / f"car_{name[-4:]}_{j}.jpg"
            cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
            scenes.append({"file": str(path), "kind": "car", "truth": {"green": [float(v) for v in box]}})
    return scenes


def evaluate(eng, scenes, res: int, min_score: float = 0.4, tie_ratio: float = 0.85) -> dict:
    import cv2

    eng.set_resolution(res)
    rows = []
    for sc in scenes:
        img = cv2.imread(sc["file"])
        work = work_image(img, False)
        queries = ([(f"person in {c} shirt", c, "person") for c in sc["truth"]] if sc["kind"] == "people"
                   else [("green car", "green", "car")])
        for phrase, color, cat in queries:
            t0 = time.perf_counter()
            dets = eng.detect(img, phrase, threshold=0.2)
            ms = (time.perf_counter() - t0) * 1e3
            truth = tuple(sc["truth"][color])
            raw_ok = bool(dets) and iou(dets[0].box, truth) > 0.5
            ranked = sorted(((d.score * color_score(color_fraction(work, d.box, d.mask, color, cat))
                              * size_prior(d.box), d) for d in dets if d.score >= min_score),
                            key=lambda x: -x[0])
            rank_ok = bool(ranked) and iou(ranked[0][1].box, truth) > 0.5
            tie = len(ranked) >= 2 and ranked[1][0] >= tie_ratio * ranked[0][0] and iou(ranked[0][1].box, ranked[1][1].box) < 0.3
            others = [d.score for d in dets if iou(d.box, truth) < 0.3]
            rows.append({"file": Path(sc["file"]).name, "phrase": phrase, "ms": round(ms, 1),
                         "raw_top1_ok": raw_ok, "ranked_top1_ok": rank_ok, "tie": tie,
                         "right_score": max([d.score for d in dets if iou(d.box, truth) > 0.5], default=0.0),
                         "best_wrong_score": max(others, default=0.0), "n": len(dets)})
    def frac(key, kind):
        sel = [r for r in rows if (kind in r["phrase"])]
        return round(sum(r[key] for r in sel) / max(1, len(sel)), 3), len(sel)
    return {"rows": rows,
            "people_raw_top1": frac("raw_top1_ok", "shirt"), "people_ranked_top1": frac("ranked_top1_ok", "shirt"),
            "people_ties": frac("tie", "shirt"),
            "car_raw_top1": frac("raw_top1_ok", "car"), "car_ranked_top1": frac("ranked_top1_ok", "car"),
            "car_ties": frac("tie", "car")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="media/bench")
    ap.add_argument("--n-people", type=int, default=12)
    ap.add_argument("--cars-per-image", type=int, default=4)
    ap.add_argument("--res", default="672,1008")
    ap.add_argument("--out", default="logs/bench_g1_accuracy.json")
    args = ap.parse_args()
    from mlx_vlm.generate import wired_limit

    from .detectors.sam_mlx import SamMlxEngine

    bench = Path(args.bench)
    out_dir = bench / "composites"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    eng = SamMlxEngine(resolution=1008)
    result = {"when": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "load_before": load_snapshot()}
    with wired_limit(eng.model):
        scenes = build_people(eng, bench, out_dir, args.n_people, rng) + build_cars(eng, bench, out_dir, args.cars_per_image)
        result["scenes"] = scenes
        for res in [int(r) for r in args.res.split(",")]:
            result[str(res)] = evaluate(eng, scenes, res)
            print(res, {k: v for k, v in result[str(res)].items() if k != "rows"})
    result["load_after"] = load_snapshot()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
