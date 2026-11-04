"""G0 for SAM: hosted sam-3.1 round trip on single images through the Meta Model API.

  $MHD_ML_PY -m mhd_ml.perception.bench_api --n640 30 --n1280 10 --out logs/bench_g0.json

Each call: one noun phrase + one JPEG data URL, metadata include_confidence/one_bit, parsed with
meta_sam_parser. The benchmark uses a 10 s deadline so slow calls are measured, and reports how
many would have missed the service's 2.5 s deadline. Costs $0.0025 per call (spend cap enforced).
"""
from __future__ import annotations

import argparse
import json
import os
import time

from .benchutil import load_bench_images, load_snapshot, stats
from .detectors.sam_api import USD_PER_IMAGE, SamApiDetector
from .spend import SpendMeter


def run_batch(images, n: int, w: int, h: int, meter: SpendMeter) -> dict:
    det = SamApiDetector(send_w=w, send_h=h, timeout_s=10.0, meter=meter)
    rows = []
    try:
        for i in range(n):
            name, img, phrase = images[i % len(images)]
            det._cache.clear()                 # measure real calls, not cache hits
            dets = det.detect(img, phrase)
            t = dict(det.last_timings)
            rows.append({"name": name, "phrase": phrase, "n": len(dets), "err": det.last_error,
                         "top": dets[0].score if dets else None, "text": det.last_text,
                         "boxes": [d.box for d in dets], **t})
            print(i, name, phrase, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in t.items()},
                  "err" if det.last_error else "", (det.last_error or "")[:120])
    finally:
        det.close()
    ok = [r for r in rows if not r["err"] and "rtt_ms" in r]
    return {
        "send_size": f"{w}x{h}", "calls": n, "ok": len(ok),
        "errors": [r["err"] for r in rows if r["err"]][:5],
        "rtt_ms": stats([r["rtt_ms"] for r in ok]),
        "parse_ms": stats([r["parse_ms"] for r in ok]),
        "encode_ms": stats([r["encode_ms"] for r in ok]),
        "jpeg_kb": round(sum(r["bytes_out"] for r in ok) / max(1, len(ok)) / 1024, 1),
        "over_2500ms": sum(1 for r in ok if r["rtt_ms"] > 2500),
        "mean_dets": round(sum(r["n"] for r in ok) / max(1, len(ok)), 2),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="media/bench/coco")
    ap.add_argument("--n640", type=int, default=30)
    ap.add_argument("--n1280", type=int, default=10)
    ap.add_argument("--out", default="logs/bench_g0.json")
    ap.add_argument("--cap", type=float, default=8.0)
    args = ap.parse_args()
    images = load_bench_images(args.images)
    meter = SpendMeter("meta-api sam-3.1 image (G0 bench)", USD_PER_IMAGE, args.cap, flush_every=10)
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "load_before": load_snapshot(),
           "spent_before_usd": round(meter.total_usd, 4)}
    if args.n640:
        out["640x360"] = run_batch(images, args.n640, 640, 360, meter)
    if args.n1280:
        out["1280x720"] = run_batch(images, args.n1280, 1280, 720, meter)
    meter.flush()
    out["spent_after_usd"] = round(meter.total_usd, 4)
    out["load_after"] = load_snapshot()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for k in ("640x360", "1280x720"):
        if k in out:
            print(k, {kk: vv for kk, vv in out[k].items() if kk != "rows"})
    print("spent", out["spent_before_usd"], "->", out["spent_after_usd"])


if __name__ == "__main__":
    main()
