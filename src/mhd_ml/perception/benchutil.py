"""Helpers shared by the perception benchmarks: test images, pan sequences, stats, load labels."""
from __future__ import annotations

import glob
import os
import re
import subprocess
from pathlib import Path

import numpy as np

# COCO val2017 images picked for the benchmark (CC BY 4.0, kept out of git under media/bench/coco).
CAR_IMAGES = {"000000001532", "000000006723", "000000005037"}


def to_1280x720(img: np.ndarray) -> np.ndarray:
    """Center-crop to 16:9 and resize to 1280x720 (the drone's frame size), no distortion."""
    import cv2

    h, w = img.shape[:2]
    target = 16 / 9
    if w / h > target:            # too wide: crop width
        nw = int(round(h * target))
        x0 = (w - nw) // 2
        img = img[:, x0:x0 + nw]
    else:                         # too tall: crop height
        nh = int(round(w / target))
        y0 = (h - nh) // 2
        img = img[y0:y0 + nh]
    return cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)


def load_bench_images(folder: str | Path) -> list[tuple[str, np.ndarray, str]]:
    """(name, 1280x720 BGR image, phrase) for every jpg in the folder; phrase is car or person."""
    import cv2

    out = []
    for f in sorted(glob.glob(str(Path(folder) / "*.jpg"))):
        name = Path(f).stem
        img = cv2.imread(f)
        if img is None:
            continue
        out.append((name, to_1280x720(img), "car" if name in CAR_IMAGES else "person"))
    return out


def pan_sequence(img: np.ndarray, n: int = 40, step_px: float = 20.0, zoom: float = 1.25):
    """Simulate a camera panning across a scene: upscale, then slide a 1280x720 window.

    step_px is the shift per frame in 1280x720 pixels (20 px per frame is roughly a 5 Hz tracker
    watching a subject drift across the frame at walking pace).
    Returns (frames, offsets): offsets[i] = (x, y) of the window in the upscaled image, so a box B
    in the 1280x720 source maps to B * zoom - (x, y) in frame i."""
    import cv2

    big = cv2.resize(img, (int(1280 * zoom), int(720 * zoom)), interpolation=cv2.INTER_LINEAR)
    max_x = big.shape[1] - 1280
    max_y = big.shape[0] - 720
    frames, offsets = [], []
    for i in range(n):
        x = int(round((i * step_px) % (2 * max_x)))
        if x > max_x:                       # bounce back and forth
            x = 2 * max_x - x
        y = int(round(max_y / 2 + (max_y / 2 - 1) * np.sin(i / 7.0)))
        frames.append(np.ascontiguousarray(big[y:y + 720, x:x + 1280]))
        offsets.append((x, y))
    return frames, offsets


def pan_truth(box, zoom: float, offset) -> tuple[float, float, float, float]:
    """Where a source-frame box appears in a pan_sequence frame (clipped to 1280x720)."""
    x, y = offset
    b = (box[0] * zoom - x, box[1] * zoom - y, box[2] * zoom - x, box[3] * zoom - y)
    return (max(0.0, b[0]), max(0.0, b[1]), min(1280.0, b[2]), min(720.0, b[3]))


def stats(values_ms: list[float]) -> dict:
    v = np.asarray(values_ms, dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "p50": round(float(np.percentile(v, 50)), 1),
            "p95": round(float(np.percentile(v, 95)), 1), "mean": round(float(v.mean()), 1),
            "min": round(float(v.min()), 1), "max": round(float(v.max()), 1)}


def gpu_utilization() -> int | None:
    """Apple GPU 'Device Utilization %' from ioreg (no root needed)."""
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                             capture_output=True, text=True, timeout=5).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def load_snapshot() -> dict:
    """What else is running that could compete for the GPU, CPU, or network (for labelling)."""
    try:
        ps = subprocess.run(["ps", "-Ao", "pcpu,command"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        ps = ""
    lines = ps.splitlines()[1:]

    def any_match(pat: str) -> bool:
        return any(re.search(pat, ln) for ln in lines)

    fetch_done = False
    log = Path(os.environ.get("MHD_HOME", ".")) / "logs" / "fetch.log"
    try:
        fetch_done = "ALL DONE" in log.read_text(errors="ignore")
    except Exception:
        pass
    busy = sorted(((float(ln.split(None, 1)[0]), ln.split(None, 1)[1][:80]) for ln in lines
                   if ln.strip() and ln.split(None, 1)[0].replace(".", "", 1).isdigit()), reverse=True)[:5]
    return {
        "gpu_util_pct": gpu_utilization(),
        "twin_chrome": any_match(r"chrome-twin|chrome-ui|user-data-dir=.*MetaHackathonDrone"),
        "glimmer_mlx_lm": any_match(r"mlx_lm|Muse-Glimmer"),
        "llama_server": any_match(r"llama-server"),
        "downloads_running": any_match(r"\bcurl\b.*huggingface|fetch_models"),
        "fetch_all_done": fetch_done,
        "top_cpu": [f"{c:.0f}% {cmd}" for c, cmd in busy],
    }
