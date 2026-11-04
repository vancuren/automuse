"""Printable ArUco marker for Session B system identification (mhd.governor.sysid).

sysid detects cv2.aruco.DICT_4X4_50 (any id unless --marker-id is given) and turns the marker's
side in pixels into range with --marker-size (metres, default 0.15). This page carries id 7 (the
screen-counter page uses ids 0-3, so the two can never be confused) at 150 mm, on US Letter at
300 DPI, with a 100 mm scale bar to check the print scale.

    source env.sh
    $MHD_PY scripts/make_aruco.py                      # -> docs/img/aruco_sysid.png
    $MHD_PY scripts/make_aruco.py --id 7 --size-mm 150 --out docs/img/aruco_sysid.png

Print at 100% ("Actual size", never "Fit to page"), measure the black square, and pass that side
in metres to sysid: --marker-id 7 --marker-size 0.150.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TREE = Path(__file__).resolve().parents[1]
DPI = 300
PAGE_IN = (8.5, 11.0)        # US Letter
FONTS = ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"]


def mm_to_px(mm: float) -> float:
    return mm / 25.4 * DPI


def font(size: int):
    for f in FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make(marker_id: int, size_mm: float) -> tuple[Image.Image, dict]:
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    cells = 6                                            # 4x4 bits + 1-cell black border each side
    cell_px = int(round(mm_to_px(size_mm) / cells))      # whole pixels per cell: crisp edges
    side_px = cell_px * cells
    marker = cv2.aruco.generateImageMarker(d, marker_id, side_px)
    W, H = int(PAGE_IN[0] * DPI), int(PAGE_IN[1] * DPI)
    page = Image.new("L", (W, H), 255)
    x0 = (W - side_px) // 2
    y0 = int(mm_to_px(30))
    page.paste(Image.fromarray(marker), (x0, y0))
    draw = ImageDraw.Draw(page)
    real_mm = side_px / DPI * 25.4
    # 100 mm scale bar with 10 mm ticks, to check the printer did not scale the page
    by = y0 + side_px + int(mm_to_px(22))
    bx0 = (W - int(round(mm_to_px(100)))) // 2
    bx1 = bx0 + int(round(mm_to_px(100)))
    draw.line([(bx0, by), (bx1, by)], fill=0, width=6)
    for i in range(11):
        x = bx0 + int(round(mm_to_px(10 * i)))
        draw.line([(x, by - (40 if i % 5 == 0 else 22)), (x, by)], fill=0, width=5)
    f_big, f_mid, f_small = font(64), font(46), font(40)
    draw.text((W // 2, by + 20), "100 mm: check this with a ruler", fill=0, font=f_small, anchor="ma")
    lines = [
        (f_big, f"ArUco DICT_4X4_50  id {marker_id}  -  black square {real_mm:.1f} mm"),
        (f_mid, "Session B system identification (mhd.governor.sysid)"),
        (f_small, "Print at 100% / Actual size (never Fit to page). Measure the black square edge to edge;"),
        (f_small, f"pass it in metres: --marker-id {marker_id} --marker-size {real_mm / 1000:.3f}"),
        (f_small, "Tape it flat on the wall at camera height, 1.5-3 m in front of the drone, bright light,"),
        (f_small, "no glare, nothing else ArUco-like in view. Keep the white border around it."),
    ]
    y = by + 110
    for f, text in lines:
        draw.text((W // 2, y), text, fill=0, font=f, anchor="ma")
        y += int(f.size * 1.45)
    info = {"id": marker_id, "dictionary": "DICT_4X4_50", "side_px": side_px, "side_mm": round(real_mm, 2),
            "dpi": DPI, "page": "US Letter", "marker_box_px": [x0, y0, x0 + side_px, y0 + side_px]}
    return page, info


def verify(path: Path, marker_id: int) -> bool:
    """The page must decode with the same detector sysid uses, also in a small, skewed camera view."""
    from mhd.governor.sysid import aruco_detector, aruco_measure
    img = cv2.imread(str(path))
    det = aruco_detector()
    _, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if ids is None or marker_id not in ids.flatten().tolist():
        return False
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[180, 60], [1100, 120], [1060, 700], [220, 660]])
    view = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (1280, 720), borderValue=(90, 90, 90))
    view = cv2.GaussianBlur(view, (3, 3), 0.8)
    return aruco_measure(view, 75.0, 0.15, det, marker_id) is not None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--id", type=int, default=7, help="marker id in DICT_4X4_50 (0-49; the counter page uses 0-3)")
    ap.add_argument("--size-mm", type=float, default=150.0, help="black square side (sysid default 0.15 m)")
    ap.add_argument("--out", default=str(TREE / "docs" / "img" / "aruco_sysid.png"))
    a = ap.parse_args(argv)
    if not 0 <= a.id < 50:
        raise SystemExit("--id must be 0..49 (DICT_4X4_50)")
    if not 40 <= a.size_mm <= 190:
        raise SystemExit("--size-mm must fit a Letter page: 40..190")
    page, info = make(a.id, a.size_mm)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    page.save(out, dpi=(DPI, DPI), optimize=True)
    ok = verify(out, a.id)
    print(f"{out}: {info}; decodes with sysid's detector: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
