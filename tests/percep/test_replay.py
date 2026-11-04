"""Replay over a folder of frames: JSONL per frame, an MP4, and a summary with lock rate."""
from __future__ import annotations

import json
import shutil

import cv2
import pytest

from mhd_ml.perception.detectors.fake import ColorBlobDetector
from mhd_ml.perception.pipeline import PerceptionConfig
from mhd_ml.perception.replay import iter_frames, load_truth, replay
from synth import BLUE, RED, Person, background, render


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_replay_folder_writes_jsonl_mp4_summary(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    bg = background(11)
    people = [Person(150, 200, vx=180, shirt=BLUE), Person(900, 220, vx=-60, shirt=RED)]
    truth_lines = []
    n = 45
    for i in range(n):
        img, boxes = render(people, i / 15.0, bg)
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), img)
        truth_lines.append(json.dumps({"frame": i, "box": list(boxes[0])}))
    (tmp_path / "truth.jsonl").write_text("\n".join(truth_lines) + "\n")
    out = tmp_path / "out"
    summary = replay(iter_frames(str(frames_dir)), "person in blue shirt", "person", "blue",
                     ColorBlobDetector(), "vit", 15.0, str(out), load_truth(str(tmp_path / "truth.jsonl")),
                     det_latency_ms=300.0, cfg=PerceptionConfig(mask_counter="off"))
    lines = [json.loads(x) for x in (out / "obs.jsonl").read_text().splitlines()]
    obs = [x for x in lines if x["type"] == "subject_obs"]
    assert len(obs) == n
    assert (out / "annotated.mp4").stat().st_size > 10_000
    cap = cv2.VideoCapture(str(out / "annotated.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n
    # The fake detector boxes a person about 1.2x too wide (IoU <= 0.77 vs truth), so 0.8 here.
    assert summary["lock_rate_after_confirm"] >= 0.8 and summary["lock_rate"] >= 0.7
    assert summary["track_switches"] == 0
    assert summary["time_to_confirm_s"] is not None and summary["time_to_confirm_s"] < 2.0
    assert json.loads((out / "summary.json").read_text())["frames"] == n


COCO_SKIER = __import__("pathlib").Path(__file__).resolve().parents[2] / "media" / "bench" / "coco" / "000000000785.jpg"


@pytest.mark.skipif(not COCO_SKIER.exists() or shutil.which("ffmpeg") is None,
                    reason="COCO bench images not downloaded (media/bench is gitignored)")
def test_replay_on_a_coco_image_sequence(tmp_path):
    """A camera pan over a real photo (COCO val2017 000000000785, CC BY 4.0): the red-jacket skier
    is found by the color test double and tracked by VitTrack through the pan."""
    from mhd_ml.perception.benchutil import pan_sequence, pan_truth, to_1280x720

    src = to_1280x720(cv2.imread(str(COCO_SKIER)))
    frames, offsets = pan_sequence(src, n=40, step_px=12, zoom=1.25)
    skier = (561.0, 76.0, 985.0, 660.0)              # SAM's box in the 1280x720 source (G1 bench)
    truth = {i: pan_truth(skier, 1.25, off) for i, off in enumerate(offsets)}
    summary = replay(iter(frames), "person in red jacket", "person", "red", ColorBlobDetector(), "vit",
                     15.0, str(tmp_path / "coco"), truth, det_latency_ms=400.0,
                     cfg=PerceptionConfig(mask_counter="off", min_color_frac=0.0))
    assert summary["frames"] == 40 and summary["stats"]["tracks_started"] >= 1
    assert summary["track_switches"] == 0
    # The color double boxes the jacket (not the whole skier), so judge by overlap with truth:
    lines = [json.loads(x) for x in (tmp_path / "coco" / "obs.jsonl").read_text().splitlines()]
    boxed = [x for x in lines if x["type"] == "subject_obs" and x["box"]]
    assert len(boxed) >= 25
    from mhd_ml.perception.types import iou as _iou
    inside = sum(1 for x in boxed if _iou((x["box"]["x1"], x["box"]["y1"], x["box"]["x2"], x["box"]["y2"]),
                                          truth[x["frame_id"]]) > 0.1)
    assert inside / len(boxed) > 0.9
