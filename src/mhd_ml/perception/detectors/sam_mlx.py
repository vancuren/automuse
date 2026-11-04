"""SAM 3.1 on this Mac through mlx-vlm 0.7.11 (mlx_vlm.models.sam3_1). ml venv only.

APIs used (read from the mlx-vlm source):
- Sam3Predictor(model, processor, score_threshold) wraps text encoding with a cache.
- sam3_1.generate._get_backbone_features(model, pixel_values): the ViT backbone (the expensive part).
- sam3_1.generate._detect_with_backbone(predictor, feats, prompts, image_size, threshold):
  FPN neck + DETR encoder/decoder + mask decoder for one or more text prompts, with NMS.
- model.track_init(feats, masks) / model.track_step(state, feats, frame_idx): the multiplex
  memory tracker. It only works with backbone features at the native 1008x1008 input.
- processor.image_size: the square input size (1008 native; smaller is faster, detection only).

Threading: MLX state is not shared across threads safely. Create and call one engine from a
single thread (the service runs all SAM work on one worker thread).
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from mhd.contracts import FRAME_H, FRAME_W

from ..types import WORK_H, WORK_W, Detection, clip_box, mask_to_polygon

NATIVE_RES = 1008


def default_model_dir() -> Path:
    return Path(os.environ.get("MHD_MODELS", "")) / "sam3.1-bf16"


def _offline_if_tokenizer_cached() -> None:
    """mlx-vlm loads SAM's CLIP tokenizer by hub name (openai/clip-vit-base-patch32), which makes
    transformers contact huggingface.co on every start. On flight day the Mac may have no
    internet, so once the tokenizer is in the project's HF cache we tell the hub to stay offline."""
    hf = os.environ.get("HF_HOME")
    if hf and (Path(hf) / "hub" / "models--openai--clip-vit-base-patch32" / "snapshots").exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


class SamMlxEngine:
    """Loads SAM 3.1 MLX once and exposes detection and tracking primitives."""

    def __init__(self, model_dir: Optional[str | Path] = None, resolution: int = NATIVE_RES,
                 threshold: float = 0.25):
        _offline_if_tokenizer_cached()
        import mlx.core as mx  # noqa: F401  (fail early if MLX/Metal is missing)
        from mlx_vlm.models.sam3.generate import Sam3Predictor
        from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
        from mlx_vlm.utils import load_model

        model_dir = Path(model_dir) if model_dir else default_model_dir()
        if not (model_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"SAM 3.1 MLX weights not found in {model_dir}")
        t0 = time.perf_counter()
        self.model = load_model(model_dir)
        self.processor = Sam31Processor.from_pretrained(str(model_dir))
        self.predictor = Sam3Predictor(self.model, self.processor, score_threshold=threshold)
        self.threshold = threshold
        self.lock = threading.RLock()     # serializes every MLX call (detector and SAM tracker)
        self.set_resolution(resolution)
        self.load_s = time.perf_counter() - t0

    # ----- configuration -------------------------------------------------------------------
    def set_resolution(self, resolution: int) -> None:
        """Square input size. Only 1008 supports the SAM tracker."""
        self.resolution = int(resolution)
        self.processor.image_size = self.resolution

    @property
    def can_track(self) -> bool:
        return self.resolution >= NATIVE_RES

    # ----- primitives ----------------------------------------------------------------------
    def preprocess(self, image_bgr: np.ndarray):
        """BGR uint8 image of any size -> MLX pixel tensor (1, R, R, 3), using the official path."""
        import mlx.core as mx
        from PIL import Image

        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        inputs = self.processor.preprocess_image(Image.fromarray(rgb))
        return mx.array(inputs["pixel_values"])

    def backbone(self, pixel_values):
        """ViT backbone features (evaluated)."""
        from mlx_vlm.models.sam3_1.generate import _get_backbone_features

        return _get_backbone_features(self.model, pixel_values)

    def detect_features(self, feats, phrase: str, threshold: Optional[float] = None,
                        size: tuple[int, int] = (WORK_W, WORK_H)):
        """Text-prompted detection on backbone features. Returns the mlx-vlm DetectionResult with
        boxes in `size` pixels (default WORK_W x WORK_H) and masks (N, H, W) uint8."""
        from mlx_vlm.models.sam3_1.generate import _detect_with_backbone

        return _detect_with_backbone(self.predictor, feats, [phrase], size,
                                     threshold if threshold is not None else self.threshold)

    def to_detections(self, result, phrase: str) -> list[Detection]:
        """mlx-vlm DetectionResult at work size -> Detections in 1280x720 pixels."""
        sx, sy = FRAME_W / WORK_W, FRAME_H / WORK_H
        dets: list[Detection] = []
        for i in range(len(result.scores)):
            b = result.boxes[i]
            box = clip_box((b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy))
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            mask = result.masks[i].astype(np.uint8) if len(result.masks) > i else None
            dets.append(Detection(box=box, score=float(result.scores[i]),
                                  mask_poly=mask_to_polygon(mask, sx, sy), mask=mask, label=phrase))
        dets.sort(key=lambda d: -d.score)
        return dets

    # ----- the detector interface ----------------------------------------------------------
    def detect(self, image_bgr: np.ndarray, phrase: str, threshold: Optional[float] = None,
               timings: Optional[dict] = None) -> list[Detection]:
        with self.lock:
            t0 = time.perf_counter()
            px = self.preprocess(image_bgr)
            t1 = time.perf_counter()
            feats = self.backbone(px)
            t2 = time.perf_counter()
            result = self.detect_features(feats, phrase, threshold)
            dets = self.to_detections(result, phrase)
            t3 = time.perf_counter()
        if timings is not None:
            timings.update(pre_ms=(t1 - t0) * 1e3, backbone_ms=(t2 - t1) * 1e3,
                           head_ms=(t3 - t2) * 1e3, total_ms=(t3 - t0) * 1e3)
        self.last_features = feats
        return dets

    # ----- SAM 3.1 tracker (native resolution only) ----------------------------------------
    def track_init(self, feats, masks_work: list[np.ndarray]):
        """Start a multiplex tracking session from work-size masks on the frame `feats` came from."""
        from mlx_vlm.models.sam3_1.generate import _init_tracker_state

        if not self.can_track:
            raise RuntimeError("the SAM 3.1 tracker needs the native 1008 input")
        return _init_tracker_state(self.model, feats, [m.astype(np.float32) for m in masks_work])

    def track_step(self, state, feats, frame_idx: int, n_objects: int = 1,
                   run_mem_encoder: bool = True):
        """Propagate tracked objects to the frame `feats` came from. Returns the mlx-vlm
        DetectionResult at work size: boxes from mask extents, scores = object presence 0..1."""
        from mlx_vlm.models.sam3_1.generate import _propagate_tracker

        return _propagate_tracker(self.model, state, feats, frame_idx, n_objects,
                                  (WORK_W, WORK_H), run_mem_encoder=run_mem_encoder)


class SamMlxDetector:
    """detect(image_bgr, phrase) -> list[Detection], with SAM 3.1 MLX."""

    name = "sam_mlx"

    def __init__(self, model_dir: Optional[str] = None, resolution: int = 672,
                 threshold: float = 0.25, engine: Optional[SamMlxEngine] = None):
        self.engine = engine or SamMlxEngine(model_dir, resolution, threshold)
        self.last_timings: dict = {}

    def detect(self, image_bgr: np.ndarray, phrase: str) -> list[Detection]:
        self.last_timings = {}
        return self.engine.detect(image_bgr, phrase, timings=self.last_timings)
