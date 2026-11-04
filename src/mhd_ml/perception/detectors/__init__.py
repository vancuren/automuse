"""Detectors: detect(image_bgr, phrase) -> list[Detection] (boxes in 1280x720 pixels)."""
from __future__ import annotations


def make_detector(name: str, **kwargs):
    """Build a detector by name: "sam_mlx" (this Mac) or "sam_api" (Meta Model API)."""
    if name == "sam_mlx":
        from .sam_mlx import SamMlxDetector
        return SamMlxDetector(**kwargs)
    if name == "sam_api":
        from .sam_api import SamApiDetector
        return SamApiDetector(**kwargs)
    raise ValueError(f"unknown detector {name!r} (use sam_mlx or sam_api)")
