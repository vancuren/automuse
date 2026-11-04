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


def build_detector(name: str, sam_res: int = 672, min_score: float = 0.4, spend_cap: float = 8.0):
    """The one place the service, replay and benchmarks turn CLI options into a detector."""
    if name == "sam_mlx":
        from .sam_mlx import SamMlxDetector
        return SamMlxDetector(resolution=sam_res, threshold=min(0.25, min_score))
    if name == "sam_api":
        from .sam_api import SamApiDetector
        return SamApiDetector(cap_usd=spend_cap)
    if name == "fake":
        from .fake import ColorBlobDetector
        return ColorBlobDetector()
    raise ValueError(f"unknown detector {name!r} (sam_mlx, sam_api, fake)")


def build_tracker_factory(tracker: str, detector):
    """(resolved tracker name, zero-argument factory). The SAM tracker shares the detector's engine;
    the pipeline sets its `phrase` to the mission phrase before init."""
    from ..trackers import make_tracker, resolve_tracker_name

    name = resolve_tracker_name(tracker)
    engine = getattr(detector, "engine", None)
    if name == "sam_mlx" and engine is None:
        raise ValueError("--tracker sam_mlx needs --detector sam_mlx (they share the model)")
    return name, (lambda: make_tracker(name, engine=engine))
