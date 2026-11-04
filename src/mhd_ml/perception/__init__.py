"""Perception service: find the subject from words (SAM 3.1), track it, and report evidence.

Layout:
- detectors/sam_mlx.py  SAM 3.1 on this Mac through mlx-vlm (ml venv only).
- detectors/sam_api.py  SAM 3.1 through the Meta Model API (sam-3.1 on /v1/responses).
- trackers.py           one init/update interface over VitTrack, KCF, CSRT, and the SAM 3.1 MLX tracker.
- fusion.py             Kalman filter on bearing and size, association, re-seeding, color score.
- service.py            the process: pulls frames from the backend, sends SubjectObs.
- replay.py             run the same pipeline over a video file or a folder of frames.

Perception reports evidence (boxes, scores, ages). The autopilot alone decides state changes.
"""
