"""Integration: the real PerceptionService against a fake backend over WebSockets (port 8301).

The fake backend serves synthetic frames (a blue-shirt person walking) on /ws/frames and sends a
FindRequest on /ws/subject; the service runs the fake color detector (no GPU, no network) and the
VitTrack tracker, and must stream SubjectObs at camera rate that end up confirmed on the subject.
"""
from __future__ import annotations

import asyncio

import pytest

from mhd.contracts import FindRequest, Release, SubjectObs
from mhd_ml.perception.fake_backend import FakeBackend, synthetic_source
from mhd_ml.perception.service import PerceptionService, parse_args
from mhd_ml.perception.types import iou

PORT = 8301
FPS = 15.0


async def _scenario(seconds: float = 6.0, release_at: float = 4.5):
    find = FindRequest(mission_id="it-1", subject_phrase="person in blue shirt", category="person", color="blue")
    fb = FakeBackend(synthetic_source(fps=FPS), fps=FPS, find=find)
    server = await fb.serve(port=PORT)
    prod = asyncio.create_task(fb.produce())
    svc = PerceptionService(parse_args(["--backend", f"ws://127.0.0.1:{PORT}", "--detector", "fake",
                                        "--tracker", "vit", "--log-level", "WARNING"]))
    run = asyncio.create_task(svc.run(install_signals=False))
    await asyncio.sleep(release_at)
    n_before_release = len(fb.received)
    await fb.send_subject(Release(mission_id="it-1"))
    await asyncio.sleep(seconds - release_at)
    n_after = len(fb.received)
    svc.stop.set()
    await asyncio.wait_for(run, timeout=5)
    prod.cancel()
    server.close()
    await server.wait_closed()
    return fb, n_before_release, n_after


def test_service_streams_confirmed_subject_obs():
    fb, n_before, n_after = asyncio.run(_scenario())
    obs = [SubjectObs.model_validate({k: v for k, v in m.items() if k != "_rx_ms"})
           for m in fb.received if m.get("type") == "subject_obs"]
    assert len(obs) >= 30, f"only {len(obs)} observations"
    # Camera rate: one observation per served frame while the mission ran (about 15 Hz).
    span_s = (obs[-1].capture_ms - obs[0].capture_ms) / 1000.0
    assert len(obs) / span_s > 0.8 * FPS
    confirmed = [o for o in obs if o.state.value == "confirmed"]
    assert confirmed, "never confirmed"
    first_conf_s = (confirmed[0].capture_ms - obs[0].capture_ms) / 1000.0
    assert first_conf_s < 3.0                                   # found and confirmed within 3 s
    # Boxes match the truth of their own frame (joined by render_id = 100000 + frame id).
    ious = [iou((o.box.x1, o.box.y1, o.box.x2, o.box.y2), fb.truth[o.frame_id]) for o in confirmed]
    assert sum(ious) / len(ious) > 0.5
    assert all(o.render_id == 100000 + o.frame_id for o in obs)
    assert all(0 <= o.confirm_age_ms < 2000 for o in confirmed)
    assert all(o.tracker_score > 0.2 for o in confirmed)
    # Release stops the stream (allow a few in flight).
    assert n_after - n_before <= 5
