from mhd.contracts import (Box, FrameMeta, Intent, Mission, SubjectObs, SubjectState, ObsSource,
                           decode_frame, encode_frame, now_ms)


def test_frame_roundtrip():
    meta = FrameMeta(frame_id=7, capture_ms=1.0, receive_ms=2.0, width=1280, height=720, render_id=42)
    m2, jpeg = decode_frame(encode_frame(meta, b"\xff\xd8abc\xff\xd9"))
    assert m2 == meta and jpeg == b"\xff\xd8abc\xff\xd9"


def test_box_geometry():
    a, b = Box(x1=0, y1=0, x2=10, y2=10), Box(x1=5, y1=5, x2=15, y2=15)
    assert a.cx == 5 and a.w == 10 and abs(a.iou(b) - 25 / 175) < 1e-9


def test_intent_strict():
    i = Intent(action="follow", subject_phrase="person in blue shirt", category="person",
               color="blue", distance="medium", reason_code="none")
    m = Mission(mission_id="m1", intent=i, transcript="follow the person in the blue shirt",
                model="keyword", latency_ms=1.0)
    assert Mission.model_validate_json(m.model_dump_json()) == m


def test_obs_defaults():
    o = SubjectObs(frame_id=1, capture_ms=now_ms(), state=SubjectState.searching, source=ObsSource.oracle)
    assert o.confirm_age_ms > 1e8 and o.box is None
