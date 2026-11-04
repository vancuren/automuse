"""SAM API client: parsing, no retry on 400, cache, spend cap, deadline. Offline (MockTransport)."""
from __future__ import annotations

import json
import time

import httpx
import numpy as np
import pytest

from conftest import DATA
from mhd_ml.perception.detectors.sam_api import (SamApiDetector, parse_records_fast,
                                                  parse_records_official, parse_sam_text)
from mhd_ml.perception.spend import SpendMeter, spent_so_far

FIXTURES = ["one", "crowd", "car"]


def _key(r):
    return (r.object_id, r.left, r.top, r.right, r.bottom, r.confidence, r.mask_w, r.mask_h, r.payload)


@pytest.mark.parametrize("name", FIXTURES)
def test_fast_reader_matches_meta_sam_parser(name):
    text = (DATA / f"sam_api_{name}_640x360.txt").read_text()
    fast = sorted(map(_key, parse_records_fast(text)))
    official = sorted(map(_key, parse_records_official(text)))
    assert fast and fast == official


def test_parse_to_detections_in_1280x720():
    text = (DATA / "sam_api_one_640x360.txt").read_text()
    dets = parse_sam_text(text, 640, 360, "person")
    assert len(dets) == 1
    d = dets[0]
    # The record says x1=281;y1=12;x2=494;y2=355 at 640x360, half-open, so x2 -> 495 -> 990 px.
    assert d.box == pytest.approx((562.0, 24.0, 990.0, 712.0))
    assert 0.9 < d.score <= 1.0
    assert d.mask is not None and d.mask.shape == (360, 640) and d.mask.sum() > 1000
    ys, xs = np.nonzero(d.mask)
    assert xs.min() >= 280 and xs.max() <= 495 and ys.min() >= 11 and ys.max() <= 356
    assert len(d.mask_poly) >= 4


def test_only_top_masks_decoded():
    text = (DATA / "sam_api_crowd_640x360.txt").read_text()
    dets = parse_sam_text(text, 640, 360, "person", max_masks=3)
    assert len(dets) >= 8
    assert all(d.mask is not None for d in dets[:3])
    assert all(d.mask is None for d in dets[3:])
    assert [d.score for d in dets] == sorted((d.score for d in dets), reverse=True)


def test_garbage_and_empty_output():
    assert parse_sam_text("", 640, 360) == []
    assert parse_sam_text("hello\n<0f>x<|box;x1=a|>", 640, 360) == []


def _ok_body(text):
    return {"id": "resp_1", "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}


def _detector(handler, ledger, cap=8.0, timeout_s=2.5):
    meter = SpendMeter("test sam", 0.0025, cap, path=ledger, flush_every=1)
    return SamApiDetector(api_key="test-key", transport=httpx.MockTransport(handler),
                          meter=meter, timeout_s=timeout_s, api_base="http://sam.test/v1")


def test_request_shape_cache_and_spend(ledger):
    seen = []
    text = (DATA / "sam_api_one_640x360.txt").read_text()

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.url.path.endswith("/responses")
        return httpx.Response(200, json=_ok_body(text))

    det = _detector(handler, ledger)
    img = np.full((720, 1280, 3), 127, np.uint8)
    d1 = det.detect(img, "person")
    d2 = det.detect(img, "person")            # same frame + phrase: served from the cache
    assert len(seen) == 1 and len(d1) == len(d2) == 1
    body = seen[0]
    assert body["model"] == "sam-3.1"
    content = body["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "person"}
    assert content[1]["type"] == "input_image" and content[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert body["metadata"] == {"include_confidence": "true", "mask_encoding": "one_bit"}
    det.close()
    assert spent_so_far(ledger) == pytest.approx(0.0025)
    row = json.loads(ledger.read_text().splitlines()[0])
    assert row["ws"] == "percep" and row["units"] == 1


def test_400_is_never_retried(ledger):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "bad phrase"}})

    det = _detector(handler, ledger)
    img = np.zeros((720, 1280, 3), np.uint8)
    assert det.detect(img, "a b c d e f") == []
    assert "HTTP 400" in det.last_error
    img2 = np.full((720, 1280, 3), 9, np.uint8)
    assert det.detect(img2, "a b c d e f") == []     # new frame, same phrase: not sent again
    assert len(calls) == 1
    det.close()


def test_spend_cap_blocks_calls(ledger):
    ledger.write_text(json.dumps({"ws": "percep", "est_usd": 7.999}) + "\n")
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=_ok_body(""))

    det = _detector(handler, ledger, cap=8.0)
    assert det.detect(np.zeros((720, 1280, 3), np.uint8), "person") == []
    assert calls == [] and "spend cap" in det.last_error
    det.close()


def test_deadline_returns_fast(ledger):
    def handler(request):
        time.sleep(1.5)
        return httpx.Response(200, json=_ok_body(""))

    det = _detector(handler, ledger, timeout_s=0.3)
    t0 = time.monotonic()
    assert det.detect(np.zeros((720, 1280, 3), np.uint8), "person") == []
    assert time.monotonic() - t0 < 0.8
    assert "exceeded" in det.last_error
    det.close()


def test_missing_key_disables(ledger, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    det = SamApiDetector(api_key="", meter=SpendMeter("t", 0.0025, 8.0, path=ledger))
    assert det.detect(np.zeros((720, 1280, 3), np.uint8), "person") == []
    assert "MODEL_API_KEY" in det.last_error
    det.close()
