"""Muse Voice Transcribe clients against a local fake server, plus opt-in real calls."""
import asyncio
import json
import threading
import time

import numpy as np
import pytest

from mhd_ml.agent.asr_meta import (AsrError, MetaFileASR, MetaRealtimeASR, RealtimeConfig, WarmRealtimeASR,
                                   pcm_chunks_from_array)
from mhd_ml.agent.audio import load_wav
from mhd_ml.agent.fakes import FakeRealtimeASR

import os

REAL_API = os.environ.get("MHD_REAL_API") == "1" and bool(os.environ.get("MODEL_API_KEY"))
real_api = pytest.mark.skipif(not REAL_API, reason="set MHD_REAL_API=1 (and MODEL_API_KEY) to call the Meta API")

PORT = 8485


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    # The fake server only checks the shape "Bearer <something>"; never the real key in unit tests.
    monkeypatch.setenv("MODEL_API_KEY", "unit-test-key")


@pytest.fixture
def fake():
    f = FakeRealtimeASR(port=PORT, transcript="follow the person in the blue shirt", expected_key="unit-test-key").start()
    yield f
    f.stop()


def cfg_for(fake, rate=24000, **kw):
    return RealtimeConfig(url=fake.url, sample_rate=rate, **kw)


@pytest.mark.parametrize("rate,encoding,chunk", [(24000, "PCM_24KHZ", 3840), (16000, "PCM_16KHZ", 2560)])
def test_handshake_and_chunks(fake, audio_items, rate, encoding, chunk):
    it = next(i for i in audio_items["commands"] if i["id"] == "c01" and i["sample_rate"] == rate)
    partials = []
    t = asyncio.run(MetaRealtimeASR(cfg_for(fake, rate)).transcribe_file(it["abspath"], realtime=False,
                                                                          on_partial=partials.append))
    assert t.text == "follow the person in the blue shirt"
    s = fake.sessions[-1]
    hs = s["handshake"]
    assert s["has_bearer"] and "authorization" not in hs
    assert hs["audioEncoding"] == encoding and hs["model"] == "muse-voice-transcribe-1.0"
    assert hs["mode"] == "PUSH_TO_TALK" and hs["languageBias"] == ["English"]
    assert "follow" in hs["keywords"] and "blue" in hs["keywords"]
    assert max(s["chunk_sizes"]) == chunk                  # 80 ms frames
    x, sr = load_wav(it["abspath"])
    assert s["bytes"] == len(x) * 2                        # every sample sent, 16-bit
    assert s["ended"] and partials and t.details["close_code"] == 1000


def test_realtime_pacing(fake):
    """Paced mode sends 80 ms chunks on a real-time schedule (the server closes on bursts or stalls)."""
    x = np.zeros(24000, dtype=np.float32)                  # 1.0 s
    t0 = time.monotonic()
    asyncio.run(MetaRealtimeASR(cfg_for(fake)).transcribe_array(x, 24000, realtime=True))
    s = fake.sessions[-1]
    spread = s["last_audio"] - s["first_audio"]
    assert 0.8 <= spread <= 1.3, spread
    assert time.monotonic() - t0 < 2.5


def test_audio_buffered_until_ack(audio_items):
    """The socket opens while the user speaks; audio before the ack is buffered, none is lost."""
    f = FakeRealtimeASR(port=PORT + 1, ack_delay_s=0.4, expected_key="unit-test-key").start()
    try:
        x = np.ones(24000, dtype=np.float32) * 0.1
        t = asyncio.run(MetaRealtimeASR(cfg_for(f)).transcribe_array(x, 24000, realtime=True))
        assert t.text and f.sessions[-1]["bytes"] == 24000 * 2
    finally:
        f.stop()


def test_handshake_rejected_raises():
    f = FakeRealtimeASR(port=PORT + 2, expected_key="some-other-key").start()
    try:
        with pytest.raises(AsrError):
            asyncio.run(MetaRealtimeASR(cfg_for(f)).transcribe_array(np.zeros(4800, np.float32), 24000,
                                                                       realtime=False))
    finally:
        f.stop()


def test_missing_key_raises(monkeypatch, fake):
    monkeypatch.delenv("MODEL_API_KEY")
    with pytest.raises(AsrError):
        asyncio.run(MetaRealtimeASR(cfg_for(fake)).transcribe_array(np.zeros(4800, np.float32), 24000,
                                                                      realtime=False))


def test_bad_sample_rate():
    with pytest.raises(ValueError):
        RealtimeConfig(sample_rate=44100).encoding


def test_warm_session_streams_silence_then_hands_over(fake):
    async def go():
        asr = WarmRealtimeASR(cfg_for(fake))
        await asr.start()
        await asyncio.sleep(0.8)                       # warm session open, streaming silence
        warm = fake.sessions[-1]
        assert warm["bytes"] > 0 and not warm["ended"]
        t_take = time.monotonic()
        s = await asr.take()
        take_ms = (time.monotonic() - t_take) * 1000
        assert s is not None and s.session_id
        chunks = pcm_chunks_from_array(np.zeros(4800, np.float32), 24000)

        async def src():
            for c in chunks:
                yield c

        t = await asr.transcribe_stream(src(), session=s)
        await asyncio.sleep(0.6)                       # the next warm session is opening
        n_sessions = len(fake.sessions)
        await asr.close()
        return t, take_ms, asr.warm_hits, n_sessions

    t, take_ms, hits, n = asyncio.run(go())
    assert t.text == "follow the person in the blue shirt"
    assert take_ms < 50 and hits == 1
    assert n >= 2                                       # a fresh warm session replaced the used one


def test_warm_falls_back_to_fresh_session_when_none_ready(fake):
    async def go():
        asr = WarmRealtimeASR(cfg_for(fake))           # start() never called: nothing warm
        t = await asr.transcribe_array(np.zeros(4800, np.float32), 24000, realtime=False)
        await asr.close()
        return t, asr.warm_misses

    t, misses = asyncio.run(go())
    assert t.text and misses == 1


# ---- file endpoint against a local fake ----------------------------------------------------------------
def test_file_endpoint_multipart(audio_items):
    import uvicorn
    from fastapi import FastAPI, Request

    seen = {}
    app = FastAPI()

    @app.post("/v1/asr/transcribe")
    async def transcribe(request: Request):
        form = await request.form()
        seen["auth"] = request.headers.get("authorization", "")
        seen["accept"] = request.headers.get("accept")
        seen["request"] = json.loads(form["request"]) if isinstance(form["request"], str) else \
            json.loads((await form["request"].read()).decode())
        audio = form["audio"]
        seen["audio_bytes"] = len(await audio.read())
        return {"sessionId": "fake", "transcript": "follow the red car", "audioDurationMs": 1000, "turns": []}

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8488, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.02)
    try:
        it = next(i for i in audio_items["commands"] if i["id"] == "c02" and i["sample_rate"] == 24000)
        t = asyncio.run(MetaFileASR(url="http://127.0.0.1:8488/v1/asr/transcribe").transcribe_file(it["abspath"]))
    finally:
        server.should_exit = True
        th.join(timeout=5)
    assert t.text == "follow the red car"
    assert seen["auth"] == "Bearer unit-test-key" and seen["accept"] == "application/json"
    assert seen["request"]["model"] == "muse-voice-transcribe-1.0" and seen["request"]["audioEncoding"] == "WAV"
    assert seen["audio_bytes"] > 44


# ---- real API (opt-in) ------------------------------------------------------------------------------------
@real_api
def test_real_realtime_call(audio_items, monkeypatch):
    monkeypatch.undo()                                  # use the real key from env.sh
    it = next(i for i in audio_items["commands"] if i["id"] == "c02" and i["sample_rate"] == 24000)
    t = asyncio.run(MetaRealtimeASR().transcribe_file(it["abspath"], realtime=True))
    assert "red car" in t.text.lower()
    assert t.details["close_code"] == 1000


@real_api
def test_real_file_call(audio_items, monkeypatch):
    monkeypatch.undo()
    it = next(i for i in audio_items["commands"] if i["id"] == "c03" and i["sample_rate"] == 16000)
    t = asyncio.run(MetaFileASR().transcribe_file(it["abspath"]))
    assert "brown dog" in t.text.lower()


def test_final_before_endstream_does_not_end_the_wait():
    """If the server ever sends final:true mid-stream, finish() still waits for the post-endStream final."""
    f = FakeRealtimeASR(port=PORT + 3, transcript="follow the red car", early_final="follow the",
                        final_delay_s=0.3, expected_key="unit-test-key").start()
    try:
        t = asyncio.run(MetaRealtimeASR(cfg_for(f)).transcribe_array(np.zeros(24000, np.float32), 24000,
                                                                       realtime=False))
    finally:
        f.stop()
    assert t.text == "follow the red car"                 # cumulative: the last final holds everything
    assert t.details["finals"] == ["follow the", "follow the red car"]


def test_chunked_resample_has_no_boundary_dips():
    from mhd_ml.agent.audio import resample

    x = np.full(1920, 0.5, dtype=np.float32)              # one 80 ms chunk at 24 kHz
    y = resample(x, 24000, 16000)
    assert len(y) == 1280 and np.allclose(y, 0.5, atol=1e-6)
