"""TTS smoke test (MMS-TTS and the `say` fallback) and Omnilingual push-to-talk transcription.

Nothing is played through the speakers unless MHD_TTS_PLAY=1 (tests run at night next to a sleeper).
"""
import os
import time

import numpy as np
import pytest

from mhd.contracts import Intent
from mhd.intent_spec import build_reply
from mhd_ml.agent.asr_local import LocalASR
from mhd_ml.agent.audio import load_wav
from mhd_ml.agent.tts import MmsTts, Speaker, say_to_file

PLAY = os.environ.get("MHD_TTS_PLAY") == "1"


@pytest.fixture(scope="module")
def mms():
    m = MmsTts()
    m.load()
    return m


def test_mms_synthesizes_and_is_intelligible(mms, recognizer):
    reply = build_reply(Intent(action="land", subject_phrase="", category="none", color="none",
                               distance="medium", reason_code="none"))
    assert reply == "Landing now."
    t0 = time.monotonic()
    wav = mms.synthesize(reply)
    synth_s = time.monotonic() - t0
    assert mms.sample_rate == 16000 and len(wav) > 0.5 * 16000
    assert np.max(np.abs(wav)) <= 0.91 and synth_s < 2.0
    # Closed loop: our own local ASR understands the synthesized reply.
    text = recognizer.transcribe(wav, mms.sample_rate)
    assert "land" in text, text


def test_mms_follow_reply(mms, recognizer):
    wav = mms.synthesize("Following the person in blue shirt.")
    text = recognizer.transcribe(wav, mms.sample_rate)
    # MMS's "blue" is heard as "valu" by Omnilingual; check the words both models agree on.
    assert "follow" in text and "person" in text and "shirt" in text, text


def test_say_fallback_renders_file(tmp_path):
    p = say_to_file("Holding position.", tmp_path / "reply.aiff")
    x, sr = load_wav(p)
    assert len(x) > 0.3 * sr


def test_speaker_queue_without_playback():
    s = Speaker(engine="mms", play=PLAY).start(warm_up=True)
    try:
        assert s.wait_ready(60)
        assert not s.mms_failed and s.warmup_ms is not None
        s.speak("Landing now.")
        time.sleep(0.05)
        assert s.wait_idle(20)
        rec = s.history[-1]
        assert rec["engine"] == "mms" and rec["started_ms"] >= rec["queued_ms"]
        # queue-to-start latency after warm-up (synthesis only, nothing else heavy running)
        assert rec["started_ms"] - rec["queued_ms"] < 1500
    finally:
        s.close()


def test_speaker_falls_back_to_say_when_mms_missing(tmp_path):
    s = Speaker(engine="mms", model_dir=str(tmp_path / "missing"), play=False).start()
    try:
        assert s.wait_ready(60) and s.mms_failed
        s.speak("Holding position.")
        time.sleep(0.05)
        assert s.wait_idle(10)
        assert s.history[-1]["engine"] == "say"
    finally:
        s.close()


def test_engine_none_is_silent():
    s = Speaker(engine="none").start()
    s.speak("anything")
    assert s.wait_idle(2) and s.history == []
    s.close()


def test_local_asr_push_to_talk(audio_items, recognizer):
    asr = LocalASR(recognizer)
    for cid, must in (("c01", ("follow", "blue", "shirt")), ("c02", ("red", "car")), ("c07", ("land",))):
        it = next(i for i in audio_items["commands"] if i["id"] == cid and i["sample_rate"] == 24000)
        t = asr.transcribe_file(it["abspath"])
        assert all(w in t.text for w in must), (cid, t.text)
        assert t.latency_ms < 1500


def test_audible_words_window():
    s = Speaker(engine="none")
    t = time.monotonic()
    s._mark_audible("Landing now.", t - 1.0, float("inf"))        # still playing
    assert s.audible_words(now=t) == {"landing", "now"}
    s._mark_audible("Landing now.", t - 1.0, t - 0.5)             # finished 0.5 s ago
    assert "landing" in s.audible_words(now=t, tail_s=1.5)
    assert s.audible_words(now=t + 2.0, tail_s=1.5) == set()       # long gone


def test_queue_keeps_non_interrupting_lines_and_drops_stale_ones():
    s = Speaker(engine="say", play=False)          # synthesizes nothing, records the order
    s.start(warm_up=False)
    try:
        s.wait_ready(10)
        s.speak("Okay.")                             # interrupting
        s.speak("Following the red car.", interrupt=False)
        s.speak("Got them. Following.", interrupt=False)
        time.sleep(0.05)
        s.wait_idle(5)
        assert [h["text"] for h in s.history] == ["Okay.", "Following the red car.", "Got them. Following."]
    finally:
        s.close()


def test_interrupting_line_drops_older_queued_lines():
    s = Speaker(engine="say", play=False)
    # Fill the queue before the worker starts, then start it: only lines of the newest generation play.
    s.speak("one", interrupt=False)
    s.speak("two", interrupt=False)
    s.speak("Landing now.")                          # interrupting: "one" and "two" are stale
    s.speak("Got them.", interrupt=False)
    s.start(warm_up=False)
    try:
        s.wait_ready(10)
        time.sleep(0.05)
        s.wait_idle(5)
        assert [h["text"] for h in s.history] == ["Landing now.", "Got them."]
    finally:
        s.close()
