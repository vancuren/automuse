"""The agent process end to end, with every external piece faked on loopback:
fake backend (8405), mock Glimmer (8482), fake Voice Transcribe (8487). Audio is injected into the mic hub.
"""
import asyncio
import json
import time

import pytest

from mhd_ml.agent.audio import MicHub, load_wav, play_file_into
from mhd_ml.agent.fakes import FakeBackend, FakeRealtimeASR
from mhd_ml.agent.mock_llm import MockLLM
from mhd_ml.agent.service import AgentService
from mhd_ml.agent.tts import Speaker

BACKEND_PORT, LLM_PORT, ASR_PORT = 8405, 8482, 8487


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "unit-test-key")


@pytest.fixture(scope="module")
def backend():
    fb = FakeBackend(port=BACKEND_PORT).start()
    yield fb
    fb.stop()


@pytest.fixture(scope="module")
def llm():
    m = MockLLM(port=LLM_PORT, default={"delay_s": 0.2}).start()
    yield m
    m.stop()


def make_cfg(asr_url, intent="glimmer"):
    return {
        "backend": f"ws://127.0.0.1:{BACKEND_PORT}", "asr": "meta", "intent": intent,
        "mic": {"sample_rate": 24000, "preroll_s": 0.0},
        "voice_transcribe": {"url": asr_url, "warm_session": True, "final_timeout_s": 3.0},
        "glimmer": {"base_url": f"http://127.0.0.1:{LLM_PORT}/v1", "model": "glimmer-m", "label": "glimmer-30b+lora",
                    "api_key_env": None, "timeout_s": 3.0},
        "spark": {"enabled": False},
        "router": {"glimmer_window_s": 2.5, "total_timeout_s": 4.0},
        "safety": {"enabled": True},
        "tts": {"engine": "none"},
    }


def item(audio_items, group, cid, rate):
    return next(i for i in audio_items[group] if i["id"] == cid and i["sample_rate"] == rate)


async def wait_for(pred, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


async def run_session(backend, cfg, recognizer, script):
    """Start the agent, connect a fake UI, run `script(svc, ui, hub)`, return what the UI saw."""
    from websockets.asyncio.client import connect

    hub = MicHub(sample_rate=24000)
    svc = AgentService(cfg, hub=hub, start_mic=False, speaker=Speaker(engine="none"))
    seen = []
    await svc.start()
    try:
        async with connect(backend.ws_base + "/ws/agent?role=ui") as ui:
            async def reader():
                try:
                    async for raw in ui:
                        seen.append(json.loads(raw))
                except Exception:
                    pass
            rt = asyncio.create_task(reader())
            assert await wait_for(lambda: backend.agent_count() >= 1, 5)
            await script(svc, ui, hub)
            await svc.wait_idle(10)
            await asyncio.sleep(0.2)
            rt.cancel()
    finally:
        await svc.stop()
    return svc, seen


async def push_to_talk(ui, hub, path):
    x, sr = load_wav(path)
    await ui.send(json.dumps({"type": "agent_event", "kind": "ptt_down"}))
    await asyncio.sleep(0.1)
    await asyncio.get_running_loop().run_in_executor(None, play_file_into, hub, x, sr, False)
    await asyncio.sleep(0.1)
    await ui.send(json.dumps({"type": "agent_event", "kind": "ptt_up"}))


def test_push_to_talk_to_mission(backend, llm, audio_items, recognizer):
    asr = FakeRealtimeASR(port=ASR_PORT, transcript="follow the person in the blue shirt").start()
    n0 = len(backend.missions)
    try:
        async def script(svc, ui, hub):
            await push_to_talk(ui, hub, item(audio_items, "commands", "c01", 24000)["abspath"])
            assert await wait_for(lambda: len(backend.missions) > n0, 10)

        svc, seen = asyncio.run(run_session(backend, make_cfg(asr.url), recognizer, script))
    finally:
        asr.stop()
    m = backend.missions[-1]
    assert m["via"] == "http" and m["type"] == "mission"
    assert m["intent"]["action"] == "follow" and m["intent"]["subject_phrase"] == "person in blue shirt"
    assert m["model"] == "glimmer-30b+lora" and m["latency_ms"] > 0
    assert m["reply"] == "Following the person in blue shirt."
    kinds = [e.get("kind") for e in seen if e.get("type") == "agent_event"]
    for k in ("partial", "final", "intent", "reply"):
        assert k in kinds, (k, kinds)
    status = [e for e in svc.events if e["kind"] == "status"][0]
    assert status["data"]["safety_listener"] and status["data"]["backend_connected"]
    assert "safety_word" not in kinds               # the mission command has no safety word
    assert asr.sessions and asr.sessions[0]["handshake"]["audioEncoding"] == "PCM_24KHZ"


def test_clarify_sends_no_mission(backend, llm, audio_items, recognizer):
    asr = FakeRealtimeASR(port=ASR_PORT, transcript="follow him").start()
    n0 = len(backend.missions)
    try:
        async def script(svc, ui, hub):
            await push_to_talk(ui, hub, item(audio_items, "commands", "c09", 24000)["abspath"])
            assert await wait_for(lambda: any(e["kind"] == "reply" for e in svc.events), 10)

        svc, seen = asyncio.run(run_session(backend, make_cfg(asr.url), recognizer, script))
    finally:
        asr.stop()
    assert len(backend.missions) == n0
    reply = [e for e in svc.events if e["kind"] == "reply"][-1]
    assert reply["text"].startswith("Who should I follow?")


def test_spoken_land_during_flight_reaches_backend(backend, llm, audio_items, recognizer):
    """No push-to-talk: the safety listener alone hears "land" and POSTs /command."""
    asr = FakeRealtimeASR(port=ASR_PORT).start()
    n0 = len(backend.commands)
    try:
        async def script(svc, ui, hub):
            it = item(audio_items, "safety", "land_mid", 24000)
            x, sr = load_wav(it["abspath"])
            t0 = time.monotonic()
            await asyncio.get_running_loop().run_in_executor(None, play_file_into, hub, x, sr, True)
            assert await wait_for(lambda: len(backend.commands) > n0, 5)
            svc._t_cmd = backend.commands[-1]["t"] - t0
            svc._audio_s = len(x) / sr

        svc, seen = asyncio.run(run_session(backend, make_cfg(asr.url), recognizer, script))
    finally:
        asr.stop()
    c = backend.commands[-1]
    assert c["command"] == "land" and c["via"] == "http"
    # Real-time replay: the command must arrive within 1.5 s of the end of the speech in the file
    # (speech ends 1.0 s before the end of the file: the generator pads 1.0 s of silence).
    speech_end = svc._audio_s - 1.0
    assert svc._t_cmd - speech_end <= 1.5, (svc._t_cmd, speech_end)
    assert any(e.get("kind") == "safety_word" for e in seen)


def test_voice_transcribe_down_falls_back_to_local_asr(backend, llm, audio_items, recognizer):
    n0 = len(backend.missions)
    cfg = make_cfg("ws://127.0.0.1:8489")          # nothing listens there

    async def script(svc, ui, hub):
        await push_to_talk(ui, hub, item(audio_items, "commands", "c02", 24000)["abspath"])
        assert await wait_for(lambda: len(backend.missions) > n0, 15)

    svc, seen = asyncio.run(run_session(backend, cfg, recognizer, script))
    m = backend.missions[-1]
    assert m["intent"]["subject_phrase"] == "red car", m
    assert any(e["kind"] == "status" and "local" in (e.get("text") or "") for e in svc.events)


def test_keyword_only_mode_and_hover_command(backend, llm, audio_items, recognizer):
    asr = FakeRealtimeASR(port=ASR_PORT, transcript="hover right here").start()
    n_cmd, n_m = len(backend.commands), len(backend.missions)
    n_req = len(llm.requests)
    try:
        async def script(svc, ui, hub):
            await push_to_talk(ui, hub, item(audio_items, "commands", "c06", 24000)["abspath"])
            assert await wait_for(lambda: len(backend.missions) > n_m, 10)

        svc, seen = asyncio.run(run_session(backend, make_cfg(asr.url, intent="keyword"), recognizer, script))
    finally:
        asr.stop()
    assert backend.missions[-1]["intent"]["action"] == "hover" and backend.missions[-1]["model"] == "keyword"
    assert any(c["command"] == "pause" for c in backend.commands[n_cmd:])
    assert len(llm.requests) == n_req                  # keyword mode never calls a model
