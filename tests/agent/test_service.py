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
    ev = [e for e in svc.events if e["kind"] == "intent"][-1]
    # what the UI's missionFromIntentEvent() reads
    assert ev["data"]["mission_id"] == m["mission_id"] and ev["data"]["reply"] == m["reply"]
    assert ev["data"]["model"] == m["model"] and ev["data"]["intent"] == m["intent"]
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


class RecordingSpeaker:
    """Stands in for tts.Speaker: records what would be spoken."""

    def __init__(self):
        self.lines = []

    def speak(self, text, interrupt=True):
        self.lines.append((text, interrupt))

    def wait_ready(self, timeout=0):
        return True

    def wait_idle(self, timeout=0):
        return True

    def close(self, interrupt=True):
        pass


async def run_with_speaker(backend, cfg, script, speaker):
    hub = MicHub(sample_rate=24000)
    svc = AgentService(cfg, hub=hub, start_mic=False, speaker=speaker)
    await svc.start()
    try:
        assert await wait_for(lambda: backend.agent_count() >= 1, 5)
        await script(svc, hub)
        await svc.wait_idle(15)
    finally:
        await svc.stop()
    return svc


def test_slow_intent_gets_a_short_acknowledgment(backend, llm, audio_items):
    asr = FakeRealtimeASR(port=ASR_PORT, transcript="follow the red car").start()
    llm.behaviours = {"glimmer-slow": {"delay_s": 1.0}}
    cfg = make_cfg(asr.url)
    cfg["glimmer"]["model"] = "glimmer-slow"
    cfg["tts"] = {"engine": "mms", "ack_after_s": 0.3, "ack_text": "Okay."}
    sp = RecordingSpeaker()
    try:
        async def script(svc, hub):
            await svc.simulate_file(item(audio_items, "commands", "c02", 24000)["abspath"], realtime=False)

        asyncio.run(run_with_speaker(backend, cfg, script, sp))
    finally:
        asr.stop()
        llm.behaviours = {}
    assert sp.lines[0] == ("Okay.", True)
    assert sp.lines[1] == ("Following the red car.", False)      # waits for the ack to finish


def test_fast_intent_has_no_acknowledgment(backend, llm, audio_items):
    asr = FakeRealtimeASR(port=ASR_PORT, transcript="follow the red car").start()
    cfg = make_cfg(asr.url)
    cfg["tts"] = {"engine": "mms", "ack_after_s": 1.0}
    sp = RecordingSpeaker()
    try:
        async def script(svc, hub):
            await svc.simulate_file(item(audio_items, "commands", "c02", 24000)["abspath"], realtime=False)

        asyncio.run(run_with_speaker(backend, cfg, script, sp))
    finally:
        asr.stop()
    assert sp.lines == [("Following the red car.", True)]


def test_lost_ptt_up_ends_utterance(backend, llm, audio_items):
    asr = FakeRealtimeASR(port=ASR_PORT, transcript="track the brown dog").start()
    cfg = make_cfg(asr.url)
    cfg["mic"]["max_ptt_s"] = 1.0
    n0 = len(backend.missions)
    try:
        async def script(svc, hub):
            x, sr = load_wav(item(audio_items, "commands", "c03", 24000)["abspath"])
            await svc.ptt_down()
            await asyncio.get_running_loop().run_in_executor(None, play_file_into, hub, x[: sr // 2], sr, False)
            # no ptt_up: the watchdog must end the utterance
            assert await wait_for(lambda: len(backend.missions) > n0, 6)

        svc = asyncio.run(run_with_speaker(backend, cfg, script, RecordingSpeaker()))
    finally:
        asr.stop()
    assert backend.missions[-1]["intent"]["subject_phrase"] == "brown dog"
    assert any("released automatically" in (e.get("text") or "") for e in svc.events)


def test_local_asr_mode(backend, llm, audio_items):
    cfg = make_cfg("ws://127.0.0.1:8489")
    cfg["asr"] = "local"
    n0 = len(backend.missions)

    async def script(svc, hub):
        await svc.simulate_file(item(audio_items, "commands", "c16", 24000)["abspath"], realtime=False)
        assert await wait_for(lambda: len(backend.missions) > n0, 10)

    svc = asyncio.run(run_with_speaker(backend, cfg, script, RecordingSpeaker()))
    m = backend.missions[-1]
    assert m["intent"]["subject_phrase"] == "gray cat"
    final = [e for e in svc.events if e["kind"] == "final"][-1]
    assert final["data"]["model"] == "omnilingual-ctc-300m"


def test_autopilot_announcements_are_spoken(backend, llm):
    asr = FakeRealtimeASR(port=ASR_PORT).start()
    sp = RecordingSpeaker()
    try:
        async def script(svc, hub):
            # what the gov backend relays: AgentEvent(kind="status", text=..., data={"kind": "announce"})
            backend.send_to_agents({"type": "agent_event", "kind": "status", "text": "Got them. Following.",
                                    "data": {"kind": "announce", "text": "Got them. Following."}})
            backend.send_to_agents({"type": "agent_event", "kind": "status", "text": "flight AIRBORNE -> LANDING",
                                    "data": {"kind": "flight_state"}})
            backend.send_to_agents({"type": "mission_ack", "ok": True})
            assert await wait_for(lambda: sp.lines, 3)
            await asyncio.sleep(0.2)

        asyncio.run(run_with_speaker(backend, make_cfg(asr.url), script, sp))
    finally:
        asr.stop()
    assert sp.lines == [("Got them. Following.", False)]


class EchoSpeaker(RecordingSpeaker):
    """Pretends a line is coming out of the laptop speakers right now."""

    def __init__(self, words):
        super().__init__()
        self.words = set(words)

    def audible_words(self):
        return self.words


def _hit(word, action):
    from mhd_ml.agent.words import SafetyHit

    return SafetyHit(action=action, command="land" if action == "land" else "pause", word=word,
                     canonical=word, fuzzy=False, transcript=word)


def test_own_voice_is_not_a_command(backend, llm):
    asr = FakeRealtimeASR(port=ASR_PORT).start()
    n0 = len(backend.commands)
    sp = EchoSpeaker({"landing", "now"})                 # the agent is saying "Landing now."
    try:
        async def script(svc, hub):
            svc._on_safety_hit(_hit("landing", "land"))  # its own echo: ignored
            svc._on_safety_hit(_hit("land", "land"))     # "land in now": ignored (already landing)
            svc._on_safety_hit(_hit("stop", "hover"))    # the operator: delivered
            assert await wait_for(lambda: len(backend.commands) > n0, 3)
            await asyncio.sleep(0.5)

        svc = asyncio.run(run_with_speaker(backend, make_cfg(asr.url), script, sp))
    finally:
        asr.stop()
    sent = [c["command"] for c in backend.commands[n0:]]
    assert sent == ["pause"], sent
    assert sum(1 for h in svc.safety_hits if h["result"].get("via") == "echo") == 2


def test_operator_stop_is_never_taken_for_echo(backend, llm):
    asr = FakeRealtimeASR(port=ASR_PORT).start()
    n0 = len(backend.commands)
    sp = EchoSpeaker({"following", "the", "person", "by", "stop", "sign"})   # a reply with "stop" in it
    try:
        async def script(svc, hub):
            svc._on_safety_hit(_hit("stop", "hover"))
            assert await wait_for(lambda: len(backend.commands) > n0, 3)

        asyncio.run(run_with_speaker(backend, make_cfg(asr.url), script, sp))
    finally:
        asr.stop()
    assert [c["command"] for c in backend.commands[n0:]] == ["pause"]


def test_governor_refusal_is_reported():
    from mhd_ml.agent.fakes import FakeBackend as FB

    with FB(port=8403, refuse_commands="estopped") as fb:
        asr = FakeRealtimeASR(port=ASR_PORT).start()
        try:
            cfg = make_cfg(asr.url)
            cfg["backend"] = fb.ws_base

            async def script(svc, hub):
                svc._on_safety_hit(_hit("land", "land"))
                assert await wait_for(lambda: svc.safety_hits, 3)
                await asyncio.sleep(0.3)

            hub = MicHub(sample_rate=24000)

            async def go():
                svc = AgentService(cfg, hub=hub, start_mic=False, speaker=RecordingSpeaker())
                await svc.start()
                try:
                    await script(svc, hub)
                finally:
                    await svc.stop()
                return svc

            svc = asyncio.run(go())
        finally:
            asr.stop()
    r = svc.safety_hits[0]["result"]
    assert r["ok"] and r["accepted"] is False and r["reason"] == "estopped"
    assert any(e["kind"] == "error" and "refused land" in e["text"] for e in svc.events)


def test_speakable_never_says_a_hover_word():
    from mhd_ml.agent.service import _has_hover_word, _speakable

    assert _speakable("Following the person by stop sign.", "follow") == "Following now."
    assert _speakable("Following the person in blue shirt.", "follow") == "Following the person in blue shirt."
    assert _speakable("Landing now.", "land") == "Landing now."
    assert not _has_hover_word("Holding position.") and not _has_hover_word("I lost the subject. Looking around.")
    assert _has_hover_word("Hold on, stop")
