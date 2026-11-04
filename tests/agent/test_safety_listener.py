"""Safety listener on `say` audio: detection, latency after speech end, and no false triggers."""
import time

import numpy as np
import pytest

from mhd_ml.agent.audio import load_wav, speech_bounds
from mhd_ml.agent.safety_listener import CommandSender, SafetyConfig, SafetyListener, SampleRing, latency_after_speech_end
from mhd_ml.agent.words import edit_distance, match_safety, safety_action_for_word

TARGET_S = 1.5        # PLAN 4.6: "land" detected within 1.5 s of speech end


# ---- word matching (no audio) ---------------------------------------------------------------------
@pytest.mark.parametrize("text,action", [
    ("land", "land"), ("okay drone please land now", "land"), ("LAND!", "land"), ("lan", "land"),
    ("abort abort", "land"), ("emergency", "land"), ("stop", "hover"), ("hover there", "hover"),
    ("hold your position", "hover"), ("freeze", "hover"), ("stopp", "hover"), ("stop nowland", "land"),
    ("landd", "land"), ("hovr right there", "hover"), ("stop, then land", "land"), ("aboard aboard", "land"),
])
def test_match_positive(text, action):
    h = match_safety(text)
    assert h is not None and h.action == action
    assert h.command == ("land" if action == "land" else "pause")


@pytest.mark.parametrize("text", [
    "the island looks beautiful", "take a picture of the landscape", "hand me the remote", "a gold necklace",
    "the coffee shop", "the top of the hill", "jumped over the fence", "cover the camera lens",
    "please cover the camera lands", "follow the person in the blue shirt", "lane", "stops here", "story time",
    "", "holland", "portland",
])
def test_match_negative(text):
    assert match_safety(text) is None


def test_partial_ignores_last_word_and_truncated_first_word():
    assert match_safety("take a picture of the land", partial=True) is None       # may become "landscape"
    assert match_safety("please land now", partial=True).action == "land"
    assert match_safety("land the drone", partial=True, truncated_start=True) is None   # may be "is-land"
    assert match_safety("so land the drone", partial=True, truncated_start=True).action == "land"


def test_edit_distance():
    assert edit_distance("land", "lend") == 1 and edit_distance("stop", "shop") == 1
    assert edit_distance("abort", "aboard") == 2 and edit_distance("hover", "hovr") == 1


def test_fuzzy_off_is_exact_only():
    assert safety_action_for_word("stolk", fuzzy=False) is None
    assert safety_action_for_word("stolk", fuzzy=True)[0] == "hover"
    assert safety_action_for_word("stalk", fuzzy=True) is None       # English: utterance-level rule only
    assert safety_action_for_word("land", fuzzy=False)[0] == "land"


def test_sample_ring():
    r = SampleRing(10)
    r.append(np.arange(7, dtype=np.float32))
    r.append(np.arange(7, 15, dtype=np.float32))
    assert list(r.get(0, 15)) == list(range(5, 15))
    r.append(np.arange(15, 40, dtype=np.float32))
    assert list(r.get(30, 40)) == list(range(30, 40))


# ---- on audio files ---------------------------------------------------------------------------------
def run_file(recognizer, path):
    lst = SafetyListener(recognizer=recognizer)
    x, sr = load_wav(path, target_sr=16000)
    _, end = speech_bounds(x, sr)
    hits = lst.process_array(x)
    return hits, end


def test_safety_words_detected_fast(audio_items, recognizer):
    results = []
    for it in audio_items["safety"]:
        hits, end = run_file(recognizer, it["abspath"])
        match = [h for h in hits if h.action == it["expected"]]
        assert match, f"{it['id']}: expected {it['expected']}, got {[(h.word, h.transcript) for h in hits]}"
        lat = latency_after_speech_end(match[0], end)
        results.append((it["id"], lat))
        assert lat <= TARGET_S, f"{it['id']}: {lat:.2f} s after speech end"
    assert len(results) >= 15


def test_land_mid_sentence_latency(audio_items, recognizer):
    it = next(i for i in audio_items["safety"] if i["id"] == "land_mid" and i["sample_rate"] == 16000)
    hits, end = run_file(recognizer, it["abspath"])
    assert hits[0].command == "land"
    assert latency_after_speech_end(hits[0], end) <= TARGET_S


def test_24k_input_is_resampled(audio_items, recognizer):
    it = next(i for i in audio_items["safety"] if i["sample_rate"] == 24000)
    lst = SafetyListener(recognizer=recognizer)
    x, sr = load_wav(it["abspath"])            # feed 24 kHz chunks directly
    assert sr == 24000
    hits = []
    for i in range(0, len(x), 1920):
        hits += lst.feed(x[i:i + 1920], sample_rate=24000)
    hits += lst.flush()
    assert any(h.action == "land" for h in hits)


def test_no_false_triggers_on_neutral_sentences(audio_items, recognizer):
    neutral = audio_items["neutral"]
    assert len(neutral) >= 20
    false = []
    for it in neutral:
        hits, _ = run_file(recognizer, it["abspath"])
        if hits:
            false.append((it["id"], [(h.word, h.transcript) for h in hits]))
    assert not false, false


def test_silence_and_noise_do_not_trigger(recognizer):
    lst = SafetyListener(recognizer=recognizer)
    rng = np.random.default_rng(1)
    x = np.concatenate([np.zeros(16000 * 2, np.float32), rng.normal(0, 0.01, 16000 * 3).astype(np.float32)])
    assert lst.process_array(x) == []


def test_one_command_per_utterance(audio_items, recognizer):
    it = next(i for i in audio_items["safety"] if i["id"] == "abort")        # "abort, abort"
    hits, _ = run_file(recognizer, it["abspath"])
    assert len([h for h in hits if h.action == "land"]) == 1


def test_stop_then_land_escalates(audio_items, recognizer):
    it = next(i for i in audio_items["safety"] if i["id"] == "stop_then_land")
    hits, _ = run_file(recognizer, it["abspath"])
    assert hits[0].action == "hover" and hits[-1].action == "land"


def test_on_hit_callback_and_config(audio_items, recognizer):
    seen = []
    cfg = SafetyConfig.from_dict({"min_silence_s": 0.3, "debounce_s": 2.0, "unknown_key": 1})
    lst = SafetyListener(cfg, recognizer=recognizer, on_hit=seen.append)
    it = next(i for i in audio_items["safety"] if i["id"] == "stop_single")
    lst.process_file(it["abspath"])
    assert seen and seen[0].command == "pause" and cfg.debounce_s == 2.0


# ---- delivery to the backend --------------------------------------------------------------------------
def test_command_sender_http(audio_items, recognizer):
    from mhd_ml.agent.fakes import FakeBackend

    with FakeBackend(port=8404) as fb:
        results = []
        sender = CommandSender(fb.http_base, on_result=lambda h, r: results.append(r))
        lst = SafetyListener(recognizer=recognizer, on_hit=sender.submit)
        it = next(i for i in audio_items["safety"] if i["id"] == "land_mid" and i["sample_rate"] == 16000)
        lst.process_file(it["abspath"])
        t0 = time.monotonic()
        while not results and time.monotonic() - t0 < 3:
            time.sleep(0.02)
        sender.close()
        assert results and results[0]["ok"] and results[0]["via"] == "http"
        assert results[0]["reply"]["flight_state"] == "LANDING"
        assert fb.commands[0]["command"] == "land" and fb.commands[0]["type"] == "cmd"
        assert results[0]["ms"] < 500


def test_command_sender_falls_back_to_ws_callable():
    from mhd_ml.agent.words import SafetyHit

    sent = []
    sender = CommandSender("http://127.0.0.1:8409", ws_send=lambda cmd: sent.append(cmd) or True,
                           timeout_s=0.3)
    results = []
    sender.on_result = lambda h, r: results.append(r)
    sender.submit(SafetyHit(action="hover", command="pause", word="stop", canonical="stop", fuzzy=False,
                            transcript="stop"))
    t0 = time.monotonic()
    while not results and time.monotonic() - t0 < 3:
        time.sleep(0.02)
    sender.close()
    assert results[0]["ok"] and results[0]["via"] == "ws"
    assert sent[0].command == "pause"


# ---- propeller noise (synthetic proxy; the real check is Session B) -----------------------------------
def test_prop_noise_alone_never_triggers(recognizer):
    from mhd_ml.agent.make_test_audio import prop_noise

    n = prop_noise(30, seed=7)
    env = 1 + 0.8 * np.sin(2 * np.pi * np.arange(len(n)) / 16000 / 7.0)     # drone coming and going
    lst = SafetyListener(recognizer=recognizer)
    assert lst.process_array((n * env * 0.2).astype(np.float32)) == []


def test_prop_noise_10db(audio_items, recognizer):
    """At 10 dB SNR: most safety words still caught, no false triggers on neutral speech."""
    from mhd_ml.agent.make_test_audio import mix_at_snr, prop_noise

    det, total, false = 0, 0, []
    for group in ("safety", "neutral"):
        for i, it in enumerate(audio_items[group]):
            if it["sample_rate"] != 16000:
                continue
            x, _ = load_wav(it["abspath"], target_sr=16000)
            y = mix_at_snr(x, prop_noise(len(x) / 16000 + 0.1, seed=i), 10.0)
            hits = SafetyListener(recognizer=recognizer).process_array(y)
            if group == "safety":
                total += 1
                det += any(h.action == it["expected"] for h in hits)
            elif hits:
                false.append((it["id"], [h.transcript for h in hits]))
    assert not false, false
    assert det >= 0.7 * total, f"{det}/{total}"      # measured 13/17 at 10 dB


def test_backlog_skips_partials_but_still_detects(audio_items, recognizer):
    """A 10 s backlog (CPU was busy): partial decodes are skipped, the segment decode still fires."""
    import queue
    import threading

    it = next(i for i in audio_items["safety"] if i["id"] == "land_long")
    x, sr = load_wav(it["abspath"], target_sr=16000)
    q = queue.Queue()
    for i in range(0, len(x), 1280):
        q.put(x[i:i + 1280])
    q.put(np.zeros(16000, np.float32))            # trailing silence closes the segment
    lst = SafetyListener(recognizer=recognizer)
    stop = threading.Event()
    th = threading.Thread(target=lst.run_queue, args=(q, 16000, stop), daemon=True)
    th.start()
    t0 = time.monotonic()
    while not lst.hits and time.monotonic() - t0 < 10:
        time.sleep(0.02)
    stop.set()
    th.join(timeout=2)
    assert lst.hits and lst.hits[0].action == "land"
    assert lst.partials_skipped > 0 and lst.max_backlog_s > 1.0


# ---- the matcher must not fire on everyday English (code review finding) ----------------------------
@pytest.mark.parametrize("text", [
    "tell me about the person in red", "they laid it down", "a lank rope", "the emergence of spring",
    "they are lending us a car", "all aboard", "stalk him", "the stops are close", "it stopped raining",
    "she hovered nearby", "the lands up north",
])
def test_everyday_english_is_not_a_command(text):
    assert match_safety(text) is None, text


def test_dictionary_sweep_only_known_words_trigger():
    """No word of /usr/share/dict/words triggers unless it is a listed safety word or spelling."""
    import os

    from mhd.intent_spec import SAFETY_WORDS
    from mhd_ml.agent.words import CTC_VARIANTS, INFLECTIONS, _dictionary

    if not os.path.exists("/usr/share/dict/words"):
        pytest.skip("no system dictionary")
    allowed = set(SAFETY_WORDS) | set(INFLECTIONS) | set(CTC_VARIANTS)
    hits = {w for w in _dictionary() if w.isalpha() and safety_action_for_word(w)}
    assert hits <= allowed, sorted(hits - allowed)


def test_ambiguous_spellings_need_a_pure_safety_utterance():
    assert match_safety("aboard aboard").action == "land"
    assert match_safety("aboard now").action == "land"
    assert match_safety("stalk stalk").action == "hover"
    assert match_safety("all aboard the train") is None


class ScriptedRecognizer:
    """Returns the queued transcripts in order (for partial-decode logic tests)."""

    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, audio, sample_rate=16000):
        return self.texts.pop(0) if self.texts else ""


def test_trailing_safety_word_fires_once_it_stops_changing():
    """Noise holds the VAD open after "okay land": the second identical partial accepts the last word."""
    lst = SafetyListener(recognizer=ScriptedRecognizer(["okay land", "okay land"]))
    lst._ring.append(np.zeros(16000 * 2, np.float32))
    lst._fed, lst._speech_start = 16000 * 2, 0
    assert lst._partial_decode() == []                 # could still become "landscape"
    hits = lst._partial_decode()                       # 0.4 s later, nothing new: it was "land"
    assert hits and hits[0].action == "land"


def test_growing_partial_does_not_fire_on_a_prefix():
    lst = SafetyListener(recognizer=ScriptedRecognizer(["a picture of the land", "a picture of the landscape"]))
    lst._ring.append(np.zeros(16000 * 2, np.float32))
    lst._fed, lst._speech_start = 16000 * 2, 0
    assert lst._partial_decode() == [] and lst._partial_decode() == []
