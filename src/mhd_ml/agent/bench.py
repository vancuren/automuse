"""Gate measurements for the agent (PLAN.md 4.9 G0, and the 4.6 acceptance numbers).

    $MHD_ML_PY -m mhd_ml.agent.bench spark  --n 20      # Spark: minimal vs low, strict schema
    $MHD_ML_PY -m mhd_ml.agent.bench voice  --n 10      # Voice Transcribe: fresh session, warm session, file
    $MHD_ML_PY -m mhd_ml.agent.bench safety             # local listener on the `say` test set
    $MHD_ML_PY -m mhd_ml.agent.bench noise              # same, with synthetic propeller noise
    $MHD_ML_PY -m mhd_ml.agent.bench e2e                # 20 scripted commands: release -> reply start

Results go to $MHD_TREE/logs/agent/bench_<name>.json; paid calls are appended to the spend log.
Always say what else was running when quoting a number.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from .spend import log_spend, spark_cost, voice_cost


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100.0
    lo, hi = int(np.floor(k)), int(np.ceil(k))
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(xs) -> dict:
    return {"n": len(xs), "p50": _r(pct(xs, 50)), "p95": _r(pct(xs, 95)), "min": _r(min(xs) if xs else None),
            "max": _r(max(xs) if xs else None), "mean": _r(statistics.fmean(xs) if xs else None)}


def _r(x):
    return None if x is None else round(float(x), 1)


def out_path(name: str) -> Path:
    p = Path(os.environ.get("MHD_TREE", ".")) / "logs" / "agent" / f"bench_{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


SPARK_COMMANDS = [
    ("follow the person in the blue shirt", "follow"), ("follow the red car", "follow"),
    ("track the brown dog", "follow"), ("stay close to the woman in the green jacket", "follow"),
    ("follow the kid with the yellow balloon", "follow"), ("hover right here", "hover"), ("land now", "land"),
    ("follow john", "refuse"), ("follow him", "clarify"), ("follow the man in the black hoodie from far away", "follow"),
    ("followed the cyclist in the orange vest", "follow"), ("track the white truck", "follow"),
    ("who is that guy in the red hat", "refuse"), ("ok so um follow the uh girl in the pink dress", "follow"),
    ("folow the gray cat", "follow"), ("can you keep an eye on the black dog", "follow"),
    ("follow the person wearing a yellow raincoat", "follow"), ("spy on my neighbor", "refuse"),
    ("follow the guy with the green backpack", "follow"), ("the woman in red", "follow"),
]


async def bench_spark(n: int, efforts: list[str]) -> dict:
    from .intent import spark_client

    cmds = (SPARK_COMMANDS * ((n + len(SPARK_COMMANDS) - 1) // len(SPARK_COMMANDS)))[:n]
    clients = {e: spark_client(reasoning_effort=e, timeout_s=20.0) for e in efforts}
    rows = []
    # Interleave efforts so both see the same network conditions.
    for text, expected in cmds:
        for e in efforts:
            a = await clients[e].infer(text, drain=True)
            u = a.usage or {}
            rows.append({"effort": e, "text": text, "expected": expected, "valid": a.valid,
                         "action": a.intent.action if a.intent else None, "correct": bool(a.intent and a.intent.action == expected),
                         "subject": a.intent.subject_phrase if a.intent else None,
                         "latency_ms": round(a.latency_ms, 1), "ttft_ms": a.ttft_ms and round(a.ttft_ms, 1),
                         "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                         "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                         "error": a.error})
            print(json.dumps(rows[-1]), flush=True)
    res = {"ts": time.time(), "n_per_effort": n, "by_effort": {}}
    total_cost, total_tokens = 0.0, 0
    for e in efforts:
        rs = [r for r in rows if r["effort"] == e]
        lat = [r["latency_ms"] for r in rs if r["valid"]]
        pin = sum(r["prompt_tokens"] or 440 for r in rs)
        pout = sum(r["completion_tokens"] or 160 for r in rs)
        cost = spark_cost(pin, pout)
        total_cost += cost
        total_tokens += pin + pout
        res["by_effort"][e] = {"latency_ms": summarize(lat), "valid_rate": sum(r["valid"] for r in rs) / len(rs),
                               "action_accuracy": sum(r["correct"] for r in rs) / len(rs),
                               "ttft_ms": summarize([r["ttft_ms"] for r in rs if r["ttft_ms"]]),
                               "mean_completion_tokens": statistics.fmean([r["completion_tokens"] or 0 for r in rs]),
                               "mean_reasoning_tokens": statistics.fmean([r["reasoning_tokens"] or 0 for r in rs]),
                               "est_usd": round(cost, 4)}
    res["rows"] = rows
    log_spend("meta:muse-spark-1.3", total_tokens, total_cost, note=f"agent G0 Spark bench {n}x{len(efforts)}")
    return res


async def bench_voice(n: int, rate: int = 24000) -> dict:
    from .asr_meta import MetaFileASR, MetaRealtimeASR, RealtimeConfig, WarmRealtimeASR
    from .audio import load_wav, speech_bounds
    from .make_test_audio import items

    cmds = [i for i in items("commands") if i["sample_rate"] == rate][:n]
    cfg = RealtimeConfig(sample_rate=rate)
    fresh, warm, filer = MetaRealtimeASR(cfg), WarmRealtimeASR(cfg), MetaFileASR()
    await warm.start()
    await asyncio.sleep(2.0)
    rows, audio_s = [], 0.0
    for it in cmds:
        x, sr = load_wav(it["abspath"])
        start, end = speech_bounds(x, sr)
        # Push-to-talk released 150 ms after speech ends.
        clip = x[: int((end + 0.15) * sr)]
        audio_s += 3 * len(clip) / sr
        for kind, asr in (("realtime_fresh", fresh), ("realtime_warm", warm)):
            t0 = time.monotonic()
            t = await asr.transcribe_array(clip, sr, realtime=True)
            # wall time at which the end-of-speech sample was sent (paced from t0)
            t_speech_end = t0 + end
            t_final = t0 + (time.monotonic() - t0)
            rows.append({"kind": kind, "id": it["id"], "text": t.text, "expected": it["text"],
                         "speech_end_to_final_ms": round((t_final - t_speech_end) * 1000, 1),
                         "endstream_to_final_ms": round(t.latency_ms, 1),
                         "connect_ms": round(t.details.get("connect_ms", 0), 1)})
            print(json.dumps(rows[-1]), flush=True)
            await asyncio.sleep(0.5)
        t = await filer.transcribe_array(clip, sr)
        rows.append({"kind": "file", "id": it["id"], "text": t.text, "expected": it["text"],
                     "speech_end_to_final_ms": round(t.latency_ms + 150, 1),   # upload starts at release
                     "request_ms": round(t.latency_ms, 1)})
        print(json.dumps(rows[-1]), flush=True)
    await warm.close()
    res = {"ts": time.time(), "n": len(cmds), "sample_rate": rate, "by_kind": {}}
    for kind in ("realtime_fresh", "realtime_warm", "file"):
        rs = [r for r in rows if r["kind"] == kind]
        exact = sum(_norm(r["text"]) == _norm(r["expected"]) for r in rs)
        res["by_kind"][kind] = {"speech_end_to_final_ms": summarize([r["speech_end_to_final_ms"] for r in rs]),
                                "exact_transcripts": f"{exact}/{len(rs)}"}
        if kind != "file":
            res["by_kind"][kind]["endstream_to_final_ms"] = summarize([r["endstream_to_final_ms"] for r in rs])
            res["by_kind"][kind]["connect_ms"] = summarize([r["connect_ms"] for r in rs])
    res["rows"] = rows
    log_spend("meta:muse-voice-transcribe-1.0", round(audio_s, 1), voice_cost(audio_s + 60),
              note="agent G0 voice bench (audio seconds; +60 s warm-session silence)")
    return res


def _norm(t: str) -> str:
    from .words import normalize

    return normalize(t)


def bench_safety() -> dict:
    from .asr_local import get_shared_recognizer
    from .audio import load_wav, speech_bounds
    from .make_test_audio import items
    from .safety_listener import SafetyListener, latency_after_speech_end

    rec = get_shared_recognizer()
    rows = []
    for group in ("safety", "neutral"):
        for it in items(group):
            lst = SafetyListener(recognizer=rec)
            x, sr = load_wav(it["abspath"], target_sr=16000)
            _, end = speech_bounds(x, sr)
            t0 = time.monotonic()
            hits = lst.process_array(x)
            wall = time.monotonic() - t0
            first = next((h for h in hits if h.action == it["expected"]), None)
            rows.append({"group": group, "id": it["id"], "sr": it["sample_rate"], "expected": it["expected"],
                         "hits": [(h.command, h.word, h.kind) for h in hits],
                         "latency_s": round(latency_after_speech_end(first, end), 3) if first else None,
                         "decodes": lst.decodes, "decode_ms_mean": round(lst.decode_ms_total / max(1, lst.decodes), 1),
                         "cpu_fraction": round(wall / (len(x) / sr), 3)})
    pos = [r for r in rows if r["group"] == "safety"]
    neg = [r for r in rows if r["group"] == "neutral"]
    lats = [r["latency_s"] for r in pos if r["latency_s"] is not None]
    return {"ts": time.time(), "detected": f"{len(lats)}/{len(pos)}",
            "false_triggers": f"{sum(bool(r['hits']) for r in neg)}/{len(neg)}",
            "latency_after_speech_end_s": {"p50": round(pct(lats, 50), 3), "max": round(max(lats), 3)},
            "decode_ms_mean": round(statistics.fmean(r["decode_ms_mean"] for r in rows), 1),
            "listener_cpu_fraction_mean": round(statistics.fmean(r["cpu_fraction"] for r in rows), 3),
            "rows": rows}


def bench_noise(snrs=(None, 15, 10, 5, 0), threshold: float = 0.5) -> dict:
    """Safety listener with synthetic propeller noise mixed in at each SNR (same seeds every run)."""
    from .asr_local import get_shared_recognizer
    from .audio import load_wav, speech_bounds
    from .make_test_audio import items, mix_at_snr, prop_noise
    from .safety_listener import SafetyConfig, SafetyListener, latency_after_speech_end

    rec = get_shared_recognizer()
    data = {g: [(it, load_wav(it["abspath"], 16000)[0], i) for i, it in enumerate(items(g)) if it["sample_rate"] == 16000]
            for g in ("safety", "neutral")}
    table = {}
    for snr in snrs:
        det, fp, lat = 0, 0, []
        for g, lst_ in data.items():
            for it, x, i in lst_:
                y = x if snr is None else mix_at_snr(x, prop_noise(len(x) / 16000 + 0.1, seed=i), snr)
                _, end = speech_bounds(x, 16000)
                hits = SafetyListener(SafetyConfig(vad_threshold=threshold), recognizer=rec).process_array(y)
                if g == "safety":
                    m = [h for h in hits if h.action == it["expected"]]
                    if m:
                        det += 1
                        lat.append(latency_after_speech_end(m[0], end))
                elif hits:
                    fp += 1
        table["clean" if snr is None else f"{snr} dB"] = {
            "detected": f"{det}/{len(data['safety'])}", "false_triggers": f"{fp}/{len(data['neutral'])}",
            "max_latency_s": round(max(lat), 3) if lat else None}
        print(json.dumps({"snr": snr, **table["clean" if snr is None else f"{snr} dB"]}), flush=True)
    return {"ts": time.time(), "vad_threshold": threshold, "table": table}


async def bench_e2e(intent_mode: str = "spark") -> dict:
    """20 scripted commands through the real service: fake backend, real Voice Transcribe (warm),
    real Spark, MMS-TTS synthesized but muted. Release -> reply start = ptt_up -> TTS playback start."""
    from mhd.contracts import now_ms

    from .audio import MicHub, load_wav, play_file_into
    from .fakes import FakeBackend
    from .make_test_audio import items
    from .service import AgentService, load_config
    from .tts import Speaker

    cmds = [i for i in items("commands") if i["sample_rate"] == 24000]
    fb = FakeBackend(port=8406).start()
    cfg = load_config()
    cfg.update({"backend": fb.ws_base, "intent": intent_mode})
    cfg.setdefault("safety", {})["enabled"] = True
    hub = MicHub(sample_rate=24000)
    speaker = Speaker(engine="mms", play=False).start()
    svc = AgentService(cfg, hub=hub, start_mic=False, speaker=speaker,
                       metrics_path=out_path("e2e_utterances").with_suffix(".jsonl"))
    await svc.start()
    speaker.wait_ready(60)
    await asyncio.sleep(2.0)                # warm Voice Transcribe session ready
    rows, audio_s = [], 0.0
    try:
        for it in cmds:
            from .audio import speech_bounds

            x, sr = load_wav(it["abspath"])
            _, end = speech_bounds(x, sr)
            clip = x[: int((end + 0.15) * sr)]
            audio_s += len(clip) / sr
            n_ev = len(svc.events)
            n_tts = len(speaker.history)
            await svc.ptt_down()
            await asyncio.get_running_loop().run_in_executor(None, play_file_into, hub, clip, sr, True)
            t_up = now_ms()
            await svc.ptt_up()
            await svc.wait_idle(20)
            t_wait = time.monotonic()
            while (len(speaker.history) <= n_tts or speaker.history[-1]["started_ms"] is None) and \
                    time.monotonic() - t_wait < 10:
                await asyncio.sleep(0.02)
            evs = svc.events[n_ev:]
            final = next((e for e in evs if e["kind"] == "final"), None)
            intent = next((e for e in evs if e["kind"] == "intent"), None)
            tts = speaker.history[-1] if len(speaker.history) > n_tts else {}
            ok_action = bool(intent and intent["data"]["intent"]["action"] == it["expected_action"])
            phrase = intent["data"]["intent"]["subject_phrase"] if intent else ""
            ok_subject = (not it["expected_subject"]) or _subject_ok(phrase, it["expected_subject"])
            rows.append({"id": it["id"], "text": it["text"], "transcript": final and final["text"],
                         "action": intent and intent["data"]["intent"]["action"], "subject": phrase,
                         "model": intent and intent["data"]["model"], "reason": intent and intent["data"]["reason"],
                         "correct": ok_action and ok_subject,
                         "release_to_final_ms": final and round(final["ts_ms"] - t_up, 1),
                         "release_to_intent_ms": intent and round(intent["ts_ms"] - t_up, 1),
                         "release_to_reply_start_ms": tts.get("started_ms") and round(tts["started_ms"] - t_up, 1),
                         "tts_synth_ms": tts.get("synth_ms") and round(tts["synth_ms"], 1)})
            print(json.dumps(rows[-1]), flush=True)
            await asyncio.sleep(0.8)
    finally:
        await svc.stop()
        fb.stop()
    lat = [r["release_to_reply_start_ms"] for r in rows if r["release_to_reply_start_ms"]]
    res = {"ts": time.time(), "intent_mode": intent_mode, "n": len(rows),
           "correct": f"{sum(r['correct'] for r in rows)}/{len(rows)}",
           "release_to_reply_start_ms": summarize(lat),
           "release_to_final_ms": summarize([r["release_to_final_ms"] for r in rows if r["release_to_final_ms"]]),
           "release_to_intent_ms": summarize([r["release_to_intent_ms"] for r in rows if r["release_to_intent_ms"]]),
           "safety_hits": len(svc.safety_hits), "rows": rows}
    log_spend("meta:muse-voice-transcribe-1.0", round(audio_s, 1), voice_cost(audio_s + 60),
              note="agent e2e bench audio seconds")
    n_spark = sum(1 for r in rows if r["model"] == "muse-spark-1.3" or r["reason"] != "safety_fast_path")
    log_spend("meta:muse-spark-1.3", n_spark * 600, spark_cost(n_spark * 440, n_spark * 160),
              note="agent e2e bench (estimated tokens)")
    return res


def _subject_ok(got: str, want: str) -> bool:
    """Same subject for segmentation: every color and thing word of the expected phrase appears, and
    the head noun matches, where "person" stands for any person noun (man, guy, cyclist, woman...)."""
    from .intent import PERSON

    syn = {"guy": "man", "kid": "child"}
    g = [syn.get(w, w) for w in got.lower().split()]
    gset = set(g)
    for w in want.lower().split():
        w = syn.get(w, w)
        if w in ("in", "with", "the", "a", "wearing"):
            continue
        if w in gset:
            continue
        if w in PERSON and ("person" in gset or any(x in PERSON for x in gset)):
            continue
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["spark", "voice", "safety", "noise", "e2e"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--efforts", default="minimal,low")
    ap.add_argument("--rate", type=int, default=24000)
    ap.add_argument("--intent", default="spark")
    ap.add_argument("--note", default="", help="what else was running (stored with the result)")
    args = ap.parse_args()
    if args.what == "spark":
        res = asyncio.run(bench_spark(args.n, args.efforts.split(",")))
    elif args.what == "voice":
        res = asyncio.run(bench_voice(min(args.n, 20), args.rate))
    elif args.what == "safety":
        res = bench_safety()
    elif args.what == "noise":
        res = bench_noise()
    else:
        res = asyncio.run(bench_e2e(args.intent))
    res["note"] = args.note
    p = out_path(args.what if args.what != "voice" else f"voice_{args.rate // 1000}k")
    p.write_text(json.dumps(res, indent=1, default=str))
    summary = {k: v for k, v in res.items() if k != "rows"}
    print(json.dumps(summary, indent=1, default=str))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
