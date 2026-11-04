"""Spend log: one JSON line per paid batch, appended to $MHD_HOME/logs/spend.jsonl (shared by workstreams)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Published prices (dev.meta.ai/docs/pricing-rate-limits, checked 2026-11-03).
SPARK_IN_PER_M = 1.25
SPARK_OUT_PER_M = 4.25
VOICE_PER_HOUR = 0.18


def spark_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens * SPARK_IN_PER_M / 1e6 + completion_tokens * SPARK_OUT_PER_M / 1e6


def voice_cost(audio_seconds: float) -> float:
    return audio_seconds / 3600.0 * VOICE_PER_HOUR


def log_spend(service: str, units: float, est_usd: float, note: str = "") -> None:
    """service like "meta:muse-spark-1.3" (the format the other workstreams use); units are tokens for
    language models and audio seconds for Voice Transcribe."""
    home = os.environ.get("MHD_HOME")
    if not home:
        return
    p = Path(home) / "logs" / "spend.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": round(time.time(), 3), "ws": "agent", "service": service,
           "units": units, "est_usd": round(est_usd, 6)}
    if note:
        rec["note"] = note
    with p.open("a") as f:
        f.write(json.dumps(rec) + "\n")
