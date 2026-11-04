"""Shared helpers for the training workstream: paths, spend logging, and the Meta Model API client.

Secrets come from the environment (env.sh loads $MHD_HOME/.env). Nothing here prints or logs a key.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx

TREE = Path(os.environ.get("MHD_TREE", Path(__file__).resolve().parents[1]))
HOME = Path(os.environ.get("MHD_HOME", TREE))
DATA_DIR = TREE / "training" / "data"
LOG_DIR = TREE / "logs"
SPEND_LOG = HOME / "logs" / "spend.jsonl"

META_BASE = "https://api.meta.ai/v1"
SPARK_MODEL = "muse-spark-1.3"
# Published Meta Model API prices for muse-spark-1.3, USD per token (dossier, 2026-11-03).
SPARK_IN_USD = 1.25e-6
SPARK_CACHED_USD = 0.15e-6
SPARK_OUT_USD = 4.25e-6


def log_spend(service: str, units: int, est_usd: float, note: str = "") -> None:
    """Append one JSON line per paid batch to the shared spend log (never includes secrets)."""
    SPEND_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": round(time.time(), 3), "ws": "train", "service": service,
           "units": int(units), "est_usd": round(float(est_usd), 6)}
    if note:
        rec["note"] = note
    with open(SPEND_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def spent_usd(service_prefix: str) -> float:
    """Total estimated spend logged by this workstream for services starting with the prefix."""
    total = 0.0
    if not SPEND_LOG.exists():
        return 0.0
    for line in SPEND_LOG.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("ws") == "train" and str(rec.get("service", "")).startswith(service_prefix):
            total += float(rec.get("est_usd", 0.0))
    return total


def spark_cost(usage: dict) -> float:
    """USD estimate from an OpenAI-style usage block (reasoning tokens are billed as output)."""
    pin = int(usage.get("prompt_tokens", 0))
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
    pout = int(usage.get("completion_tokens", 0))
    return (pin - cached) * SPARK_IN_USD + cached * SPARK_CACHED_USD + pout * SPARK_OUT_USD


def meta_chat(messages: list[dict], *, schema: Optional[dict] = None, schema_name: str = "out",
              model: str = SPARK_MODEL, reasoning_effort: str = "minimal", max_tokens: int = 2000,
              timeout: float = 120.0, client: Optional[httpx.Client] = None) -> dict[str, Any]:
    """One chat completion on the Meta Model API. Returns the parsed JSON response.

    Raises httpx.HTTPStatusError on a non-2xx answer (the error body never contains our key).
    """
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set (source env.sh)")
    body: dict[str, Any] = {"model": model, "messages": messages,
                            "reasoning_effort": reasoning_effort, "max_completion_tokens": max_tokens}
    if schema is not None:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
    own = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        r = client.post(f"{META_BASE}/chat/completions", json=body,
                        headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    finally:
        if own:
            client.close()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
