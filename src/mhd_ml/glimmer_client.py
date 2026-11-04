"""Minimal client for the fine-tuned Muse Glimmer intent model, for the agent to call.

Edge mode:  mlx_lm.server on this Mac (scripts/serve_glimmer.sh), http://127.0.0.1:8080/v1, no key.
Cloud mode: Fireworks on-demand deployment (training/fireworks.py), https://api.fireworks.ai/inference/v1,
            Bearer FIREWORKS_API_KEY, model '<model>#accounts/<acct>/deployments/<deployment>'.

Both are OpenAI-style chat completions. The request uses GLIMMER_SYSTEM_PROMPT exactly as in training and
temperature 0; the reply's "content" holds the intent JSON (Glimmer's reasoning channel arrives separately
as "reasoning" / "reasoning_content"), parsed with mhd.intent_spec.parse_intent like every other path.

    from mhd_ml.glimmer_client import GlimmerClient
    g = GlimmerClient.edge()                        # or GlimmerClient.cloud()
    intent, info = g.intent("follow the person in the blue shirt")   # (Intent | None, {...})

This module never raises on a bad answer or a timeout: it returns (None, info) so the caller can fall
back (hedged routing with Spark, then the keyword grammar).
"""
from __future__ import annotations

import os
import time
from typing import Optional

import httpx

from mhd.contracts import Intent
from mhd.intent_spec import GLIMMER_SYSTEM_PROMPT, parse_intent

EDGE_URL = "http://127.0.0.1:8080/v1"
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1"
BASE_MODEL = "accounts/fireworks/models/muse-glimmer-30b"
DEFAULT_DEPLOYMENT = "glimmer-intent"
DEFAULT_LORA = "glimmer-intent-v1"


class GlimmerClient:
    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None, timeout_s: float = 6.0,
                 max_tokens: int = 200, label: str = "glimmer-30b+lora"):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.label = label          # goes into Mission.model
        self._client = httpx.Client(timeout=timeout_s)

    @classmethod
    def edge(cls, url: str = EDGE_URL, timeout_s: float = 8.0) -> "GlimmerClient":
        """Local mlx_lm.server with the LoRA (about 4-5 s per command on the M5 Pro)."""
        return cls(url, "default_model", None, timeout_s, label="glimmer-30b+lora")

    @classmethod
    def cloud(cls, deployment: str = DEFAULT_DEPLOYMENT, lora: Optional[str] = DEFAULT_LORA,
              timeout_s: float = 2.5) -> "GlimmerClient":
        """Fireworks deployment. lora=None means base Glimmer on that deployment."""
        acct = os.environ["FIREWORKS_ACCOUNT_ID"]
        dep = f"accounts/{acct}/deployments/{deployment}"
        model = f"accounts/{acct}/models/{lora}#{dep}" if lora else f"{BASE_MODEL}#{dep}"
        return cls(FIREWORKS_URL, model, os.environ["FIREWORKS_API_KEY"], timeout_s,
                   max_tokens=200 if lora else 1200, label="glimmer-30b+lora" if lora else "glimmer-30b")

    def intent(self, transcript: str) -> tuple[Optional[Intent], dict]:
        body = {"model": self.model, "temperature": 0.0, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": GLIMMER_SYSTEM_PROMPT},
                             {"role": "user", "content": transcript}]}
        t0 = time.perf_counter()
        info: dict = {"model": self.label}
        try:
            r = self._client.post(self.url, json=body, headers=self.headers)
            info["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            if r.status_code != 200:
                info["error"] = f"HTTP {r.status_code}: {r.text[:200]}"
                return None, info
            msg = r.json()["choices"][0]["message"]
            content = msg.get("content") or ""
            info["raw"] = content
            info["reasoning"] = msg.get("reasoning_content") or msg.get("reasoning")
            got = parse_intent(content)
            if got is None and "<|message|>" in content:     # channel markers leaked into content
                got = parse_intent(content.rsplit("<|message|>", 1)[1])
            return got, info
        except httpx.HTTPError as e:
            info["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            info["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            return None, info

    def close(self) -> None:
        self._client.close()
