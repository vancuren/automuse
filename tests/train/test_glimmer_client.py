"""Tests for mhd_ml.glimmer_client with a mocked OpenAI-style server."""
import json

import httpx

from mhd_ml.glimmer_client import GlimmerClient
from mhd.intent_spec import GLIMMER_SYSTEM_PROMPT

JSON = '{"action":"land","subject_phrase":"","category":"none","color":"none","distance":"medium","reason_code":"none"}'


def _client(handler, **kw):
    g = GlimmerClient("http://127.0.0.1:9/v1", "default_model", **kw)
    g._client = httpx.Client(transport=httpx.MockTransport(handler))
    return g


def test_ok_and_request_shape():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": JSON, "reasoning": "Land now."}}]})

    intent, info = _client(handler).intent("land now")
    assert intent and intent.action == "land"
    assert info["reasoning"] == "Land now." and info["latency_ms"] >= 0
    assert seen["messages"][0] == {"role": "system", "content": GLIMMER_SYSTEM_PROMPT}
    assert seen["temperature"] == 0.0


def test_leaked_channels_and_bad_answers():
    leaked = "to=self<|message|>x {y}<|eom|><|start|>assistant to=user<|message|>" + JSON
    g = _client(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": leaked}}]}))
    assert g.intent("land")[0].action == "land"
    g = _client(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "sure!"}}]}))
    assert g.intent("land")[0] is None
    g = _client(lambda r: httpx.Response(503, json={"error": "DEPLOYMENT_SCALING_UP"}))
    intent, info = g.intent("land")
    assert intent is None and "503" in info["error"]


def test_timeout_returns_none():
    def handler(req):
        raise httpx.ReadTimeout("slow", request=req)

    intent, info = _client(handler).intent("land")
    assert intent is None and "ReadTimeout" in info["error"]


def test_cloud_model_string(monkeypatch):
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "acct")
    monkeypatch.setenv("FIREWORKS_API_KEY", "k")
    g = GlimmerClient.cloud()
    assert g.model == "accounts/acct/models/glimmer-intent-v1#accounts/acct/deployments/glimmer-intent"
    assert g.headers == {"Authorization": "Bearer k"}
    b = GlimmerClient.cloud(lora=None)
    assert b.model == "accounts/fireworks/models/muse-glimmer-30b#accounts/acct/deployments/glimmer-intent"
    assert b.label == "glimmer-30b"


def test_split_channels_mlx_raw_format():
    from mhd_ml.glimmer_client import split_channels
    raw = ('to=self<|message|>Draft: {"action": maybe follow}<|eom|><|start|>assistant to=user<|message|>'
           + JSON + "<|eot|>")
    answer, reasoning = split_channels(raw)
    assert answer == JSON and reasoning.startswith("Draft:")
    assert split_channels(JSON) == (JSON, None)
    # a brace in the reasoning must not break parsing
    g = _client(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": raw}}]}))
    intent, info = g.intent("land")
    assert intent.action == "land" and info["reasoning"].startswith("Draft:")
