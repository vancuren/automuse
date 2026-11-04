"""Intent parsing and the OpenAI-compatible clients, against a local mock server on port 8480."""
import asyncio
import json

import pytest

from mhd.intent_spec import GLIMMER_SYSTEM_PROMPT, SYSTEM_PROMPT
from mhd_ml.agent.intent import ChatConfig, ChatIntentClient, glimmer_client, parse_model_output
from mhd_ml.agent.mock_llm import MockLLM

PORT = 8480
GOOD = ('{"action":"follow","subject_phrase":"person in blue shirt","category":"person",'
        '"color":"blue","distance":"medium","reason_code":"none"}')


# ---- parsing --------------------------------------------------------------------------------------
def test_parse_plain_json():
    i, repaired = parse_model_output(GOOD)
    assert i.subject_phrase == "person in blue shirt" and not repaired


def test_parse_with_reasoning_text_and_braces():
    text = "Reasoning: the user wants {a person}. Answer:\n" + GOOD
    i, _ = parse_model_output(text)
    assert i is not None and i.action == "follow"


def test_parse_think_block_and_code_fence():
    text = "<think>maybe {\"action\": \"land\"}</think>\n```json\n" + GOOD + "\n```"
    i, _ = parse_model_output(text)
    assert i is not None and i.action == "follow"


def test_parse_repairs_case_and_missing_fields():
    i, repaired = parse_model_output('{"action":"Hover","subject_phrase":"","category":"None","color":null}')
    assert i is not None and i.action == "hover" and i.distance == "medium" and repaired


def test_parse_rejects_bad_values():
    assert parse_model_output('{"action":"takeoff","subject_phrase":"","category":"none","color":"none",'
                              '"distance":"medium","reason_code":"none"}')[0] is None
    assert parse_model_output("I will follow the person in blue.")[0] is None
    assert parse_model_output(GOOD[:40])[0] is None


def test_parse_rejects_follow_without_subject():
    bad = GOOD.replace('"person in blue shirt"', '""')
    assert parse_model_output(bad)[0] is None


# ---- clients against the mock -----------------------------------------------------------------------
@pytest.fixture(scope="module")
def mock():
    m = MockLLM(port=PORT, behaviours={
        "slow": {"delay_s": 1.5},
        "invalid": {"mode": "invalid"},
        "partial": {"mode": "partial"},
        "error": {"mode": "error"},
        "chatty": {"mode": "valid", "prefix": "Sure. Here is the JSON: "},
        "dribble": {"chunk_chars": 3, "chunk_delay_s": 0.01},
    }).start()
    yield m
    m.stop()


def client(mock, model, stream=True, **kw) -> ChatIntentClient:
    cfg = ChatConfig(base_url=mock.base_url, model=model, label=f"test-{model}", api_key_env=None,
                     system_prompt=GLIMMER_SYSTEM_PROMPT, stream=stream, timeout_s=kw.pop("timeout_s", 5.0), **kw)
    return ChatIntentClient(cfg)


@pytest.mark.parametrize("stream", [True, False])
def test_valid_answer(mock, stream):
    a = asyncio.run(client(mock, "glimmer", stream=stream).infer("follow the person in the blue shirt"))
    assert a.valid, a.error
    assert a.intent.subject_phrase == "person in blue shirt" and a.intent.category == "person"
    body = mock.requests[-1]["body"]
    assert body["messages"][0] == {"role": "system", "content": GLIMMER_SYSTEM_PROMPT}
    assert body["messages"][1]["content"] == "follow the person in the blue shirt"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
    assert body["stream"] is stream


def test_streaming_stops_at_complete_json(mock):
    a = asyncio.run(client(mock, "dribble").infer("track the brown dog"))
    assert a.valid and a.intent.subject_phrase == "brown dog"
    assert a.ttft_ms is not None and a.ttft_ms <= a.latency_ms


def test_chatty_prefix_still_parses(mock):
    a = asyncio.run(client(mock, "chatty").infer("follow the red car"))
    assert a.valid and a.intent.subject_phrase == "red car"


@pytest.mark.parametrize("model,err", [("invalid", "invalid"), ("partial", "invalid"), ("error", "HTTP 500")])
def test_bad_answers_are_invalid_not_exceptions(mock, model, err):
    a = asyncio.run(client(mock, model).infer("follow the red car"))
    assert not a.valid and err in (a.error or "")


def test_timeout(mock):
    a = asyncio.run(client(mock, "slow", timeout_s=0.5).infer("follow the red car"))
    assert not a.valid and a.error == "timeout" and a.latency_ms < 1200


def test_glimmer_helper_uses_glimmer_prompt(mock):
    c = glimmer_client(mock.base_url, "glimmer", api_key_env=None)
    a = asyncio.run(c.infer("land now"))
    assert a.valid and a.intent.action == "land" and a.model == "glimmer-30b+lora"
    assert mock.requests[-1]["body"]["messages"][0]["content"] == GLIMMER_SYSTEM_PROMPT
    assert not mock.requests[-1]["auth"]          # local servers get no Authorization header


def test_spark_style_request_body(mock, monkeypatch):
    """The Spark client sends strict json_schema, reasoning_effort, max_completion_tokens, no temperature."""
    from mhd_ml.agent.intent import spark_client

    monkeypatch.setenv("MODEL_API_KEY", "test-key-not-real")
    c = spark_client("minimal", base_url=mock.base_url, model="spark")
    a = asyncio.run(c.infer("follow the red car"))
    assert a.valid
    req = mock.requests[-1]
    body = req["body"]
    assert req["auth"] is True
    assert body["reasoning_effort"] == "minimal"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "max_completion_tokens" in body and "temperature" not in body
    assert body["messages"][0]["content"] == SYSTEM_PROMPT
    assert "tools" not in body and "tool_choice" not in body


def test_parse_glimmer_raw_channels_from_mlx_lm_server():
    """mlx_lm.server returns Glimmer's channels raw; the answer is after the last <|message|>."""
    raw = ("to=self<|message|>Follow by appearance: {person} blue shirt.<|eom|><|start|>assistant "
           "to=user<|message|>" + GOOD + "<|eot|>")
    i, repaired = parse_model_output(raw)
    assert i is not None and i.subject_phrase == "person in blue shirt" and not repaired


def test_glimmer_request_has_no_constrained_decoding(mock):
    c = glimmer_client(mock.base_url, "default_model", api_key_env=None)
    a = asyncio.run(c.infer("follow the red car"))
    body = mock.requests[-1]["body"]
    assert a.valid and "response_format" not in body and body["max_tokens"] == 1200
    assert body["temperature"] == 0.0 and body["model"] == "default_model"


def test_edge_mode_gets_longer_deadline():
    from mhd_ml.agent.service import AgentService

    cfg = {"intent": "glimmer", "glimmer": {"base_url": "http://127.0.0.1:8080/v1", "model": "default_model",
                                            "api_key_env": None}, "spark": {"enabled": False}}
    r = AgentService(cfg, start_mic=False)._build_router()
    assert r.total_timeout_s == 9.0
    cfg["glimmer"]["base_url"] = "https://api.fireworks.ai/inference/v1"
    assert AgentService(cfg, start_mic=False)._build_router().total_timeout_s == 5.0
