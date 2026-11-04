"""Hedged router timing: Glimmer preferred within 2.5 s, else the first valid answer, else keyword."""
import asyncio
import time

import pytest

from mhd_ml.agent.intent import ChatConfig, ChatIntentClient, HedgedRouter
from mhd_ml.agent.mock_llm import MockLLM

PORT = 8481
WINDOW = 2.5


@pytest.fixture(scope="module")
def mock():
    m = MockLLM(port=PORT).start()
    yield m
    m.stop()


def make(mock, model, label):
    return ChatIntentClient(ChatConfig(base_url=mock.base_url, model=model, label=label, api_key_env=None,
                                       timeout_s=6.0))


def route(mock, glimmer_b, spark_b, text="follow the person in the blue shirt", total=8.0, mode="auto"):
    mock.behaviours = {"glimmer-m": glimmer_b, "spark-m": spark_b}
    r = HedgedRouter(glimmer=make(mock, "glimmer-m", "glimmer-30b+lora"), spark=make(mock, "spark-m", "muse-spark-1.3"),
                     mode=mode, prefer_window_s=WINDOW, total_timeout_s=total)

    async def go():
        t0 = time.monotonic()
        res = await r.route(text)
        dt = time.monotonic() - t0
        await asyncio.sleep(0)
        await r.aclose()
        return res, dt

    return asyncio.run(go())


def test_both_fast_glimmer_wins(mock):
    res, dt = route(mock, {"delay_s": 0.3}, {"delay_s": 0.1})
    assert res.model == "glimmer-30b+lora" and res.reason == "glimmer_in_window"
    assert dt < 1.0
    assert res.intent.subject_phrase == "person in blue shirt"


def test_slow_glimmer_spark_wins_at_window(mock):
    res, dt = route(mock, {"delay_s": 4.0}, {"delay_s": 0.3})
    assert res.model == "muse-spark-1.3" and res.reason == "first_valid"
    assert WINDOW - 0.05 <= dt < WINDOW + 0.6, dt          # waited for the Glimmer window, no longer


def test_invalid_glimmer_does_not_wait_for_window(mock):
    res, dt = route(mock, {"delay_s": 0.1, "mode": "invalid"}, {"delay_s": 0.4})
    assert res.model == "muse-spark-1.3"
    assert dt < 1.2, dt
    assert res.attempts["glimmer-30b+lora"]["valid"] is False


def test_glimmer_just_after_window_still_first_valid(mock):
    res, dt = route(mock, {"delay_s": 2.8}, {"delay_s": 4.5})
    assert res.model == "glimmer-30b+lora" and res.reason == "first_valid"
    assert 2.7 <= dt < 3.6, dt


def test_both_invalid_keyword_fallback(mock):
    res, dt = route(mock, {"mode": "invalid"}, {"mode": "error"})
    assert res.model == "keyword" and res.reason == "keyword_fallback"
    assert res.intent.subject_phrase == "person in blue shirt"
    assert dt < 1.5


def test_total_timeout_keyword_fallback(mock):
    res, dt = route(mock, {"delay_s": 5.0}, {"delay_s": 5.0}, total=1.0)
    assert res.model == "keyword" and res.reason == "keyword_fallback"
    assert 0.95 <= dt < 1.6, dt


def test_safety_fast_path_skips_models(mock):
    n = len(mock.requests)
    res, dt = route(mock, {"delay_s": 1.0}, {"delay_s": 1.0}, text="stop, land the drone")
    assert res.model == "keyword" and res.reason == "safety_fast_path" and res.intent.action == "land"
    assert dt < 0.1 and len(mock.requests) == n


def test_mode_spark_only(mock):
    res, dt = route(mock, {"delay_s": 0.1}, {"delay_s": 0.2}, mode="spark")
    assert res.model == "muse-spark-1.3" and dt < 1.0


def test_mode_keyword_only(mock):
    n = len(mock.requests)
    res, _ = route(mock, {}, {}, mode="keyword")
    assert res.model == "keyword" and res.reason == "keyword_only" and len(mock.requests) == n


def test_late_answer_recorded_in_history(mock):
    mock.behaviours = {"glimmer-m": {"delay_s": 0.1}, "spark-m": {"delay_s": 0.6}}
    r = HedgedRouter(glimmer=make(mock, "glimmer-m", "glimmer-30b+lora"), spark=make(mock, "spark-m", "muse-spark-1.3"))

    async def go():
        res = await r.route("follow the red car")
        await asyncio.sleep(1.0)          # let Spark finish in the background
        return res

    res = asyncio.run(go())
    assert res.model == "glimmer-30b+lora"
    late = [h for h in r.history if h.get("late")]
    assert late and late[0]["model"] == "muse-spark-1.3" and late[0]["valid"]


def test_safety_word_inside_a_follow_command_goes_to_the_models(mock):
    js = ('{"action":"follow","subject_phrase":"man at bus stop","category":"person","color":"none",'
          '"distance":"medium","reason_code":"none"}')
    res, dt = route(mock, {"mode": "fixed", "content": js, "delay_s": 0.2}, {"mode": "fixed", "content": js},
                    text="follow the man at the bus stop")
    assert res.reason != "safety_fast_path" and res.intent.action == "follow"
