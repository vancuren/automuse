"""Tests for training/eval.py: parsing, subject normalization, metrics, report files."""
import json

import pytest

from training import eval as ev

GOOD = '{"action":"follow","subject_phrase":"person in blue shirt","category":"person","color":"blue","distance":"medium","reason_code":"none"}'


def _intent(action="follow", subject="person in blue shirt", category="person", color="blue", distance="medium",
            reason="none"):
    if action != "follow":
        subject, category, color = "", "none", "none"
    return {"action": action, "subject_phrase": subject, "category": category, "color": color,
            "distance": distance, "reason_code": reason}


def test_parse_reply_plain_and_leaked_channels():
    assert ev.parse_reply(GOOD).subject_phrase == "person in blue shirt"
    leaked = ("to=self<|message|>Follow by appearance: {maybe}.<|eom|><|start|>assistant to=user<|message|>"
              + GOOD + "<|eot|>")
    assert ev.parse_reply(leaked).action == "follow"
    assert ev.parse_reply("") is None and ev.parse_reply(None) is None
    assert ev.parse_reply('{"action":"fly"}') is None


@pytest.mark.parametrize("a,b", [
    ("person in blue shirt", "man wearing a blue t-shirt"),
    ("person in gray hoodie", "guy in grey hoody"),
    ("white pickup truck", "white pickup"),
    ("green toy car", "green rc car"),
    ("person with red backpack", "kid with the red rucksack"),
    ("black dog", "black lab"),
])
def test_norm_subject_equivalent(a, b):
    assert ev.norm_subject(a) == ev.norm_subject(b)


@pytest.mark.parametrize("a,b", [
    ("person in blue shirt", "person in red shirt"),
    ("red car", "red truck"),
    ("person in blue shirt", "person in blue jacket"),
])
def test_norm_subject_different(a, b):
    assert ev.norm_subject(a) != ev.norm_subject(b)


def _rec(gold, pred, ms=100.0):
    return {"id": "x", "set": "test", "gold": gold, "pred": pred, "latency_ms": ms, "error": None,
            "usage": {"completion_tokens": 50}}


def test_metrics_counts():
    recs = [
        _rec(_intent(), _intent(subject="man in a blue shirt")),             # follow, normalized match only
        _rec(_intent(), _intent()),                                          # exact
        _rec(_intent(), None),                                               # invalid
        _rec(_intent("clarify", reason="ambiguous"), _intent("clarify", reason="ambiguous")),
        _rec(_intent("refuse", reason="identity_request"), _intent("clarify", reason="ambiguous")),
        _rec(_intent("land"), _intent("land")),
        _rec(_intent("hover"), _intent("follow")),
    ]
    m = ev.metrics(recs)
    assert m["n"] == 7
    assert m["valid_json_rate"] == pytest.approx(6 / 7, abs=1e-4)
    assert m["action_accuracy"] == pytest.approx(4 / 7, abs=1e-4)
    assert m["n_follow"] == 3
    assert m["subject_exact"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["subject_normalized"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["clarify"]["precision"] == 0.5 and m["clarify"]["recall"] == 1.0
    assert m["refuse"]["recall"] == 0.0 and m["refuse"]["support"] == 1
    assert m["land_hover_accuracy"] == 0.5
    assert m["confusion"]["follow"]["invalid"] == 1
    assert m["latency_ms"]["p50"] == 100.0


def test_report_writes_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "EVAL_LOG", tmp_path / "eval")
    monkeypatch.setattr(ev, "DOCS", tmp_path / "docs")
    (tmp_path / "eval").mkdir()
    rows = [dict(_rec(_intent(), _intent()), set=s, id=f"{s}-{k}", model="glimmer-lora-mlx", note="unit test")
            for s in ("test", "gold") for k in range(3)]
    (tmp_path / "eval" / "glimmer-lora-mlx.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = ev.report(type("A", (), {"only": None})())
    assert out["models"]["glimmer-lora-mlx"]["sets"]["test"]["action_accuracy"] == 1.0
    assert (tmp_path / "docs" / "results.json").exists()
    assert "Glimmer + LoRA" in (tmp_path / "docs" / "summary.md").read_text()
    assert (tmp_path / "docs" / "intent_eval.png").stat().st_size > 10_000


def test_final_answer_strips_markers():
    assert ev.final_answer("to=self<|message|>x<|eom|><|start|>assistant to=user<|message|>{}<|eot|>") == "{}"
