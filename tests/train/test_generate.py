"""Tests for training/generate.py: labels from construction, splits, formats, noise."""
import json
import random
from collections import Counter

import pytest

from mhd.contracts import Intent
from mhd.intent_spec import GLIMMER_SYSTEM_PROMPT, parse_intent
from training import generate as g


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen")
    # copy the hand-written gold set so dedup against it is exercised
    gold_src = g.DATA_DIR / "gold.jsonl"
    if gold_src.exists():
        (out / "gold.jsonl").write_text(gold_src.read_text())
    stats = g.generate(seed=7, paraphrase=False, out_dir=out)
    rows = {s: [json.loads(l) for l in (out / f"{s}.jsonl").read_text().splitlines()] for s in ("train", "valid", "test")}
    return out, stats, rows


def test_sizes(data):
    _, stats, rows = data
    assert stats["sizes"] == {"test": 300, "valid": 150, "train": 2200}
    assert {k: len(v) for k, v in rows.items()} == {"train": 2200, "valid": 150, "test": 300}


def test_every_label_is_a_valid_intent(data):
    _, _, rows = data
    for split in rows.values():
        for r in split:
            i = Intent.model_validate(r["intent"])
            if i.action == "follow":
                assert i.subject_phrase and len(i.subject_phrase.split()) <= 4
                assert i.category != "none" and i.reason_code == "none"
            else:
                assert i.subject_phrase == "" and i.category == "none" and i.color == "none"
                assert i.distance == "medium"
            if i.action == "clarify":
                assert i.reason_code == "ambiguous"
            if i.action == "refuse":
                assert i.reason_code in ("identity_request", "unsafe", "unsupported")
            if i.action in ("hover", "land"):
                assert i.reason_code == "none"


def test_families_are_disjoint_across_splits(data):
    _, _, rows = data
    fams = {s: {r["family"] for r in v} for s, v in rows.items()}
    assert not fams["train"] & fams["test"]
    assert not fams["train"] & fams["valid"]
    assert not fams["valid"] & fams["test"]


def test_no_transcript_leaks_between_splits_or_gold(data):
    out, _, rows = data
    keys = {s: {g.norm_key(r["transcript"]) for r in v} for s, v in rows.items()}
    assert not keys["train"] & keys["test"]
    assert not keys["train"] & keys["valid"]
    gold = out / "gold.jsonl"
    if gold.exists():
        gk = {g.norm_key(json.loads(l)["transcript"]) for l in gold.read_text().splitlines()}
        assert not gk & (keys["train"] | keys["valid"] | keys["test"])


def test_action_mix(data):
    _, _, rows = data
    for split, v in rows.items():
        c = Counter(r["intent"]["action"] for r in v)
        n = len(v)
        assert 0.45 < c["follow"] / n < 0.70, (split, c)
        assert c["clarify"] / n > 0.10, (split, c)
        assert c["refuse"] / n > 0.08, (split, c)
        assert c["hover"] / n > 0.05 and c["land"] / n > 0.05, (split, c)
    reasons = Counter(r["intent"]["reason_code"] for r in rows["test"] if r["intent"]["action"] == "refuse")
    assert set(reasons) == {"identity_request", "unsafe", "unsupported"}


def test_chat_format_round_trips(data):
    out, _, rows = data
    for name in ("mlx/train.jsonl", "mlx/valid.jsonl", "mlx/test.jsonl", "fireworks/train.jsonl",
                 "fireworks/valid.jsonl", "fireworks/smoke30.jsonl"):
        lines = (out / name).read_text().splitlines()
        assert lines, name
        for line in lines[:50]:
            row = json.loads(line)
            assert list(row) == ["messages"]
            sys_m, user_m, asst = row["messages"]
            assert sys_m == {"role": "system", "content": GLIMMER_SYSTEM_PROMPT}
            assert user_m["role"] == "user" and user_m["content"]
            assert asst["role"] == "assistant" and asst["reasoning_content"]
            assert parse_intent(asst["content"]) is not None
    assert len((out / "fireworks/smoke30.jsonl").read_text().splitlines()) == 30
    # chat rows line up with the metadata rows
    first = json.loads((out / "mlx/test.jsonl").read_text().splitlines()[0])
    assert first["messages"][1]["content"] == rows["test"][0]["transcript"]
    assert json.loads(first["messages"][2]["content"]) == rows["test"][0]["intent"]


@pytest.mark.parametrize("clean,expect", [
    ("follow the person in the blue shirt", {"subject": "person in blue shirt"}),
])
def test_render_examples(clean, expect):
    fam = next(f for f in g.FAMILIES if f.fid == "person_follow")
    rng = random.Random(0)
    for _ in range(200):
        text, intent = g.render(fam, rng.choice(fam.templates), rng)
        assert intent.action == "follow" and intent.subject_phrase.startswith("person ")
        assert intent.color in g.COLORS


def test_distance_labels_follow_the_words():
    rng = random.Random(3)
    for _ in range(500):
        text, label = g._distance(rng, "random")
        if label == "close":
            assert any(w in text for w in ("close", "tight", "1 ", "2 ", "one ", "two ")), text
        if label == "far":
            assert any(w in text for w in ("far", "distance", "back", "behind", "away", "6", "8", "10", "12", "15",
                                           "20", "six", "eight", "ten", "twelve", "fifteen", "twenty")), text


def test_asr_noise_keeps_key_words_mostly():
    rng = random.Random(5)
    clean = "follow the person in the blue shirt"
    outs = [g.asr_noise(clean, rng) for _ in range(300)]
    assert any(o == o.lower() and "," not in o for o in outs)       # Omnilingual style
    assert any(o[:1].isupper() and o.endswith(".") for o in outs)   # Voice Transcribe style
    assert any(o.lower().startswith(("hey", "okay", "ok", "drone", "yo", "alright", "hi")) for o in outs)
    kept = sum(("blue" in o.lower() or "blew" in o.lower()) and ("shirt" in o.lower() or "shert" in o.lower())
               for o in outs)
    assert kept / len(outs) > 0.9


def test_named_with_appearance_drops_the_name():
    fam = next(f for f in g.FAMILIES if f.fid == "named_appearance")
    rng = random.Random(11)
    for _ in range(100):
        text, intent = g.render(fam, rng.choice(fam.templates), rng)
        assert intent.action == "follow"
        assert not set(intent.subject_phrase.split()) & set(g.NAMES)


def test_paraphrase_filter():
    ex = g.Example("x", "train", "person", "person_follow", "seed", "follow the guy in the red hoodie", "",
                   g.make_intent("follow", "person in red hoodie", "person", "red"))
    assert g.paraphrase_ok("can you keep up with the dude wearing the red hoodie", ex)
    assert not g.paraphrase_ok("follow the guy in the blue hoodie", ex)          # color changed
    assert not g.paraphrase_ok("follow the guy in red", ex)                       # garment lost
    assert not g.paraphrase_ok("follow the guy in the red hoodie up close", ex)  # distance added
    land = g.Example("y", "train", "land", "land_land", "seed", "land now", "", g.make_intent("land"))
    assert g.paraphrase_ok("okay bring it down and land", land)
    assert not g.paraphrase_ok("follow the dog", land)


def test_gold_file_is_valid():
    path = g.DATA_DIR / "gold.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows) == 80
    assert len({r["id"] for r in rows}) == 80
    acts = Counter(Intent.model_validate(r["intent"]).action for r in rows)
    assert set(acts) == {"follow", "hover", "land", "clarify", "refuse"}
