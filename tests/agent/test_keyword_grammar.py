"""Keyword grammar: transcript -> Intent without any model (fallback and safety fast path)."""
import pytest

from mhd.contracts import Intent
from mhd_ml.agent.intent import KeywordIntent, keyword_intent

# (transcript, action, subject_phrase, category, color, distance, reason_code)
CASES = [
    # follow: "<color> <thing>"
    ("follow the red car", "follow", "red car", "car", "red", "medium", "none"),
    ("Follow the red car.", "follow", "red car", "car", "red", "medium", "none"),
    ("track the brown dog", "follow", "brown dog", "animal", "brown", "medium", "none"),
    ("follow the gray cat", "follow", "gray cat", "animal", "gray", "medium", "none"),
    ("follow the grey cat", "follow", "gray cat", "animal", "gray", "medium", "none"),
    ("track the black suv", "follow", "black suv", "car", "black", "medium", "none"),
    ("follow the purple balloon", "follow", "purple balloon", "object", "purple", "medium", "none"),
    ("watch the brown horse", "follow", "brown horse", "animal", "brown", "medium", "none"),
    ("lock onto the green kite", "follow", "green kite", "object", "green", "medium", "none"),
    ("keep up with the white truck", "follow", "white truck", "car", "white", "medium", "none"),
    ("folow the red car", "follow", "red car", "car", "red", "medium", "none"),          # ASR misspelling
    ("follow the dog", "follow", "dog", "animal", "none", "medium", "none"),
    ("follow the ball", "follow", "ball", "object", "none", "medium", "none"),
    ("follow the bus", "follow", "bus", "car", "none", "medium", "none"),
    # follow: "<person> in/with <color> <garment or object>"
    ("follow the person in the blue shirt", "follow", "person in blue shirt", "person", "blue", "medium", "none"),
    ("followed the person in the blue shirt", "follow", "person in blue shirt", "person", "blue", "medium", "none"),
    ("follow the person in the blue shirt.", "follow", "person in blue shirt", "person", "blue", "medium", "none"),
    ("film the girl wearing a pink dress", "follow", "girl in pink dress", "person", "pink", "medium", "none"),
    ("follow the kid with the yellow balloon", "follow", "kid with yellow balloon", "person", "yellow", "medium", "none"),
    ("follow the woman with the red umbrella", "follow", "woman with red umbrella", "person", "red", "medium", "none"),
    ("follow the cyclist in the orange vest", "follow", "cyclist in orange vest", "person", "orange", "medium", "none"),
    ("follow the man in the blue t shirt", "follow", "man in blue shirt", "person", "blue", "medium", "none"),
    ("follow the guy in red", "follow", "man in red", "person", "red", "medium", "none"),
    ("find the person in yellow", "follow", "person in yellow", "person", "yellow", "medium", "none"),
    ("follow the red jacket", "follow", "person in red jacket", "person", "red", "medium", "none"),
    ("follow the boy on the red bike", "follow", "boy with red bike", "person", "red", "medium", "none"),
    ("the person in the blue shirt", "follow", "person in blue shirt", "person", "blue", "medium", "none"),
    ("follow the runner", "follow", "runner", "person", "none", "medium", "none"),
    # distance
    ("stay close to the woman in the green jacket", "follow", "woman in green jacket", "person", "green", "close", "none"),
    ("please follow that blue car closely", "follow", "blue car", "car", "blue", "close", "none"),
    ("follow the man in the black hoodie from far away", "follow", "man in black hoodie", "person", "black", "far", "none"),
    ("follow the white van from a distance", "follow", "white van", "car", "white", "far", "none"),
    # safety words: land beats hover, no model needed
    ("land", "land", "", "none", "none", "medium", "none"),
    ("land now", "land", "", "none", "none", "medium", "none"),
    ("okay drone please land", "land", "", "none", "none", "medium", "none"),
    ("abort", "land", "", "none", "none", "medium", "none"),
    ("emergency", "land", "", "none", "none", "medium", "none"),
    ("stop and land", "land", "", "none", "none", "medium", "none"),
    ("stop", "hover", "", "none", "none", "medium", "none"),
    ("stop following", "hover", "", "none", "none", "medium", "none"),
    ("hover right here", "hover", "", "none", "none", "medium", "none"),
    ("hold on", "hover", "", "none", "none", "medium", "none"),
    ("freeze", "hover", "", "none", "none", "medium", "none"),
    # refusals
    ("follow John", "refuse", "", "none", "none", "medium", "identity_request"),
    ("follow sarah", "refuse", "", "none", "none", "medium", "identity_request"),
    ("who is that man", "refuse", "", "none", "none", "medium", "identity_request"),
    ("identify the person in the red shirt", "refuse", "", "none", "none", "medium", "identity_request"),
    ("stalk that woman", "refuse", "", "none", "none", "medium", "unsafe"),
    ("spy on my neighbor", "refuse", "", "none", "none", "medium", "unsafe"),
    ("follow her home", "refuse", "", "none", "none", "medium", "unsafe"),
    ("take off", "refuse", "", "none", "none", "medium", "unsupported"),     # voice never takes off
    ("do a flip", "refuse", "", "none", "none", "medium", "unsupported"),
    ("fly higher", "refuse", "", "none", "none", "medium", "unsupported"),
    # clarify
    ("follow him", "clarify", "", "none", "none", "medium", "ambiguous"),
    ("follow me", "clarify", "", "none", "none", "medium", "ambiguous"),
    ("follow", "clarify", "", "none", "none", "medium", "ambiguous"),
    ("follow the two kids", "clarify", "", "none", "none", "medium", "ambiguous"),
    ("follow the children", "clarify", "", "none", "none", "medium", "ambiguous"),
    ("follow the dogs", "clarify", "", "none", "none", "medium", "ambiguous"),
    ("", "clarify", "", "none", "none", "medium", "ambiguous"),
    ("uh", "clarify", "", "none", "none", "medium", "ambiguous"),
]


@pytest.mark.parametrize("text,action,phrase,category,color,distance,reason", CASES)
def test_keyword_grammar(text, action, phrase, category, color, distance, reason):
    i = keyword_intent(text)
    assert isinstance(i, Intent)
    assert (i.action, i.subject_phrase, i.category, i.color, i.distance, i.reason_code) == (
        action, phrase, category, color, distance, reason)


def test_case_count():
    assert len(CASES) >= 40


def test_phrases_are_short_enough_for_sam():
    for text, action, *_ in CASES:
        i = keyword_intent(text)
        assert len(i.subject_phrase.split()) <= 4
        if i.action != "follow":
            assert i.subject_phrase == "" and i.category == "none" and i.color == "none"


def test_name_with_appearance_follows_by_appearance_only():
    i = keyword_intent("follow mike in the red shirt")
    assert i.action == "follow" and i.subject_phrase == "person in red shirt"
    assert "mike" not in i.subject_phrase


def test_ctc_variants_do_not_leak_into_grammar():
    # "stalk" is a CTC spelling of "stop" for the safety listener, but in a command it is a refusal.
    assert keyword_intent("stalk him").action == "refuse"


def test_keyword_client_attempt():
    a = KeywordIntent().infer_sync("follow the red car")
    assert a.valid and a.model == "keyword" and a.latency_ms < 50
