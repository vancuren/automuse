"""Template-and-slot generator for the Muse Glimmer intent fine-tune.

Labels come from construction: every example is built from a structured slot sample, and the Intent
label is derived from those slots, never from a model. Optional Spark paraphrases only rewrite the
wording of a seed sentence; they keep the seed's label and are filtered so the key words survive.

Label conventions (also in docs/ws/train.md):
- People are always "person" in subject_phrase (never man/woman/kid: clothing only, no gender or age
  judgement): "person in blue shirt", "person with red backpack", "person in blue" (color only).
- Vehicles: "<color> <type>" (car, truck, pickup truck, van, suv, jeep, taxi, bus, toy car, sports car).
- Animals: "<color> <animal>"; objects: "<color> <object>". No color -> just the noun, color "none".
- If several garments are described, only the first one goes in the label (at most two attributes).
- distance: medium unless asked; numbers of meters map to close (<=2), medium (3-5), far (>=6).
- clarify / ambiguous: pronouns or pointing words with no description, a missing subject, "someone",
  "follow me", a color with no noun ("the red one"), or two subjects at once.
- refuse / identity_request: a person by name, relationship ("my neighbor"), role or fame, or any
  request to identify or recognize someone. A name plus a clothing description is followed by the
  description only (the name is dropped).
- refuse / unsafe: surveillance (everyone, a house, a window, following someone home, covertly) and
  harm (fly into, hit, scare). unsafe wins over identity_request.
- refuse / unsupported: things this drone cannot do by voice: take off (button only), flips, flying
  somewhere, altitude changes, photos, and non-drone requests.

Usage:
  $MHD_PY training/generate.py                 # templates only (deterministic, no network)
  $MHD_PY training/generate.py --paraphrase    # plus Muse Spark 1.3 paraphrases (Meta API, logged spend)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mhd.contracts import Intent  # noqa: E402
from mhd.intent_spec import GLIMMER_SYSTEM_PROMPT, PROMPT_VERSION  # noqa: E402

from training.common import DATA_DIR, read_jsonl, write_jsonl  # noqa: E402

# ----------------------------------------------------------------------------------------------
# Slot vocabularies. Canonical label word -> surface words a speaker might use.
# ----------------------------------------------------------------------------------------------
COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "gray", "brown"]
COLOR_SURFACE = {c: [c] for c in COLORS}
COLOR_SURFACE["gray"] = ["gray", "grey"]
VEHICLE_COLORS = COLORS + ["silver"]
COLOR_SURFACE["silver"] = ["silver"]
ANIMAL_COLORS = ["brown", "black", "white", "gray"]
# Shades the speaker may add; the label keeps the base color ("light blue shirt" -> "blue").
SHADES = ["light", "dark", "bright"]

# garment canonical -> (preposition used in the label, surface forms, is_plural)
GARMENTS: dict[str, tuple[str, list[str], bool]] = {
    "shirt": ("in", ["shirt", "t-shirt", "t shirt", "tee shirt", "top"], False),
    "jacket": ("in", ["jacket", "windbreaker"], False),
    "hoodie": ("in", ["hoodie", "hoody", "hooded sweatshirt"], False),
    "coat": ("in", ["coat", "raincoat", "overcoat"], False),
    "dress": ("in", ["dress"], False),
    "sweater": ("in", ["sweater", "jumper", "sweatshirt"], False),
    "jersey": ("in", ["jersey"], False),
    "vest": ("in", ["vest"], False),
    "shorts": ("in", ["shorts"], True),
    "pants": ("in", ["pants", "trousers"], True),
    "skirt": ("in", ["skirt"], False),
    "hat": ("with", ["hat"], False),
    "cap": ("with", ["cap", "baseball cap", "ball cap"], False),
    "backpack": ("with", ["backpack", "rucksack"], False),
    "helmet": ("with", ["helmet"], False),
    "umbrella": ("with", ["umbrella"], False),
    "scarf": ("with", ["scarf"], False),
}
WORN = [g for g, v in GARMENTS.items() if v[0] == "in"]
# People nouns a speaker may use. The label always says "person".
PERSON_NOUNS = ["person", "guy", "man", "woman", "lady", "kid", "girl", "boy", "dude", "runner", "jogger",
                "skater", "cyclist", "fellow", "gentleman", "child"]
SHORT_PERSON_NOUNS = ["guy", "lady", "kid", "person", "dude", "man", "woman", "girl", "boy"]

VEHICLES: dict[str, list[str]] = {
    "car": ["car", "sedan", "hatchback"],
    "truck": ["truck"],
    "pickup truck": ["pickup truck", "pickup", "pick up truck"],
    "van": ["van", "minivan"],
    "suv": ["suv", "s u v"],
    "jeep": ["jeep"],
    "taxi": ["taxi", "cab"],
    "bus": ["bus"],
    "toy car": ["toy car", "rc car", "remote control car", "little toy car"],
    "sports car": ["sports car"],
}
ANIMALS: dict[str, list[str]] = {
    "dog": ["dog", "puppy", "doggy", "pup", "doggo"],
    "cat": ["cat", "kitty", "kitten"],
    "horse": ["horse", "pony"],
    "cow": ["cow"],
    "goat": ["goat"],
    "sheep": ["sheep"],
    "duck": ["duck"],
    "goose": ["goose"],
    "chicken": ["chicken", "hen"],
    "rabbit": ["rabbit", "bunny"],
    "deer": ["deer"],
    "pig": ["pig"],
}
OBJECTS: dict[str, list[str]] = {
    "ball": ["ball", "soccer ball", "beach ball"],
    "kite": ["kite"],
    "balloon": ["balloon"],
    "bike": ["bike", "bicycle"],
    "scooter": ["scooter"],
    "skateboard": ["skateboard"],
    "boat": ["boat"],
    "stroller": ["stroller", "pram"],
    "frisbee": ["frisbee"],
    "shopping cart": ["shopping cart"],
}
NAMES = ["john", "dave", "sarah", "mike", "emily", "maria", "chris", "alex", "priya", "jamal", "wei", "carlos",
         "olivia", "liam", "noah", "emma", "sophia", "james", "kevin", "lisa", "tom", "anna", "raj", "fatima",
         "diego", "yuki", "omar", "grace", "ben", "chloe", "marcus", "nina", "steve", "rachel", "ahmed", "lucy"]
LAST_NAMES = ["smith", "johnson", "garcia", "lee", "patel", "nguyen", "kim", "chen", "martinez", "cohen",
              "okafor", "rossi", "murphy", "singh"]
RELATIONS = ["neighbor", "wife", "husband", "son", "daughter", "boss", "ex", "roommate", "brother", "sister",
             "coworker", "mom", "dad", "friend", "teacher", "landlord", "kid", "girlfriend", "boyfriend", "cousin"]
FAMOUS = ["taylor swift", "the mayor", "the president", "that celebrity", "the senator", "lebron james",
          "the famous guy from tv", "that youtuber", "the police chief", "my favorite actor"]
LANDMARKS = ["tree", "door", "bench", "fence", "wall", "stairs", "fountain", "gate", "corner", "couch", "table",
             "lamp post", "hedge", "steps"]
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 10: "ten",
             12: "twelve", 15: "fifteen", 20: "twenty"}

WAKE_WORDS = ["hey drone", "okay drone", "ok drone", "drone", "hey dron", "yo drone", "alright drone",
              "hey drome", "hi drone", "drone drone", "hey buddy"]
FILLERS = ["um", "uh", "uhh", "er", "like", "so", "okay", "hmm", "you know", "alright", "well", "erm"]
PRE = ["", "", "", "please", "can you", "could you", "i want you to", "go ahead and", "i need you to", "now",
       "just", "okay", "alright"]
POST = ["", "", "", "", "please", "now", "thanks", "thank you", "okay", "right now", "for me"]

# ASR-style confusions: mostly phonetic, like what Omnilingual CTC and Voice Transcribe emit.
MISSPELL: dict[str, list[str]] = {
    "meters": ["metrs", "meter", "metres", "meeters"], "drone": ["dron", "drome", "drown"],
    "follow": ["folow", "fallow", "follo"], "following": ["folowing", "fallowing"],
    "blue": ["blew"], "red": ["read", "rid"], "white": ["wite"], "green": ["grean", "greene"],
    "yellow": ["yello", "yelow"], "gray": ["grey"], "grey": ["gray"], "purple": ["purpel", "perple"],
    "orange": ["orang", "orenge"], "black": ["blak"], "brown": ["browne", "broun"], "pink": ["pinc"],
    "shirt": ["shert", "shirts"], "jacket": ["jackit", "jaket"], "hoodie": ["hoody", "hudi", "hood e"],
    "sweater": ["swetter", "sweat her"], "backpack": ["back pack", "bagpack"], "dress": ["dres"],
    "car": ["cars", "kar"], "truck": ["truk"], "dog": ["dogg", "dawg"], "cat": ["kat"],
    "person": ["persin", "purson"], "land": ["lan", "lend"], "hover": ["hova", "hoover"],
    "stop": ["stopp", "stap"], "position": ["posision", "positon"], "please": ["pleas", "plz"],
    "track": ["trak"], "camera": ["camra", "kamera"], "umbrella": ["umbrela"], "helmet": ["helmit"],
    "horse": ["horce"], "rabbit": ["rabit"], "skateboard": ["skate board"], "balloon": ["baloon"],
    "bicycle": ["bycicle"], "scooter": ["scooder"], "distance": ["distanse"], "close": ["clothes"],
}
DROPPABLE = {"the", "a", "an", "that", "please", "to", "you", "just", "now", "can", "and", "it", "on"}

# ----------------------------------------------------------------------------------------------
# Labels and short reasoning (Glimmer's "assistant to=self" channel; see docs/ws/train.md).
# ----------------------------------------------------------------------------------------------


def make_intent(action: str, subject: str = "", category: str = "none", color: str = "none",
                distance: str = "medium", reason: str = "none") -> Intent:
    return Intent(action=action, subject_phrase=subject, category=category, color=color,
                  distance=distance, reason_code=reason)


def reasoning_for(intent: Intent) -> str:
    """One short line for the reasoning channel. Kept tiny so latency stays near 60 output tokens."""
    if intent.action == "follow":
        s = f"Follow by appearance: {intent.subject_phrase}."
        if intent.distance != "medium":
            s += f" Distance {intent.distance}."
        return s
    if intent.action == "hover":
        return "Stop and hold position."
    if intent.action == "land":
        return "Land now."
    if intent.action == "clarify":
        return "No single clear subject, so ask who to follow."
    if intent.reason_code == "identity_request":
        return "Identity-based request; I follow by appearance only, so refuse."
    if intent.reason_code == "unsafe":
        return "Surveillance or harmful request, so refuse."
    return "Not something this drone can do by voice, so refuse."


def intent_json(intent: Intent) -> str:
    """Compact JSON in schema field order (fewest output tokens)."""
    return intent.model_dump_json()


def to_chat(transcript: str, intent: Intent, with_reasoning: bool = True) -> dict:
    """The training row: system prompt (frozen), transcript, and the assistant target."""
    assistant: dict = {"role": "assistant", "content": intent_json(intent)}
    if with_reasoning:
        assistant["reasoning_content"] = reasoning_for(intent)
    return {"messages": [{"role": "system", "content": GLIMMER_SYSTEM_PROMPT},
                         {"role": "user", "content": transcript},
                         assistant]}


# ----------------------------------------------------------------------------------------------
# Slot samplers. Each returns (slots for the template, Intent). `dist` is filled per template.
# ----------------------------------------------------------------------------------------------


def _a(word: str) -> str:
    """Indefinite article for a word."""
    return "an" if word[:1] in "aeiou" else "a"


def _color_word(rng: random.Random, color: str, allow_shade: bool = True) -> str:
    surface = rng.choice(COLOR_SURFACE[color])
    if allow_shade and rng.random() < 0.08:
        surface = f"{rng.choice(SHADES)} {surface}"
    return surface


def _distance(rng: random.Random, kind: str) -> tuple[str, str]:
    """Return (suffix text, distance label). kind: random|close|far|none."""
    if kind == "none":
        return "", "medium"
    if kind == "random":
        r = rng.random()
        kind = "none" if r < 0.68 else "close" if r < 0.79 else "far" if r < 0.90 else "num"
    if kind == "none":
        return "", "medium"
    if kind == "close":
        return rng.choice([" and stay close", " up close", " nice and close", " and keep it tight",
                           " from close up", " and get close", " real close"]), "close"
    if kind == "far":
        return rng.choice([" from far away", " from a distance", " and keep your distance", " but stay back",
                           " from way back", " and hang back", " from far back", " from a ways back"]), "far"
    # numeric meters
    n = rng.choice([1, 2, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
    num = NUM_WORDS[n] if rng.random() < 0.5 else str(n)
    unit = "meter" if n == 1 else "meters"
    text = rng.choice([f" from {num} {unit}", f" {num} {unit} back", f" at {num} {unit}",
                       f" about {num} {unit} away", f" and stay {num} {unit} behind"])
    label = "close" if n <= 2 else "medium" if n <= 5 else "far"
    return text, label


def sample_person(rng: random.Random) -> tuple[dict, dict]:
    """A person described by one garment or accessory (sometimes a color only)."""
    garment = rng.choice(list(GARMENTS))
    prep, surfaces, plural = GARMENTS[garment]
    color = rng.choice(COLORS)
    g = rng.choice(surfaces)
    c = _color_word(rng, color)
    if prep == "in":
        forms = [f"in the {c} {g}", f"wearing the {c} {g}", f"in {c} {g}"]
        if not plural:
            forms += [f"in {_a(c)} {c} {g}", f"wearing {_a(c)} {c} {g}"]
    else:
        verb = "carrying" if garment in ("umbrella", "backpack") else "wearing"
        forms = [f"with the {c} {g}", f"with {_a(c)} {c} {g}", f"{verb} the {c} {g}", f"{verb} {_a(c)} {c} {g}"]
    desc = rng.choice(forms)
    noun = rng.choice(PERSON_NOUNS)
    short = rng.choice(SHORT_PERSON_NOUNS)
    # Extra garment for the verbose family: never the same garment or color, always worn.
    g2 = rng.choice([x for x in WORN if x != garment])
    c2 = rng.choice([x for x in COLORS if x != color])
    extra = f"{c2} {rng.choice(GARMENTS[g2][1])}"
    slots = {"noun": noun, "short": short, "desc": desc, "c": c, "g": g, "extra": extra,
             "lm": rng.choice(LANDMARKS), "an": _a(c)}
    label = {"subject": f"person {prep} {color} {garment}", "category": "person", "color": color}
    return slots, label


def sample_person_bare(rng: random.Random) -> tuple[dict, dict]:
    """A person noun with no description ("follow that guy"): follow "person"; perception disambiguates."""
    return ({"noun": rng.choice(PERSON_NOUNS), "short": rng.choice(SHORT_PERSON_NOUNS), "lm": rng.choice(LANDMARKS)},
            {"subject": "person", "category": "person", "color": "none"})


def sample_person_color_only(rng: random.Random) -> tuple[dict, dict]:
    color = rng.choice(COLORS)
    c = _color_word(rng, color, allow_shade=False)
    slots = {"noun": rng.choice(PERSON_NOUNS), "short": rng.choice(SHORT_PERSON_NOUNS), "c": c}
    return slots, {"subject": f"person in {color}", "category": "person", "color": color}


def sample_vehicle(rng: random.Random, colored: bool = True) -> tuple[dict, dict]:
    vtype = rng.choice(list(VEHICLES))
    v = rng.choice(VEHICLES[vtype])
    if not colored:
        return {"v": v, "lm": rng.choice(LANDMARKS)}, {"subject": vtype, "category": "car", "color": "none"}
    color = rng.choice(VEHICLE_COLORS)
    c = _color_word(rng, color)
    return ({"v": v, "c": c, "an": _a(c), "lm": rng.choice(LANDMARKS)},
            {"subject": f"{color} {vtype}", "category": "car", "color": color})


def sample_animal(rng: random.Random, colored: bool = True) -> tuple[dict, dict]:
    atype = rng.choice(list(ANIMALS))
    a = rng.choice(ANIMALS[atype])
    if not colored:
        return {"v": a, "lm": rng.choice(LANDMARKS)}, {"subject": atype, "category": "animal", "color": "none"}
    palette = ANIMAL_COLORS + (["orange"] if atype == "cat" else [])
    color = rng.choice(palette)
    c = _color_word(rng, color, allow_shade=False)
    return ({"v": a, "c": c, "an": _a(c), "lm": rng.choice(LANDMARKS)},
            {"subject": f"{color} {atype}", "category": "animal", "color": color})


def sample_object(rng: random.Random, colored: bool = True) -> tuple[dict, dict]:
    otype = rng.choice(list(OBJECTS))
    o = rng.choice(OBJECTS[otype])
    if not colored:
        return {"v": o}, {"subject": otype, "category": "object", "color": "none"}
    color = rng.choice(COLORS)
    c = _color_word(rng, color, allow_shade=False)
    return {"v": o, "c": c, "an": _a(c)}, {"subject": f"{color} {otype}", "category": "object", "color": color}


def _name(rng: random.Random) -> str:
    n = rng.choice(NAMES)
    if rng.random() < 0.2:
        n = f"{n} {rng.choice(LAST_NAMES)}"
    return n


def _pron(rng: random.Random) -> dict:
    p = rng.choice([("he", "him", "he's"), ("she", "her", "she's"), ("they", "them", "they're")])
    return {"he": p[0], "him": p[1], "hes": p[2]}


# ----------------------------------------------------------------------------------------------
# Template families. A family is a phrasing style; whole families are held out for valid and test.
# Template strings use {pre} {post} {dist} plus the group's slots. A template may force a distance
# by prefixing "close|" or "far|" (then {dist} must not appear in it).
# ----------------------------------------------------------------------------------------------


@dataclass
class Family:
    fid: str
    group: str
    split: str                     # train | valid | test
    templates: list[str]
    sampler: Callable[[random.Random], tuple[dict, dict]] = None  # type: ignore[assignment]
    action: str = "follow"
    reason: str = "none"
    use_dist: bool = True          # follow families: whether {dist} gets a random distance


def _fixed(action: str, reason: str = "none"):
    def sampler(rng: random.Random) -> tuple[dict, dict]:
        slots = {"name": _name(rng), "rel": rng.choice(RELATIONS), "famous": rng.choice(FAMOUS),
                 "noun": rng.choice(PERSON_NOUNS), "short": rng.choice(SHORT_PERSON_NOUNS),
                 "lm": rng.choice(LANDMARKS)}
        slots.update(_pron(rng))
        # Some refusal and clarify templates mention a described person, vehicle or animal.
        ps, _ = sample_person(rng)
        slots["desc"] = ps["desc"]
        vs, _ = sample_vehicle(rng)
        slots["v"], slots["c"] = vs["v"], vs["c"]
        an_s, _ = sample_animal(rng)
        slots["animal"], slots["ac"] = an_s["v"], an_s["c"]
        slots["n"] = str(rng.choice([20, 30, 50, 100, 200]))
        return slots, {"subject": "", "category": "none", "color": "none"}
    return sampler


def _named_sampler(rng: random.Random) -> tuple[dict, dict]:
    slots, label = sample_person(rng)
    slots.update({"name": _name(rng), "rel": rng.choice(RELATIONS)})
    slots.update(_pron(rng))
    return slots, label


FAMILIES: list[Family] = [
    # ---------------- follow: people ----------------
    Family("person_follow", "person", "train", [
        "{pre} follow the {noun} {desc}{dist} {post}", "follow that {noun} {desc}{dist}",
        "{pre} start following the {noun} {desc}{dist}", "follow the {c} {g} {short}{dist}",
        "follow the one {desc}{dist}", "{pre} follow someone {desc}{dist}"], sample_person),
    Family("person_track", "person", "train", [
        "track the {noun} {desc}{dist}", "keep the camera on the {noun} {desc}{dist}",
        "{pre} keep tracking the {noun} {desc} {post}", "lock on to the {noun} {desc}{dist}",
        "lock onto the {c} {g} {short}", "{pre} track that {noun} {desc}{dist}"], sample_person),
    Family("person_question", "person", "train", [
        "can you follow the {noun} {desc}{dist}", "could you go after the {noun} {desc}",
        "would you mind following the {noun} {desc}{dist}", "will you follow the {noun} {desc} {post}",
        "can you keep up with the {noun} {desc}{dist}"], sample_person),
    Family("person_deictic", "person", "train", [
        "see the {noun} {desc} follow {post}", "the {noun} {desc} follow that one{dist}",
        "you see that {noun} {desc} go follow{dist}", "there's a {noun} {desc} by the {lm} follow them{dist}",
        "the {noun} over by the {lm} {desc} follow"], sample_person),
    Family("person_garment_only", "person", "train", [
        "follow the {c} {g}{dist}", "go after the {c} {g} {post}", "track {c} {g}{dist}",
        "{pre} follow {c} {g}", "keep the camera on the {c} {g}"], sample_person),
    Family("person_verbose", "person", "train", [
        "follow the tall {noun} {desc} and {extra}{dist}",
        "{pre} follow the {noun} {desc}, the one with the {extra}",
        "there is a {noun} {desc} and {extra} near the {lm}, follow them{dist}",
        "track the {noun} {desc} who also has {extra} on"], sample_person),
    Family("person_distance_first", "person", "train", [
        "close|stay close to the {noun} {desc}", "far|keep your distance from the {noun} {desc} but follow them",
        "far|hang back and follow the {noun} {desc}", "close|get close and follow the {noun} {desc}",
        "far|from a distance follow the {noun} {desc}", "close|follow closely behind the {noun} {desc}"],
        sample_person, use_dist=False),
    Family("person_color_only", "person", "train", [
        "follow the {noun} in {c}{dist}", "track the {noun} dressed in {c}", "{pre} follow the {short} in all {c}",
        "go after the {noun} wearing {c}{dist}"], sample_person_color_only),
    Family("person_bare", "person", "train", [
        "follow the {noun}{dist}", "track that {short}", "{pre} follow that {noun} over there{dist}",
        "keep the camera on the {noun} by the {lm}"], sample_person_bare),
    Family("person_valid", "person", "valid", [
        "{pre} go follow the {noun} {desc}{dist}", "film the {noun} {desc}{dist}",
        "stick with the {noun} {desc}{dist} {post}", "the {c} {g} {short} follow them",
        "stay on the {noun} {desc}"], sample_person),
    Family("person_test", "person", "test", [
        "{pre} keep following the {noun} {desc}{dist}", "i want the drone to follow the {noun} {desc}{dist}",
        "focus on the {noun} {desc} and follow{dist}", "go with the {noun} {desc} {post}",
        "pick the {noun} {desc} and follow them{dist}", "shadow the {noun} {desc}{dist}"], sample_person),
    # ---------------- follow: a name plus an appearance -> follow the appearance only ----------------
    Family("named_appearance", "person_named", "train", [
        "follow {name} {hes} the one {desc}", "{name} is the {noun} {desc}, follow {him}",
        "follow my {rel}, the {noun} {desc}", "follow {name} {desc}{dist}",
        "track my {rel} {desc}{dist}"], _named_sampler),
    Family("named_appearance_train2", "person_named", "train", [
        "{name} is wearing the {c} {g}, follow {him}", "my {rel} has the {c} {g} on, follow {him}{dist}",
        "follow {name}, look for the {c} {g}"], _named_sampler),
    Family("named_appearance_valid", "person_named", "valid", [
        "go after {name} {desc}", "my {rel} is the {noun} {desc} go follow {him}"], _named_sampler),
    Family("named_appearance_test", "person_named", "test", [
        "that's {name} {desc}, follow {him}{dist}", "follow my {rel}, {hes} {desc}"], _named_sampler),
    # ---------------- follow: vehicles ----------------
    Family("car_follow", "car", "train", [
        "{pre} follow the {c} {v}{dist} {post}", "follow that {c} {v}{dist}", "start following the {c} {v}{dist}",
        "{pre} follow {an} {c} {v}"], sample_vehicle),
    Family("car_track", "car", "train", [
        "track the {c} {v}{dist}", "keep the camera on the {c} {v}", "lock onto the {c} {v}{dist}",
        "{pre} keep tracking the {c} {v}"], sample_vehicle),
    Family("car_keepup", "car", "train", [
        "go after the {c} {v}{dist}", "keep up with the {c} {v}{dist} {post}", "stay with the {c} {v}",
        "chase the {c} {v}{dist}"], sample_vehicle),
    Family("car_deictic", "car", "train", [
        "the {c} {v} follow it{dist}", "see that {c} {v} follow it", "there's {an} {c} {v} by the {lm} follow it",
        "you see the {c} {v} go after it{dist}"], sample_vehicle),
    Family("car_color_after", "car", "train", [
        "follow the {v}, the {c} one{dist}", "follow the {v} that's {c}", "track the {v} the {c} one"],
        sample_vehicle),
    Family("car_no_color", "car", "train", [
        "follow the {v}{dist}", "{pre} track the {v}", "keep the camera on that {v}{dist}"],
        lambda r: sample_vehicle(r, colored=False)),
    Family("car_question", "car", "train", [
        "can you follow the {c} {v}{dist}", "could you keep up with the {c} {v}", "will you track the {c} {v}{dist}"],
        sample_vehicle),
    Family("car_distance_first", "car", "train", [
        "close|stay close to the {c} {v}", "far|keep your distance from the {c} {v} and follow it",
        "far|hang back behind the {c} {v}", "close|get close to the {c} {v} and follow it"],
        sample_vehicle, use_dist=False),
    Family("car_valid", "car", "valid", [
        "film the {c} {v}{dist}", "stick with the {c} {v} {post}", "go follow that {c} {v}{dist}"], sample_vehicle),
    Family("car_test", "car", "test", [
        "i want you on the {c} {v}{dist}", "shadow the {c} {v}{dist}", "focus on the {c} {v} and follow it",
        "pick the {c} {v} and stay with it{dist}", "follow that {v}"], sample_vehicle),
    # ---------------- follow: animals ----------------
    Family("animal_follow", "animal", "train", [
        "{pre} follow the {c} {v}{dist} {post}", "follow that {c} {v}{dist}", "start following the {c} {v}"],
        sample_animal),
    Family("animal_track", "animal", "train", [
        "track the {c} {v}{dist}", "keep the camera on the {c} {v}", "lock onto the {c} {v}{dist}"], sample_animal),
    Family("animal_keepup", "animal", "train", [
        "keep up with the {c} {v}{dist}", "go after the {c} {v} {post}", "stay with the {c} {v}{dist}"],
        sample_animal),
    Family("animal_deictic", "animal", "train", [
        "see that {c} {v} follow it{dist}", "the {c} {v} by the {lm} follow it", "there's {an} {c} {v} go follow it"],
        sample_animal),
    Family("animal_no_color", "animal", "train", [
        "follow the {v}{dist}", "{pre} track the {v}", "follow my {v}{dist}", "keep up with our {v}"],
        lambda r: sample_animal(r, colored=False)),
    Family("animal_possessive", "animal", "train", [
        "follow my {c} {v}{dist}", "track our {c} {v}", "{pre} follow my {v}, the {c} one"], sample_animal),
    Family("animal_valid", "animal", "valid", [
        "film the {c} {v}{dist}", "stick with the {c} {v}", "go follow the {c} {v} {post}"], sample_animal),
    Family("animal_test", "animal", "test", [
        "shadow the {c} {v}{dist}", "i want you to stay on the {c} {v}", "focus on the {c} {v} and follow it{dist}",
        "follow the {v}"], sample_animal),
    # ---------------- follow: objects ----------------
    Family("object_follow", "object", "train", [
        "follow the {c} {v}{dist}", "{pre} track the {c} {v}", "keep the camera on the {c} {v}{dist}"], sample_object),
    Family("object_misc", "object", "train", [
        "go after the {c} {v} {post}", "follow that {v}{dist}", "can you follow the {c} {v}"], sample_object),
    Family("object_valid", "object", "valid", ["stick with the {c} {v}", "film the {c} {v}{dist}"], sample_object),
    Family("object_test", "object", "test", ["shadow the {c} {v}", "stay on the {c} {v}{dist}"], sample_object),
    # ---------------- hover ----------------
    Family("hover_stop", "hover", "train", ["stop", "stop stop", "stop now", "stop right there", "{pre} stop {post}",
                                            "okay stop"], _fixed("hover"), "hover"),
    Family("hover_hover", "hover", "train", ["hover", "hover here", "just hover", "hover in place {post}",
                                             "{pre} hover for a sec"], _fixed("hover"), "hover"),
    Family("hover_hold", "hover", "train", ["hold position", "hold", "hold it", "hold on", "hold still",
                                            "{pre} hold your position {post}"], _fixed("hover"), "hover"),
    Family("hover_stay", "hover", "train", ["stay there", "stay put", "stay right there", "stay where you are",
                                            "wait there", "wait", "wait a second"], _fixed("hover"), "hover"),
    Family("hover_cancel", "hover", "train", ["pause", "freeze", "don't move", "stop moving", "stop following",
                                              "quit following", "stop tracking {post}"], _fixed("hover"), "hover"),
    Family("hover_valid", "hover", "valid", ["can you stop", "could you just hover", "hang there", "hold up"],
           _fixed("hover"), "hover"),
    Family("hover_test", "hover", "test", ["stay in place", "hold steady", "just stay right here {post}",
                                           "stop where you are", "enough following just hover"],
           _fixed("hover"), "hover"),
    # ---------------- land ----------------
    Family("land_land", "land", "train", ["land", "land now", "{pre} land {post}", "land the drone", "land it"],
           _fixed("land"), "land"),
    Family("land_down", "land", "train", ["come down", "come on down", "bring it down", "come down now",
                                          "come back down and land"], _fixed("land"), "land"),
    Family("land_touch", "land", "train", ["touch down", "set down", "set it down", "put it down",
                                           "go down and land"], _fixed("land"), "land"),
    Family("land_done", "land", "train", ["time to land", "okay that's enough land", "we're done land",
                                          "let's land", "that's it land now"], _fixed("land"), "land"),
    Family("land_abort", "land", "train", ["abort", "emergency land", "abort abort", "emergency",
                                           "abort and land"], _fixed("land"), "land"),
    Family("land_valid", "land", "valid", ["can you land", "could you land now", "take it down"],
           _fixed("land"), "land"),
    Family("land_test", "land", "test", ["would you land please", "go ahead and land it", "landing time",
                                         "bring the drone down now", "drop down and land"], _fixed("land"), "land"),
    # ---------------- clarify (ambiguous) ----------------
    Family("clarify_pronoun", "clarify", "train", ["follow them", "follow {him}", "follow it", "track them",
                                                   "go after {him}", "keep up with them", "{pre} follow {him} {post}"],
           _fixed("clarify", "ambiguous"), "clarify", "ambiguous"),
    Family("clarify_deictic", "clarify", "train", ["go after that", "follow that one", "track that one",
                                                   "follow this", "follow that thing over there",
                                                   "that one follow it", "follow the one over there"],
           _fixed("clarify", "ambiguous"), "clarify", "ambiguous"),
    Family("clarify_missing", "clarify", "train", ["follow", "follow the", "start following", "follow uh",
                                                   "follow um the", "can you follow", "go follow", "track",
                                                   "follow the uh"],
           _fixed("clarify", "ambiguous"), "clarify", "ambiguous"),
    Family("clarify_vague", "clarify", "train", ["follow someone", "follow somebody", "follow anyone",
                                                 "follow something", "follow whatever moves", "follow the red one",
                                                 "follow the blue thing", "go after the green one"],
           _fixed("clarify", "ambiguous"), "clarify", "ambiguous"),
    Family("clarify_me", "clarify", "train", ["follow me", "come follow me", "track me", "follow me around",
                                              "keep following me", "film me"],
           _fixed("clarify", "ambiguous"), "clarify", "ambiguous"),
    Family("clarify_multi", "clarify", "train", ["follow the {c} {v} and the {noun} {desc}",
                                                 "follow the {ac} {animal} and the {c} {v}",
                                                 "follow both of them", "follow those two",
                                                 "follow the {noun} {desc} and the {ac} {animal}"],
           _fixed("clarify", "ambiguous"), "clarify", "ambiguous"),
    Family("clarify_valid", "clarify", "valid", ["go get {him}", "follow that over there", "follow the",
                                                 "follow whoever", "follow us"],
           _fixed("clarify", "ambiguous"), "clarify", "ambiguous"),
    Family("clarify_test", "clarify", "test", ["stay with {him}", "follow the thing", "go after it",
                                               "follow that one there", "follow uh the um", "chase after them",
                                               "follow the purple one", "follow me please",
                                               "follow the {noun} and the {c} {v}"],
           _fixed("clarify", "ambiguous"), "clarify", "ambiguous"),
    # ---------------- refuse: identity ----------------
    Family("refuse_name", "refuse_identity", "train", ["follow {name}", "find {name}", "track {name}",
                                                       "go find {name} {post}", "where is {name} follow {him}",
                                                       "{pre} follow {name}"],
           _fixed("refuse", "identity_request"), "refuse", "identity_request"),
    Family("refuse_relation", "refuse_identity", "train", ["follow my {rel}", "find my {rel}", "track my {rel}",
                                                           "follow my neighbor {name}", "go find my {rel} {name}",
                                                           "{pre} follow my {rel} {post}"],
           _fixed("refuse", "identity_request"), "refuse", "identity_request"),
    Family("refuse_identify", "refuse_identity", "train", ["who is that", "who's that {short}",
                                                           "identify that {noun}", "tell me who that {noun} is",
                                                           "find out who that person is",
                                                           "recognize the {noun} {desc}",
                                                           "who is the {noun} {desc}"],
           _fixed("refuse", "identity_request"), "refuse", "identity_request"),
    Family("refuse_famous", "refuse_identity", "train", ["follow {famous}", "find {famous}", "track {famous}"],
           _fixed("refuse", "identity_request"), "refuse", "identity_request"),
    Family("refuse_identity_valid", "refuse_identity", "valid", ["go after {name}", "look for my {rel}",
                                                                 "is that {name} over there"],
           _fixed("refuse", "identity_request"), "refuse", "identity_request"),
    Family("refuse_identity_test", "refuse_identity", "test", ["shadow {name}", "follow my {rel} please",
                                                               "can you figure out who the {noun} {desc} is",
                                                               "keep the camera on {name}", "follow {famous} {post}"],
           _fixed("refuse", "identity_request"), "refuse", "identity_request"),
    # ---------------- refuse: surveillance and harm (unsafe) ----------------
    Family("refuse_everyone", "refuse_unsafe", "train", ["track everyone", "follow everyone",
                                                         "track everybody in the park", "follow all the people",
                                                         "track every car on the street",
                                                         "follow every person who walks by"],
           _fixed("refuse", "unsafe"), "refuse", "unsafe"),
    Family("refuse_house", "refuse_unsafe", "train", ["watch that house", "keep an eye on the house across the street",
                                                      "look in that window", "watch my neighbor's yard",
                                                      "monitor the parking lot", "hover outside that window",
                                                      "spy on the people next door"],
           _fixed("refuse", "unsafe"), "refuse", "unsafe"),
    Family("refuse_stalk", "refuse_unsafe", "train", ["follow that {noun} home", "follow the {noun} {desc} home",
                                                      "find out where that {noun} lives", "secretly follow the {noun} {desc}",
                                                      "follow the {noun} {desc} without them noticing",
                                                      "follow {name} all day and see where {he} goes"],
           _fixed("refuse", "unsafe"), "refuse", "unsafe"),
    Family("refuse_harm", "refuse_unsafe", "train", ["fly into that {noun}", "crash into the {c} {v}",
                                                     "hit the {ac} {animal}", "dive bomb them",
                                                     "chase the {ac} {animal} away", "scare the {animal}",
                                                     "fly at the {noun} {desc}", "ram the {c} {v}"],
           _fixed("refuse", "unsafe"), "refuse", "unsafe"),
    Family("refuse_unsafe_valid", "refuse_unsafe", "valid", ["record everyone in the room", "watch the front door of that house",
                                                             "buzz that {noun}'s head"],
           _fixed("refuse", "unsafe"), "refuse", "unsafe"),
    Family("refuse_unsafe_test", "refuse_unsafe", "test", ["track all the cars", "peek into the neighbor's window",
                                                           "follow the {noun} {desc} to their house",
                                                           "smash into the {c} {v}", "spy on everyone outside"],
           _fixed("refuse", "unsafe"), "refuse", "unsafe"),
    # ---------------- refuse: unsupported ----------------
    Family("refuse_takeoff", "refuse_unsupported", "train", ["take off", "takeoff", "launch", "lift off",
                                                             "start the motors", "fly up and take off"],
           _fixed("refuse", "unsupported"), "refuse", "unsupported"),
    Family("refuse_tricks", "refuse_unsupported", "train", ["do a flip", "do a barrel roll", "do a backflip",
                                                            "spin around really fast", "do some tricks"],
           _fixed("refuse", "unsupported"), "refuse", "unsupported"),
    Family("refuse_go", "refuse_unsupported", "train", ["fly to the store", "fly home", "go to the kitchen",
                                                        "deliver this package", "fly over the highway",
                                                        "go up {n} meters", "fly {n} feet high"],
           _fixed("refuse", "unsupported"), "refuse", "unsupported"),
    Family("refuse_misc", "refuse_unsupported", "train", ["take a picture", "take a selfie", "play some music",
                                                          "turn on the lights", "what's the weather",
                                                          "tell me a joke"],
           _fixed("refuse", "unsupported"), "refuse", "unsupported"),
    Family("refuse_unsupported_valid", "refuse_unsupported", "valid", ["get airborne", "fly to the park",
                                                                       "record a video and post it"],
           _fixed("refuse", "unsupported"), "refuse", "unsupported"),
    Family("refuse_unsupported_test", "refuse_unsupported", "test", ["take off now", "do a loop",
                                                                     "fly to my car", "climb {n} meters",
                                                                     "snap a photo"],
           _fixed("refuse", "unsupported"), "refuse", "unsupported"),
]

# Share of each group in every split before dedup (about 51% follow, 22% hover+land, 15% clarify,
# 12% refuse). Dedup drops repeats of short commands, so the final hover+land share is a bit lower.
GROUP_WEIGHTS = {
    "person": 0.25, "person_named": 0.03, "car": 0.11, "animal": 0.08, "object": 0.04,
    "hover": 0.11, "land": 0.11, "clarify": 0.15,
    "refuse_identity": 0.055, "refuse_unsafe": 0.045, "refuse_unsupported": 0.02,
}
SPLIT_SIZES = {"train": 2200, "valid": 150, "test": 300}

# ----------------------------------------------------------------------------------------------
# Rendering and ASR noise
# ----------------------------------------------------------------------------------------------


class _Slots(dict):
    """format_map dict that tolerates unused slots and fails loudly on missing ones."""

    def __missing__(self, key):  # pragma: no cover - a template bug
        raise KeyError(f"template slot {{{key}}} not provided")


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+,", ",", text)
    return text


def render(family: Family, template: str, rng: random.Random) -> tuple[str, Intent]:
    """Fill one template; returns the clean (pre-noise) text and the constructed label."""
    forced = None
    if "|" in template:
        forced, template = template.split("|", 1)
    slots, label = family.sampler(rng)
    dist_text, dist_label = "", "medium"
    if family.action == "follow":
        if forced:
            dist_label = forced
        elif family.use_dist and "{dist}" in template:
            dist_text, dist_label = _distance(rng, "random")
    s = _Slots(slots)
    s.update({"pre": rng.choice(PRE), "post": rng.choice(POST), "dist": dist_text})
    s.setdefault("him", "them")
    text = _tidy(template.format_map(s))
    if family.action == "follow":
        intent = make_intent("follow", label["subject"], label["category"], label["color"], dist_label)
    else:
        intent = make_intent(family.action, reason=family.reason)
    return text, intent


def asr_noise(text: str, rng: random.Random, protect: frozenset[str] = frozenset()) -> str:
    """Make a clean sentence look like a speech-recognition transcript."""
    words = text.lower().replace(",", " ,").split()
    # 1) phonetic misspellings, at most two per utterance
    n_miss = 0
    for i, w in enumerate(words):
        if n_miss < 2 and w in MISSPELL and w not in protect and rng.random() < 0.10:
            words[i] = rng.choice(MISSPELL[w])
            n_miss += 1
    # 2) a dropped function word
    if len(words) > 3 and rng.random() < 0.15:
        idx = [i for i, w in enumerate(words) if w in DROPPABLE]
        if idx:
            del words[rng.choice(idx)]
    # 3) a duplicated word ("the the", "follow follow")
    if len(words) > 1 and rng.random() < 0.08:
        i = rng.randrange(len(words))
        if words[i] != ",":
            words.insert(i, words[i])
    # 4) a filler at the start or in the middle
    if rng.random() < 0.25:
        i = 0 if rng.random() < 0.6 else rng.randrange(len(words) + 1)
        words.insert(i, rng.choice(FILLERS))
    # 5) a wake word
    if rng.random() < 0.30:
        words = rng.choice(WAKE_WORDS).split() + ([","] if rng.random() < 0.5 else []) + words
    out = " ".join(words).replace(" ,", ",").strip(" ,")
    out = re.sub(r",+", ",", out)
    # 6) transcript style: Omnilingual CTC style (lowercase, no punctuation) or Voice Transcribe style
    if rng.random() < 0.55:
        out = out.translate(str.maketrans("", "", string.punctuation.replace("'", "").replace("-", "")))
    else:
        out = out[:1].upper() + out[1:]
        out = re.sub(r"\bi\b", "I", out)
        q = out.lower().startswith(("can ", "could ", "would ", "will ", "who ", "where ", "what ", "is "))
        out = out.rstrip(".?!") + ("?" if q else ".")
    return _tidy(out)


def norm_key(text: str) -> str:
    """Dedup key: lowercase letters and spaces only."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


# ----------------------------------------------------------------------------------------------
# Assembling the splits
# ----------------------------------------------------------------------------------------------


@dataclass
class Example:
    id: str
    split: str
    group: str
    family: str
    source: str            # template | paraphrase | gold
    clean: str
    transcript: str
    intent: Intent
    extra: dict = field(default_factory=dict)

    def record(self) -> dict:
        return {"id": self.id, "split": self.split, "group": self.group, "family": self.family,
                "source": self.source, "clean": self.clean, "transcript": self.transcript,
                "intent": self.intent.model_dump(), **self.extra}


def families_for(split: str, group: str) -> list[Family]:
    return [f for f in FAMILIES if f.split == split and f.group == group]


def _protect(intent: Intent, clean: str) -> frozenset[str]:
    """Words whose misspelling is not interesting (names) are protected from noise."""
    return frozenset(w for w in clean.lower().split() if w in NAMES or w in LAST_NAMES)


def build_split(split: str, n: int, seed: int, taken: set[str]) -> list[Example]:
    """Sample n examples for one split from its own template families, by group weight."""
    rng = random.Random(f"{seed}-{split}")
    groups = list(GROUP_WEIGHTS)
    weights = [GROUP_WEIGHTS[g] for g in groups]
    out: list[Example] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        group = rng.choices(groups, weights)[0]
        fams = families_for(split, group)
        if not fams:
            raise RuntimeError(f"no {split} family for group {group}")
        fam = rng.choice(fams)
        template = rng.choice(fam.templates)
        clean, intent = render(fam, template, rng)
        transcript = asr_noise(clean, rng, _protect(intent, clean))
        key = norm_key(transcript)
        # Short commands repeat a lot ("stop"); allow a few exact repeats in train only.
        dup_ok = split == "train" and len(key.split()) <= 3 and rng.random() < 0.5
        if key in taken or (key in seen and not dup_ok):
            continue
        seen.add(key)
        out.append(Example(f"{split}-{len(out):05d}", split, group, fam.fid, "template", clean, transcript, intent))
    if len(out) < n:
        raise RuntimeError(f"could only build {len(out)} of {n} {split} examples")
    return out


# ----------------------------------------------------------------------------------------------
# Optional Muse Spark 1.3 paraphrases (wording only; labels stay from construction)
# ----------------------------------------------------------------------------------------------

PARA_SYSTEM = """You rewrite short voice commands for a small camera drone into different natural spoken wording.
For each item, write ONE new casual way a person might say the same command out loud.
Keep the exact meaning and every key detail: the same colors, the same clothing or vehicle or animal words (synonyms are fine), the same names, the same distance request.
Do not add new details, colors, names, or extra requests. Keep it under 20 words. No quotes."""

PARA_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "text"],
        "properties": {"id": {"type": "integer"}, "text": {"type": "string"}}}}},
}
PARA_GROUPS = {"person", "car", "animal", "object", "hover", "land", "clarify", "refuse_identity"}

_SYN = {
    "shirt": {"shirt", "t-shirt", "tshirt", "tee", "top"}, "hoodie": {"hoodie", "hoody", "sweatshirt"},
    "sweater": {"sweater", "jumper", "sweatshirt"}, "pants": {"pants", "trousers"}, "cap": {"cap"},
    "coat": {"coat", "raincoat", "overcoat"}, "jacket": {"jacket", "windbreaker"},
    "pickup truck": {"pickup", "truck"}, "toy car": {"toy", "car", "rc"}, "taxi": {"taxi", "cab"},
    "van": {"van", "minivan"}, "suv": {"suv"}, "car": {"car", "sedan", "hatchback"},
    "dog": {"dog", "puppy", "pup", "doggy", "doggo"}, "cat": {"cat", "kitty", "kitten"},
    "rabbit": {"rabbit", "bunny"}, "horse": {"horse", "pony"}, "bike": {"bike", "bicycle"},
    "stroller": {"stroller", "pram"},
}
_COLOR_WORDS = set(COLORS) | {"grey", "silver"}


def paraphrase_ok(text: str, ex: Example) -> bool:
    """Keep a paraphrase only if the label is still obviously right for it."""
    words = set(norm_key(text).split())
    if not words or len(words) > 24:
        return False
    i = ex.intent
    colors_in = (words & _COLOR_WORDS) - {"grey"} | ({"gray"} if "grey" in words else set())
    if i.action == "follow":
        if i.color != "none" and colors_in != {i.color}:
            return False
        if i.color == "none" and colors_in:
            return False
        head = i.subject_phrase.split()[-1] if i.category != "person" else None
        if i.category == "person":
            parts = i.subject_phrase.split()
            garment = " ".join(parts[3:]) if len(parts) > 3 else None
            if garment and not (words & _SYN.get(garment, {garment})):
                return False
        elif head:
            canon = i.subject_phrase.replace(i.color + " ", "") if i.color != "none" else i.subject_phrase
            if not (words & _SYN.get(canon, {head})):
                return False
        if i.distance == "medium" and words & {"close", "closely", "distance", "far", "back", "meters", "tight"}:
            return False
        if i.distance != "medium" and not (words & {"close", "closely", "distance", "far", "back", "meters",
                                                     "tight", "away", "behind", "near"}):
            return False
        return not (words & set(NAMES))
    if i.action == "land":
        return bool(words & {"land", "landing", "down"}) and not (words & {"follow", "track"})
    if i.action == "hover":
        return bool(words & {"stop", "hover", "hold", "stay", "wait", "pause", "freeze", "still", "put"}) \
            and "land" not in words and "down" not in words
    if i.action == "clarify":
        return not colors_in and not (words & {"person", "car", "dog", "cat", "shirt", "truck"})
    if i.reason_code == "identity_request":
        names = set(norm_key(ex.clean).split()) & (set(NAMES) | set(RELATIONS))
        return bool(names) and names <= words
    return False


def spark_paraphrase(examples: list[Example], seed: int, budget_usd: float) -> list[Example]:
    """Paraphrase clean seed sentences in batches of 10 (one strict-schema call each)."""
    import httpx

    from training.common import log_spend, meta_chat, spark_cost, spent_usd

    rng = random.Random(f"{seed}-para")
    out: list[Example] = []
    client = httpx.Client(timeout=120)
    batches = [examples[i:i + 10] for i in range(0, len(examples), 10)]
    total = 0.0
    for b_idx, batch in enumerate(batches):
        if spent_usd("meta:") + 0.02 > budget_usd:
            print(f"[paraphrase] stopping: spend cap ${budget_usd:.2f} reached", file=sys.stderr)
            break
        items = [{"id": k, "command": ex.clean} for k, ex in enumerate(batch)]
        msgs = [{"role": "system", "content": PARA_SYSTEM},
                {"role": "user", "content": json.dumps({"items": items})}]
        try:
            resp = meta_chat(msgs, schema=PARA_SCHEMA, schema_name="paraphrases", client=client, max_tokens=3000)
        except Exception as e:  # keep going; the template data is complete without paraphrases
            print(f"[paraphrase] batch {b_idx} failed: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
            continue
        usage = resp.get("usage", {})
        cost = spark_cost(usage)
        total += cost
        log_spend("meta:muse-spark-1.3", usage.get("total_tokens", 0), cost, note=f"paraphrase batch {b_idx}")
        try:
            got = json.loads(resp["choices"][0]["message"]["content"])["items"]
        except Exception:
            continue
        for item in got:
            k = item.get("id")
            if not isinstance(k, int) or not 0 <= k < len(batch):
                continue
            ex = batch[k]
            text = _tidy(str(item.get("text", "")).strip().strip('"'))
            if not paraphrase_ok(text, ex):
                continue
            transcript = asr_noise(text, rng, _protect(ex.intent, text))
            out.append(Example(f"{ex.split}-p{len(out):04d}", ex.split, ex.group, ex.family, "paraphrase",
                               text, transcript, ex.intent, {"seed_batch": b_idx, "seed_text": ex.clean}))
    client.close()
    print(f"[paraphrase] kept {len(out)} of {sum(len(b) for b in batches)}; est cost ${total:.4f}", file=sys.stderr)
    return out


def paraphrase_seeds(split: str, n: int, seed: int) -> list[Example]:
    """Fresh clean sentences from the split's families (paraphrasable groups only)."""
    rng = random.Random(f"{seed}-{split}-seeds")
    groups = [g for g in GROUP_WEIGHTS if g in PARA_GROUPS]
    weights = [GROUP_WEIGHTS[g] for g in groups]
    seeds = []
    for _ in range(n):
        g = rng.choices(groups, weights)[0]
        fam = rng.choice(families_for(split, g))
        clean, intent = render(fam, rng.choice(fam.templates), rng)
        seeds.append(Example("seed", split, g, fam.fid, "seed", clean, clean, intent))
    return seeds


# ----------------------------------------------------------------------------------------------
# Writing files
# ----------------------------------------------------------------------------------------------


def write_all(splits: dict[str, list[Example]], out_dir: Path, gold: list[Example]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Full records with metadata (used by eval.py and the tests)
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", [e.record() for e in rows])
    # mlx_lm chat format: {train,valid,test}.jsonl with only "messages"
    for name, rows in splits.items():
        write_jsonl(out_dir / "mlx" / f"{name}.jsonl", [to_chat(e.transcript, e.intent) for e in rows])
    # Fireworks chat JSONL: same messages; assistant thinking goes in reasoning_content
    write_jsonl(out_dir / "fireworks" / "train.jsonl", [to_chat(e.transcript, e.intent) for e in splits["train"]])
    write_jsonl(out_dir / "fireworks" / "valid.jsonl", [to_chat(e.transcript, e.intent) for e in splits["valid"]])
    smoke = splits["train"][:30]
    write_jsonl(out_dir / "fireworks" / "smoke30.jsonl", [to_chat(e.transcript, e.intent) for e in smoke])
    if gold:
        write_jsonl(out_dir / "mlx" / "gold.jsonl", [to_chat(e.transcript, e.intent) for e in gold])
    stats = {"prompt_version": PROMPT_VERSION, "sizes": {k: len(v) for k, v in splits.items()},
             "gold": len(gold), "by_group": {}, "by_action": {}, "by_source": {}}
    for name, rows in splits.items():
        for key, fn in (("by_group", lambda e: e.group), ("by_action", lambda e: e.intent.action),
                        ("by_source", lambda e: e.source)):
            d = stats[key].setdefault(name, {})
            for e in rows:
                d[fn(e)] = d.get(fn(e), 0) + 1
    stats["families"] = {s: sorted({f.fid for f in FAMILIES if f.split == s}) for s in ("train", "valid", "test")}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def load_gold(path: Path) -> list[Example]:
    """Hand-written gold set (never trained on)."""
    if not path.exists():
        return []
    out = []
    for r in read_jsonl(path):
        intent = Intent.model_validate(r["intent"])
        out.append(Example(r["id"], "gold", r.get("group", "gold"), "gold", "gold", r["transcript"],
                           r["transcript"], intent, {"note": r.get("note", "")}))
    return out


def generate(seed: int = 7, paraphrase: bool = False, para_train: int = 450, para_valid: int = 30,
             para_test: int = 60, budget_usd: float = 10.0, out_dir: Path = DATA_DIR) -> dict:
    gold = load_gold(out_dir / "gold.jsonl")
    gold_keys = {norm_key(g.transcript) for g in gold}
    # Test first, then valid, then train; each later split skips transcripts already used.
    taken: set[str] = set(gold_keys)
    splits: dict[str, list[Example]] = {}
    para: dict[str, list[Example]] = {"train": [], "valid": [], "test": []}
    if paraphrase:
        for split, n in (("test", para_test), ("valid", para_valid), ("train", para_train)):
            para[split] = spark_paraphrase(paraphrase_seeds(split, n, seed), seed, budget_usd)
    for split in ("test", "valid", "train"):
        p = [e for e in para[split] if norm_key(e.transcript) not in taken]
        p_keys = set()
        p_unique = []
        for e in p:
            k = norm_key(e.transcript)
            if k not in p_keys:
                p_keys.add(k)
                p_unique.append(e)
        n_template = SPLIT_SIZES[split] - len(p_unique)
        rows = build_split(split, n_template, seed, taken | p_keys) + p_unique
        random.Random(f"{seed}-{split}-shuffle").shuffle(rows)
        for k, e in enumerate(rows):
            e.id = f"{split}-{k:05d}"
        splits[split] = rows
        taken |= {norm_key(e.transcript) for e in rows}
    return write_all(splits, out_dir, gold)


def rebuild(out_dir: Path = DATA_DIR) -> dict:
    """Rewrite the derived chat files (mlx/, fireworks/) from the committed records. No network."""
    splits = {}
    for split in ("test", "valid", "train"):
        rows = []
        for r in read_jsonl(out_dir / f"{split}.jsonl"):
            extra = {k: v for k, v in r.items() if k not in
                     ("id", "split", "group", "family", "source", "clean", "transcript", "intent")}
            rows.append(Example(r["id"], r["split"], r["group"], r["family"], r["source"], r["clean"],
                                r["transcript"], Intent.model_validate(r["intent"]), extra))
        splits[split] = rows
    return write_all(splits, out_dir, load_gold(out_dir / "gold.jsonl"))


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--paraphrase", action="store_true", help="add Muse Spark 1.3 paraphrases (Meta API)")
    ap.add_argument("--para-train", type=int, default=450)
    ap.add_argument("--para-valid", type=int, default=30)
    ap.add_argument("--para-test", type=int, default=60)
    ap.add_argument("--budget-usd", type=float, default=10.0, help="cap on total logged Meta API spend")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--rebuild", action="store_true",
                    help="only rewrite mlx/ and fireworks/ chat files from the committed records")
    a = ap.parse_args(argv)
    if a.rebuild:
        stats = rebuild(a.out)
        print(json.dumps(stats["sizes"]))
        return
    stats = generate(a.seed, a.paraphrase, a.para_train, a.para_valid, a.para_test, a.budget_usd, a.out)
    print(json.dumps({k: stats[k] for k in ("sizes", "gold", "by_action", "by_source")}, indent=1))


if __name__ == "__main__":
    main()
