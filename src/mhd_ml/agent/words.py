"""Transcript normalization and safety-word matching (shared by the safety listener and the keyword grammar).

CTC output from Omnilingual is lowercase, unpunctuated, and sometimes misspelled ("lan", "stopp").
Matching works on whole words, never substrings, so "island" and "landscape" do not trigger "land".

Fuzzy matching is deliberately narrow. A false alarm only ever hovers or lands, but in a demo a false
hover in the middle of "follow the person in the gold shirt" would still be a failure, so:
- exact matches against mhd.intent_spec.SAFETY_WORDS (which already lists the common CTC misspellings);
- plus edit distance 1 to a short list of canonical words, only for words of 4+ letters that share the
  first letter, and never for a real English word that happens to be one edit away (NEAR_MISSES).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from mhd.intent_spec import SAFETY_WORDS

# Canonical safety words eligible for fuzzy matching, and what they mean. "hold" is exact-only: it is
# one edit from too many everyday words (gold, cold, bold, hole, held, hood).
FUZZY_CANON = {
    "land": "land", "landing": "land", "abort": "land", "emergency": "land",
    "stop": "hover", "hover": "hover", "freeze": "hover",
}

# Real words one edit away from a canonical safety word. These never trigger.
NEAR_MISSES = {
    # land
    "lane", "lad", "lard", "laud", "lent", "lens", "lind", "band", "hand", "sand", "wand", "bland",
    "glad", "lander", "lands",
    # stop
    "shop", "step", "stoop", "stomp", "stow", "slop", "sop", "stops", "top", "atop", "stoppable", "swop",
    "stood", "stork", "store", "story", "stone", "stove",
    # hover
    "hovel", "hoover", "homer", "hovers", "cover", "over", "lover", "rover", "mover", "power",
    # freeze
    "frieze", "freezer", "breeze", "frees", "freed",
    # abort
    "abbot", "aborts",
}

# Inflections that mean the same thing as their safety word (exact match only).
INFLECTIONS = {"landed": "land", "aborting": "land", "stopping": "hover", "hovering": "hover"}

# Misspellings seen from Omnilingual CTC on `say` test audio (listener only; the keyword grammar, which
# also reads clean Voice Transcribe text, must not turn "stalk" into hover).
CTC_VARIANTS = {
    "landin": "land", "aboard": "land", "abord": "land",
    "stob": "hover", "stap": "hover", "stalk": "hover", "stog": "hover",
    "preeze": "hover", "preese": "hover", "freese": "hover", "freez": "hover",
}

# CTC sometimes glues a safety word to a neighbour ("stop now land" -> "stop nowland"). A word that is a
# safety word plus one of these glue words counts ("nowland", "landnow", "stopit"); "island" does not.
GLUE = {"now", "please", "then", "and", "it", "right", "so", "just", "ok", "okay", "go", "the", "here",
        "there", "you"}

_PUNCT_RE = re.compile(r"[^a-z0-9' ]+")


def normalize(text: str) -> str:
    """Lowercase, replace punctuation with spaces, collapse whitespace."""
    t = (text or "").lower().replace("-", " ")
    t = _PUNCT_RE.sub(" ", t)
    return " ".join(t.split())


def words(text: str) -> list[str]:
    return [w.strip("'") for w in normalize(text).split() if w.strip("'")]


def edit_distance(a: str, b: str, limit: int = 2) -> int:
    """Levenshtein distance, stopping early once it exceeds `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        best = cur[0]
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            best = min(best, cur[j])
        if best > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def safety_action_for_word(word: str, fuzzy: bool = True) -> Optional[tuple[str, str, bool]]:
    """(action, canonical, was_fuzzy) for one normalized word, or None. action is 'land' or 'hover'.

    fuzzy=False: SAFETY_WORDS and inflections only (the keyword grammar). fuzzy=True adds the CTC
    misspellings, glued words ("nowland"), and edit distance 1 (the safety listener)."""
    if word in SAFETY_WORDS:
        return SAFETY_WORDS[word], word, False
    if word in INFLECTIONS:
        return INFLECTIONS[word], word, False
    if not fuzzy:
        return None
    if word in CTC_VARIANTS:
        return CTC_VARIANTS[word], word, True
    for canon in ("land", "stop", "hover", "abort"):
        if len(word) > len(canon) + 1:
            if word.startswith(canon) and word[len(canon):] in GLUE:
                return FUZZY_CANON[canon], canon, True
            if word.endswith(canon) and word[:-len(canon)] in GLUE:
                return FUZZY_CANON[canon], canon, True
    if len(word) < 4 or word in NEAR_MISSES:
        return None
    for canon, action in FUZZY_CANON.items():
        if word[0] == canon[0] and edit_distance(word, canon, limit=1) <= 1:
            return action, canon, True
    return None


# Operator command sent for each safety action (see OperatorCommand in mhd.contracts).
COMMAND_FOR_ACTION = {"land": "land", "hover": "pause"}


@dataclass
class SafetyHit:
    action: str            # "land" or "hover"
    command: str           # OperatorCommand.command: "land" or "pause"
    word: str              # the word as transcribed
    canonical: str         # the safety word it matched
    fuzzy: bool
    transcript: str
    kind: str = "final"    # "final" (whole VAD segment) or "partial" (speech still going)
    audio_t_s: float = 0.0     # stream position (seconds) when the hit was produced
    compute_ms: float = 0.0    # decode time of the transcript that produced the hit
    wall_ms: float = 0.0       # mhd.contracts.now_ms() when the hit was produced

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def match_safety(text: str, partial: bool = False, truncated_start: bool = False,
                 fuzzy: bool = True) -> Optional[SafetyHit]:
    """Find a safety word in a transcript. Land beats hover when both appear.

    partial=True: the speaker is still talking, so the last word may be cut off ("land" from
    "landscape"); it is ignored until more audio arrives. truncated_start=True: the window began
    mid-utterance, so the first word may be a fragment ("land" from "island"); it is ignored too.
    """
    ws = words(text)
    lo = 1 if truncated_start else 0
    hi = len(ws) - 1 if partial else len(ws)
    best: Optional[SafetyHit] = None
    for w in ws[lo:hi]:
        r = safety_action_for_word(w, fuzzy=fuzzy)
        if r is None:
            continue
        action, canon, was_fuzzy = r
        hit = SafetyHit(action=action, command=COMMAND_FOR_ACTION[action], word=w, canonical=canon,
                        fuzzy=was_fuzzy, transcript=text, kind="partial" if partial else "final")
        if best is None or (best.action == "hover" and action == "land"):
            best = hit
        if action == "land":
            break
    return best
