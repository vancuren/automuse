"""The single source of truth for turning a voice transcript into a mission.

Used by the agent (Glimmer, Spark, keyword fallback) and by training (data generation, evaluation),
so the fine-tuned model sees exactly the prompt it was trained on.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from mhd.contracts import Intent

PROMPT_VERSION = "intent-v1"

SYSTEM_PROMPT = """You turn a spoken command for a small camera drone into one JSON object. The drone follows people or things by appearance only.
Rules:
- action: follow (follow and track a subject), hover (stop moving, hold position), land, clarify (the request is ambiguous or missing a subject), refuse (identity-based, surveillance, harmful, or impossible).
- subject_phrase: a short noun phrase a segmentation model understands, at most 4 words, one concept, at most two attributes, e.g. "person in blue shirt", "green car", "brown dog". Empty string unless action is follow.
- category: person, car, animal, object, or none.
- color: the main color word in the description, or none.
- distance: close, medium, or far. Use medium unless the user asks to stay close or far.
- reason_code: none, ambiguous, identity_request (the user names a specific person or asks to identify someone), unsafe, or unsupported.
- Never follow someone by name or identity. Appearance only.
- Transcripts come from speech recognition: they may be lowercase, unpunctuated, and misspelled. Interpret them generously.
Reply with the JSON object only."""

GLIMMER_SYSTEM_PROMPT = "Reasoning strength: low\n" + SYSTEM_PROMPT


def intent_json_schema() -> dict:
    """Strict JSON schema (additionalProperties false, all fields required) for structured output."""
    schema = Intent.model_json_schema()
    schema["additionalProperties"] = False
    schema["required"] = list(schema["properties"].keys())
    schema.pop("title", None)
    return schema


def build_reply(intent: Intent) -> str:
    """The spoken line, built locally so models emit fewer tokens."""
    if intent.action == "follow":
        return f"Following the {intent.subject_phrase}."
    if intent.action == "hover":
        return "Holding position."
    if intent.action == "land":
        return "Landing now."
    if intent.action == "refuse":
        if intent.reason_code == "identity_request":
            return "I can't follow someone by identity. Tell me what they're wearing instead."
        return "I can't do that one."
    return "Who should I follow? Describe them, like the person in the blue shirt."


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_intent(text: str) -> Optional[Intent]:
    """Extract and validate the first JSON object in a model reply. Returns None if invalid."""
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return Intent.model_validate(json.loads(m.group(0)))
    except Exception:
        return None


# Safety words act immediately, without any language model (the local listener and the fast path).
SAFETY_WORDS = {
    "land": "land", "landing": "land", "lan": "land", "lend": "land",
    "stop": "hover", "stopp": "hover", "hover": "hover", "hold": "hover", "freeze": "hover",
    "abort": "land", "emergency": "land",
}
