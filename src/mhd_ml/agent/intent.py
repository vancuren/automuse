"""Transcript -> Intent: keyword grammar, Glimmer and Spark clients, and the hedged router.

Language models choose missions; they never move the sticks, and no output of theirs can produce an
E-stop or a takeoff (the Intent schema has no such action).

- KeywordIntent: deterministic grammar. Always available, instant, used as the fallback and as the
  safety fast path ("stop", "land" never wait for a model).
- ChatIntentClient: any OpenAI-compatible /chat/completions endpoint, streamed; returns as soon as the
  streamed text holds a complete, valid Intent JSON object.
    * GlimmerClient: Muse Glimmer 30B (+ our LoRA) on Fireworks, or mlx_lm.server / llama-server
      locally. System prompt GLIMMER_SYSTEM_PROMPT (exactly what the LoRA was trained on).
    * SparkClient: muse-spark-1.3 on the Meta Model API with response_format json_schema strict
      (intent_json_schema()), reasoning_effort "minimal" (reasoning cannot be disabled; "none" is a
      400), no tools (tool_choice only supports "auto", so the schema route is the strict one).
- HedgedRouter: sends the transcript to Glimmer and Spark at once. Glimmer's answer is used if it is
  valid within 2.5 s; otherwise the first valid answer wins; if none is valid in time, the keyword
  grammar answers. Records which model answered and every model's latency.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from pydantic import ValidationError

from mhd.contracts import Intent
from mhd.intent_spec import GLIMMER_SYSTEM_PROMPT, SYSTEM_PROMPT, intent_json_schema, parse_intent

from .words import normalize, safety_action_for_word, words

log = logging.getLogger("mhd.agent.intent")


# ---------------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------------
@dataclass
class IntentAttempt:
    model: str
    intent: Optional[Intent]
    latency_ms: float                     # request start -> complete valid JSON (or failure)
    raw: str = ""
    error: Optional[str] = None
    ttft_ms: Optional[float] = None       # first streamed content token
    repaired: bool = False                # JSON needed a lenient fix-up to validate
    usage: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.intent is not None

    def summary(self) -> dict:
        return {"model": self.model, "valid": self.valid, "latency_ms": round(self.latency_ms, 1),
                "ttft_ms": None if self.ttft_ms is None else round(self.ttft_ms, 1),
                "error": self.error, "repaired": self.repaired,
                "action": self.intent.action if self.intent else None}


@dataclass
class RouteResult:
    intent: Intent
    model: str
    latency_ms: float                     # router start -> decision
    reason: str                           # glimmer_in_window | first_valid | keyword_fallback | ...
    attempts: dict = field(default_factory=dict)   # model -> IntentAttempt.summary()


# ---------------------------------------------------------------------------------------------------
# Lenient parsing of model output
# ---------------------------------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_ALLOWED = {
    "action": {"follow", "hover", "land", "clarify", "refuse"},
    "category": {"person", "car", "animal", "object", "none"},
    "distance": {"close", "medium", "far"},
    "reason_code": {"none", "ambiguous", "identity_request", "unsafe", "unsupported"},
}


def _json_objects(text: str) -> list[str]:
    """Every balanced top-level {...} substring, in order (string-aware)."""
    out, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                out.append(text[start:i + 1])
    return out


def _repair(d: dict) -> Optional[dict]:
    """Fix small, unambiguous deviations: case, None/null values, missing distance or reason_code."""
    if not isinstance(d, dict) or "action" not in d:
        return None
    r = dict(d)
    for k in ("action", "category", "distance", "reason_code", "color"):
        v = r.get(k)
        if v is None:
            continue
        r[k] = str(v).strip().lower()
    r.setdefault("distance", "medium")
    r.setdefault("reason_code", "none")
    r.setdefault("category", "none")
    if r.get("color") in (None, "", "null"):
        r["color"] = "none"
    if r.get("subject_phrase") is None:
        r["subject_phrase"] = ""
    r = {k: r[k] for k in ("action", "subject_phrase", "category", "color", "distance", "reason_code") if k in r}
    for k, allowed in _ALLOWED.items():
        if r.get(k) not in allowed:
            return None
    return r


def check_intent(i: Intent) -> Optional[str]:
    """Semantic checks beyond the schema. Returns a problem description, or None if usable."""
    if i.action == "follow" and not i.subject_phrase.strip():
        return "follow without a subject"
    if i.action == "follow" and len(i.subject_phrase.split()) > 6:
        return "subject phrase too long for segmentation"
    return None


def parse_model_output(text: str) -> tuple[Optional[Intent], bool]:
    """(intent, repaired). Strict parse first (mhd.intent_spec.parse_intent), then each balanced JSON
    object, then a lenient repair. Reasoning blocks (<think>...</think>) are ignored."""
    i = parse_intent(text)
    if i is not None and check_intent(i) is None:
        return i, False
    body = _THINK_RE.sub("", text or "")
    for cand in _json_objects(body):
        try:
            d = json.loads(cand)
        except json.JSONDecodeError:
            continue
        try:
            i = Intent.model_validate(d)
            if check_intent(i) is None:
                return i, False
        except ValidationError:
            pass
        r = _repair(d)
        if r is not None:
            try:
                i = Intent.model_validate(r)
                if check_intent(i) is None:
                    return i, True
            except ValidationError:
                pass
    return None, False


def _complete_json_prefix(text: str) -> bool:
    """True once the text contains at least one balanced JSON object (cheap check while streaming)."""
    return "}" in text and bool(_json_objects(_THINK_RE.sub("", text)))


# ---------------------------------------------------------------------------------------------------
# Keyword grammar
# ---------------------------------------------------------------------------------------------------
COLORS = {"red", "orange", "yellow", "green", "blue", "purple", "violet", "pink", "brown", "black",
          "white", "gray", "grey", "silver", "gold", "golden", "beige", "tan", "navy", "maroon", "teal",
          "turquoise", "cyan"}
COLOR_CANON = {"grey": "gray", "golden": "gold", "violet": "purple"}
PERSON = {"person", "man", "woman", "guy", "girl", "boy", "kid", "child", "lady", "gentleman", "runner",
          "jogger", "cyclist", "biker", "skater", "skateboarder", "dancer", "player", "walker", "hiker",
          "rider", "someone", "somebody", "dude", "teenager", "toddler", "baby", "adult", "student"}
PERSON_CANON = {"guy": "man", "dude": "man", "gentleman": "man", "lady": "woman", "someone": "person",
                "somebody": "person"}
CARS = {"car", "truck", "van", "suv", "jeep", "taxi", "cab", "bus", "pickup", "sedan", "vehicle",
        "motorcycle", "motorbike", "scooter", "tractor"}
ANIMALS = {"dog", "puppy", "cat", "kitten", "horse", "bird", "duck", "goose", "cow", "sheep", "goat",
           "deer", "rabbit", "bunny", "animal", "pet", "chicken", "pony"}
OBJECTS = {"ball", "balloon", "kite", "bike", "bicycle", "skateboard", "umbrella", "backpack", "bag",
           "box", "frisbee", "toy", "boat", "stroller", "cart", "suitcase", "bottle", "drone", "robot",
           "wagon", "sled"}
GARMENTS = {"shirt", "tshirt", "jacket", "hoodie", "sweater", "sweatshirt", "coat", "dress", "hat",
            "cap", "pants", "jeans", "shorts", "skirt", "top", "jersey", "vest", "scarf", "shoes",
            "sneakers", "helmet", "uniform", "suit", "beanie", "raincoat", "blouse", "costume", "gloves",
            "glasses", "sunglasses", "boots", "hood"}
GARMENT_CANON = {"tshirt": "shirt"}
FOLLOW_VERBS = {"follow", "track", "film", "chase", "watch", "record", "shoot", "find", "tail",
                "accompany"}
FOLLOW_PHRASES = ("keep up with", "go after", "stay with", "stick with", "lock on", "lock onto",
                  "keep an eye on", "look for", "focus on", "keep the camera on", "go with", "stay on",
                  "stay close to", "stay near", "keep close to", "point at", "aim at")
PRONOUNS = {"him", "her", "them", "it", "that", "this", "me", "us", "those", "these", "one", "there"}
COMMON_NAMES = {"john", "mike", "michael", "sarah", "david", "james", "robert", "mary", "jennifer",
                "linda", "chris", "emma", "olivia", "liam", "noah", "sophia", "ava", "mia", "lucas",
                "ethan", "jake", "alex", "sam", "tom", "anna", "kate", "bob", "joe", "dan", "daniel",
                "maria", "jose", "luis", "carlos", "juan", "ana", "rachel", "steve", "tony", "lisa",
                "amy", "matt", "ben", "nick", "ryan", "kevin", "brian", "jason", "laura", "emily",
                "jessica", "ashley", "mark", "paul", "peter", "susan", "karen", "nancy", "betty", "mom",
                "dad", "grandma", "grandpa", "russell", "russ", "elon", "zuck", "mark's"}
IDENTITY_PHRASES = ("who is", "who's", "identify", "recognize", "recognise", "facial", "face recognition",
                    "his name", "her name", "their name", "name of", "find out who", "track down",
                    "look up", "what is his name", "what's his name", "id that", "id this")
UNSAFE_WORDS = {"stalk", "stalking", "spy", "spying", "surveil", "surveillance", "attack", "hit", "ram",
                "crash", "dive", "scare", "harass", "peep", "peek", "hunt", "weapon", "bomb", "drop"}
UNSAFE_PHRASES = ("follow her home", "follow him home", "follow them home", "through the window",
                  "into the window", "fly at", "fly into", "crash into", "without them knowing",
                  "without her knowing", "without him knowing", "secretly")
UNSUPPORTED_PHRASES = ("take off", "takeoff", "lift off", "liftoff", "flip", "backflip", "frontflip",
                       "somersault", "barrel roll", "spin",
                       "do a trick", "fly higher", "go higher", "go up", "go down", "fly lower",
                       "come back", "return home", "go home", "take a picture", "take a photo",
                       "take a selfie", "turn left", "turn right", "fly to", "go to", "fly over",
                       "speed up", "faster", "slower", "orbit", "circle around", "loop")
CLOSE_WORDS = ("close", "closer", "closely", "near", "nearby", "tight", "tightly")
FAR_WORDS = ("far", "farther", "further", "distance", "back off", "wide", "far away", "from afar")
MULTI_WORDS = {"two", "three", "both", "all", "everyone", "everybody", "group", "people", "crowd"}


IRREGULAR_PLURALS = {"men": "man", "women": "woman", "children": "child", "people": "person",
                     "kids": "kid", "geese": "goose", "sheep": "sheep"}


def _canon_noun(w: str) -> Optional[tuple[str, str, bool]]:
    """(canonical noun, category, is_plural) for a subject noun."""
    plural = False
    if w in IRREGULAR_PLURALS:
        w, plural = IRREGULAR_PLURALS[w], w != "sheep"
    cands = [(w, plural)]
    if not plural and w.endswith("es") and len(w) > 4:
        cands.append((w[:-2], True))
    if not plural and w.endswith("s") and len(w) > 3:
        cands.append((w[:-1], True))
    for cand, pl in cands:
        if cand in PERSON:
            return PERSON_CANON.get(cand, cand), "person", pl
        if cand in CARS:
            return cand, "car", pl
        if cand in ANIMALS:
            return cand, "animal", pl
        if cand in OBJECTS:
            return cand, "object", pl
    return None


def _is_follow_verb(w: str) -> bool:
    for v in FOLLOW_VERBS:
        if w == v or (w.startswith(v) and len(w) - len(v) <= 3):   # follows, followed, following
            return True
    # ASR misspellings of "follow" (folow, fallow, follo); not "fellow"
    return w in {"folow", "fallow", "follo", "foll", "phollow", "fallo", "folo", "followe"}


def _intent(action, phrase="", category="none", color="none", distance="medium", reason="none") -> Intent:
    return Intent(action=action, subject_phrase=phrase, category=category, color=color,
                  distance=distance, reason_code=reason)


def _has_phrase(text: str, phrases) -> bool:
    padded = f" {text} "
    return any(f" {p} " in padded for p in phrases)


def keyword_intent(transcript: str) -> Intent:
    """Deterministic transcript -> Intent. Handles safety words, refusals, and
    "follow the <color> <thing>" / "follow the <thing> in the <color> <garment>"."""
    text = normalize(transcript).replace("t shirt", "tshirt")
    ws = words(text)
    if not ws:
        return _intent("clarify", reason="ambiguous")

    # 1. Safety words win (exact words only, no fuzzy guesses). Land beats hover.
    acts = [safety_action_for_word(w, fuzzy=False) for w in ws]
    acts = [a[0] for a in acts if a]
    if "land" in acts:
        return _intent("land")
    if "hover" in acts:
        return _intent("hover")

    # 2. Refusals: surveillance and harm, then identity, then things this drone does not do by voice.
    if any(w in UNSAFE_WORDS for w in ws) or _has_phrase(text, UNSAFE_PHRASES):
        return _intent("refuse", reason="unsafe")
    if _has_phrase(text, IDENTITY_PHRASES):
        return _intent("refuse", reason="identity_request")
    if _has_phrase(text, UNSUPPORTED_PHRASES):
        return _intent("refuse", reason="unsupported")

    distance = "medium"
    if _has_phrase(text, FAR_WORDS):
        distance = "far"
    elif _has_phrase(text, CLOSE_WORDS):
        distance = "close"

    # 3. Find the subject: the first subject noun, an adjacent color, and "in/with/wearing <color> <garment>".
    verb_idx = next((i for i, w in enumerate(ws) if _is_follow_verb(w)), None)
    has_verb = verb_idx is not None or _has_phrase(text, FOLLOW_PHRASES)
    start = (verb_idx + 1) if verb_idx is not None else 0
    noun_idx, noun, category = None, None, None
    plural = False
    for i in range(start, len(ws)):
        r = _canon_noun(ws[i])
        if r:
            noun_idx, (noun, category, plural) = i, r
            break
    garment_idx = next((i for i in range(start, len(ws)) if ws[i] in GARMENTS), None)

    def color_before(i: Optional[int], lo: int) -> Optional[str]:
        """A color word within two words before position i (e.g. "the red car", "red sports car")."""
        if i is None:
            return None
        for j in range(i - 1, max(lo, i - 3) - 1, -1):
            if ws[j] in COLORS:
                return COLOR_CANON.get(ws[j], ws[j])
        return None

    if noun is None and garment_idx is not None:
        noun, category, noun_idx = "person", "person", None    # "follow the red jacket" -> a person

    if noun is not None:
        # Several subjects ("the two kids", "the dogs"): the drone follows one, so ask which.
        if plural or any(w in MULTI_WORDS for w in ws[start:(noun_idx if noun_idx is not None else len(ws))]):
            return _intent("clarify", reason="ambiguous")
        noun_color = color_before(noun_idx, start) if noun_idx is not None else None
        attr = None   # ("in"|"with", color, thing)
        if category == "person":
            # attribute after the noun: in/wearing/with + [color] + garment/object
            lo = (noun_idx + 1) if noun_idx is not None else start
            for i in range(lo, len(ws)):
                w = ws[i]
                if w in GARMENTS:
                    attr = ("in", color_before(i, lo), GARMENT_CANON.get(w, w))
                    break
                obj = _canon_noun(w)
                if obj and obj[1] in ("object", "animal") and not obj[2] and i > lo:
                    prep = "with"
                    attr = (prep, color_before(i, lo), obj[0])
                    break
            if attr is None:
                # "the person in blue" (color without garment)
                for i in range(lo, min(len(ws), lo + 4)):
                    if ws[i] in COLORS:
                        attr = ("in", COLOR_CANON.get(ws[i], ws[i]), None)
                        break
        if category == "person" and attr is not None:
            prep, acolor, thing = attr
            head = noun_color + " " + noun if (noun_color and thing) else noun
            parts = [head, prep] + ([acolor] if acolor else []) + ([thing] if thing else [])
            phrase = " ".join(parts)
            color = acolor or noun_color or "none"
        else:
            phrase = f"{noun_color} {noun}" if noun_color else noun
            color = noun_color or "none"
        phrase = " ".join(phrase.split()[:4])
        if has_verb or noun_color or attr or len(ws) <= 4:
            return _intent("follow", phrase, category, color, distance)

    # 4. No describable subject.
    after = ws[start:] if has_verb else ws
    if has_verb and after and (after[0] in COMMON_NAMES or (len(after) == 1 and after[0] not in PRONOUNS
                                                             and after[0] not in COLORS)):
        return _intent("refuse", reason="identity_request")
    if any(w in COMMON_NAMES for w in after[:2]) and has_verb:
        return _intent("refuse", reason="identity_request")
    return _intent("clarify", reason="ambiguous")


def mentions_subject(transcript: str) -> bool:
    """True if the words go beyond a bare safety command: a follow verb, a color, a garment, or a
    subject noun other than "drone"."""
    text = normalize(transcript)
    ws = words(text)
    if any(_is_follow_verb(w) for w in ws) or _has_phrase(text, FOLLOW_PHRASES):
        return True
    for w in ws:
        if w in COLORS or w in GARMENTS:
            return True
        r = _canon_noun(w)
        if r and r[0] != "drone":
            return True
    return False


class KeywordIntent:
    name = "keyword"

    async def infer(self, transcript: str) -> IntentAttempt:
        return self.infer_sync(transcript)

    def infer_sync(self, transcript: str) -> IntentAttempt:
        t0 = time.monotonic()
        i = keyword_intent(transcript)
        return IntentAttempt(model=self.name, intent=i, latency_ms=(time.monotonic() - t0) * 1000.0,
                             raw=i.model_dump_json())


# ---------------------------------------------------------------------------------------------------
# OpenAI-compatible chat clients
# ---------------------------------------------------------------------------------------------------
@dataclass
class ChatConfig:
    base_url: str
    model: str
    label: str                                # name recorded in Mission.model
    api_key_env: Optional[str] = None         # env var holding the key; None for local servers
    system_prompt: str = SYSTEM_PROMPT
    response_format: str = "json_schema"      # json_schema | json_object | none
    strict: bool = True
    stream: bool = True
    max_tokens: int = 1024
    max_tokens_param: str = "max_tokens"      # or "max_completion_tokens"
    temperature: Optional[float] = 0.0
    timeout_s: float = 8.0
    extra: dict = field(default_factory=dict) # e.g. {"reasoning_effort": "minimal"}

    @classmethod
    def from_dict(cls, d: dict) -> "ChatConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class ChatIntentClient:
    """Streams /chat/completions and stops reading as soon as a valid Intent JSON has arrived."""

    def __init__(self, cfg: ChatConfig, http_client=None):
        self.cfg = cfg
        self.name = cfg.label
        self._client = http_client          # injected (tests); never closed here
        self._own = None                    # one long-lived client: no TLS handshake per command
        self._own_loop = None

    def _http(self):
        """The client to use: the injected one, else a long-lived one bound to the running loop."""
        import httpx

        if self._client is not None:
            return self._client
        loop = asyncio.get_running_loop()
        if self._own is None or self._own_loop is not loop or self._own.is_closed:
            self._own = httpx.AsyncClient(timeout=httpx.Timeout(self.cfg.timeout_s, connect=3.0),
                                          limits=httpx.Limits(keepalive_expiry=120.0))
            self._own_loop = loop
        return self._own

    async def warm(self) -> Optional[float]:
        """Open the connection ahead of the first command (GET /models, free). Returns ms or None."""
        t0 = time.monotonic()
        try:
            h = self._headers()
            h.pop("Accept", None)
            r = await self._http().get(self.cfg.base_url.rstrip("/") + "/models", headers=h)
            return _ms(t0) if r.status_code < 500 else None
        except Exception:
            return None

    async def aclose(self) -> None:
        if self._own is not None and not self._own.is_closed:
            try:
                await self._own.aclose()
            except Exception:
                pass
        self._own = None

    def _body(self, transcript: str) -> dict:
        c = self.cfg
        body = {
            "model": c.model,
            "messages": [{"role": "system", "content": c.system_prompt},
                         {"role": "user", "content": transcript}],
            "stream": c.stream,
            c.max_tokens_param: c.max_tokens,
        }
        if c.temperature is not None:
            body["temperature"] = c.temperature
        if c.response_format == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "intent", "schema": intent_json_schema(), "strict": c.strict}}
        elif c.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        if c.stream:
            body["stream_options"] = {"include_usage": True}
        body.update(c.extra)
        return body

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key_env:
            key = os.environ.get(self.cfg.api_key_env, "")
            if not key:
                raise RuntimeError(f"{self.cfg.api_key_env} is not set")
            h["Authorization"] = f"Bearer {key}"
        if self.cfg.stream:
            h["Accept"] = "text/event-stream"
        return h

    async def infer(self, transcript: str, drain: bool = False) -> IntentAttempt:
        """drain=True (benchmarks only): keep reading after the JSON is complete so the final usage
        chunk arrives; latency_ms still marks the moment the JSON was complete."""
        import httpx

        t0 = time.monotonic()
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        client = self._http()
        content, ttft, usage = "", None, {}
        try:
            headers = self._headers()
            body = self._body(transcript)
            if not self.cfg.stream:
                r = await asyncio.wait_for(client.post(url, json=body, headers=headers), self.cfg.timeout_s)
                if r.status_code != 200:
                    return IntentAttempt(self.name, None, _ms(t0), r.text[:300], f"HTTP {r.status_code}")
                j = r.json()
                msg = (j.get("choices") or [{}])[0].get("message") or {}
                content = msg.get("content") or ""
                usage = j.get("usage") or {}
                intent, repaired = parse_model_output(content)
                return IntentAttempt(self.name, intent, _ms(t0), content,
                                     None if intent else "invalid intent JSON", _ms(t0), repaired, usage)

            async def run_stream():
                nonlocal content, ttft, usage, done_at
                async with client.stream("POST", url, json=body, headers=headers) as r:
                    if r.status_code != 200:
                        err = (await r.aread()).decode("utf-8", "replace")[:300]
                        return None, False, f"HTTP {r.status_code}: {err}"
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            j = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if j.get("usage"):
                            usage = j["usage"]
                        for ch in j.get("choices") or []:
                            piece = (ch.get("delta") or {}).get("content") or ""
                            if piece:
                                if ttft is None:
                                    ttft = _ms(t0)
                                content += piece
                        if done_at is None and content and _complete_json_prefix(content):
                            intent, repaired = parse_model_output(content)
                            if intent is not None:
                                done_at = _ms(t0)
                                if not drain:
                                    return intent, repaired, None
                intent, repaired = parse_model_output(content)
                return intent, repaired, None if intent else "invalid intent JSON"

            done_at = None
            intent, repaired, err = await asyncio.wait_for(run_stream(), timeout=self.cfg.timeout_s)
            return IntentAttempt(self.name, intent, done_at or _ms(t0), content, err, ttft, repaired, usage)
        except asyncio.TimeoutError:
            return IntentAttempt(self.name, None, _ms(t0), content, "timeout", ttft, False, usage)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return IntentAttempt(self.name, None, _ms(t0), content, f"{e.__class__.__name__}: {e}"[:300],
                                 ttft, False, usage)


def _ms(t0: float) -> float:
    return (time.monotonic() - t0) * 1000.0


def glimmer_client(base_url: str, model: str, label: str = "glimmer-30b+lora",
                   api_key_env: Optional[str] = "FIREWORKS_API_KEY", response_format: str = "json_schema",
                   timeout_s: float = 8.0, max_tokens: int = 256, **extra) -> ChatIntentClient:
    """Muse Glimmer behind any OpenAI-compatible server. Fireworks: base_url
    https://api.fireworks.ai/inference/v1 and model accounts/<acct>/models/<name> or a deployment.
    Local: http://127.0.0.1:8080/v1 (mlx_lm.server or llama-server), api_key_env None."""
    return ChatIntentClient(ChatConfig(
        base_url=base_url, model=model, label=label, api_key_env=api_key_env,
        system_prompt=GLIMMER_SYSTEM_PROMPT, response_format=response_format, max_tokens=max_tokens,
        timeout_s=timeout_s, extra=extra))


def spark_client(reasoning_effort: str = "minimal", base_url: str = "https://api.meta.ai/v1",
                 model: str = "muse-spark-1.3", timeout_s: float = 8.0, stream: bool = True,
                 max_tokens: int = 1024) -> ChatIntentClient:
    """muse-spark-1.3 on the Meta Model API: strict json_schema, reasoning_effort minimal by default.
    Reasoning tokens count against the token limit, so the limit is generous (the JSON itself is ~50)."""
    return ChatIntentClient(ChatConfig(
        base_url=base_url, model=model, label=model, api_key_env="MODEL_API_KEY",
        system_prompt=SYSTEM_PROMPT, response_format="json_schema", strict=True, stream=stream,
        max_tokens=max_tokens, max_tokens_param="max_completion_tokens", temperature=None,
        timeout_s=timeout_s, extra={"reasoning_effort": reasoning_effort}))


# ---------------------------------------------------------------------------------------------------
# Hedged router
# ---------------------------------------------------------------------------------------------------
class HedgedRouter:
    """Race Glimmer and Spark; prefer Glimmer inside the window; fall back to the keyword grammar.

    mode: auto (both), glimmer (Glimmer, keyword fallback), spark (Spark, keyword fallback), keyword.
    """

    def __init__(self, glimmer: Optional[ChatIntentClient] = None, spark: Optional[ChatIntentClient] = None,
                 mode: str = "auto", prefer_window_s: float = 2.5, total_timeout_s: float = 5.0,
                 safety_fast_path: bool = True):
        self.glimmer = glimmer if mode in ("auto", "glimmer") else None
        self.spark = spark if mode in ("auto", "spark") else None
        self.keyword = KeywordIntent()
        self.mode = mode
        self.prefer_window_s = prefer_window_s
        self.total_timeout_s = total_timeout_s
        self.safety_fast_path = safety_fast_path
        self.history: list[dict] = []          # late finishers are recorded here too
        self._background: set[asyncio.Task] = set()

    async def route(self, transcript: str) -> RouteResult:
        t0 = time.monotonic()
        kw = self.keyword.infer_sync(transcript)
        # Safety fast path: a plain "stop" / "land now" never waits for a language model. A sentence
        # that also names a subject ("follow the man at the bus stop") goes to the models; the local
        # listener has already paused on its "stop" in either case.
        if self.safety_fast_path and kw.intent.action in ("land", "hover") and not mentions_subject(transcript):
            return self._done(kw, t0, "safety_fast_path", {"keyword": kw.summary()})
        clients = [c for c in (self.glimmer, self.spark) if c is not None]
        if not clients:
            return self._done(kw, t0, "keyword_only", {"keyword": kw.summary()})

        tasks: dict[asyncio.Task, ChatIntentClient] = {
            asyncio.create_task(c.infer(transcript), name=f"intent-{c.name}"): c for c in clients}
        results: dict[str, IntentAttempt] = {}
        order: list[str] = []                   # completion order
        pending = set(tasks)
        pref_deadline = t0 + self.prefer_window_s
        hard_deadline = t0 + self.total_timeout_s
        g = self.glimmer.name if self.glimmer else None
        chosen: Optional[IntentAttempt] = None
        reason = ""
        while True:
            now = time.monotonic()
            gr = results.get(g) if g else None
            if gr is not None and gr.valid:
                chosen = gr
                reason = "glimmer_in_window" if gr.latency_ms <= self.prefer_window_s * 1000 else "first_valid"
                break
            glimmer_out = g is None or (gr is not None and not gr.valid) or now >= pref_deadline
            if glimmer_out:
                valid = [results[m] for m in order if results[m].valid]
                if valid:
                    chosen, reason = valid[0], "first_valid"
                    break
            if not pending or now >= hard_deadline:
                break
            wait_until = hard_deadline if glimmer_out else min(pref_deadline, hard_deadline)
            done, pending = await asyncio.wait(pending, timeout=max(0.0, wait_until - now),
                                               return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                c = tasks[t]
                try:
                    results[c.name] = t.result()
                except Exception as e:
                    results[c.name] = IntentAttempt(c.name, None, _ms(t0), error=f"{e.__class__.__name__}")
                order.append(c.name)

        attempts = {m: a.summary() for m, a in results.items()}
        # Losers keep running in the background (bounded by their own timeouts) so their latency and
        # validity still get logged for the evaluation; they never change the decision.
        for t in pending:
            self._background.add(t)
            t.add_done_callback(lambda t, c=tasks[t], t0=t0: self._late(t, c, transcript))
        if chosen is None:
            attempts["keyword"] = kw.summary()
            return self._done(kw, t0, "keyword_fallback", attempts)
        return self._done(chosen, t0, reason, attempts)

    def _done(self, a: IntentAttempt, t0: float, reason: str, attempts: dict) -> RouteResult:
        res = RouteResult(intent=a.intent, model=a.model, latency_ms=_ms(t0), reason=reason, attempts=attempts)
        self.history.append({"model": res.model, "reason": reason, "latency_ms": round(res.latency_ms, 1),
                             "attempts": attempts})
        log.info("intent %s by %s in %.0f ms (%s)", a.intent.action, a.model, res.latency_ms, reason)
        return res

    def _late(self, t: asyncio.Task, c: ChatIntentClient, transcript: str) -> None:
        self._background.discard(t)
        if t.cancelled():
            return
        try:
            a = t.result()
        except Exception:
            return
        self.history.append({"late": True, "model": c.name, "transcript": transcript, **a.summary()})

    async def warm(self) -> dict:
        """Pre-open the HTTPS connections to every model server (call at startup and every minute)."""
        clients = [c for c in (self.glimmer, self.spark) if c is not None]
        res = await asyncio.gather(*(c.warm() for c in clients), return_exceptions=True)
        return {c.name: (None if isinstance(r, Exception) else r) for c, r in zip(clients, res)}

    async def aclose(self) -> None:
        for t in list(self._background):
            t.cancel()
        self._background.clear()
        for c in (self.glimmer, self.spark):
            if c is not None:
                await c.aclose()
