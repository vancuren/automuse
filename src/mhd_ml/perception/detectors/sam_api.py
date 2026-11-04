"""SAM 3.1 through the Meta Model API: POST https://api.meta.ai/v1/responses, model "sam-3.1".

Request (OpenAI Responses shape): one user message with one input_text noun phrase and one
input_image base64 JPEG data URL; metadata include_confidence="true", mask_encoding="one_bit".
Response: special-token text in the output_text part, one line per frame, for example
  <0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334;c=0.93|><|mask;x=0;y=0;data=27,60,!...|>,1<...>
parsed by meta_sam_parser (boxes in source pixels, half-open; masks box-local base85 rasters).
A concept that is absent gives an empty output_text.

Rules this client enforces:
- a 2.5 s total deadline per call (connect + upload + response), fail fast, return no candidates;
- no retries at all (the next detector tick is the retry); a phrase that got HTTP 400 is never
  sent again by this process; 401/403 disables the client; 429 backs off for 5 s;
- results cached by (frame hash, phrase), so the same frame is never paid for twice;
- a workstream spend cap (default $8) via SpendMeter; $0.0025 per image.
The API key comes from MODEL_API_KEY (loaded by env.sh) and is never printed or logged.
"""
from __future__ import annotations

import base64
import concurrent.futures
import re
from dataclasses import dataclass
import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from mhd.contracts import FRAME_H, FRAME_W

from ..spend import SpendMeter
from ..types import WORK_H, WORK_W, Detection, clip_box, mask_to_polygon

log = logging.getLogger("percep.sam_api")

API_BASE = "https://api.meta.ai/v1"
MODEL = "sam-3.1"
USD_PER_IMAGE = 0.0025


class ApiDisabled(RuntimeError):
    pass


def extract_output_text(body: dict) -> str:
    """Concatenate every output_text part of a Responses API body."""
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    parts = []
    for item in body.get("output") or []:
        for c in item.get("content") or []:
            if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                parts.append(c["text"])
    return "".join(parts)


# ----- parsing -------------------------------------------------------------------------------
# The record grammar, as documented by meta_sam_parser: a line is a frame header <Nf> followed by
# records; a record is a decimal object id followed by exactly one <|box;...|> and one
# <|mask;...|> token, comma-separated. Box fields are key=value (x1, y1, x2, y2 inclusive, w, h of
# the source image, optional c); mask fields are x=0, y=0, data=H,W,<!|~>base85 payload, optional c.
# The payload alphabet never contains | or ; so [^|]* ends exactly at a token's |>.
_LINE_RE = re.compile(r"^<([0-9]+)f(?:;[^>]*)?>(.*)$")
_RECORD_RE = re.compile(r",?([0-9]+)((?:<\|[A-Za-z][A-Za-z0-9_.-]*(?:;[^|]*)?\|>)+)")
_TOKEN_RE = re.compile(r"<\|([A-Za-z][A-Za-z0-9_.-]*)(?:;([^|]*))?\|>")
_DATA_RE = re.compile(r"([0-9]+),([0-9]+),([!~][^|]+)")


@dataclass
class SamRecord:
    """One parsed object: half-open box in source pixels plus its (still encoded) mask."""
    object_id: str
    left: int
    top: int
    right: int
    bottom: int
    src_w: int
    src_h: int
    confidence: Optional[float]
    mask_w: int
    mask_h: int
    payload: str


def _fields(body: Optional[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (body or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _confidence(fields: dict[str, str]) -> Optional[float]:
    try:
        c = float(fields["c"])
    except (KeyError, ValueError):
        return None
    return c if 0.0 <= c <= 1.0 else None


def parse_records_fast(text: str) -> list[SamRecord]:
    """Fast reader for image output (frame 0). Malformed records are skipped with a warning."""
    out: list[SamRecord] = []
    for line in (text or "").splitlines():
        m = _LINE_RE.match(line.strip())
        if not m or m.group(1) != "0":
            continue
        for rec in _RECORD_RE.finditer(m.group(2)):
            toks = {t.group(1): _fields(t.group(2)) for t in _TOKEN_RE.finditer(rec.group(2))}
            box, mask = toks.get("box"), toks.get("mask")
            try:
                x1, y1, x2, y2 = (int(box[k]) for k in ("x1", "y1", "x2", "y2"))
                w, h = int(box["w"]), int(box["h"])
                data = _DATA_RE.fullmatch(mask["data"])
                if data is None or mask.get("x") != "0" or mask.get("y") != "0":
                    raise ValueError("bad mask token")
                if not (0 <= x1 <= x2 < w and 0 <= y1 <= y2 < h):
                    raise ValueError("box outside the image")
            except (TypeError, KeyError, ValueError) as e:
                log.warning("skipping malformed SAM record %s: %s", rec.group(1), e)
                continue
            conf = _confidence(box)
            if conf is None:
                conf = _confidence(mask)
            out.append(SamRecord(rec.group(1), x1, y1, x2 + 1, y2 + 1, w, h, conf,
                                 int(data.group(2)), int(data.group(1)), data.group(3)))
    return out


def parse_records_official(text: str) -> list[SamRecord]:
    """The same records through meta_sam_parser (validates and decodes every mask: slow)."""
    from meta_sam_parser import CompletedOutcome, image_segmentation_format

    parser = image_segmentation_format().create_parser()
    parser.push(text or "", emit=False)
    result = parser.finish(CompletedOutcome()).result
    for d in result.diagnostics:
        log.warning("sam parser %s: %s", d.code, d.message)
    boxes = {r.object_id: r for r in result.records if r.kind == "box"}
    masks = {r.object_id: r for r in result.records if r.kind == "mask"}
    out = []
    for oid, b in boxes.items():
        m = masks.get(oid)
        if m is None:
            continue
        conf = b.confidence if b.confidence is not None else m.confidence
        # The parser does not keep w/h; the caller passes the size it sent.
        out.append(SamRecord(oid, int(b.left), int(b.top), int(b.right), int(b.bottom), 0, 0, conf,
                             m.mask.width, m.mask.height, m.mask.payload))
    return out


def decode_mask(rec: SamRecord) -> np.ndarray:
    """Decode a record's box-local mask to a (mask_h, mask_w) uint8 0/1 array.

    Uses meta_sam_parser's raw decoder (one pass, 0.8 us per pixel in pure Python) when available,
    else the public decode_mask_to_raster (which also re-encodes to check, about twice as slow)."""
    try:
        from meta_sam_parser._mask_codec import _decode_raster  # pinned 0.1.0; private but stable here
        if rec.payload.startswith("!"):
            raster = _decode_raster(rec.payload, rec.mask_w, rec.mask_h)
        else:
            raise ImportError("lossless payload: use the public decoder")
    except ImportError:
        from meta_sam_parser import SegmentationMask, decode_mask_to_raster
        raster = decode_mask_to_raster(SegmentationMask(
            encoding="one_bit" if rec.payload.startswith("!") else "lossless",
            payload=rec.payload, width=rec.mask_w, height=rec.mask_h))
    return np.frombuffer(raster, dtype=np.uint8).reshape(rec.mask_h, rec.mask_w)


def parse_sam_text(text: str, src_w: int, src_h: int, phrase: str = "", max_masks: int = 3,
                   official: bool = False) -> list[Detection]:
    """Parse SAM special-token text for one image sent at src_w x src_h.

    Returns Detections in 1280x720 pixels, best first. Only the best `max_masks` masks are decoded
    (decoding is pure Python and costs about 20-50 ms per person-sized mask at 640x360); the rest
    get box-only Detections (mask None, empty polygon)."""
    import cv2

    recs = parse_records_official(text) if official else parse_records_fast(text)
    # The API only returns matches. A missing c means "not reported", not zero: use 0.9.
    recs.sort(key=lambda r: -(r.confidence if r.confidence is not None else 0.9))
    sx, sy = FRAME_W / src_w, FRAME_H / src_h          # source -> 1280x720
    wx, wy = WORK_W / src_w, WORK_H / src_h            # source -> 640x360
    out: list[Detection] = []
    for i, r in enumerate(recs):
        work_mask = None
        if i < max_masks:
            try:
                local = decode_mask(r)
                x0, y0 = int(round(r.left * wx)), int(round(r.top * wy))
                x1 = min(max(int(round(r.right * wx)), x0 + 1), WORK_W)
                y1 = min(max(int(round(r.bottom * wy)), y0 + 1), WORK_H)
                if x1 > x0 and y1 > y0:
                    work_mask = np.zeros((WORK_H, WORK_W), np.uint8)
                    work_mask[y0:y1, x0:x1] = cv2.resize(local, (x1 - x0, y1 - y0),
                                                         interpolation=cv2.INTER_NEAREST)
            except Exception as e:  # a bad mask must not lose the box
                log.warning("sam mask decode failed for object %s: %s", r.object_id, e)
                work_mask = None
        box = clip_box((r.left * sx, r.top * sy, r.right * sx, r.bottom * sy))
        poly = mask_to_polygon(work_mask, FRAME_W / WORK_W, FRAME_H / WORK_H) if work_mask is not None else []
        score = float(r.confidence) if r.confidence is not None else 0.9
        out.append(Detection(box=box, score=score, mask_poly=poly, mask=work_mask, label=phrase))
    return out


class SamApiDetector:
    """detect(image_bgr, phrase) -> list[Detection] through the hosted sam-3.1 model."""

    name = "sam_api"

    def __init__(self, send_w: int = 640, send_h: int = 360, timeout_s: float = 2.5,
                 cap_usd: float = 8.0, jpeg_quality: int = 85, cache_size: int = 64,
                 api_base: Optional[str] = None, api_key: Optional[str] = None,
                 meter: Optional[SpendMeter] = None, transport=None):
        import httpx

        self.send_w, self.send_h = send_w, send_h
        self.timeout_s = timeout_s
        self.jpeg_quality = jpeg_quality
        self.api_base = (api_base or os.environ.get("MODEL_API_BASE") or API_BASE).rstrip("/")
        self._key = api_key if api_key is not None else os.environ.get("MODEL_API_KEY", "")
        self.meter = meter or SpendMeter("meta-api sam-3.1 image", USD_PER_IMAGE, cap_usd)
        self._cache: OrderedDict[str, list[Detection]] = OrderedDict()
        self._cache_size = cache_size
        self._bad_phrases: set[str] = set()
        self._disabled_reason: Optional[str] = None
        self._backoff_until = 0.0
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s), transport=transport)
        # Requests run on a tiny pool so the caller can stop waiting at the deadline even when the
        # socket is stuck (httpx timeouts are per phase, not total). Two workers bound the number
        # of abandoned requests still finishing in the background.
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sam_api")
        self._lock = threading.Lock()
        self._inflight = 0
        self.last_timings: dict = {}
        self.last_error: Optional[str] = None
        self.last_text: str = ""          # raw special-token output of the last call (debugging)

    # ----- helpers -------------------------------------------------------------------------
    def encode(self, image_bgr: np.ndarray) -> bytes:
        import cv2

        img = image_bgr
        if img.shape[1] != self.send_w or img.shape[0] != self.send_h:
            img = cv2.resize(img, (self.send_w, self.send_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes()

    def request_body(self, jpeg: bytes, phrase: str) -> dict:
        url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        return {
            "model": MODEL,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": phrase},
                {"type": "input_image", "image_url": url},
            ]}],
            "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
        }

    def _post_blocking(self, body: dict, deadline: float) -> tuple[int, bytes]:
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        try:
            with self._client.stream("POST", f"{self.api_base}/responses", json=body,
                                     headers=headers) as r:
                chunks = []
                for chunk in r.iter_bytes():
                    chunks.append(chunk)
                    if time.monotonic() > deadline:
                        raise TimeoutError("sam-3.1 response exceeded the total deadline")
                return r.status_code, b"".join(chunks)
        finally:
            with self._lock:
                self._inflight -= 1

    def _post(self, body: dict) -> tuple[int, bytes]:
        """POST with a hard total deadline of timeout_s seconds, measured from now."""
        deadline = time.monotonic() + self.timeout_s
        with self._lock:
            if self._inflight >= 2:
                raise TimeoutError("two earlier sam-3.1 calls are still pending")
            self._inflight += 1
        fut = self._pool.submit(self._post_blocking, body, deadline)
        try:
            return fut.result(timeout=max(0.0, deadline - time.monotonic()))
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"sam-3.1 call exceeded {self.timeout_s:.1f} s") from None

    # ----- the detector interface ----------------------------------------------------------
    def detect(self, image_bgr: np.ndarray, phrase: str) -> list[Detection]:
        import json

        self.last_timings, self.last_error = {}, None
        if self._disabled_reason:
            self.last_error = self._disabled_reason
            return []
        if not self._key:
            self._disabled_reason = "MODEL_API_KEY is not set"
            self.last_error = self._disabled_reason
            log.error("sam_api disabled: %s", self._disabled_reason)
            return []
        if phrase in self._bad_phrases:
            self.last_error = "phrase rejected earlier with HTTP 400"
            return []
        if time.monotonic() < self._backoff_until:
            self.last_error = "rate-limited, backing off"
            return []

        t0 = time.perf_counter()
        jpeg = self.encode(image_bgr)
        key = hashlib.sha1(jpeg + b"\x00" + phrase.encode("utf-8")).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            self.last_timings = {"cache_hit": True, "total_ms": (time.perf_counter() - t0) * 1e3}
            return list(self._cache[key])
        if not self.meter.can_spend(1):
            self.last_error = f"spend cap ${self.meter.cap_usd:.2f} reached"
            log.error("sam_api: %s", self.last_error)
            return []

        body = self.request_body(jpeg, phrase)
        t1 = time.perf_counter()
        try:
            self.meter.charge(1)          # counted once the request is on its way
            status, raw = self._post(body)
        except Exception as e:            # timeout, connection error: fail fast, no retry
            self.last_error = f"{type(e).__name__}: {e}"[:200]
            self.last_timings = {"rtt_ms": (time.perf_counter() - t1) * 1e3, "error": True}
            log.warning("sam_api call failed: %s", self.last_error)
            return []
        t2 = time.perf_counter()
        if status != 200:
            snippet = raw[:300].decode("utf-8", "replace")
            self.last_error = f"HTTP {status}: {snippet}"
            self.last_timings = {"rtt_ms": (t2 - t1) * 1e3, "status": status}
            if status == 400:
                self._bad_phrases.add(phrase)          # never retry a 400
            elif status in (401, 403):
                self._disabled_reason = f"HTTP {status} (check MODEL_API_KEY)"
            elif status == 429:
                self._backoff_until = time.monotonic() + 5.0
            log.warning("sam_api %s", self.last_error)
            return []
        try:
            text = extract_output_text(json.loads(raw))
            self.last_text = text
            dets = parse_sam_text(text, self.send_w, self.send_h, phrase)
        except Exception as e:
            self.last_error = f"parse error: {e}"[:200]
            log.warning("sam_api %s", self.last_error)
            return []
        t3 = time.perf_counter()
        self.last_timings = {"encode_ms": (t1 - t0) * 1e3, "rtt_ms": (t2 - t1) * 1e3,
                             "parse_ms": (t3 - t2) * 1e3, "total_ms": (t3 - t0) * 1e3,
                             "bytes_out": len(jpeg), "n": len(dets)}
        self._cache[key] = dets
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return list(dets)

    def close(self) -> None:
        self.meter.flush()
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._client.close()
