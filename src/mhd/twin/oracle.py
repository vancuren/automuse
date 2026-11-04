"""Oracle perception for the wire-level twin: SubjectObs from ground truth, with realistic flaws.

    $MHD_PY -m mhd.twin.oracle --twin http://127.0.0.1:8766 --backend ws://127.0.0.1:8000 \
        [--subject "person in blue shirt"] [--latency-ms 300] [--noise-px 4] [--dropout 0.05] [--log obs.jsonl]

It plays the perception service's role on the backend's /ws/subject (docs/interfaces.md): it
receives FindRequest/Release from the backend and sends SubjectObs (source "oracle") at the
camera rate. Each observation is computed from the twin's Truth for that frame, delivered after
--latency-ms (perception latency), with box noise, random dropouts, a find delay (searching,
then candidate, then confirmed), and a simulated 2 Hz detector for confirm_age_ms.
capture_ms is the frame's render_ms and frame_id is its render_id, so metrics.py can join.
Without --backend it only writes the observation log. Twin only; never use it on the real drone.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from typing import Optional

import websockets

from mhd.contracts import Box, FindRequest, ObsSource, SubjectObs, SubjectState, now_ms


def pick_actor(truth: dict, phrase: str) -> Optional[dict]:
    """The truth actor whose label matches the subject phrase (exact, then best word overlap)."""
    phrase_l = phrase.lower().strip()
    for a in truth["actors"]:
        if a["label"].lower() == phrase_l or a["actor_id"] == phrase:
            return a
    words = set(phrase_l.split())
    best, score = None, 0
    for a in truth["actors"]:
        s = len(words & set(a["label"].lower().split()))
        if s > score:
            best, score = a, s
    return best if score >= 2 else None


class Oracle:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
        self.find: Optional[FindRequest] = None
        if args.subject:
            self.find = FindRequest(mission_id="oracle", subject_phrase=args.subject)
        self.find_t0: Optional[float] = None
        self.last_detect_ms = -1e9
        self.last_seen_ms: Optional[float] = None
        self.confirmed = False
        self.track_id = 0
        self.out_ws = None
        self.log = open(args.log, "a", encoding="utf-8") if args.log else None
        self.sent = 0

    def on_backend_msg(self, text: str) -> None:
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            return
        if d.get("type") == "find":
            self.find = FindRequest.model_validate(d)
            self.find_t0 = None
            self.confirmed = False
        elif d.get("type") == "release":
            self.find = None

    def observe(self, truth: dict) -> Optional[SubjectObs]:
        if self.find is None:
            return None
        a = self.args
        t = truth["render_ms"]
        if self.find_t0 is None:
            self.find_t0 = t
            self.track_id += 1
        common = dict(mission_id=self.find.mission_id, frame_id=truth["render_id"], capture_ms=t,
                      render_id=truth["render_id"], source=ObsSource.oracle)
        actor = pick_actor(truth, self.find.subject_phrase)
        visible = actor is not None and actor.get("box") and actor.get("visible_frac", 0) >= a.min_visible
        dropped = self.rng.random() < a.dropout
        if visible and dropped:
            # A random dropout is a frame the tracker missed: real perception keeps its track and
            # simply sends nothing for it (it reports "lost" only after several failed frames).
            return None
        if not visible:
            lost = self.confirmed and self.last_seen_ms is not None and t - self.last_seen_ms > 1000
            state = SubjectState.lost if lost else SubjectState.searching
            if lost:
                self.confirmed = False
            return SubjectObs(state=state, confirm_age_ms=t - self.last_detect_ms, **common)
        self.last_seen_ms = t
        since = t - self.find_t0
        if since < a.find_delay_ms:
            return SubjectObs(state=SubjectState.searching, **common)
        if t - self.last_detect_ms >= 1000.0 / a.detector_hz:
            self.last_detect_ms = t
        b = actor["box"]
        n = a.noise_px
        box = Box(x1=max(0.0, b["x1"] + self.rng.gauss(0, n)), y1=max(0.0, b["y1"] + self.rng.gauss(0, n)),
                  x2=min(1280.0, b["x2"] + self.rng.gauss(0, n)), y2=min(720.0, b["y2"] + self.rng.gauss(0, n)))
        state = SubjectState.confirmed if since >= a.find_delay_ms + a.confirm_ms else SubjectState.candidate
        self.confirmed = state == SubjectState.confirmed
        return SubjectObs(state=state, track_id=self.track_id, box=box, confidence=0.9, tracker_score=0.95,
                          confirm_age_ms=t - self.last_detect_ms, **common)

    async def emit_later(self, obs: SubjectObs) -> None:
        await asyncio.sleep(self.args.latency_ms / 1000.0)
        line = obs.model_dump_json()
        if self.log:
            self.log.write(line + "\n")
        if self.out_ws is not None:
            try:
                await self.out_ws.send(line)
            except websockets.ConnectionClosed:
                self.out_ws = None
        self.sent += 1

    async def backend_loop(self) -> None:
        while True:
            try:
                async with websockets.connect(self.args.backend.rstrip("/") + "/ws/subject") as ws:
                    self.out_ws = ws
                    async for msg in ws:
                        if isinstance(msg, str):
                            self.on_backend_msg(msg)
            except (OSError, websockets.WebSocketException):
                self.out_ws = None
                await asyncio.sleep(1.0)

    async def truth_loop(self) -> None:
        url = self.args.twin.replace("http://", "ws://").rstrip("/") + "/ws/truth"
        while True:
            try:
                async with websockets.connect(url, max_size=None) as ws:
                    async for msg in ws:
                        obs = self.observe(json.loads(msg))
                        if obs is not None:
                            asyncio.create_task(self.emit_later(obs))
            except (OSError, websockets.WebSocketException):
                await asyncio.sleep(0.5)

    async def run(self) -> None:
        tasks = [asyncio.create_task(self.truth_loop())]
        if self.args.backend and self.args.backend != "none":
            tasks.append(asyncio.create_task(self.backend_loop()))
        if self.args.duration_s:
            await asyncio.sleep(self.args.duration_s)
            for t in tasks:
                t.cancel()
        else:
            await asyncio.gather(*tasks)
        if self.log:
            self.log.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Oracle perception from twin ground truth (twin only)")
    ap.add_argument("--twin", default="http://127.0.0.1:8766")
    ap.add_argument("--backend", default="none", help="ws://127.0.0.1:8000, or none")
    ap.add_argument("--subject", default=None, help="follow this label without waiting for a FindRequest")
    ap.add_argument("--latency-ms", type=float, default=300.0, help="perception latency after render")
    ap.add_argument("--noise-px", type=float, default=4.0)
    ap.add_argument("--dropout", type=float, default=0.03)
    ap.add_argument("--min-visible", type=float, default=0.4)
    ap.add_argument("--find-delay-ms", type=float, default=1200.0)
    ap.add_argument("--confirm-ms", type=float, default=800.0)
    ap.add_argument("--detector-hz", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None, help="append SubjectObs JSONL here")
    ap.add_argument("--duration-s", type=float, default=0.0)
    a = ap.parse_args(argv)
    for url in (a.twin, a.backend):
        if url and url != "none" and not any(h in url for h in ("127.0.0.1", "localhost")):
            raise SystemExit("the oracle only talks to loopback services")
    asyncio.run(Oracle(a).run())


if __name__ == "__main__":
    main()
