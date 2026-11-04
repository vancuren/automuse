"""The perception pipeline, synchronous and thread-safe, shared by service.py and replay.py.

    core = PerceptionCore(cfg, detector_name="sam_mlx", make_tracker=lambda: make_tracker("vit"))
    core.on_find(FindRequest(...))
    msgs = core.on_frame(meta, img_bgr)          # every camera frame -> [SubjectObs]
    job = core.detection_due(now_ms)             # a frame for the detector, when its turn comes
    msgs = core.on_detection(job, dets, latency) # detector result (later) -> maybe [Candidates]

What perception reports is evidence only (docs: PLAN 4.3); the autopilot owns state changes.
SubjectObs.state means:
- searching: a mission is active and no track exists.
- candidate: a track exists with fewer than `confirm_needed` detector confirmations, or
  look-alikes tie and perception waits for a Select (then box is None and Candidates was sent).
- confirmed: a track with at least `confirm_needed` consistent detector confirmations.
- lost: the track's tracker failed `lost_frames` frames in a row; box is None until a detection
  re-finds a subject (which always starts a new track_id, so re-finding goes through CONFIRM).
tracker_score is the in-loop tracker's own score; confirm_age_ms is this frame's capture time
minus the capture time of the frame the detector last confirmed the subject on.
"""
from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from mhd.contracts import (Box, Candidate, Candidates, FindRequest, FrameMeta, ObsSource, Release,
                           Select, SubjectObs, SubjectState)

from .fusion import (DEFAULT_HFOV_DEG, BoxKalman, Camera, FrameRing, associate, box_offset,
                     color_fraction, color_score, hist_intersection, hue_hist, mask_counter,
                     overlaps_counter, reseed, size_prior, work_image)
from .types import Detection, clip_box, iou

log = logging.getLogger("percep.pipeline")


@dataclass
class PerceptionConfig:
    min_score: float = 0.4            # detector confidence needed to be a candidate (PLAN 4.3)
    tie_ratio: float = 0.85           # 2nd-best rank >= this x best rank -> look-alikes tie
    find_period_ms: float = 500.0     # detector at 2 Hz while finding, candidate, lost or in doubt
    follow_period_ms: float = 1000.0  # 1 Hz while following a confirmed, healthy track
    doubt_score: float = 0.5          # tracker score below this -> detector back at 2 Hz
    lost_score: float = 0.15          # tracker score below this counts as a miss
    lost_frames: int = 3              # consecutive misses before the track is reported lost
    confirm_needed: int = 2           # consistent detector confirmations for "confirmed"
    reseed_max_offset: float = 0.3    # reject a re-seed whose replay lands > 0.3 box sizes away
    max_reject_streak: int = 2        # rejected re-seeds in a row -> restart from the detection
    det_miss_switch: int = 3          # detections in a row without our subject -> may switch
    candidates_resend_ms: float = 2000.0
    replay_max_steps: int = 30
    mask_counter: str = "auto"        # auto (twin frames only), on, off
    hfov_deg: float = DEFAULT_HFOV_DEG
    history_ms: float = 2500.0
    max_blind_start_ms: float = 300.0  # start without replay only from a detection this fresh
    min_box_px: float = 8.0           # a fused box smaller than this (at the frame edge) = lost
    min_color_frac: float = 0.08      # with a color word, candidates whose shirt region matches
                                      # less than this are dropped (SAM boxes the blue walker for
                                      # "person in red shirt" at 0.47-0.72 when red is off screen)


@dataclass
class MissionSpec:
    mission_id: str
    phrase: str
    category: str = "none"
    color: str = "none"


@dataclass
class DetJob:
    """One frame handed to the detector."""
    mission_id: str
    phrase: str
    frame_id: int
    capture_ms: float
    image: np.ndarray            # full-size BGR, counter block masked if needed
    submit_ms: float


@dataclass
class Cand:
    det: Detection
    rank: float
    color: float                 # color score factor (1.0 when no color word)
    hist: Optional[np.ndarray]


@dataclass
class Track:
    track_id: int
    tracker: object
    kf: BoxKalman
    n_confirms: int
    last_confirm_capture_ms: float
    confidence: float
    hist: Optional[np.ndarray]
    poly_rel: list = field(default_factory=list)
    history: deque = field(default_factory=lambda: deque(maxlen=200))  # (frame_id, capture_ms, box)
    last_score: float = 1.0
    misses: int = 0
    lost: bool = False
    reject_streak: int = 0
    det_misses: int = 0
    pending_source: Optional[str] = None
    pending_latency_ms: Optional[float] = None

    def box_at(self, frame_id: int, capture_ms: float):
        """The track's box on a given frame (or the nearest earlier one in its history)."""
        best = None
        for fid, t, b in self.history:
            if fid == frame_id:
                return b
            if t <= capture_ms:
                best = b
        if best is None and self.history:
            best = self.history[0][2]
        return best


@dataclass
class PendingTie:
    frame_id: int
    capture_ms: float
    items: list                  # list[Cand], numbered 1..N left to right
    sent_ms: float = -1e18


def _poly_rel(det: Detection) -> list:
    x1, y1, x2, y2 = det.box
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    return [((x - x1) / w, (y - y1) / h) for x, y in det.mask_poly]


class PerceptionCore:
    def __init__(self, cfg: PerceptionConfig, detector_name: str, make_tracker: Callable[[], object]):
        self.cfg = cfg
        self.detector_name = detector_name
        self.make_tracker = make_tracker
        self.cam = Camera(cfg.hfov_deg)
        self.lock = threading.RLock()
        self.mission: Optional[MissionSpec] = None
        self.track: Optional[Track] = None
        self.pending: Optional[PendingTie] = None
        self.selected: Optional[Cand] = None      # the user's pick, waiting for a fresh detection
        self.ring = FrameRing(span_ms=2000.0)
        self.det_inflight = False
        self.last_det_submit_ms = -1e18
        self._latest: Optional[tuple[FrameMeta, np.ndarray]] = None
        self._track_seq = 0
        self.stats = {"frames": 0, "detections": 0, "reseed_ok": 0, "reseed_rejected": 0,
                      "tracks_started": 0, "candidates_sent": 0}

    # ----- source names ---------------------------------------------------------------------
    def _det_source(self) -> ObsSource:
        try:
            return ObsSource(self.detector_name)
        except ValueError:
            return ObsSource.oracle      # test doubles (detectors/fake.py)

    def _mask_counter_for(self, meta: FrameMeta) -> bool:
        if self.cfg.mask_counter == "on":
            return True
        if self.cfg.mask_counter == "off":
            return False
        return meta.flight_mode == "twin"

    # ----- commands from the backend --------------------------------------------------------
    def on_find(self, msg: FindRequest) -> None:
        with self.lock:
            phrase = msg.subject_phrase.strip() or (msg.category if msg.category != "none" else "")
            if not phrase:
                log.warning("find %s has no subject phrase; ignored", msg.mission_id)
                return
            spec = MissionSpec(msg.mission_id, phrase, msg.category, msg.color)
            if self.mission == spec:
                return                            # a repeated FindRequest must not reset the track
            self.mission = spec
            self.track = None
            self.pending = None
            self.selected = None
            self.last_det_submit_ms = -1e18       # detect at once
            log.info("find %s: %r (category %s, color %s)", msg.mission_id, msg.subject_phrase,
                     msg.category, msg.color)

    def on_release(self, msg: Release) -> None:
        with self.lock:
            if self.mission is not None and msg.mission_id not in (self.mission.mission_id, "*", ""):
                log.info("release for %s ignored (active mission %s)", msg.mission_id, self.mission.mission_id)
                return
            log.info("release %s", msg.mission_id)
            self.mission = None
            self.track = None
            self.pending = None
            self.selected = None

    def on_select(self, msg: Select) -> list:
        """The user picked one of the numbered candidates."""
        with self.lock:
            if self.mission is None or msg.mission_id != self.mission.mission_id or self.pending is None:
                log.info("select %s ignored (no pending candidates)", msg.index)
                return []
            chosen = next((c for i, c in enumerate(self.pending.items, start=1) if i == msg.index), None)
            if chosen is None:
                log.info("select index %s out of range", msg.index)
                return []
            p = self.pending
            if self.ring.get(p.frame_id) is not None:
                self._start_track(chosen, p.frame_id, p.capture_ms, None, confirms=self.cfg.confirm_needed)
            else:
                # The frame the user looked at is gone from the buffer (they took > 2 s): the
                # subject has moved, so find the chosen one again in the next detection.
                self.selected = chosen
                self.pending = None
                self.last_det_submit_ms = -1e18
                log.info("select %d: re-identifying in the next detection", msg.index)
            return []

    # ----- camera frames --------------------------------------------------------------------
    def on_frame(self, meta: FrameMeta, img_bgr: np.ndarray) -> list:
        """Track on this frame and return the SubjectObs for it (nothing while idle)."""
        with self.lock:
            self.stats["frames"] += 1
            if self.mission is None:
                self._latest = (meta, img_bgr)
                return []
            newest = self.ring.latest()
            if newest is not None and meta.frame_id <= newest.frame_id:
                # Frame ids went backwards: the backend restarted. Old frames and the track's
                # history would be matched to the wrong images, so start over.
                log.warning("frame id %d after %d: backend restarted; resetting", meta.frame_id, newest.frame_id)
                self.ring = FrameRing(span_ms=self.ring.span_ms)
                if self.track is not None:
                    self.track.lost = True
            work = work_image(img_bgr, self._mask_counter_for(meta))
            self.ring.add(meta.frame_id, meta.capture_ms, work)
            self._latest = (meta, img_bgr)
            tr = self.track
            if tr is not None and not tr.lost:
                box, score = tr.tracker.update(work)
                tr.last_score = score if box is not None else 0.0
                if box is None or score < self.cfg.lost_score:
                    tr.misses += 1
                    if tr.misses >= self.cfg.lost_frames:
                        tr.lost = True
                        log.info("track %d lost (tracker score %.2f)", tr.track_id, tr.last_score)
                    else:
                        tr.kf.predict(meta.capture_ms)
                else:
                    tr.misses = 0
                    tr.kf.update(meta.capture_ms, clip_box(box), "tracker", score)
                fx1, fy1, fx2, fy2 = clip_box(tr.kf.box())
                if not tr.lost and (fx2 - fx1 < self.cfg.min_box_px or fy2 - fy1 < self.cfg.min_box_px):
                    tr.lost = True                # the subject left the frame
                    log.info("track %d left the frame", tr.track_id)
                tr.history.append((meta.frame_id, meta.capture_ms, tr.kf.box()))
            return [self._obs(meta)]

    def _obs(self, meta: FrameMeta) -> SubjectObs:
        m, tr = self.mission, self.track
        common = dict(mission_id=m.mission_id, frame_id=meta.frame_id, capture_ms=meta.capture_ms,
                      render_id=meta.render_id)
        if tr is None or tr.lost:
            state = (SubjectState.lost if tr is not None else
                     SubjectState.candidate if self.pending is not None else SubjectState.searching)
            return SubjectObs(state=state, track_id=tr.track_id if tr else None, box=None,
                              confidence=tr.confidence if tr else 0.0, tracker_score=0.0,
                              confirm_age_ms=(max(0.0, meta.capture_ms - tr.last_confirm_capture_ms)
                                              if tr else 1e9),
                              source=self._det_source() if tr is None else ObsSource.tracker, **common)
        x1, y1, x2, y2 = clip_box(tr.kf.box())
        w, h = x2 - x1, y2 - y1
        poly = [(round(x1 + u * w, 1), round(y1 + v * h, 1)) for u, v in tr.poly_rel] or None
        source = self._det_source() if tr.pending_source else ObsSource.tracker
        latency = tr.pending_latency_ms
        tr.pending_source, tr.pending_latency_ms = None, None
        state = SubjectState.confirmed if tr.n_confirms >= self.cfg.confirm_needed else SubjectState.candidate
        return SubjectObs(state=state, track_id=tr.track_id, box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                          confidence=tr.confidence, tracker_score=round(float(tr.last_score), 4),
                          confirm_age_ms=max(0.0, meta.capture_ms - tr.last_confirm_capture_ms),
                          mask_poly=poly, source=source, detector_latency_ms=latency, **common)

    # ----- detector scheduling --------------------------------------------------------------
    def detector_period_ms(self) -> float:
        tr = self.track
        # Following = a confirmed, healthy track whose last detection agreed. Any doubt (weak
        # tracker score, a rejected re-seed, a detection that missed the subject) -> 2 Hz.
        following = (tr is not None and not tr.lost and tr.n_confirms >= self.cfg.confirm_needed
                     and tr.last_score >= self.cfg.doubt_score
                     and tr.reject_streak == 0 and tr.det_misses == 0)
        return self.cfg.follow_period_ms if following else self.cfg.find_period_ms

    def detection_due(self, now_ms: float) -> Optional[DetJob]:
        """A detector job for the newest frame if it is time and none is in flight."""
        with self.lock:
            if self.mission is None or self.det_inflight or self._latest is None:
                return None
            if now_ms - self.last_det_submit_ms < self.detector_period_ms():
                return None
            meta, img = self._latest
            if self.ring.get(meta.frame_id) is None:
                return None                      # not processed yet under this mission
            image = img.copy()
            if self._mask_counter_for(meta):
                mask_counter(image)
            self.det_inflight = True
            self.last_det_submit_ms = now_ms
            return DetJob(self.mission.mission_id, self.mission.phrase, meta.frame_id,
                          meta.capture_ms, image, now_ms)

    def detection_failed(self, job: DetJob) -> None:
        with self.lock:
            self.det_inflight = False

    # ----- detector results -----------------------------------------------------------------
    def _rank(self, dets: list[Detection], work) -> list[Cand]:
        m = self.mission
        out = []
        for d in dets:
            if d.score < self.cfg.min_score or overlaps_counter(d.box):
                continue
            frac = color_fraction(work, d.box, d.mask, m.color, m.category)
            if frac is not None and frac < self.cfg.min_color_frac:
                self.stats["color_rejected"] = self.stats.get("color_rejected", 0) + 1
                continue                          # wrong color: never a candidate
            cs = color_score(frac)
            hist = hue_hist(work, d.box, d.mask, m.category)
            out.append(Cand(d, d.score * cs * size_prior(d.box), cs, hist))
        out.sort(key=lambda c: -c.rank)
        return out

    def _tie(self, cands: list[Cand]) -> bool:
        return (len(cands) >= 2 and cands[1].rank >= self.cfg.tie_ratio * cands[0].rank
                and iou(cands[0].det.box, cands[1].det.box) < 0.3)

    def on_detection(self, job: DetJob, dets: list[Detection], latency_ms: Optional[float] = None,
                     now_ms: Optional[float] = None) -> list:
        """Fuse a detector result for job's frame (captured earlier). Returns [Candidates] or []."""
        with self.lock:
            self.det_inflight = False
            if self.mission is None or job.mission_id != self.mission.mission_id:
                return []                         # from an old mission
            self.stats["detections"] += 1
            start = self.ring.get(job.frame_id)
            work = start.img if start is not None else work_image(job.image, False)
            cands = self._rank(dets, work)
            tr = self.track
            if tr is not None and not tr.lost:
                return self._update_track(tr, cands, job, latency_ms)
            return self._find(cands, job, latency_ms, now_ms if now_ms is not None else job.submit_ms)

    def _find(self, cands: list[Cand], job: DetJob, latency_ms, now_ms: float) -> list:
        if not cands:
            return []
        lost = self.track if (self.track is not None and self.track.lost) else None
        if lost is not None and lost.hist is not None:
            # After a loss, prefer candidates that look like the subject we had.
            for c in cands:
                sim = hist_intersection(lost.hist, c.hist)
                c.rank *= 0.5 + 0.5 * (sim if sim is not None else 0.5)
            cands.sort(key=lambda c: -c.rank)
        if self.selected is not None:
            # Re-identify the user's pick: closest in position and color to what they chose.
            sel = self.selected
            i = associate(sel.det.box, sel.hist, [c.det for c in cands], [c.hist for c in cands],
                          min_iou=0.0, max_offset=2.0)
            if i is None:
                return []
            self.selected = None
            self._start_track(cands[i], job.frame_id, job.capture_ms, latency_ms,
                              confirms=self.cfg.confirm_needed)
            return []
        if self._tie(cands):
            # Number the look-alikes left to right. The set the user sees stays fixed until it is
            # re-sent, so a Select index always refers to the boxes on their screen.
            if self.pending is None or now_ms - self.pending.sent_ms >= self.cfg.candidates_resend_ms:
                items = sorted([c for c in cands if c.rank >= self.cfg.tie_ratio * cands[0].rank][:6],
                               key=lambda c: c.det.box[0] + c.det.box[2])
                self.pending = PendingTie(job.frame_id, job.capture_ms, items, sent_ms=now_ms)
                self.stats["candidates_sent"] += 1
                return [Candidates(mission_id=job.mission_id, frame_id=job.frame_id, items=[
                    Candidate(index=i, box=c.det.to_box(), confidence=round(c.det.score, 4),
                              color_score=round(c.color, 4)) for i, c in enumerate(items, start=1)])]
            return []
        self.pending = None                       # the tie resolved itself
        self._start_track(cands[0], job.frame_id, job.capture_ms, latency_ms)
        return []

    def _start_track(self, c: Cand, frame_id: int, capture_ms: float, latency_ms, confirms: int = 1) -> None:
        rs = reseed(self.make_tracker, self.ring, frame_id, c.det.box, self.cfg.replay_max_steps)
        if rs is not None:
            tracker, box, t, score = rs.tracker, rs.box, rs.capture_ms, rs.score
        else:
            # The detection frame left the buffer or the replay failed. Starting on the newest
            # frame with the detection box is only safe if little time has passed since then.
            newest = self.ring.latest()
            if newest is None or newest.capture_ms - capture_ms > self.cfg.max_blind_start_ms:
                log.info("not starting a track from a %s ms old detection without replay",
                         None if newest is None else round(newest.capture_ms - capture_ms))
                return
            tracker = self.make_tracker()
            tracker.init(newest.img, c.det.box)
            box, t, score = c.det.box, newest.capture_ms, 1.0
        self._track_seq += 1
        self.stats["tracks_started"] += 1
        kf = BoxKalman(self.cam, clip_box(box), t)
        newest = self.ring.latest()
        tr = Track(track_id=self._track_seq, tracker=tracker, kf=kf, n_confirms=confirms,
                   last_confirm_capture_ms=capture_ms, confidence=c.det.score, hist=c.hist,
                   poly_rel=_poly_rel(c.det), last_score=score,
                   pending_source=self.detector_name, pending_latency_ms=latency_ms)
        if newest is not None:
            tr.history.append((newest.frame_id, newest.capture_ms, kf.box()))
        self.track = tr
        self.pending = None
        log.info("track %d started from a %.2f detection (%s)", tr.track_id, c.det.score,
                 "re-seeded" if rs else "no replay")

    def _update_track(self, tr: Track, cands: list[Cand], job: DetJob, latency_ms) -> list:
        tb = tr.box_at(job.frame_id, job.capture_ms)
        i = associate(tb, tr.hist, [c.det for c in cands], [c.hist for c in cands]) if tb else None
        if i is None:
            tr.det_misses += 1
            # Our subject is not in this detection. If the tracker is unsure, or the detector has
            # missed it several times while seeing one clear candidate elsewhere, switch to it.
            if cands and not self._tie(cands) and (tr.last_score < self.cfg.doubt_score
                                                   or tr.det_misses >= self.cfg.det_miss_switch):
                log.info("track %d: detector sees the subject elsewhere; new track", tr.track_id)
                self._start_track(cands[0], job.frame_id, job.capture_ms, latency_ms)
            return []
        c = cands[i]
        rs = reseed(self.make_tracker, self.ring, job.frame_id, c.det.box, self.cfg.replay_max_steps)
        live = tr.box_at(rs.frame_id, rs.capture_ms) if rs is not None else None
        if rs is not None and live is not None and box_offset(rs.box, live) > self.cfg.reseed_max_offset:
            tr.reject_streak += 1
            self.stats["reseed_rejected"] += 1
            log.info("track %d: re-seed rejected (offset %.2f of box size, streak %d)", tr.track_id,
                     box_offset(rs.box, live), tr.reject_streak)
            if tr.reject_streak >= self.cfg.max_reject_streak:
                self._start_track(c, job.frame_id, job.capture_ms, latency_ms)
            return []
        if rs is not None:
            tr.tracker = rs.tracker
            tr.kf.update(rs.capture_ms, clip_box(rs.box), "tracker", max(rs.score, 0.5))
            tr.last_score = rs.score
            self.stats["reseed_ok"] += 1
        tr.n_confirms += 1
        tr.last_confirm_capture_ms = job.capture_ms
        tr.confidence = c.det.score
        if c.hist is not None:
            tr.hist = c.hist if tr.hist is None else 0.7 * tr.hist + 0.3 * c.hist
        if c.det.mask_poly:
            tr.poly_rel = _poly_rel(c.det)
        tr.reject_streak = 0
        tr.det_misses = 0
        tr.misses = 0
        tr.pending_source, tr.pending_latency_ms = self.detector_name, latency_ms
        return []
