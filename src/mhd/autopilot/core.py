"""The autopilot: a pure function called by the governor on every 40 ms RC tick (PLAN.md 4.4).

    sticks, new_state, events = autopilot_step(obs, state, params, now_ms,
                                               can_move=..., applied_yaw=..., applied_pitch=...)

- `obs` is the newest SubjectObs (it repeats between perception updates; the autopilot notices
  new evidence by frame_id / capture_ms). Perception only reports evidence; every lifecycle
  transition is decided here.
- `state` is an immutable AutopilotRun; the caller keeps the returned one.
- `can_move` is False when the governor will not apply our sticks (on the ground, paused, human
  override, safety neutral). The lifecycle keeps running, but SEARCH waits for it.
- `applied_yaw` / `applied_pitch` are the normalized sticks the governor actually sent on the
  previous tick (after caps). They drive the open-loop heading estimate (search bounds,
  feed-forward) and the backward-travel budget, so they stay honest under human override.

Sticks are normalized (-1..1). Throttle and roll are always 0. The governor applies sign,
deadband compensation, caps, slew, and rounding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from mhd.contracts import AutopilotState, SubjectObs, SubjectState

from mhd.governor.config import AutopilotConfig

Mode = AutopilotState
_ZERO = {"throttle": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0}
_HEADING_HIST_MS = 4000.0
_OBS_GAP_DOUBT_S = 0.5     # FOLLOW -> DOUBT when no new observation arrives for this long
_EVIDENCE_GAP_S = 1.0      # CONFIRM -> FIND when no new observation arrives for this long


@dataclass(frozen=True)
class AutopilotRun:
    """Everything the autopilot remembers between ticks. Immutable; replaced every step."""
    mode: AutopilotState = AutopilotState.idle
    mission_id: Optional[str] = None
    distance: str = "medium"
    category: str = "none"
    entered_ms: float = 0.0
    last_step_ms: Optional[float] = None
    # newest evidence seen
    last_frame_id: int = -1
    last_capture_ms: Optional[float] = None
    last_obs_seen_ms: Optional[float] = None       # our clock when that obs first arrived
    last_confirm_age_ms: float = 1e9               # obs.confirm_age_ms of the newest obs
    last_box: Optional[tuple[float, float, float, float]] = None   # x1, y1, x2, y2 of last good obs
    confirm_track_id: Optional[int] = None
    confirm_since_capture_ms: Optional[float] = None
    doubt_since_capture_ms: Optional[float] = None
    # last good sighting
    last_seen_bearing: float = 0.0
    last_seen_abs: Optional[float] = None          # heading estimate + bearing, rad
    # open-loop heading estimate from applied yaw
    heading: float = 0.0
    heading_hist: tuple = ()                       # ((t_ms, heading), ...)
    # detector statistics (from SubjectObs.detector_latency_ms and confirm_age_ms)
    det_lat_hist: tuple = ()
    last_confirm_capture_ms: Optional[float] = None
    det_period_ms: Optional[float] = None
    # forward / back
    backward_m: float = 0.0
    standoff_since_ms: Optional[float] = None
    budget_exhausted_announced: bool = False
    # lost / search
    lost_since_ms: Optional[float] = None
    search_targets: tuple = ()
    search_idx: int = 0
    search_phase: str = "turn"
    search_phase_ms: float = 0.0
    # feed-forward on the subject's own angular rate
    ff_rate: float = 0.0
    ff_prev: Optional[tuple[float, float]] = None  # (capture_ms, absolute bearing)
    # last computed values, for telemetry and logs
    dbg_bearing: Optional[float] = None
    # integration fixes: low-passed box width (gait), doubtful-observation debounce
    bw_f: Optional[float] = None
    bw_f_ms: Optional[float] = None
    doubt_count: int = 0
    doubt_hist: tuple = ()                         # last doubt_window_obs FOLLOW obs: True = doubtful
    dbg_e_s: Optional[float] = None
    dbg_gate: str = ""


# ----------------------------------------------------------------------------------------------
# Geometry


def focal_px(params: AutopilotConfig) -> float:
    half_w = params.camera.frame_w / 2.0
    return half_w / math.tan(math.radians(params.camera.hfov_deg) / 2.0)


def bearing_rad(cx: float, params: AutopilotConfig) -> float:
    """beta = atan((cx - W/2) / f). +beta: subject right of center, so +yaw corrects it."""
    return math.atan((cx - params.camera.frame_w / 2.0) / focal_px(params))


def yaw_rate_per_unit(params: AutopilotConfig) -> float:
    """K in rad/s per normalized yaw unit (127 counts)."""
    return params.plant.yaw_rate_per_count * 127.0


def yaw_kp(params: AutopilotConfig) -> float:
    """P gain (normalized yaw per rad of bearing) for loop gain Kp*K*theta = loop_gain."""
    return params.yaw.loop_gain / (params.delay.theta_s * yaw_rate_per_unit(params))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


# ----------------------------------------------------------------------------------------------
# Lifecycle entry points used by the governor


def autopilot_start(state: AutopilotRun, mission_id: str, now_ms: float, *,
                    distance: str = "medium", category: str = "none") -> AutopilotRun:
    """Begin (or restart) a follow mission: FIND. Keeps the heading estimate and the backward travel
    used this flight (a new mission or a resume must not refill the budget), resets the rest."""
    return AutopilotRun(mode=Mode.find, mission_id=mission_id, distance=distance,
                        category=category, entered_ms=now_ms, last_step_ms=state.last_step_ms,
                        heading=state.heading, heading_hist=state.heading_hist,
                        backward_m=state.backward_m)


def autopilot_stop(state: AutopilotRun, now_ms: float) -> AutopilotRun:
    return AutopilotRun(mode=Mode.idle, entered_ms=now_ms, last_step_ms=state.last_step_ms,
                        heading=state.heading, heading_hist=state.heading_hist,
                        backward_m=state.backward_m)


# ----------------------------------------------------------------------------------------------
# The step


def autopilot_step(obs: Optional[SubjectObs], state: AutopilotRun, params: AutopilotConfig,
                   now_ms: float, *, can_move: bool = True, applied_yaw: float = 0.0,
                   applied_pitch: float = 0.0) -> tuple[dict, AutopilotRun, list]:
    w = dict(vars(state))          # mutable working copy; the input state is never modified
    events: list[dict] = []
    lc = params.lifecycle

    # 1. Dead reckoning from what the governor actually sent last tick.
    dt = 0.0 if state.last_step_ms is None else max(0.0, min(0.2, (now_ms - state.last_step_ms) / 1000.0))
    w["last_step_ms"] = now_ms
    w["heading"] = state.heading + yaw_rate_per_unit(params) * applied_yaw * dt
    hist = state.heading_hist + ((now_ms, w["heading"]),)
    while len(hist) > 2 and hist[0][0] < now_ms - _HEADING_HIST_MS:
        hist = hist[1:]
    w["heading_hist"] = hist
    speed = params.plant.forward_speed_per_count * 127.0 * applied_pitch * dt   # metres, + forward
    if state.mode != Mode.idle:
        # Backward travel during missions, open loop on the modeled speed. Forward travel does not
        # refill it (review: under a reversed sign, "forward" is the drone backing away).
        w["backward_m"] = state.backward_m + max(0.0, -speed)

    if state.mode == Mode.idle:
        return dict(_ZERO), AutopilotRun(**w), events

    # 2. Evidence bookkeeping.
    new_obs = _is_new(obs, state)
    if new_obs:
        w["last_frame_id"] = obs.frame_id
        w["last_capture_ms"] = obs.capture_ms
        w["last_obs_seen_ms"] = now_ms
        w["last_confirm_age_ms"] = obs.confirm_age_ms
        if obs.detector_latency_ms is not None and obs.detector_latency_ms > 0:
            w["det_lat_hist"] = (state.det_lat_hist + (float(obs.detector_latency_ms),))[-20:]
        if obs.confirm_age_ms < 1e8:
            conf_t = obs.capture_ms - obs.confirm_age_ms
            prev = state.last_confirm_capture_ms
            if prev is None or conf_t > prev + 1.0:
                if prev is not None and 50.0 < conf_t - prev < 10000.0:
                    p = state.det_period_ms
                    w["det_period_ms"] = (conf_t - prev) if p is None else 0.7 * p + 0.3 * (conf_t - prev)
                w["last_confirm_capture_ms"] = conf_t

    det_p95_s = _p95(w["det_lat_hist"]) / 1000.0 if w["det_lat_hist"] else lc.detector_p95_default_s
    det_period_s = (w["det_period_ms"] / 1000.0) if w["det_period_ms"] else lc.detector_period_default_s
    since_obs_s = (now_ms - w["last_obs_seen_ms"]) / 1000.0 if w["last_obs_seen_ms"] is not None else math.inf
    conf_age_now_s = w["last_confirm_age_ms"] / 1000.0 + (since_obs_s if since_obs_s != math.inf else 0.0)
    lost_age_s = max(lc.lost_confirm_age_min_s, 2.0 * det_period_s + det_p95_s)
    in_mode_s = (now_ms - state.entered_ms) / 1000.0

    sticks = dict(_ZERO)
    mode = state.mode

    def enter(new_mode: AutopilotState, reason: str, text: Optional[str] = None) -> None:
        events.append({"kind": "state", "from": w["mode"].value, "to": new_mode.value,
                       "reason": reason, "ts_ms": now_ms})
        if text:
            events.append({"kind": "announce", "text": text, "ts_ms": now_ms})
        w["mode"] = new_mode
        w["entered_ms"] = now_ms

    # 3. Lifecycle.
    if mode == Mode.find:
        if new_obs and _is_candidate(obs, params):
            _begin_confirm(w, obs)
            enter(Mode.confirm, "candidate")
        elif in_mode_s >= lc.find_timeout_s:
            enter(Mode.give_up, "find_timeout", "I can't find the subject. Holding position.")

    elif mode == Mode.confirm:
        if new_obs:
            if _consistent(obs, w, params):
                w["last_box"] = _box_t(obs)
                held_s = (obs.capture_ms - w["confirm_since_capture_ms"]) / 1000.0
                fresh = obs.confirm_age_ms / 1000.0 <= params.forward.gate.max_confirm_age_s
                if held_s >= lc.confirm_s and fresh:
                    _note_sighting(w, obs, params)
                    enter(Mode.follow, "locked", "Got them. Following.")
            else:
                w["confirm_since_capture_ms"] = None
                enter(Mode.find, "lock_broken")
        elif since_obs_s > _EVIDENCE_GAP_S:
            enter(Mode.find, "no_evidence")

    elif mode == Mode.follow:
        reason = None
        if new_obs:
            reason = _doubt_reason(obs, w, state, params)
            hist = (state.doubt_hist + (reason is not None,))[-lc.doubt_window_obs:]
            w["doubt_hist"] = hist
            if reason is None:
                w["doubt_count"] = 0
                _note_sighting(w, obs, params)
                w["last_box"] = _box_t(obs)
            elif (reason in _DEBOUNCED_REASONS and w["doubt_count"] + 1 < lc.doubt_debounce_obs
                  and sum(hist) < lc.doubt_window_k):
                # One doubtful frame (a tracker dip, a dropped box): keep acting on the last good
                # box instead of stopping. Never more than doubt_window_k - 1 of the last
                # doubt_window_obs frames. Lost, track switches and box jumps act at once.
                w["doubt_count"] += 1
                reason = None
        elif since_obs_s > _OBS_GAP_DOUBT_S:
            reason = "no_observations"
        if reason is None and conf_age_now_s > lost_age_s:
            _enter_lost(w, now_ms)
            enter(Mode.lost, "confirmation_too_old", "I lost the subject. Looking around.")
        elif reason is not None:
            w["doubt_count"] = 0
            w["doubt_hist"] = ()
            w["doubt_since_capture_ms"] = obs.capture_ms if new_obs else (w["last_capture_ms"] or now_ms)
            enter(Mode.doubt, reason)
        else:
            sticks = _follow_sticks(w, params, now_ms, conf_age_now_s, events)

    elif mode == Mode.doubt:
        recovered = (new_obs and _has_box(obs) and obs.state == SubjectState.confirmed
                     and obs.tracker_score >= lc.follow_min_score
                     and w["doubt_since_capture_ms"] is not None
                     and obs.capture_ms - obs.confirm_age_ms >= w["doubt_since_capture_ms"] - 1.0)
        if recovered:
            _note_sighting(w, obs, params)
            w["last_box"] = _box_t(obs)
            w["confirm_track_id"] = obs.track_id      # a new track needed this fresh confirmation
            enter(Mode.follow, "reconfirmed")
        elif in_mode_s >= det_p95_s + lc.doubt_extra_s:
            _enter_lost(w, now_ms)
            enter(Mode.lost, "doubt_timeout", "I lost the subject. Looking around.")

    elif mode in (Mode.lost, Mode.search):
        lost_s = (now_ms - (w["lost_since_ms"] or now_ms)) / 1000.0
        if new_obs and _is_candidate(obs, params):
            _begin_confirm(w, obs)
            enter(Mode.confirm, "candidate")       # re-finding always passes through CONFIRM
        elif lost_s >= lc.give_up_s:
            enter(Mode.give_up, "give_up_timeout", "I can't find the subject. Holding position.")
        elif mode == Mode.lost:
            if can_move and in_mode_s >= det_p95_s + lc.search.stare_extra_s:
                _plan_search(w, params, now_ms)
                enter(Mode.search, "start_search")
        else:
            sticks, done = _search_sticks(w, params, now_ms, det_p95_s, can_move)
            if done:
                enter(Mode.give_up, "search_exhausted", "I can't find the subject. Holding position.")

    elif mode == Mode.give_up:
        pass  # hover until a new mission or an explicit resume (autopilot_start)

    return sticks, AutopilotRun(**w), events


# ----------------------------------------------------------------------------------------------
# Evidence helpers


# Doubtful observations FOLLOW may ride through for a frame (keeping the last good box).
_DEBOUNCED_REASONS = frozenset({"no_box", "low_tracker_score"})


def _is_new(obs: Optional[SubjectObs], state: AutopilotRun) -> bool:
    if obs is None:
        return False
    if obs.mission_id is not None and state.mission_id is not None and obs.mission_id != state.mission_id:
        return False
    return obs.frame_id != state.last_frame_id or obs.capture_ms != state.last_capture_ms


def _has_box(obs: SubjectObs) -> bool:
    b = obs.box
    return b is not None and b.w > 1.0 and b.h > 1.0


def _box_t(obs: SubjectObs) -> tuple[float, float, float, float]:
    b = obs.box
    return (b.x1, b.y1, b.x2, b.y2)


def _is_candidate(obs: SubjectObs, params: AutopilotConfig) -> bool:
    return (_has_box(obs) and obs.state in (SubjectState.candidate, SubjectState.confirmed)
            and obs.tracker_score >= params.lifecycle.confirm_min_score)


def _begin_confirm(w: dict, obs: SubjectObs) -> None:
    w["confirm_since_capture_ms"] = obs.capture_ms
    w["confirm_track_id"] = obs.track_id
    w["last_box"] = _box_t(obs)


def _consistent(obs: SubjectObs, w: dict, params: AutopilotConfig) -> bool:
    if not _is_candidate(obs, params):
        return False
    if w["confirm_track_id"] is not None and obs.track_id is not None and obs.track_id != w["confirm_track_id"]:
        return False
    if w["last_box"] is not None:
        x1, _, x2, _ = w["last_box"]
        jump = abs(obs.box.cx - (x1 + x2) / 2.0)
        if jump > params.lifecycle.confirm_max_jump_frac * params.camera.frame_w:
            return False
    return True


def _doubt_reason(obs: SubjectObs, w: dict, state: AutopilotRun, params: AutopilotConfig) -> Optional[str]:
    lc = params.lifecycle
    if obs.state in (SubjectState.lost, SubjectState.searching):
        return f"perception_{obs.state.value}"
    if (w["confirm_track_id"] is not None and obs.track_id is not None
            and obs.track_id != w["confirm_track_id"]):
        return "track_switch"                 # perception started a new track: maybe someone else
    if not _has_box(obs):
        return "no_box"
    if obs.tracker_score < lc.follow_min_score:
        return "low_tracker_score"
    if state.last_box is not None:
        x1, y1, x2, y2 = state.last_box
        if abs(obs.box.cx - (x1 + x2) / 2.0) > lc.jump_frac * params.camera.frame_w:
            return "box_jump"
        prev_w = max(1.0, x2 - x1)
        ratio = obs.box.w / prev_w
        if ratio > lc.jump_size_ratio or ratio < 1.0 / lc.jump_size_ratio:
            return "box_size_jump"
    return None


def _p95(values: tuple) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(math.ceil(0.95 * len(s))) - 1)]


def _heading_at(w: dict, t_ms: float) -> float:
    """Open-loop heading estimate at time t (nearest sample at or before t)."""
    hist = w["heading_hist"]
    if not hist:
        return w["heading"]
    best = hist[0][1]
    for t, h in hist:
        if t <= t_ms:
            best = h
        else:
            break
    return best


def _note_sighting(w: dict, obs: SubjectObs, params: AutopilotConfig) -> None:
    beta = bearing_rad(obs.box.cx, params)
    head = _heading_at(w, obs.capture_ms - params.plant.yaw_lag_s * 1000.0)
    abs_b = head + beta
    w["last_seen_bearing"] = beta
    w["last_seen_abs"] = abs_b
    # feed-forward state: low-passed absolute angular rate of the subject
    prev = w["ff_prev"]
    if prev is not None:
        dt = (obs.capture_ms - prev[0]) / 1000.0
        if 0.02 < dt < 1.0:
            rate = _wrap(abs_b - prev[1]) / dt
            tau = max(1e-3, params.yaw.feedforward.lowpass_s)
            w["ff_rate"] = w["ff_rate"] + (dt / (tau + dt)) * (rate - w["ff_rate"])
        elif dt >= 1.0:
            w["ff_rate"] = 0.0
    w["ff_prev"] = (obs.capture_ms, abs_b)


def _enter_lost(w: dict, now_ms: float) -> None:
    w["lost_since_ms"] = now_ms
    w["bw_f"] = None
    w["bw_f_ms"] = None
    w["ff_rate"] = 0.0
    w["ff_prev"] = None
    w["standoff_since_ms"] = None


# ----------------------------------------------------------------------------------------------
# FOLLOW: yaw on bearing, forward/back on size


def _follow_sticks(w: dict, params: AutopilotConfig, now_ms: float, conf_age_now_s: float,
                   events: list) -> dict:
    sticks = dict(_ZERO)
    if w["last_box"] is None:
        return sticks
    x1, y1, x2, y2 = w["last_box"]
    W, H = float(params.camera.frame_w), float(params.camera.frame_h)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    beta = bearing_rad((x1 + x2) / 2.0, params)
    w["dbg_bearing"] = beta

    # Yaw: P-only at loop gain Kp*K*theta; optional feed-forward of the subject's own rate.
    yaw = 0.0
    if abs(beta) >= params.yaw.bearing_deadband_rad:
        yaw = yaw_kp(params) * beta
    if params.yaw.feedforward.enabled:
        yaw += params.yaw.feedforward.gain * w["ff_rate"] / yaw_rate_per_unit(params)
    sticks["yaw"] = max(-1.0, min(1.0, yaw))

    # Forward / back.
    fw = params.forward
    st = fw.standoff
    if bh > st.max_box_h_frac * H or bw > st.max_box_w_frac * W:
        if w["standoff_since_ms"] is None:
            w["standoff_since_ms"] = now_ms
        w["dbg_gate"] = "standoff"
        pitch = 0.0
        if st.back_off_pitch > 0 and (now_ms - w["standoff_since_ms"]) / 1000.0 >= st.hover_s:
            pitch = -min(st.back_off_pitch, fw.cap * fw.back_cap_ratio)
    else:
        w["standoff_since_ms"] = None
        ref_frac = getattr(fw.ref_width_frac, w["distance"], fw.ref_width_frac.medium)
        scale = getattr(fw.category_scale, w["category"], 1.0)
        w_ref = max(1.0, ref_frac * scale * W)
        tau = fw.size_lowpass_s
        if tau > 0 and w["bw_f"] is not None and w["bw_f_ms"] is not None and bw < w["bw_f"]:
            dt_s = max(0.0, (now_ms - w["bw_f_ms"]) / 1000.0)
            w["bw_f"] = w["bw_f"] + (dt_s / (tau + dt_s)) * (bw - w["bw_f"])
        else:
            # first box, filter off, or the box grew: a subject getting closer is seen at once
            # (review: a lagging width kept the drone moving toward an approaching walker)
            w["bw_f"] = bw
        w["bw_f_ms"] = now_ms
        e_s = math.log(w_ref / max(1.0, w["bw_f"]))
        w["dbg_e_s"] = e_s
        e = 0.0 if abs(e_s) <= fw.size_deadband else e_s - _sign(e_s) * fw.size_deadband
        pitch = fw.kp * e
        g = fw.gate
        trusted = abs(beta) < g.max_abs_bearing_rad and conf_age_now_s < g.max_confirm_age_s
        if pitch > 0:
            framed = (y1 >= g.edge_margin_frac * H and y2 <= (1.0 - g.edge_margin_frac) * H
                      and bh < g.max_box_h_frac * H)
            w["dbg_gate"] = "forward" if (trusted and framed) else "yaw_only"
            pitch = min(pitch, fw.cap) if (trusted and framed) else 0.0
        elif pitch < 0:
            w["dbg_gate"] = "back" if trusted else "yaw_only"
            pitch = max(pitch, -fw.cap * fw.back_cap_ratio) if trusted else 0.0
        else:
            w["dbg_gate"] = "hold"
    if pitch < 0 and w["backward_m"] >= fw.backward_budget_m:
        pitch = 0.0
        w["dbg_gate"] = "backward_budget"
        if not w["budget_exhausted_announced"]:
            w["budget_exhausted_announced"] = True
            events.append({"kind": "safety", "text": "backward travel budget used up", "ts_ms": now_ms})
    sticks["pitch"] = pitch
    return sticks


# ----------------------------------------------------------------------------------------------
# SEARCH: stop-and-stare within +/- max_deg of the last-seen bearing


def _plan_search(w: dict, params: AutopilotConfig, now_ms: float) -> None:
    sc = params.lifecycle.search
    center = w["last_seen_abs"] if w["last_seen_abs"] is not None else w["heading"]
    side = _sign(w["last_seen_bearing"]) or 1.0
    step, top = math.radians(sc.step_deg), math.radians(sc.max_deg)
    n = max(1, int(round(top / step)))
    offsets = [0.0] + [side * step * k for k in range(1, n + 1)] + [-side * step * k for k in range(1, n + 1)]
    w["search_targets"] = tuple(center + o for o in offsets)
    w["search_idx"] = 0
    w["search_phase"] = "turn"
    w["search_phase_ms"] = now_ms


def _search_sticks(w: dict, params: AutopilotConfig, now_ms: float, det_p95_s: float,
                   can_move: bool) -> tuple[dict, bool]:
    sticks = dict(_ZERO)
    sc = params.lifecycle.search
    targets = w["search_targets"]
    if w["search_idx"] >= len(targets):
        return sticks, True
    if w["search_phase"] == "turn":
        err = _wrap(targets[w["search_idx"]] - w["heading"])
        if abs(err) < math.radians(3.0):
            w["search_phase"] = "stare"
            w["search_phase_ms"] = now_ms
        elif can_move:
            sticks["yaw"] = _sign(err) * sc.yaw
    else:
        if (now_ms - w["search_phase_ms"]) / 1000.0 >= det_p95_s + sc.stare_extra_s:
            w["search_idx"] += 1
            w["search_phase"] = "turn"
            w["search_phase_ms"] = now_ms
            if w["search_idx"] >= len(targets):
                return sticks, True
    return sticks, False
