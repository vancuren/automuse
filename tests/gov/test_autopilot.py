"""Unit tests for the pure autopilot function (PLAN.md 4.4)."""
from __future__ import annotations

import math

from conftest import make_obs
from mhd.autopilot import AutopilotRun, autopilot_start, autopilot_step, bearing_rad, yaw_kp, yaw_rate_per_unit
from mhd.contracts import AutopilotState as S
from mhd.governor.config import load_autopilot_config, replace

P = load_autopilot_config()
PB = replace(P, **{"forward.backward_budget_m": 0.5})   # backing allowed (off by default since review 3)


class Driver:
    """Steps the autopilot alone at 25 Hz, assuming the governor applied exactly its output."""

    def __init__(self, params=P, can_move=True):
        self.p = params
        self.t = 0.0
        self.s = autopilot_start(AutopilotRun(), "m1", self.t)
        self.can_move = can_move
        self.applied = {"yaw": 0.0, "pitch": 0.0}
        self.fid = 0
        self.events = []

    def step(self, obs=None):
        sticks, self.s, ev = autopilot_step(obs, self.s, self.p, self.t, can_move=self.can_move,
                                            applied_yaw=self.applied["yaw"], applied_pitch=self.applied["pitch"])
        self.events += ev
        self.applied = {"yaw": sticks["yaw"] if self.can_move else 0.0,
                        "pitch": sticks["pitch"] if self.can_move else 0.0}
        self.t += 40.0
        return sticks

    def obs(self, **kw):
        self.fid += 1
        kw.setdefault("confirm_age_ms", 200.0)
        return make_obs(self.fid, self.t - 150.0, **kw)

    def run(self, seconds, **obs_kw):
        out = None
        for _ in range(int(round(seconds * 25))):
            out = self.step(self.obs(**obs_kw) if obs_kw is not None else None)
        return out

    def run_blind(self, seconds):
        out = None
        for _ in range(int(round(seconds * 25))):
            out = self.step(None)
        return out


def test_bearing_sign_and_gain_formula():
    assert bearing_rad(640, P) == 0.0
    assert bearing_rad(900, P) > 0 and bearing_rad(300, P) < 0
    f = 640 / math.tan(math.radians(75) / 2)
    assert math.isclose(bearing_rad(1280, P), math.atan(640 / f))
    assert math.isclose(yaw_kp(P) * yaw_rate_per_unit(P) * P.delay.theta_s, 0.4)


def test_find_confirm_follow_takes_about_one_second_of_lock():
    d = Driver()
    assert d.s.mode == S.find
    d.step(d.obs(cx=900))
    assert d.s.mode == S.confirm
    d.run(0.9, cx=900)
    assert d.s.mode == S.confirm             # not yet: < 1 s of consistent lock
    d.run(0.2, cx=900)
    assert d.s.mode == S.follow
    out = d.step(d.obs(cx=900))
    assert out["yaw"] > 0 and out["throttle"] == 0 and out["roll"] == 0


def test_confirm_resets_on_a_jump_or_track_switch():
    d = Driver()
    d.run(0.5, cx=900)
    d.step(d.obs(cx=300))                    # jumped across the frame
    assert d.s.mode == S.find
    d.run(0.5, cx=900)
    d.step(d.obs(cx=905, track_id=7))        # another identity
    assert d.s.mode == S.find


def test_follow_to_doubt_on_low_score_then_lost_after_p95_plus_half():
    d = Driver()
    d.run(1.3, cx=800)
    assert d.s.mode == S.follow
    d.step(d.obs(cx=800, score=0.2))         # one doubtful frame is debounced (doubt_debounce_obs = 2)
    assert d.s.mode == S.follow
    out = d.step(d.obs(cx=800, score=0.2))   # the second in a row enters DOUBT
    assert d.s.mode == S.doubt and out["yaw"] == 0 and out["pitch"] == 0
    d.run(1.9, cx=800, score=0.2)            # default detector p95 1.5 s + 0.5 s
    assert d.s.mode == S.doubt
    d.run(0.2, cx=800, score=0.2)
    assert d.s.mode in (S.lost, S.search)


def test_doubt_recovers_only_on_a_fresh_detector_confirmation():
    d = Driver()
    d.run(1.3, cx=800)
    d.step(d.obs(cx=800, score=0.2))
    d.step(d.obs(cx=800, score=0.2))             # two in a row (doubt_debounce_obs = 2)
    assert d.s.mode == S.doubt
    d.run(0.3, cx=800, confirm_age_ms=2000.0)    # tracker fine but confirmation predates the doubt
    assert d.s.mode == S.doubt
    d.step(d.obs(cx=800, confirm_age_ms=0.0))
    assert d.s.mode == S.follow


def test_lost_when_confirmation_age_too_old():
    d = Driver()
    d.run(1.3, cx=800)
    age = {"v": 200.0}
    for _ in range(100):
        age["v"] += 40.0
        d.step(d.obs(cx=800, confirm_age_ms=age["v"]))
        if d.s.mode != S.follow:
            break
    assert d.s.mode == S.lost and age["v"] > 2500.0 - 1.0


def test_search_is_stop_and_stare_within_90_degrees_then_gives_up():
    d = Driver()
    d.run(1.3, cx=1000)                      # last seen on the right
    d.step(d.obs(cx=1000, state="lost", w=0))
    assert d.s.mode == S.doubt
    heads = []
    for _ in range(int(30 * 25)):
        d.step(d.obs(state="searching", w=0))
        heads.append(d.s.heading)
        if d.s.mode == S.give_up:
            break
    assert d.s.mode == S.give_up
    center = d.s.last_seen_abs
    assert max(abs(h - center) for h in heads) <= math.radians(90) + 0.06
    assert max(heads) > center + math.radians(50)        # searched toward the last-seen side
    kinds = [e["to"] for e in d.events if e["kind"] == "state"]
    assert "SEARCH" in kinds and kinds[-1] == "GIVE_UP"
    assert any(e["kind"] == "announce" for e in d.events)


def test_refind_from_search_goes_through_confirm():
    d = Driver()
    d.run(1.3, cx=1000)
    d.step(d.obs(state="lost", w=0))
    d.run(3.0, state="searching", w=0)
    assert d.s.mode in (S.lost, S.search)
    d.step(d.obs(cx=700))
    assert d.s.mode == S.confirm


def test_search_waits_when_the_governor_will_not_move():
    d = Driver(can_move=False)
    d.run(1.3, cx=1000)
    d.step(d.obs(state="lost", w=0))
    d.run(5.0, state="searching", w=0)
    assert d.s.mode == S.lost                # LOST holds; SEARCH needs motion
    assert d.s.heading == 0.0


def test_forward_only_when_gated():
    d = Driver()
    small = dict(cx=640, w=100, h=300, cy=360)           # far away: wants forward
    d.run(1.3, **small)
    assert d.s.mode == S.follow
    assert d.step(d.obs(**small))["pitch"] > 0
    assert d.step(d.obs(cx=640, w=100, h=300, cy=155))["pitch"] == 0       # y1 = 5 px: top edge within 2%
    assert d.step(d.obs(cx=640, w=100, h=300, cy=360, confirm_age_ms=1600))["pitch"] == 0   # stale confirm
    # |bearing| >= 0.4 rad: yaw only
    cx = 640 + math.tan(0.45) * (640 / math.tan(math.radians(37.5)))
    d2 = Driver()
    d2.run(1.3, cx=cx, w=100, h=300)
    out = d2.step(d2.obs(cx=cx, w=100, h=300))
    assert out["pitch"] == 0 and out["yaw"] > 0


def test_forward_cap_and_size_deadband():
    d = Driver()
    d.run(1.3, cx=640, w=70, h=300)                       # small: far (but within the 64 px gate)
    assert math.isclose(d.step(d.obs(cx=640, w=70, h=300))["pitch"], P.forward.cap)
    d2 = Driver()
    w_ref = P.forward.ref_width_frac.medium * 1280
    d2.run(1.3, cx=640, w=w_ref * 1.05, h=300)            # within the +/-0.08 deadband
    assert d2.step(d2.obs(cx=640, w=w_ref * 1.05, h=300))["pitch"] == 0


def test_standoff_hovers_and_never_backs_off_by_default():
    # Review (2026-11-04): under a reversed pitch sign a back-off flies toward the person.
    assert P.forward.standoff.back_off_pitch == 0.0
    d = Driver()
    big = dict(cx=640, w=800, h=650, cy=360)              # too close
    d.run(1.3, **big)
    assert d.s.mode == S.follow
    pitches = [d.step(d.obs(**big))["pitch"] for _ in range(int(12 * 25))]
    assert all(p == 0 for p in pitches) and d.s.dbg_gate == "standoff"
    assert d.s.backward_m == 0.0


def test_standoff_back_off_when_configured_stops_at_the_budget():
    p = replace(PB, **{"forward.standoff.back_off_pitch": 0.08})
    d = Driver(p)
    big = dict(cx=640, w=800, h=650, cy=360)
    d.run(1.3, **big)
    assert d.step(d.obs(**big))["pitch"] == 0             # hover first
    d.run(0.6, **big)
    out = d.step(d.obs(**big))
    assert out["pitch"] < 0 and abs(out["pitch"]) <= P.forward.cap * P.forward.back_cap_ratio
    # the 0.5 m budget runs out: 0.08 x 127 x 0.012 = 0.12 m/s -> ~4 s
    d.run(6.0, **big)
    assert d.step(d.obs(**big))["pitch"] == 0
    assert d.s.backward_m >= p.forward.backward_budget_m == 0.5
    assert any(e["kind"] == "safety" for e in d.events)


def test_backward_budget_is_not_refilled_by_forward_travel_or_a_new_mission():
    d = Driver(PB)
    near = dict(cx=640, w=180, h=400)       # closer than the 128 px reference: backs up on the size error
    d.run(1.3, **near)
    assert d.s.mode == S.follow and d.step(d.obs(**near))["pitch"] < 0
    d.run(4.0, **near)                      # 0.125 x 127 x 0.012 = 0.19 m/s: 0.5 m in ~2.6 s
    assert d.step(d.obs(**near))["pitch"] == 0 and d.s.dbg_gate == "backward_budget"
    used = d.s.backward_m
    assert used >= PB.forward.backward_budget_m
    for i in range(50):                     # the subject drifts away (to 90 px): 2 s of forward flight
        d.step(d.obs(cx=640, w=180 - 1.8 * i, h=400 - 3 * i))
    assert d.applied["pitch"] > 0
    assert d.s.backward_m >= used           # forward travel does not refill the budget
    s2 = autopilot_start(d.s, "m2", d.t)
    assert s2.backward_m == d.s.backward_m  # nor does a new mission (the governor resets it at takeoff)


def test_track_switch_enters_doubt_and_recovers_on_a_fresh_confirmation():
    d = Driver()
    d.run(1.3, cx=800)
    assert d.s.mode == S.follow and d.s.confirm_track_id == 1
    out = d.step(d.obs(cx=805, track_id=2))          # perception started a new track nearby
    assert d.s.mode == S.doubt and out["yaw"] == 0 and out["pitch"] == 0
    d.run(0.3, cx=805, track_id=2, confirm_age_ms=2000.0)   # its confirmation predates the doubt
    assert d.s.mode == S.doubt
    d.step(d.obs(cx=805, track_id=2, confirm_age_ms=0.0))
    assert d.s.mode == S.follow and d.s.confirm_track_id == 2
    d.run(0.5, cx=805, track_id=2)
    assert d.s.mode == S.follow


def test_alternating_bad_frames_enter_doubt_by_the_third_in_the_window():
    d = Driver()
    d.run(1.3, cx=800)
    for good_between in (True, True):
        d.step(d.obs(cx=800, score=0.1))
        assert d.s.mode == S.follow
        d.step(d.obs(cx=800))
    d.step(d.obs(cx=800, score=0.1))                 # third doubtful frame within 8
    assert d.s.mode == S.doubt


def test_box_jumps_are_not_debounced():
    d = Driver()
    d.run(1.3, cx=640)
    d.step(d.obs(cx=1100))                           # 460 px > 0.25 W
    assert d.s.mode == S.doubt
    d2 = Driver()
    d2.run(1.3, cx=640, w=180)
    d2.step(d2.obs(cx=640, w=300))                   # width ratio 1.67 > 1.6
    assert d2.s.mode == S.doubt


def test_a_growing_box_passes_the_width_filter_at_once():
    d = Driver(PB)
    d.run(1.3, cx=640, w=110, h=330)                 # farther than the reference: forward
    assert d.step(d.obs(cx=640, w=110, h=330))["pitch"] > 0
    out = d.step(d.obs(cx=640, w=170, h=400))        # suddenly closer (ratio 1.55: not a jump)
    assert out["pitch"] < 0                          # backs off on this very frame
    d.run(0.5, cx=640, w=170, h=400)
    out = d.step(d.obs(cx=640, w=110, h=330))        # farther again: shrinking is low-passed
    assert d.s.bw_f > 160 and out["pitch"] < 0       # (a raw width would already fly forward)


def test_feedforward_off_by_default_and_bounded_when_on():
    assert P.yaw.feedforward.enabled is False
    pff = replace(P, **{"yaw.feedforward.enabled": True})
    d = Driver(pff)
    cx = {"v": 700.0}
    for _ in range(int(3 * 25)):
        cx["v"] = min(1100.0, cx["v"] + 2.0)
        d.step(d.obs(cx=cx["v"]))
    assert d.s.mode == S.follow and d.s.ff_rate > 0


def test_idle_outputs_zero_and_mission_filter():
    s = AutopilotRun()
    out, s2, ev = autopilot_step(make_obs(1, 0.0, cx=1000), s, P, 40.0)
    assert out == {"throttle": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0} and s2.mode == S.idle
    d = Driver()
    d.step(make_obs(1, d.t - 100, cx=900, mission_id="other"))
    assert d.s.mode == S.find


# ---------------------------------------------------------------------------------------------
# Integration fixes (2026-11-04): doubt debounce and box-width low-pass

def test_single_doubtful_frame_keeps_following_on_the_last_good_box():
    d = Driver()
    d.run(1.3, cx=800)
    assert d.s.mode == S.follow
    out = d.step(d.obs(cx=800, score=0.1))       # one tracker dip
    assert d.s.mode == S.follow and out["yaw"] > 0
    d.step(d.obs(cx=800))                         # a good frame resets the count
    d.step(d.obs(cx=800, score=0.1))
    assert d.s.mode == S.follow


def test_perception_lost_enters_doubt_at_once():
    from mhd.contracts import SubjectState
    d = Driver()
    d.run(1.3, cx=800)
    o = d.obs(cx=800)
    lost = o.model_copy(update={"state": SubjectState.lost, "box": None})
    out = d.step(lost)
    assert d.s.mode == S.doubt and out["yaw"] == 0


def test_width_lowpass_damps_a_gait_swing_in_the_size_error():
    """A walker's box width swings +/-20 % with its gait; the forward command must not flip-flop."""
    def pitches(params):
        d = Driver(params)
        d.run(1.3, cx=640, w=110)
        out = []
        for i in range(100):                      # 4 s of a 1 Hz gait swing around the reference width
            w = 110 * (1.0 + 0.2 * math.sin(2 * math.pi * i / 25.0))
            out.append(d.step(d.obs(cx=640, w=w, h=330))["pitch"])
        return out
    raw = pitches(replace(P, forward=replace(P.forward, size_lowpass_s=0.0)))
    filt = pitches(P)
    flips = lambda xs: sum(1 for a, b in zip(xs, xs[1:]) if a * b < 0)
    assert flips(filt) <= flips(raw)
    assert max(abs(x) for x in filt) < max(abs(x) for x in raw)


def test_the_autopilot_never_backs_up_by_default():
    # Review 3: backing is the autopilot's only path toward the person under a reversed pitch sign.
    assert P.forward.backward_budget_m == 0.0
    d = Driver()
    near = dict(cx=700, w=200, h=450)                # closer than the reference
    d.run(1.3, **near)
    outs = [d.step(d.obs(**near)) for _ in range(100)]
    assert all(o["pitch"] == 0 for o in outs) and outs[-1]["yaw"] > 0
    assert d.s.dbg_gate == "no_backing" and d.s.backward_m == 0.0
    assert not any(e["kind"] == "safety" for e in d.events)


def test_forward_only_toward_a_subject_within_about_6_m():
    d = Driver()
    far = dict(cx=700, w=60, h=180)                  # 60 px < 0.05 W: a person about 6.3 m away
    d.run(1.3, **far)
    out = d.step(d.obs(**far))
    assert out["pitch"] == 0 and out["yaw"] > 0 and d.s.dbg_gate == "too_far"
    d.step(d.obs(cx=700, w=66, h=200))               # 66 px: within the gate
    assert d.step(d.obs(cx=700, w=66, h=200))["pitch"] > 0


def test_a_broken_lock_after_a_loss_goes_back_to_lost_not_find():
    # Demo finding: after LOST, a flaky candidate flapped CONFIRM <-> FIND in the air, and FIND
    # never searches. A broken lock now returns to LOST (then SEARCH), keeping the give-up timer.
    d = Driver()
    d.run(1.3, cx=1000)
    d.step(d.obs(state="lost", w=0))
    d.run(3.0, state="searching", w=0)
    assert d.s.mode in (S.lost, S.search)
    lost_since = d.s.lost_since_ms
    d.step(d.obs(cx=700))
    assert d.s.mode == S.confirm
    d.step(d.obs(cx=700, score=0.2))                 # the lock breaks
    assert d.s.mode == S.lost and d.s.lost_since_ms == lost_since
    d2 = Driver()                                    # from FIND it still goes back to FIND
    d2.step(d2.obs(cx=700))
    d2.step(d2.obs(cx=700, score=0.2))
    assert d2.s.mode == S.find


def test_confirm_does_not_flap_on_a_score_at_the_threshold():
    d = Driver()
    states = []
    for i in range(int(1.6 * 25)):
        d.step(d.obs(cx=700, score=0.51 if i % 2 else 0.47))   # around confirm_min_score 0.5
        states.append(d.s.mode)
    flips = sum(1 for a, b in zip(states, states[1:]) if a != b)
    assert flips <= 2 and d.s.mode == S.follow       # FIND -> CONFIRM -> FOLLOW, no flapping

