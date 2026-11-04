"""scripts/session_a.py with no drone, no backend and no network (always run; the interactive end-to-end
test is opt-in): the link-loss gate's uncontrolled window, the Lewei board's RC-pause wording, and the
latency step's hint. A fake backend stands in for the real one, IO._input answers from a script, and
everything is written under tmp_path."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[2]


@pytest.fixture
def sa(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("session_a_gates", TREE / "scripts" / "session_a.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "TREE", tmp_path)          # nothing is written into the checkout
    return mod


class FakeBackend:
    """On the ground, armed, real mode; refuses /debug/rc_pause so a step ends right after asking for it."""

    http = "http://127.0.0.1:9"

    def __init__(self):
        self.cmds, self.posts = [], []

    def up(self):
        return True

    def state(self):
        return {"flight_mode": "real", "flight_state": "ON_GROUND", "armed": True, "safety_flags": [],
                "latency_ms": {}}

    def wait(self, pred, timeout, interval=0.1):
        s = self.state()
        return s if pred(s) else None

    def cmd(self, command, **kw):
        self.cmds.append(command)

    def post(self, path, body=None):
        self.posts.append((path, body))
        return {"ok": False, "reason": "refused_by_test"}

    def close(self):
        pass


def script(sa, monkeypatch, answers=None) -> list[str]:
    """IO._input answers the first matching prompt fragment, else Enter. Returns the prompts asked."""
    asked: list[str] = []

    def fake(io, prompt):
        asked.append(prompt)
        print(f"  ?  {prompt}")
        for frag, ans in (answers or {}).items():
            if frag.lower() in prompt.lower():
                return ans
        return ""
    monkeypatch.setattr(sa.IO, "_input", fake)
    return asked


GO_STEPS = {
    "preflight": {"measured": {"fails": []}},
    "ffprobe": {"measured": {"ok": True}},
    "flag_takeoff_land": {"answers": {"motors_spun": True, "land_stopped_motors": True, "arm_step": "none needed"}},
    "flag_estop": {"answers": {"estop_cut_motors": True, "estop_latched": True}},
    "flag_land_ground": {"answers": {"land_on_ground_noop": True}},
    "linkloss_kill": {"answers": {"motors": "stopped at once", "after_s": 1.0}},
    "linkloss_wifi": {"answers": {"motors": "ramped down (landing)", "after_s": 2.5}},
    "latency": {"measured": {"result": {"decoded": 2, "latency_ms": {"p05": 480.0, "p50": 500.0}}}},
    "hfov": {"measured": {"hfov_deg": 77.3}},
}


def steps(kill=1.0, wifi=2.5) -> dict:
    st = copy.deepcopy(GO_STEPS)
    st["linkloss_kill"]["answers"]["after_s"] = kill
    st["linkloss_wifi"]["answers"]["after_s"] = wifi
    return st


def make_session(sa, path: Path, *, board="lewei", auto=False, steps=None):
    path.mkdir(parents=True, exist_ok=True)
    a = argparse.Namespace(board=board, drone_ip="127.0.0.1" if auto else sa.LEWEI_IP, rtsp_url=None, backend_port=9,
                           counter_port=9, auto_twin=auto, twin="http://127.0.0.1:9", media_dir=str(path / "media"),
                           motor_s=1.0, link_s=10.0, latency_s=1.0, display_ms=0.0, skip_ml=True, twin_hfov_deg=75.0)
    rec = sa.Recorder(path / "session_a-test.json", {"mode": "test"})
    for name, kinds in (steps or {}).items():
        st = rec.step(name)
        for kind, values in kinds.items():
            st[kind].update(values)
        st["status"] = "done"
    s = sa.Session(a, sa.IO(auto, False), rec)
    s.be = FakeBackend()
    return s


def link_gate(summary: dict) -> dict:
    return next(r for r in summary["gates"] if r["gate"].startswith("link loss"))


# ------------------------------------------------------------------ link loss: the uncontrolled window

def test_a_short_uncontrolled_window_is_a_plain_go(sa, tmp_path):
    out = make_session(sa, tmp_path, steps=steps(kill=1.0, wifi=3.0)).summary()      # 3.0 s is not over 3 s
    assert out["verdict"] == "GO for Session B (props on)"
    g = link_gate(out)
    assert g["result"] == "PASS" and g["detail"].endswith("; uncontrolled window 3 s")
    assert out["carry_forward"]["link_loss_uncontrolled_window_s"] == 3.0
    assert out["carry_forward"]["MHD_CAPTURE_OFFSET_MS"] == 480          # two decoded frames are enough


@pytest.mark.parametrize("kill, wifi, how_long", [(1.0, 4.5, "up to 4.5 s (over 3 s)"),
                                                  (3.5, None, "an unknown time (the stop time was not recorded)"),
                                                  (None, None, "an unknown time")])
def test_a_long_or_unknown_uncontrolled_window_is_go_with_conditions(sa, tmp_path, capsys, kill, wifi, how_long):
    s = make_session(sa, tmp_path, steps=steps(kill=kill, wifi=wifi))
    out = s.summary()
    assert out["verdict"].startswith(f"GO WITH CONDITIONS: link loss leaves the drone uncontrolled for {how_long}")
    assert out["verdict"].endswith("it keeps flying its last stick command until the motors stop; "
                                   "keep everyone out of its path")
    g = link_gate(out)
    assert g["result"] == "CONDITIONAL" and g["hard"] and "keep everyone out of its path" in g["detail"]
    assert out["carry_forward"]["link_loss_uncontrolled_window_s"] == (4.5 if wifi == 4.5 else "unknown")
    assert json.loads(s.rec.path.read_text())["summary"]["verdict"] == out["verdict"]
    s.print_summary(out)
    printed = capsys.readouterr().out
    assert "CONDITIONAL" in printed and out["verdict"] in printed


def test_conditions_keep_the_throttle_ramp_note(sa, tmp_path):
    st = steps(kill=5.0, wifi=1.0)
    st["flag_takeoff_land"]["answers"]["land_stopped_motors"] = False
    out = make_session(sa, tmp_path, steps=st).summary()
    assert out["verdict"].startswith("GO WITH CONDITIONS: link loss leaves the drone uncontrolled for up to 5 s")
    assert out["verdict"].endswith("; with the throttle-ramp landing ON")


def test_failures_and_gaps_still_win_over_the_window(sa, tmp_path):
    st = steps(kill=5.0, wifi=1.0)
    st["linkloss_wifi"]["answers"]["motors"] = "kept spinning"
    assert make_session(sa, tmp_path / "a", steps=st).summary()["verdict"].startswith("NO-GO: link loss")
    st["remote"] = {"answers": {"remote_stop_cut_motors": True}}
    assert make_session(sa, tmp_path / "b", steps=st).summary()["verdict"] == \
        "GO WITH CONDITIONS: link loss does not stop the drone; keep the physical remote in hand"
    st = steps(kill=5.0, wifi=1.0)
    del st["preflight"]
    assert make_session(sa, tmp_path / "c", steps=st).summary()["verdict"].startswith("INCOMPLETE: preflight")


def test_auto_twin_summary_is_unchanged(sa, tmp_path):
    # The twin lands on its own 1 s link-loss timer: its 3.4 s stop time stays a plain GO, as before.
    out = make_session(sa, tmp_path, auto=True, steps=steps(kill=3.4, wifi=3.49)).summary()
    assert out["verdict"] == "GO for Session B (props on)"
    g = link_gate(out)
    assert g["result"] == "PASS" and "uncontrolled" not in g["detail"]
    assert "link_loss_uncontrolled_window_s" not in out["carry_forward"]


def test_link_loss_asks_for_the_stop_time_as_the_uncontrolled_window(sa, tmp_path, monkeypatch):
    monkeypatch.setattr(sa.IO, "countdown", lambda self, seconds, label: None)
    asked = script(sa, monkeypatch, {"what did the motors do": "r", "after how many seconds": "4.5"})
    s = make_session(sa, tmp_path)
    assert s._observe_motors("linkloss_wifi", 10.0, 0.0) == ("ramped down (landing)", 4.5)
    assert "did they stop? (until then the drone flies its last stick command)" in asked[-1]


# ------------------------------------------------------------------ RC pause: ALL packets on the Lewei board

def test_lewei_rc_pause_says_all_packets_stop(sa, tmp_path, monkeypatch, capsys):
    asked = script(sa, monkeypatch)
    s = make_session(sa, tmp_path, board="lewei")
    s.step_linkloss_rcpause("linkloss_rcpause")
    out = capsys.readouterr().out
    assert "This board has NO keepalive: the pause stops ALL packets" in out and "keepalive-only" not in out
    assert asked == ["press Enter to stop ALL packets to the drone for 10 s [Enter]"]
    assert s.be.posts == [("/debug/rc_pause", {"seconds": 10.0})]       # same request as on the other board
    assert s.rec.get("linkloss_rcpause", "measured", "rc_pause_response")["ok"] is False


def test_x69_rc_pause_wording_is_unchanged(sa, tmp_path, monkeypatch, capsys):
    asked = script(sa, monkeypatch)
    s = make_session(sa, tmp_path, board="x69_lg")
    s.step_linkloss_rcpause("linkloss_rcpause")
    out = capsys.readouterr().out
    assert "tolerates keepalive-only" in out and "ALL packets" not in out
    assert asked == ["press Enter to pause RC control packets for 10 s [Enter]"]
    assert s.be.posts == [("/debug/rc_pause", {"seconds": 10.0})]


def test_lewei_step_title(sa, capsys):
    assert sa.main(["--list", "--board", "lewei"]) == 0
    assert "Link loss 3: ALL packets stop (this board has no keepalive)" in capsys.readouterr().out
    assert sa.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "Link loss 3: RC packets stop, keepalive continues" in out and "ALL packets" not in out


# ------------------------------------------------------------------ latency: few decoded frames

LAT = {"p05": 480.0, "p50": 500.0, "p95": 520.0, "min": 470.0, "max": 530.0}


@pytest.mark.parametrize("decoded, hint", [(0, True), (2, True), (4, True), (5, False), (9, False)])
def test_latency_hint_below_five_decoded_frames(sa, tmp_path, monkeypatch, capsys, decoded, hint):
    asked = script(sa, monkeypatch, {"stopwatch": "?"})
    s = make_session(sa, tmp_path)
    res = {"decoded": decoded, "frames": 1200, "source": "test"}
    if decoded:
        res["latency_ms"] = dict(LAT)
    s.report_latency("latency", res)
    out = capsys.readouterr().out
    assert asked == ["latency from the stopwatch method (ms, ? = unknown)"]       # the same single question
    assert ("40 cm" in out and "brightness" in out and "full screen" in out) is hint
    if hint:
        assert out.index("40 cm") < out.index("?  latency from the stopwatch method")    # the hint comes first
        assert f"--board lewei --only latency --resume {s.rec.path}" in out
    if decoded:
        assert "p50 500.0 ms, p05 480.0 ms. One or two samples are acceptable" in out
    assert s.rec.get("latency", "answers", "manual_latency_ms") is None


def test_latency_with_enough_frames_asks_nothing(sa, tmp_path, monkeypatch, capsys):
    asked = script(sa, monkeypatch)
    make_session(sa, tmp_path).report_latency("latency", {"decoded": 97, "source": "test", "latency_ms": dict(LAT)})
    out = capsys.readouterr().out
    assert asked == [] and "40 cm" not in out and "MHD_CAPTURE_OFFSET_MS=480" in out
