"""Browser tests: the operator UI in Google Chrome against the mock backend.

Each test gets a fresh page and a reset mock (see conftest.py). The mock logs every message the
UI sends, with its own receive time, to a JSONL file; the tests read that file.
"""
from __future__ import annotations

import time

import pytest

HB = {"type": "hb", "client": "ui"}


def sel(testid: str) -> str:
    return f'[data-testid="{testid}"]'


def gaps(entries: list[dict]) -> list[float]:
    ts = [e["t_ms"] for e in entries]
    return [b - a for a, b in zip(ts, ts[1:])]


# ── Heartbeat ────────────────────────────────────────────────────────────────────────────────

def test_heartbeat_is_5hz_from_a_single_socket(ui, mock_backend):
    time.sleep(1.0)  # let React StrictMode's double mount settle
    t0 = mock_backend.now_ms()
    time.sleep(4.0)
    hbs = mock_backend.of_type("hb", "ws", since_ms=t0)
    rate = len(hbs) / 4.0
    assert 4.5 <= rate <= 5.5, f"heartbeat rate {rate:.2f} Hz"
    assert len({e["conn"] for e in hbs}) == 1, "exactly one heartbeat socket (the worker's)"
    assert all(e["msg"] == HB for e in hbs)
    assert max(gaps(hbs)) < 400, f"worst heartbeat gap {max(gaps(hbs)):.0f} ms"


def test_heartbeat_keeps_going_while_the_page_main_thread_is_blocked(ui, mock_backend):
    """Only a Web Worker can keep sending while the page's own thread is frozen."""
    time.sleep(0.5)
    t0 = mock_backend.now_ms()
    ui.evaluate("() => { const t = performance.now(); while (performance.now() - t < 2500) {} }")
    t1 = mock_backend.now_ms()
    hbs = mock_backend.of_type("hb", "ws", since_ms=t0)
    during = [e for e in hbs if e["t_ms"] <= t1]
    assert t1 - t0 >= 2400
    assert len(during) >= 10, f"only {len(during)} heartbeats during a 2.5 s main-thread block"
    assert max(gaps(during)) < 450


# ── Sticks ───────────────────────────────────────────────────────────────────────────────────

def test_no_axes_while_idle_and_only_nonzero_while_a_key_is_held(ui, mock_backend):
    time.sleep(2.0)
    assert mock_backend.of_type("axes") == [], "axes sent while idle"
    assert ui.get_attribute(sel("local-input"), "data-sending") == "0"

    t0 = mock_backend.now_ms()
    ui.keyboard.down("KeyW")  # pitch forward
    time.sleep(0.6)
    ui.keyboard.up("KeyW")
    t_up = mock_backend.now_ms()
    time.sleep(0.6)

    axes = mock_backend.of_type("axes", "ws", since_ms=t0)
    assert 12 <= len(axes) <= 24, f"{len(axes)} axes messages in 0.6 s (expect about 30 Hz)"
    assert all(e["msg"]["pitch"] == 1 and e["msg"]["mode"] == "inc" for e in axes)
    assert all(e["valid"] for e in axes)  # the mock rejects all-zero and extra fields
    late = [e for e in axes if e["t_ms"] > t_up + 100]
    assert late == [], f"{len(late)} axes messages after the key was released"


def test_axes_stop_on_window_blur_even_with_the_key_still_down(ui, mock_backend):
    ui.keyboard.down("KeyD")
    time.sleep(0.4)
    ui.evaluate("() => window.dispatchEvent(new Event('blur'))")
    t_blur = mock_backend.now_ms()
    time.sleep(0.6)
    ui.keyboard.up("KeyD")
    axes = mock_backend.of_type("axes", "ws")
    assert axes, "no axes were sent before the blur"
    late = [e for e in axes if e["t_ms"] > t_blur + 80]
    assert late == [], f"{len(late)} axes messages after blur"


def test_axes_stop_when_the_tab_is_hidden(ui, mock_backend):
    ui.keyboard.down("ArrowUp")  # throttle up
    time.sleep(0.4)
    ui.evaluate("""() => {
        Object.defineProperty(document, 'visibilityState', { value: 'hidden', configurable: true });
        document.dispatchEvent(new Event('visibilitychange'));
    }""")
    t_hidden = mock_backend.now_ms()
    time.sleep(0.6)
    ui.keyboard.up("ArrowUp")
    ui.evaluate("() => { delete document.visibilityState; }")
    axes = mock_backend.of_type("axes", "ws")
    assert axes and all(e["msg"]["throttle"] == 1 for e in axes)
    assert [e for e in axes if e["t_ms"] > t_hidden + 80] == []


# ── Push-to-talk and focus ──────────────────────────────────────────────────────────────────

def test_space_is_push_to_talk_and_never_clicks_a_button(ui, mock_backend):
    # Record, after our capture-phase listener ran, whether Space's default action was blocked.
    ui.evaluate("""() => {
        window.__space = [];
        addEventListener('keydown', e => { if (e.code === 'Space') __space.push(['down', e.defaultPrevented]); });
        addEventListener('keyup', e => { if (e.code === 'Space') __space.push(['up', e.defaultPrevented]); });
    }""")

    # Click TAKEOFF with the mouse: it works, but it must not keep focus.
    ui.click(sel("takeoff-btn"))
    assert len(mock_backend.wait_for(lambda: mock_backend.commands("takeoff"))) == 1
    assert ui.evaluate("() => document.activeElement.tagName") == "BODY"

    # Even a scripted focus is undone at once by the focus guard.
    ui.evaluate("() => document.querySelector('[data-testid=\"ptt-btn\"]').focus()")
    assert ui.evaluate("() => document.activeElement.tagName") != "BUTTON"

    ui.keyboard.down("Space")
    time.sleep(0.7)
    ui.keyboard.up("Space")

    events = mock_backend.wait_for(lambda: mock_backend.of_type("agent_event", "agent"), min_count=2)
    kinds = [e["msg"]["kind"] for e in events]
    assert kinds[:2] == ["ptt_down", "ptt_up"], kinds
    assert ui.evaluate("() => window.__space") == [["down", True], ["up", True]]
    time.sleep(0.3)
    assert len(mock_backend.commands("takeoff")) == 1, "Space re-fired the TAKEOFF button"

    # The mock agent answers: transcript, then a mission with a model badge and latency.
    ui.wait_for_selector(f'{sel("transcript")} >> text=follow the person in the blue shirt', timeout=3000)
    ui.wait_for_selector(f'{sel("model-badge")} >> text=glimmer-30b+lora', timeout=4000)
    assert "s" in ui.inner_text(sel("mission-latency"))


def test_ptt_button_hold_and_release(ui, mock_backend):
    box = ui.locator(sel("ptt-btn")).bounding_box()
    ui.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    ui.mouse.down()
    ui.wait_for_selector(f'{sel("ptt-indicator")}[data-active="1"]', timeout=2000)
    time.sleep(0.5)
    ui.mouse.up()
    events = mock_backend.wait_for(lambda: mock_backend.of_type("agent_event", "agent"), min_count=2)
    assert [e["msg"]["kind"] for e in events][:2] == ["ptt_down", "ptt_up"]
    assert ui.evaluate("() => document.activeElement.tagName") == "BODY"


def test_push_to_talk_is_released_when_the_window_loses_focus(ui, mock_backend):
    ui.keyboard.down("Space")
    mock_backend.wait_for(lambda: mock_backend.of_type("agent_event", "agent"))
    ui.evaluate("() => window.dispatchEvent(new Event('blur'))")
    events = mock_backend.wait_for(lambda: mock_backend.of_type("agent_event", "agent"), min_count=2)
    ui.keyboard.up("Space")
    assert [e["msg"]["kind"] for e in events][:2] == ["ptt_down", "ptt_up"]


def test_every_button_is_out_of_the_tab_order(ui):
    tab_indexes = ui.eval_on_selector_all("button", "els => els.map(e => e.tabIndex)")
    assert tab_indexes and all(t == -1 for t in tab_indexes), tab_indexes
    ui.keyboard.press("Tab")
    assert ui.evaluate("() => document.activeElement.tagName") != "BUTTON"


# ── E-STOP and LAND ─────────────────────────────────────────────────────────────────────────

def test_escape_sends_estop_over_the_socket(ui, mock_backend):
    ui.keyboard.press("Escape")
    ws = mock_backend.wait_for(lambda: mock_backend.commands("estop", "ws"))
    assert len(ws) == 1 and ws[0]["msg"] == {"type": "cmd", "command": "estop"}
    ui.wait_for_selector(f'{sel("flight-state")}[data-state="ESTOPPED"]', timeout=2000)
    time.sleep(1.0)  # telemetry confirmed it, so no HTTP re-send
    assert mock_backend.commands("estop", "http") == []
    # Reset needs two clicks.
    ui.click(sel("estop-reset-btn"))
    time.sleep(0.2)
    assert mock_backend.commands("estop_reset") == []
    ui.click(sel("estop-reset-btn"))
    mock_backend.wait_for(lambda: mock_backend.commands("estop_reset"))
    ui.wait_for_selector(f'{sel("flight-state")}[data-state="ON_GROUND"]', timeout=2000)


def test_escape_sends_estop_over_http_when_the_socket_is_down(ui, mock_backend):
    mock_backend.control(ws_mode="down")
    ui.wait_for_selector(sel("disconnected-banner"), timeout=3000)
    t0 = mock_backend.now_ms()
    ui.keyboard.press("Escape")
    http = mock_backend.wait_for(lambda: mock_backend.commands("estop", "http", since_ms=t0))
    assert len(http) >= 1 and http[0]["msg"] == {"type": "cmd", "command": "estop"}
    assert mock_backend.commands("estop", "ws") == []
    assert http[0]["t_ms"] - t0 < 1000, "HTTP fallback was slow"
    ui.wait_for_selector(f'{sel("flight-state")}[data-state="ESTOPPED"]', timeout=2000)
    assert ui.get_attribute(sel("last-command"), "data-channel") == "http"
    # Auto-reconnect: the banner goes away by itself once the socket is back.
    mock_backend.control(ws_mode="normal")
    ui.wait_for_selector(sel("disconnected-banner"), state="detached", timeout=5000)


def test_estop_is_resent_over_http_when_the_socket_is_open_but_unread(ui, mock_backend):
    mock_backend.control(ws_mode="deaf")  # accepts messages, never acts on them
    ui.click(sel("estop-btn"))
    ws = mock_backend.wait_for(lambda: mock_backend.commands("estop", "ws"))
    assert ws and ws[0].get("note") == "ignored (deaf)"
    http = mock_backend.wait_for(lambda: mock_backend.commands("estop", "http"), timeout_s=3.0)
    assert http, "no HTTP re-send after telemetry failed to confirm the E-STOP"
    assert 500 <= http[0]["t_ms"] - ws[0]["t_ms"] <= 1500
    ui.wait_for_selector(f'{sel("flight-state")}[data-state="ESTOPPED"]', timeout=2000)
    mock_backend.control(ws_mode="normal")


def test_a_confirmed_estop_is_not_reasserted_after_a_quick_reset(ui, mock_backend):
    """E-STOP confirmed by telemetry, then reset within the 600 ms re-send window: the UI must
    not fire the E-STOP again over HTTP."""
    import httpx

    ui.keyboard.press("Escape")
    ui.wait_for_selector(f'{sel("flight-state")}[data-state="ESTOPPED"]', timeout=2000)
    httpx.post(mock_backend.url + "/command", json={"type": "cmd", "command": "estop_reset"})
    ui.wait_for_selector(f'{sel("flight-state")}[data-state="ON_GROUND"]', timeout=2000)
    time.sleep(1.2)
    assert mock_backend.commands("estop", "http") == [], "E-STOP re-asserted after reset"
    assert len(mock_backend.commands("estop", "ws")) == 1


def test_l_lands_over_the_socket_and_over_http_when_down(ui, mock_backend):
    ui.click(sel("takeoff-btn"))
    ui.wait_for_selector(f'{sel("flight-state")}[data-state="TAKING_OFF"]', timeout=2000)
    ui.keyboard.press("KeyL")
    assert mock_backend.wait_for(lambda: mock_backend.commands("land", "ws"))
    ui.wait_for_selector(f'{sel("flight-state")}[data-state="LANDING"]', timeout=2000)
    ui.click(sel("touchdown-btn"))
    ui.wait_for_selector(f'{sel("flight-state")}[data-state="ON_GROUND"]', timeout=2000)

    mock_backend.control(ws_mode="down")
    ui.wait_for_selector(sel("disconnected-banner"), timeout=3000)
    ui.keyboard.press("KeyL")
    assert mock_backend.wait_for(lambda: mock_backend.commands("land", "http"))
    # The LAND button does the same thing as the key.
    t0 = mock_backend.now_ms()
    ui.click(sel("land-btn"))
    assert mock_backend.wait_for(lambda: mock_backend.commands("land", "http", since_ms=t0))
    mock_backend.control(ws_mode="normal")
    ui.wait_for_selector(sel("disconnected-banner"), state="detached", timeout=5000)


def test_land_and_estop_buttons_are_far_apart(ui):
    land = ui.locator(sel("land-btn")).bounding_box()
    estop = ui.locator(sel("estop-btn")).bounding_box()
    gap = estop["x"] - (land["x"] + land["width"])
    assert gap > 800, f"only {gap:.0f} px between LAND and E-STOP at 1920 px wide"
    assert estop["width"] >= 240 and land["width"] >= 180


# ── REAL / TWIN and the ARM gate ────────────────────────────────────────────────────────────

def test_twin_banner_and_takeoff_without_arm(ui, mock_backend):
    assert ui.get_attribute(sel("mode-banner"), "data-mode") == "twin"
    assert ui.locator(sel("arm-gate")).count() == 0
    assert ui.is_enabled(sel("takeoff-btn"))
    ui.wait_for_selector(sel("chase-view"), timeout=3000)


def test_real_banner_and_arm_gate(ui, mock_backend):
    mock_backend.control(flight_mode="real")
    ui.wait_for_selector(f'{sel("mode-banner")}[data-mode="real"]', timeout=2000)
    assert "REAL DRONE" in ui.inner_text(sel("mode-banner"))
    assert ui.locator(sel("chase-view")).count() == 0, "no twin chase view in real mode"

    # Takeoff is locked until the operator types ARM.
    assert ui.is_disabled(sel("takeoff-btn"))
    assert ui.inner_text(sel("takeoff-blocker")) == "type ARM first"
    ui.click(sel("takeoff-btn"), force=True)  # a click on the disabled button does nothing
    assert ui.is_disabled(sel("arm-btn"))

    ui.click(sel("arm-input"))
    ui.keyboard.type("ar")
    assert ui.input_value(sel("arm-input")) == "AR"
    assert ui.is_disabled(sel("arm-btn"))
    ui.keyboard.type("m")
    assert ui.is_enabled(sel("arm-btn"))
    ui.keyboard.press("Enter")

    arm = mock_backend.wait_for(lambda: mock_backend.commands("arm"))
    assert arm and arm[0]["msg"] == {"type": "cmd", "command": "arm", "arm_phrase": "ARM"}
    ui.wait_for_selector(sel("armed-ok"), timeout=2000)
    assert ui.evaluate("() => document.activeElement.tagName") == "BODY", "ARM box keeps focus"
    assert ui.is_enabled(sel("takeoff-btn"))
    assert mock_backend.commands("takeoff") == []

    ui.click(sel("takeoff-btn"))
    assert mock_backend.wait_for(lambda: mock_backend.commands("takeoff"))
    ui.wait_for_selector(f'{sel("flight-state")}[data-state="TAKING_OFF"]', timeout=2000)


def test_stick_keys_are_ignored_while_typing_arm(ui, mock_backend):
    mock_backend.control(flight_mode="real")
    ui.wait_for_selector(sel("arm-input"), timeout=2000)
    ui.click(sel("arm-input"))
    ui.keyboard.type("wasdl")  # would be sticks and LAND outside the box
    time.sleep(0.3)
    assert mock_backend.of_type("axes") == []
    assert mock_backend.commands("land") == []
    ui.keyboard.press("Escape")  # Esc still E-STOPs, even from the text box
    assert mock_backend.wait_for(lambda: mock_backend.commands("estop"))


# ── Candidates, pack timer, panels ──────────────────────────────────────────────────────────

def test_candidate_picker_sends_select(ui, mock_backend):
    mock_backend.control(flight_state="AIRBORNE", mission="default", autopilot_state="CONFIRM", freeze_autopilot=True)
    ui.wait_for_selector(f'{sel("autopilot-state")}[data-state="CONFIRM"]', timeout=2000)
    mock_backend.candidates()
    ui.wait_for_selector(sel("candidate-2"), timeout=2000)
    ui.wait_for_selector(sel("candidate-box-2"), timeout=2000)
    ui.click(sel("candidate-2"))
    sel_msgs = mock_backend.wait_for(lambda: mock_backend.of_type("select", "agent"))
    assert sel_msgs and sel_msgs[0]["msg"] == {"type": "select", "mission_id": "m001", "index": 2}


def test_mask_and_box_follow_the_subject_colored_by_state(ui, mock_backend):
    mock_backend.control(flight_state="AIRBORNE", mission="default", autopilot_state="FOLLOW", freeze_autopilot=True)
    ui.wait_for_selector(f'{sel("subject-box")}[data-state="FOLLOW"]', timeout=2000)
    fill = ui.get_attribute(sel("subject-mask"), "fill")
    assert fill == "#22c55e"  # FOLLOW = green
    mock_backend.control(autopilot_state="DOUBT")
    ui.wait_for_selector(f'{sel("subject-box")}[data-state="DOUBT"]', timeout=2000)
    assert ui.get_attribute(sel("subject-mask"), "fill") == "#fb923c"  # DOUBT = orange


def test_pack_timer_warns_and_pack_swap_only_on_ground(ui, mock_backend):
    mock_backend.control(pack_time_s=305)
    ui.wait_for_selector(f'{sel("pack-timer")}[data-level="amber"]', timeout=2000)
    mock_backend.control(pack_time_s=370)
    ui.wait_for_selector(f'{sel("pack-timer")}[data-level="red"]', timeout=2000)
    ui.click(sel("pack-2"))
    swap = mock_backend.wait_for(lambda: mock_backend.commands("pack_swap"))
    assert swap and swap[0]["msg"] == {"type": "cmd", "command": "pack_swap", "pack_id": "2"}
    ui.wait_for_selector(f'{sel("pack-timer")}[data-level="ok"]', timeout=2000)
    mock_backend.control(flight_state="AIRBORNE")
    ui.wait_for_selector(f'{sel("flight-state")}[data-state="AIRBORNE"]', timeout=2000)
    assert ui.is_disabled(sel("pack-1"))


def test_footer_credits_turbodrone(ui):
    assert "UI shell from turbodrone (Apache-2.0)" in ui.inner_text(sel("footer"))


def test_telemetry_loss_shows_unknown_and_locks_takeoff(ui, browser_ctx):
    """A page whose telemetry socket can never connect: state UNKNOWN, takeoff locked,
    LAND and E-STOP still available."""
    page = browser_ctx.new_page()
    page.add_init_script("""
        const RealWS = window.WebSocket;
        window.WebSocket = function (url, protocols) {
            if (String(url).includes('/ws/telemetry')) url = 'ws://127.0.0.1:9/ws/telemetry';
            return new RealWS(url, protocols);
        };
        window.WebSocket.prototype = RealWS.prototype;
        Object.assign(window.WebSocket, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });
    """)
    page.goto(ui.url)
    page.wait_for_selector(sel("telemetry-banner"), timeout=5000)
    assert page.get_attribute(sel("flight-state"), "data-state") == "UNKNOWN"
    assert page.get_attribute(sel("mode-banner"), "data-mode") == "unknown"
    assert page.is_disabled(sel("takeoff-btn"))
    assert page.is_enabled(sel("land-btn")) and page.is_enabled(sel("estop-btn"))
    page.close()


# ── Agent and governor events ───────────────────────────────────────────────────────────────

def test_agent_and_governor_events_go_to_the_right_place(ui, mock_backend):
    ui.wait_for_selector(sel("agent-health"), timeout=3000)  # "agent ready" on connect
    assert "listener on" in ui.inner_text(sel("agent-health"))
    assert "agent alive" not in ui.inner_text(sel("transcript"))
    assert "agent ready" not in ui.inner_text(sel("transcript"))

    # A governor transition, as the backend hub relays it, lands in the Safety feed only.
    mock_backend.control(agent_event={"type": "agent_event", "kind": "status",
                                      "text": "autopilot FOLLOW -> DOUBT (obs_age)",
                                      "data": {"kind": "state", "from": "FOLLOW", "to": "DOUBT"}})
    ui.wait_for_selector(f'{sel("system-events")} >> text=FOLLOW -> DOUBT', timeout=2000)
    assert "DOUBT" not in ui.inner_text(sel("transcript"))

    # The local listener heard a safety word.
    mock_backend.control(agent_event={"type": "agent_event", "kind": "safety_word", "text": "land",
                                      "data": {"command": "land", "action": "land", "decode_ms": 140.0}})
    ui.wait_for_selector(f'{sel("transcript")} >> text=heard “land” → land', timeout=2000)

    # An intent from the agent (its real shape has no mission_id) shows before telemetry has it.
    mock_backend.control(agent_event={"type": "agent_event", "kind": "intent", "text": "refuse", "data": {
        "intent": {"action": "refuse", "subject_phrase": "", "category": "none", "color": "none",
                   "distance": "medium", "reason_code": "identity_request"},
        "model": "muse-spark-1.3", "latency_ms": 2310.0, "reason": "spark answered first",
        "transcript": "follow my friend dave"}})
    ui.wait_for_selector(f'{sel("model-badge")} >> text=muse-spark-1.3', timeout=2000)
    assert "identity_request" in ui.inner_text(sel("intent-json"))
    assert "2.31 s" in ui.inner_text(sel("mission-latency"))
