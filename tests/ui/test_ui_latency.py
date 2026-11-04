"""Latency measurements for the operator UI (loopback, mock backend, headless Chrome).

time.monotonic is one clock for every process on this Mac (docs/conventions.md), so the
pytest process and the mock's receive stamps can be compared directly. Results go to
logs/ui_tests/measurements.json; the asserts are loose regression bounds, not targets.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import time

import httpx

from conftest import LOG_DIR


def sel(testid: str) -> str:
    return f'[data-testid="{testid}"]'


def now_ms() -> float:
    return time.monotonic() * 1000.0


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summary(xs: list[float]) -> dict:
    return {"n": len(xs), "p50_ms": round(pct(xs, 50), 1), "p95_ms": round(pct(xs, 95), 1),
            "max_ms": round(max(xs), 1), "mean_ms": round(statistics.mean(xs), 1)}


def reset_estop(ui, mock) -> None:
    httpx.post(mock.url + "/command", json={"type": "cmd", "command": "estop_reset"})
    ui.wait_for_selector(f'{sel("flight-state")}[data-state="ON_GROUND"]', timeout=3000)
    time.sleep(0.7)  # let the UI's 600 ms confirmation window close


def key_to_wire(ui, mock, channel: str, n: int = 10) -> list[float]:
    out = []
    for _ in range(n):
        before = len(mock.commands("estop", channel))
        t0 = now_ms()
        ui.keyboard.press("Escape")
        got = mock.wait_for(lambda: mock.commands("estop", channel), min_count=before + 1)
        out.append(got[before]["t_ms"] - t0)
        ui.wait_for_selector(f'{sel("flight-state")}[data-state="ESTOPPED"]', timeout=3000)
        reset_estop(ui, mock)
    return out


def test_measure_latencies(ui, mock_backend):
    results: dict = {"conditions": {
        "what": "loopback 127.0.0.1, mock backend, vite dev server, headless Google Chrome 157",
        "load_avg": os.getloadavg(),
        "note": "other overnight workstreams (twin Chrome/ffmpeg, percep/train Python) were running",
    }}

    # 1) Esc -> E-STOP arrives over the control socket
    results["esc_to_estop_ws"] = summary(key_to_wire(ui, mock_backend, "ws"))

    # 2) Esc -> E-STOP arrives over HTTP POST /command while the socket is down
    mock_backend.control(ws_mode="down")
    ui.wait_for_selector(sel("disconnected-banner"), timeout=3000)
    results["esc_to_estop_http_socket_down"] = summary(key_to_wire(ui, mock_backend, "http"))
    t_up = now_ms()
    mock_backend.control(ws_mode="normal")
    ui.wait_for_selector(sel("disconnected-banner"), state="detached", timeout=5000)
    results["control_socket_reconnect_ms"] = round(now_ms() - t_up, 1)

    # 3) backend state change -> in the DOM. A MutationObserver stamps the change on the wall
    #    clock in the page (Playwright's own polling would add up to 100 ms).
    ui.evaluate("""() => {
        window.__changes = [];
        const el = document.querySelector('[data-testid="flight-state"]');
        new MutationObserver(() => __changes.push([el.dataset.state, performance.timeOrigin + performance.now()]))
            .observe(el, { attributes: true, attributeFilter: ['data-state'] });
    }""")
    shown = []
    for i in range(20):
        state = "AIRBORNE" if i % 2 == 0 else "ON_GROUND"
        time.sleep(random.uniform(0.0, 0.1))  # random phase against the 10 Hz telemetry tick
        t0 = time.time() * 1000.0
        mock_backend.control(flight_state=state)
        ui.wait_for_function(f"() => __changes.some(c => c[0] === '{state}' && c[1] >= {t0})", timeout=3000)
        t_dom = ui.evaluate(f"() => __changes.find(c => c[0] === '{state}' && c[1] >= {t0})[1]")
        shown.append(t_dom - t0)
    results["backend_state_to_dom"] = summary(shown) | {
        "includes": "HTTP control call, wait for the next 10 Hz telemetry tick (0-100 ms), WS, React render"}

    # 4) heartbeat inter-arrival at the backend, over 5 s
    t0 = mock_backend.now_ms()
    time.sleep(5.0)
    hbs = mock_backend.of_type("hb", "ws", since_ms=t0)
    gaps = [b["t_ms"] - a["t_ms"] for a, b in zip(hbs, hbs[1:])]
    results["heartbeat_interarrival"] = summary(gaps) | {"rate_hz": round(len(hbs) / 5.0, 2)}

    # 5) axes rate while a stick key is held for 2 s
    t0 = mock_backend.now_ms()
    ui.keyboard.down("KeyA")
    time.sleep(2.0)
    ui.keyboard.up("KeyA")
    axes = mock_backend.of_type("axes", "ws", since_ms=t0)
    results["axes_rate_hz_while_held"] = round(len(axes) / 2.0, 1)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "measurements.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    assert results["esc_to_estop_ws"]["p95_ms"] < 150
    assert results["esc_to_estop_http_socket_down"]["p95_ms"] < 300
    assert results["backend_state_to_dom"]["p95_ms"] < 250
    assert results["heartbeat_interarrival"]["max_ms"] < 400
    assert 25 <= results["axes_rate_hz_while_held"] <= 33
