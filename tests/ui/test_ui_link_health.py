"""Lewei board link health in the UI. The hub raises drone_link_stale when the drone stops replying (3 s,
8 s while LANDING): TAKEOFF is disabled on it, LAND and E-STOP never are. The battery volts stop looking
live once the drone's last reply is over 3 s old. Source-level checks that need no browser; the browser
checks are test_ui_browser.py::test_drone_link_stale_blocks_takeoff_but_never_land_or_estop and
test_ui_browser.py::test_lewei_battery_volts_grey_out_when_the_drone_stops_replying."""
from __future__ import annotations

import re
from pathlib import Path

from mhd.backend.hub import DRONE_LINK_STALE_FLAG, DRONE_LINK_STALE_MS

SRC = Path(__file__).resolve().parents[2] / "turbodrone" / "frontend" / "src"
SOURCES = sorted(p for p in SRC.rglob("*") if p.suffix in {".ts", ".tsx"})


def _blocking_flags() -> dict[str, str]:
    gates = (SRC / "lib" / "gates.ts").read_text()
    block = re.search(r"TAKEOFF_BLOCKING_FLAGS[^=]*= \{(.*?)\};", gates, re.S).group(1)
    return dict(re.findall(r'^\s*(\w+):\s*"([^"]*)",', block, re.M))


def _button(src: str, testid: str) -> str:
    """The whole <UiButton ...>...</UiButton> element with this data-testid."""
    i = src.index(f'data-testid="{testid}"')
    return src[src.rindex("<UiButton", 0, i):src.index("</UiButton>", i)]


def test_drone_link_stale_disables_takeoff_under_the_backends_flag_name():
    flags = _blocking_flags()
    assert flags.get(DRONE_LINK_STALE_FLAG) == "drone link lost"
    assert {"video_stale", "heartbeat_missing", "governor_fault", "rc_tick_stale"} <= set(flags)


def test_only_the_takeoff_button_reads_the_blocking_flags():
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", (SRC / "lib" / "gates.ts").read_text(), flags=re.S)
    assert "export const TAKEOFF_BLOCKING_FLAGS" not in code     # module-private...
    decl = code.index("const TAKEOFF_BLOCKING_FLAGS") + len("const ")
    fn = code.index("export function takeoffBlocker(")
    uses = [m.start() for m in re.finditer("TAKEOFF_BLOCKING_FLAGS", code)]
    assert len(uses) > 1 and all(u == decl or u > fn for u in uses)   # ...and read only by takeoffBlocker
    users = sorted(p.relative_to(SRC).as_posix() for p in SOURCES if "takeoffBlocker" in p.read_text())
    assert users == ["components/CommandBar.tsx", "lib/gates.ts"]
    bar = (SRC / "components" / "CommandBar.tsx").read_text()
    assert bar.count("takeoffBlocker(") == 1
    assert "disabled={blocker !== null}" in _button(bar, "takeoff-btn")
    land, estop = _button(bar, "land-btn"), _button(bar, "estop-btn")
    for el in (land, estop):
        assert "disabled" not in el and "blocker" not in el and "safety_flags" not in el
    assert "onPointerDown={onLand}" in land and "onPointerDown={onEstop}" in estop
    hotkeys = (SRC / "hooks" / "useHotkeys.ts").read_text()   # Esc and L: no gate either
    assert "gates" not in hotkeys and "safety_flags" not in hotkeys and "blocker" not in hotkeys.lower()


def test_battery_volts_are_not_shown_as_live_once_the_drone_stops_replying():
    fs = (SRC / "components" / "FlightStatus.tsx").read_text()
    assert f"const FC_STALE_MS = {int(DRONE_LINK_STALE_MS)};" in fs
    assert "telemetry?.latency_ms?.fc_telemetry_age" in fs and "fcAge > FC_STALE_MS" in fs
    old = fs[fs.index('data-old="1"'):fs.index('data-old="0"')]
    shown = old.split("title=")[0] + old.split("Lewei board)`}")[1]
    assert "— V" in shown and "volts" not in shown            # a grey dash, never the last reading
    assert "text-slate-500" in old and "amber" not in old
    assert 'data-testid="fc-battery-age"' in old and "{fcOldFor} ago" in old
    fresh = fs[fs.index('data-old="0"'):]
    assert 'volts < 3.5 ? "text-amber-300"' in fresh.split("</span>")[0]   # amber below 3.5 V while fresh
