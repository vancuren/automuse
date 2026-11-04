"""The UI's TypeScript mirror of the contracts (src/lib/contracts.ts) must match the frozen
pydantic models in src/mhd/contracts field for field and literal for literal."""
from __future__ import annotations

import re
import typing
from enum import Enum
from pathlib import Path

import pytest

from mhd import contracts as C

TS = (Path(__file__).resolve().parents[2] / "turbodrone/frontend/src/lib/contracts.ts").read_text()

MIRRORED = ["Box", "SubjectObs", "Candidate", "Candidates", "Select", "OperatorInput", "Heartbeat",
            "OperatorCommand", "Intent", "Mission", "AgentEvent", "Telemetry"]


def ts_interface_fields(name: str) -> dict[str, str]:
    m = re.search(rf"export interface {name} \{{(.*?)\n\}}", TS, re.S)
    assert m, f"interface {name} missing from contracts.ts"
    fields = {}
    for line in m.group(1).splitlines():
        fm = re.match(r"\s+(\w+)\??:\s*(.+?);", line)
        if fm:
            fields[fm.group(1)] = fm.group(2)
    return fields


def ts_const_array(name: str) -> list[str]:
    m = re.search(rf"export const {name} = \[(.*?)\] as const;", TS, re.S)
    assert m, f"const {name} missing"
    return re.findall(r'"([^"]+)"', m.group(1))


def literal_values(annotation) -> list[str]:
    """Literal[...] values, unwrapping Optional."""
    args = typing.get_args(annotation)
    if typing.get_origin(annotation) is typing.Literal:
        return list(args)
    for a in args:
        if typing.get_origin(a) is typing.Literal:
            return list(typing.get_args(a))
    raise AssertionError(f"not a Literal: {annotation}")


def ts_string_union(type_text: str) -> list[str]:
    return re.findall(r'"([^"]+)"', type_text)


@pytest.mark.parametrize("name", MIRRORED)
def test_field_names_match(name):
    model = getattr(C, name)
    assert set(ts_interface_fields(name)) == set(model.model_fields), name


def test_command_names_match():
    assert ts_const_array("COMMAND_NAMES") == literal_values(C.OperatorCommand.model_fields["command"].annotation)


def test_agent_event_kinds_match():
    assert ts_const_array("AGENT_EVENT_KINDS") == literal_values(C.AgentEvent.model_fields["kind"].annotation)


@pytest.mark.parametrize("const,enum", [("FLIGHT_STATES", C.FlightState), ("AUTOPILOT_STATES", C.AutopilotState)])
def test_state_enums_match(const, enum: type[Enum]):
    assert ts_const_array(const) == [e.value for e in enum]


@pytest.mark.parametrize("alias,enum", [("SubjectState", C.SubjectState), ("ObsSource", C.ObsSource)])
def test_union_aliases_match(alias, enum):
    m = re.search(rf'export type {alias} = (.+?);', TS)
    assert m and ts_string_union(m.group(1)) == [e.value for e in enum]


@pytest.mark.parametrize("model,field", [
    ("Intent", "action"), ("Intent", "category"), ("Intent", "distance"), ("Intent", "reason_code"),
    ("OperatorInput", "mode"), ("Telemetry", "flight_mode"),
])
def test_literal_fields_match(model, field):
    ts_type = ts_interface_fields(model)[field]
    py = literal_values(getattr(C, model).model_fields[field].annotation)
    assert ts_string_union(ts_type) == py


def test_frame_size_and_version():
    assert f'CONTRACT_VERSION = "{C.CONTRACT_VERSION}"' in TS
    assert f"FRAME_W = {C.FRAME_W}" in TS and f"FRAME_H = {C.FRAME_H}" in TS
