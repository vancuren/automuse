"""Message contracts v0.1 (pydantic v2).

Conventions (also in docs/conventions.md):
- Pixel space: boxes are always in a 1280x720 frame (FRAME_W x FRAME_H). Scale if you work smaller.
- Clock: `now_ms()` = time.monotonic() * 1000. It is shared by all processes on this Mac.
- Normalized sticks are -1..1. +yaw = turn clockwise seen from above (nose right). +pitch = forward.
  +roll = right. +throttle = up. Wire counts are 0..255 with 128 = center. Per-axis sign and
  deadband live in the governor's params, not here.
- Bearing: +beta = subject right of image center, in radians.
- The person or object being followed is the "subject" (never "target") in code and copy.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "0.1"
FRAME_W = 1280
FRAME_H = 720

__all__ = [
    "CONTRACT_VERSION", "FRAME_W", "FRAME_H",
    "Box", "FrameMeta", "SubjectState", "ObsSource", "FindRequest", "Release", "SubjectObs",
    "Candidate", "Candidates", "Select", "OperatorInput", "Heartbeat", "OperatorCommand",
    "Intent", "Mission", "AgentEvent", "FlightState", "AutopilotState", "Telemetry",
    "TruthActor", "Truth", "TwinControl", "NextFrame",
]


class _Msg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Box(_Msg):
    """Axis-aligned box in 1280x720 pixels."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def iou(self, other: "Box") -> float:
        ix = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        inter = ix * iy
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union > 0 else 0.0


class FrameMeta(_Msg):
    """Header of a binary frame message (see framing.py). backend -> perception."""
    type: Literal["frame"] = "frame"
    frame_id: int
    capture_ms: float          # best estimate of camera capture time (mono ms)
    receive_ms: float          # when the backend had the decoded frame (mono ms)
    width: int
    height: int
    render_id: Optional[int] = None   # twin counter-block render id, when decoded
    flight_mode: Literal["twin", "real"] = "twin"


class NextFrame(_Msg):
    """Pull request: perception asks for the newest frame newer than `after_frame_id`."""
    type: Literal["next"] = "next"
    after_frame_id: int = -1


class SubjectState(str, Enum):
    searching = "searching"
    candidate = "candidate"
    confirmed = "confirmed"
    lost = "lost"


class ObsSource(str, Enum):
    sam_api = "sam_api"
    sam_mlx = "sam_mlx"
    tracker = "tracker"
    oracle = "oracle"


class FindRequest(_Msg):
    """backend -> perception: start finding/following a subject described in words."""
    type: Literal["find"] = "find"
    mission_id: str
    subject_phrase: str        # SAM-ready noun phrase, <= 4 words, e.g. "person in blue shirt"
    category: Literal["person", "car", "animal", "object", "none"] = "none"
    color: str = "none"


class Release(_Msg):
    type: Literal["release"] = "release"
    mission_id: str


class SubjectObs(_Msg):
    """perception -> backend, at camera rate. Evidence only; the autopilot owns state changes."""
    type: Literal["subject_obs"] = "subject_obs"
    mission_id: Optional[str] = None
    frame_id: int
    capture_ms: float                      # capture time of the frame this observation describes
    state: SubjectState
    track_id: Optional[int] = None
    box: Optional[Box] = None
    confidence: float = 0.0                # detector confidence at the last confirmation
    tracker_score: float = 0.0             # 0..1 from the in-loop tracker (1 for detector frames)
    confirm_age_ms: float = 1e9            # age of the last detector confirmation, at capture_ms
    mask_poly: Optional[list[tuple[float, float]]] = None   # simplified polygon, pixels
    source: ObsSource
    detector_latency_ms: Optional[float] = None
    render_id: Optional[int] = None        # twin only, for joining with Truth


class Candidate(_Msg):
    index: int
    box: Box
    confidence: float
    color_score: float = 0.0


class Candidates(_Msg):
    """perception -> backend -> UI/agent when look-alikes tie."""
    type: Literal["candidates"] = "candidates"
    mission_id: str
    frame_id: int
    items: list[Candidate]


class Select(_Msg):
    type: Literal["select"] = "select"
    mission_id: str
    index: int


class OperatorInput(_Msg):
    """UI -> backend. Sent only while nonzero. All-zero messages are not input."""
    type: Literal["axes"] = "axes"
    throttle: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    mode: Literal["abs", "inc"] = "abs"
    ts_ms: Optional[float] = None


class Heartbeat(_Msg):
    """UI -> backend at 5 Hz from a Web Worker."""
    type: Literal["hb"] = "hb"
    client: str = "ui"
    ts_ms: Optional[float] = None


class OperatorCommand(_Msg):
    """UI -> backend (WebSocket, or HTTP POST /command as fallback)."""
    type: Literal["cmd"] = "cmd"
    command: Literal[
        "takeoff", "land", "estop", "estop_reset", "pause", "resume", "confirm_touchdown",
        "arm", "disarm", "tilt_up", "tilt_down", "tilt_stop", "pack_swap",
    ]
    arm_phrase: Optional[str] = None
    pack_id: Optional[str] = None


class Intent(_Msg):
    """Strict schema the intent model must emit (about 50 output tokens)."""
    action: Literal["follow", "hover", "land", "clarify", "refuse"]
    subject_phrase: str
    category: Literal["person", "car", "animal", "object", "none"]
    color: str
    distance: Literal["close", "medium", "far"]
    reason_code: Literal["none", "ambiguous", "identity_request", "unsafe", "unsupported"]


class Mission(_Msg):
    """agent -> backend."""
    type: Literal["mission"] = "mission"
    mission_id: str
    intent: Intent
    transcript: str
    model: str                 # "glimmer-30b+lora", "glimmer-30b", "muse-spark-1.3", "keyword"
    latency_ms: float
    reply: Optional[str] = None    # spoken line, built locally from the intent


class AgentEvent(_Msg):
    """agent <-> backend <-> UI."""
    type: Literal["agent_event"] = "agent_event"
    kind: Literal["ptt_down", "ptt_up", "partial", "final", "intent", "reply", "error",
                  "safety_word", "status"]
    text: Optional[str] = None
    data: Optional[dict] = None
    ts_ms: Optional[float] = None


class FlightState(str, Enum):
    on_ground = "ON_GROUND"
    taking_off = "TAKING_OFF"
    airborne = "AIRBORNE"
    landing = "LANDING"
    estopped = "ESTOPPED"


class AutopilotState(str, Enum):
    idle = "IDLE"
    find = "FIND"
    confirm = "CONFIRM"
    follow = "FOLLOW"
    doubt = "DOUBT"
    lost = "LOST"
    search = "SEARCH"
    give_up = "GIVE_UP"


class Telemetry(_Msg):
    """backend -> UI at 10 Hz."""
    type: Literal["telemetry"] = "telemetry"
    ts_ms: float
    flight_mode: Literal["twin", "real"]
    flight_state: FlightState
    autopilot_state: AutopilotState
    armed: bool
    sticks_wire: dict[str, int]                       # final 0..255 counts sent this tick
    sticks_by_source: dict[str, dict[str, float]]     # normalized, per source
    active_source: str
    frame_age_ms: Optional[float] = None
    obs_age_ms: Optional[float] = None
    latency_ms: dict[str, float] = Field(default_factory=dict)
    flight_time_s: float = 0.0
    pack_id: Optional[str] = None
    safety_flags: list[str] = Field(default_factory=list)
    mission: Optional[Mission] = None
    subject: Optional[SubjectObs] = None


class TruthActor(_Msg):
    actor_id: str
    label: str                  # e.g. "person in blue shirt", "green car"
    box: Optional[Box] = None   # projected into the drone camera; None if off-screen
    visible_frac: float = 0.0
    distance_m: float = 0.0
    bearing_rad: float = 0.0


class Truth(_Msg):
    """twin -> metrics/oracle: ground truth for one rendered frame."""
    type: Literal["truth"] = "truth"
    render_id: int
    render_ms: float
    drone: dict[str, float]     # x, y, z, yaw, pitch, roll, vx, vy, vz (m, rad, m/s)
    actors: list[TruthActor]


class TwinControl(_Msg):
    type: Literal["twin_control"] = "twin_control"
    scenario: Optional[str] = None
    reset: bool = False
    latency_ms: Optional[float] = None     # target TOTAL camera-to-backend latency
    drift: Optional[float] = None
    rc_loss: Optional[float] = None
    stall: Optional[dict] = None           # {"every_s": 20, "duration_ms": 1500}
