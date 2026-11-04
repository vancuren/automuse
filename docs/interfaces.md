# Process interfaces (v0.1)

All messages are the pydantic models in `src/mhd/contracts`. JSON over WebSocket unless noted. Ports are defaults; every service takes `--port`/`--backend` flags so parallel test runs never collide.

## drone-backend (forked turbodrone FastAPI), default :8000
| Endpoint | Direction | Messages |
|---|---|---|
| `WS /ws` (patched upstream path) | UI -> backend | `OperatorInput` (`type:"axes"`, sent only while nonzero), `Heartbeat` (`type:"hb"`, 5 Hz from a Web Worker), `OperatorCommand` (`type:"cmd"`). Legacy upstream messages (`takeoff`, `land`, `estop`, `set_camera_tilt`) map to governor commands; `set_profile` and `set_speed_index` are ignored. |
| `POST /command` | UI -> backend | `OperatorCommand`; HTTP fallback for LAND and E-STOP. Returns `{"ok": true, "flight_state": "..."}`. |
| `WS /ws/telemetry` | backend -> UI | `Telemetry` at 10 Hz. |
| `GET /state` | any | latest `Telemetry` as JSON. |
| `GET /mjpeg` | backend -> UI | upstream MJPEG stream for display. |
| `WS /ws/frames` | perception <-> backend | client sends `NextFrame`; server replies with one binary frame (`mhd.contracts.framing`). Newest only, one in flight. |
| `WS /ws/subject` | perception <-> backend | server -> client: `FindRequest`, `Release`, `Select`. client -> server: `SubjectObs` (camera rate), `Candidates`. |
| `WS /ws/agent?role=agent` / `?role=ui` | agent <-> backend <-> UI | `AgentEvent` relayed between roles. agent -> server: `Mission` (or `POST /mission`), `OperatorCommand` for safety words. ui -> server: `AgentEvent` kind `ptt_down`/`ptt_up` (relayed to agent), `Select` (relayed to perception). |
| `POST /mission` | agent -> backend | `Mission`. |

Environment: `FLIGHT_MODE=twin|real` (required), `DRONE_TYPE=x69_lg`, `DRONE_IP` (twin: 127.0.0.1 only; real: 172.16.11.1 only), `CONTROL_PORT` (twin may use a non-default UDP port), `X69_LG_VIDEO_MODE=rtsp`, `X69_LG_RTSP_URL`, `PLUGINS_ENABLED=false`, `GOVERNOR_CONFIG=config/governor.yaml`.

## twin, default HTTP/WS :8766, UDP :23458, RTSP :8554
| Endpoint | Messages |
|---|---|
| UDP `127.0.0.1:23458` | X69 RC datagrams in (28-byte control + 8-byte keepalive), exactly as the real drone receives them. |
| `rtsp://127.0.0.1:8554/live/ch00_1` | HEVC 1280x720 out (mediamtx, loopback only), with the counter block stamped in the bottom-right corner. |
| `GET /` and `GET /chase` | renderer page (drone camera) and chase view for the UI and demo video. |
| `WS /ws/render` | twin <-> renderer: pose/scene state out, JPEG frames in (pull, one in flight). |
| `WS /ws/truth` | `Truth` per rendered frame. |
| `POST /control` | `TwinControl` (scenario, reset, total latency target, drift, RC loss, stalls). |
| `GET /status` | twin health: fps, measured pipeline baseline latency, RC inter-arrival p50/p99, link state. |

## perception (client only)
`$MHD_ML_PY -m mhd_ml.perception.service --backend ws://127.0.0.1:8000 --detector sam_mlx|sam_api --tracker sam_mlx|vit|kcf|csrt`

## agent (client only)
`$MHD_ML_PY -m mhd_ml.agent.service --backend ws://127.0.0.1:8000 --asr meta|local --intent auto|glimmer|spark|keyword --glimmer-url http://127.0.0.1:8080/v1`

## glimmer server (edge mode / local eval)
`$MHD_ML_PY -m mlx_lm.server --model $MHD_MODELS/Muse-Glimmer-30B-4bit [--adapter-path $MHD_MODELS/adapters/glimmer-intent-v1] --port 8080` (OpenAI-compatible). Only one Glimmer process may run at a time (about 17-25 GB of memory).
