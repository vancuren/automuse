# Conventions every component follows

- **Subject, not target.** The person or object being followed is the "subject" in code, UI, and copy.
- **Pixels.** Boxes are always in a 1280x720 frame. Work smaller if you like, but convert at the boundary.
- **Clock.** `mhd.contracts.now_ms()` (time.monotonic in ms). Shared by all processes on this Mac. Wall-clock time appears only in log file names.
- **Sticks.** Normalized -1..1. +yaw turns clockwise seen from above (nose right). +pitch flies forward. +roll moves right. +throttle climbs. Wire counts are 0..255, center 128. The governor alone converts normalized to wire, applying per-axis sign, deadband compensation, caps, and symmetric rounding (params in `config/governor.yaml`). The X69 speed tier is pinned at full scale (index 2), so wire counts are exact.
- **Bearing.** beta = atan((cx - 640) / f) radians, f = 640 / tan(HFOV/2). +beta means the subject is right of center, so +yaw corrects it.
- **Frames.** `/ws/frames` is pull-based: the consumer sends `{"type":"next","after_frame_id":N}` and receives one binary frame (see `mhd.contracts.framing`). One frame in flight, newest only.
- **Modes.** `FLIGHT_MODE=twin|real` is required. Twin mode refuses non-loopback `DRONE_IP`.
- **Paths.** Source `env.sh`. Code runs with `PYTHONPATH=$MHD_TREE/src`, interpreters `$MHD_PY` (core) and `$MHD_ML_PY` (ml). Models live in `$MHD_MODELS`, tools in `$MHD_TOOLS`. Nothing is written outside the project folder.
