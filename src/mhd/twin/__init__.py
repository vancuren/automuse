"""Wire-level digital twin of the Loiley X69 (see PLAN.md section 4.2 and docs/ws/twin.md).

Modules:
- protocol:   decode the X69 RC datagrams exactly as turbodrone's x69_lg adapter builds them.
- dynamics:   100 Hz kinematic model of the drone (params in config/twin/x69.yaml).
- scenarios:  actors and world layout (config/twin/scenarios/*.yaml) plus ground-truth projection.
- counter:    corner counter block (render id and render time) that survives HEVC.
- server:     asyncio process: UDP RC in, renderer bridge, HEVC/RTSP out, truth and control API.
- metrics:    join Truth with SubjectObs/Telemetry logs by render id.
"""
