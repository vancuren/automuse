"""Safety governor for the X69: owns every RC packet (PLAN.md 4.5). Pure Python: no MLX, no torch.

- config:          strict loaders for config/governor.yaml and config/autopilot.yaml
- core.Governor:   flight state machine, priorities, staleness checks, caps, flags (tick at 25 Hz)
- turbodrone_glue: GovernorStrategy (the per-packet choke point) and the guarded X69 model
                   (import separately; it needs turbodrone/backend on sys.path)
- telemetry:       contract Telemetry builder
"""
from .config import (  # noqa: F401
    AutopilotConfig, ConfigError, GovernorConfig, autopilot_config_from_dict, governor_config_from_dict,
    load_autopilot_config, load_governor_config,
)
from .core import CommandResult, Governor, GovernorError, TickOutput  # noqa: F401
