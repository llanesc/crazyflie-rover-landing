"""Generic LandingEnvConfig field overrides from key=value CLI strings."""

from __future__ import annotations

from crazyflie_rover_landing.envs.landing_config import LandingEnvConfig


def apply_overrides(env_cfg: LandingEnvConfig, overrides: list[str]) -> dict:
    """Apply key=value overrides to env_cfg. Returns dict of applied overrides."""
    applied = {}
    for item in overrides:
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        if not hasattr(env_cfg, key):
            raise ValueError(f"Unknown LandingEnvConfig field: {key}")
        current = getattr(env_cfg, key)
        if isinstance(current, bool):
            parsed = val.lower() in ("true", "1", "yes")
        elif isinstance(current, int):
            parsed = int(val)
        elif isinstance(current, float) or current is None:
            parsed = float(val)
        elif isinstance(current, str):
            parsed = val
        else:
            raise ValueError(
                f"Cannot auto-coerce override for field {key} (type={type(current).__name__})"
            )
        setattr(env_cfg, key, parsed)
        applied[key] = parsed
    return applied
