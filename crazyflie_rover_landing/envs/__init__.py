"""Environment modules for the drone-rover landing task."""

from crazyflie_rover_landing.envs.landing_config import LandingEnvConfig

__all__ = ["LandingEnvConfig", "LandingEnv", "RescaleActionWrapper"]


def __getattr__(name):
    if name == "LandingEnv":
        from crazyflie_rover_landing.envs.landing_env import LandingEnv
        return LandingEnv
    if name == "RescaleActionWrapper":
        from crazyflie_rover_landing.envs.wrappers import RescaleActionWrapper
        return RescaleActionWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
