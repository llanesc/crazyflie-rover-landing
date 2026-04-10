"""Policy modules for drone and rover agents.

Lazy imports to avoid pulling in acados/leap_c when only MLP is needed.
"""


def __getattr__(name):
    if name == "DroneACMPCGaussianPolicy":
        from crazyflie_rover_landing.policies.drone_policy_linear_ls import DroneACMPCGaussianPolicy
        return DroneACMPCGaussianPolicy
    elif name == "MLPGaussianPolicy":
        from crazyflie_rover_landing.policies.mlp_policy import MLPGaussianPolicy
        return MLPGaussianPolicy
    elif name == "SharedCritic":
        from crazyflie_rover_landing.policies.shared_critic import SharedCritic
        return SharedCritic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DroneACMPCGaussianPolicy",
    "MLPGaussianPolicy",
    "SharedCritic",
]
