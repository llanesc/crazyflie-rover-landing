"""LEAP-C OCP definitions for drone and rover.

Lazy imports to avoid pulling in acados when not needed.
"""


def __getattr__(name):
    if name in ("DRONE_NX_EULER", "DRONE_NX_QUAT", "DRONE_NU",
                "create_drone_params_linear_ls", "export_drone_ocp_linear_ls"):
        from crazyflie_rover_landing.leap_c import drone_ocp_linear_ls as _m
        mapping = {
            "DRONE_NX_EULER": _m.NX_EULER,
            "DRONE_NX_QUAT": _m.NX_QUAT,
            "DRONE_NU": _m.NU,
            "create_drone_params_linear_ls": _m.create_drone_params_linear_ls,
            "export_drone_ocp_linear_ls": _m.export_drone_ocp_linear_ls,
        }
        return mapping[name]
    elif name in ("NX_ROVER", "NU_ROVER", "NY_ROVER",
                  "create_rover_params_linear_ls", "export_rover_ocp_linear_ls"):
        from crazyflie_rover_landing.leap_c import rover_ocp_linear_ls as _m
        mapping = {
            "NX_ROVER": _m.NX_ROVER,
            "NU_ROVER": _m.NU_ROVER,
            "NY_ROVER": _m.NY_ROVER,
            "create_rover_params_linear_ls": _m.create_rover_params_linear_ls,
            "export_rover_ocp_linear_ls": _m.export_rover_ocp_linear_ls,
        }
        return mapping[name]
    elif name in ("DronePlanner", "DronePlannerConfig"):
        from crazyflie_rover_landing.leap_c.drone_planner import DronePlanner, DronePlannerConfig
        return DronePlanner if name == "DronePlanner" else DronePlannerConfig
    elif name in ("RoverPlanner", "RoverPlannerConfig"):
        from crazyflie_rover_landing.leap_c.rover_planner import RoverPlanner, RoverPlannerConfig
        return RoverPlanner if name == "RoverPlanner" else RoverPlannerConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DRONE_NX_EULER", "DRONE_NX_QUAT", "DRONE_NU",
    "create_drone_params_linear_ls", "export_drone_ocp_linear_ls",
    "NX_ROVER", "NU_ROVER", "NY_ROVER",
    "create_rover_params_linear_ls", "export_rover_ocp_linear_ls",
    "DronePlanner", "DronePlannerConfig",
    "RoverPlanner", "RoverPlannerConfig",
]
