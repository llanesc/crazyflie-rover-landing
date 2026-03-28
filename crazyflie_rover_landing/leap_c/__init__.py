"""LEAP-C OCP definitions for drone and rover."""

from crazyflie_rover_landing.leap_c.drone_ocp_linear_ls import (
    NX_EULER as DRONE_NX_EULER,
    NX_QUAT as DRONE_NX_QUAT,
    NU as DRONE_NU,
    create_drone_params_linear_ls,
    export_drone_ocp_linear_ls,
)
from crazyflie_rover_landing.leap_c.rover_ocp_linear_ls import (
    NX_ROVER,
    NU_ROVER,
    NY_ROVER,
    create_rover_params_linear_ls,
    export_rover_ocp_linear_ls,
)
from crazyflie_rover_landing.leap_c.drone_planner import DronePlanner, DronePlannerConfig
from crazyflie_rover_landing.leap_c.rover_planner import RoverPlanner, RoverPlannerConfig

__all__ = [
    "DRONE_NX_EULER",
    "DRONE_NX_QUAT",
    "DRONE_NU",
    "create_drone_params_linear_ls",
    "export_drone_ocp_linear_ls",
    "NX_ROVER",
    "NU_ROVER",
    "NY_ROVER",
    "create_rover_params_linear_ls",
    "export_rover_ocp_linear_ls",
    "DronePlanner",
    "DronePlannerConfig",
    "RoverPlanner",
    "RoverPlannerConfig",
]
