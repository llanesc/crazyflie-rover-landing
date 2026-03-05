"""Acados OCP for the Crazyflie drone attitude control with LINEAR_LS cost.

Adapted from the crazyflie-mape-crazyflow quadrotor_ocp_linear_ls.py.
The main change is the default drone model (cf2x_T350 instead of cf2x_L250).

Cost: J = 0.5 * (y - y_ref)' W (y - y_ref),  y = [x; u]

The neural network outputs:
  1. W (weights)  — log-scaled diagonal entries
  2. y_ref        — linearly scaled to physical bounds

State:   [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw] (12D)
Control: [roll_cmd, pitch_cmd, yaw_cmd, thrust]                         (4D)
"""

from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp

from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import cs_rpy2matrix
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

# State / control dimensions
NX = 12   # [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]
NU = 4    # [roll_cmd, pitch_cmd, yaw_cmd, thrust]
NY = NX + NU   # 16 — combined output for LINEAR_LS

W_SIZE = NY       # diagonal weights
YREF_SIZE = NY    # reference vector

QuadrotorAcadosParamInterface = Literal["global", "stagewise"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _integrate_erk4(f_expl: ca.SX, x: ca.SX, u: ca.SX, p: ca.SX, dt: float) -> ca.SX:
    ode = ca.Function("ode", [x, u, p], [f_expl])
    k1 = ode(x, u, p)
    k2 = ode(x + dt / 2 * k1, u, p)
    k3 = ode(x + dt / 2 * k2, u, p)
    k4 = ode(x + dt * k3, u, p)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def define_so_rpy_euler_dynamics(
    dt: float,
    drone_model: str = "cf2x_T350",
    mass: float | None = None,
    gravity: float | None = None,
) -> tuple[ca.SX, ca.SX, ca.SX]:
    """Define discrete quadrotor dynamics with so_rpy Euler model + RK4.

    Args:
        dt: Integration timestep [s].
        drone_model: Drone model identifier.
        mass: Drone mass [kg]. None to load from drone_model.
        gravity: Gravitational acceleration [m/s²]. None to load from drone_model.

    Returns:
        (x_next, x, u) CasADi SX expressions.
    """
    params = load_params("so_rpy", drone_model)
    if mass is None:
        mass = float(params["mass"])
    if gravity is None:
        gravity = float(np.abs(params["gravity_vec"][2]))

    acc_coef = float(params["acc_coef"])
    cmd_f_coef = float(params["cmd_f_coef"])
    rpy_coef = np.array(params["rpy_coef"])
    rpy_rates_coef = np.array(params["rpy_rates_coef"])
    cmd_rpy_coef = np.array(params["cmd_rpy_coef"])

    X = ca.SX.sym("x", NX)
    U = ca.SX.sym("u", NU)

    pos = X[0:3]
    rpy = X[3:6]
    vel = X[6:9]
    drpy = X[9:12]

    roll_cmd, pitch_cmd, yaw_cmd, thrust = U[0], U[1], U[2], U[3]

    R = cs_rpy2matrix(rpy)

    pos_dot = vel
    rpy_dot = drpy
    thrust_z = acc_coef + cmd_f_coef * thrust
    vel_dot = R @ ca.vertcat(0, 0, thrust_z / mass) + ca.vertcat(0, 0, -gravity)
    drpy_dot = ca.vertcat(
        rpy_coef[0] * rpy[0] + rpy_rates_coef[0] * drpy[0] + cmd_rpy_coef[0] * roll_cmd,
        rpy_coef[1] * rpy[1] + rpy_rates_coef[1] * drpy[1] + cmd_rpy_coef[1] * pitch_cmd,
        rpy_coef[2] * rpy[2] + rpy_rates_coef[2] * drpy[2] + cmd_rpy_coef[2] * yaw_cmd,
    )

    X_dot = ca.vertcat(pos_dot, rpy_dot, vel_dot, drpy_dot)
    p = ca.SX.sym("p_empty", 0)
    X_next = _integrate_erk4(X_dot, X, U, p, dt)

    return X_next, X, U


def create_drone_params_linear_ls(
    N_horizon: int = 2,
    param_interface: QuadrotorAcadosParamInterface = "global",
    drone_model: str = "cf2x_T350",
    roll_pitch_max: float = 0.5,
    yaw_max: float = 0.5,
    pos_offset_max: float = 2.0,
    thrust_min: float | None = None,
    thrust_max: float | None = None,
    mass: float | None = None,
    gravity: float | None = None,
) -> list[AcadosParameter]:
    """Create learnable parameters for drone MPC with LINEAR_LS cost.

    Args:
        N_horizon: MPC horizon steps.
        param_interface: "global" or "stagewise".
        drone_model: Drone model identifier.
        roll_pitch_max: Max roll/pitch command [rad].
        yaw_max: Max yaw command [rad].
        pos_offset_max: Max position reference offset [m].
        thrust_min: Min collective thrust [N]. None to load from drone_model.
        thrust_max: Max collective thrust [N]. None to load from drone_model.
        mass: Drone mass [kg]. None to load from drone_model.
        gravity: Gravitational acceleration [m/s²]. None to load from drone_model.

    Returns:
        List of AcadosParameter objects.
    """
    drone_params = load_params("so_rpy", drone_model)
    if mass is None:
        mass = float(drone_params["mass"])
    if gravity is None:
        gravity = float(np.abs(drone_params["gravity_vec"][2]))
    if thrust_min is None:
        thrust_min = float(drone_params["thrust_min"]) * 4
    if thrust_max is None:
        thrust_max = float(drone_params["thrust_max"]) * 4
    cmd_f_coef = float(drone_params["cmd_f_coef"])
    hover_thrust = (mass * gravity) / cmd_f_coef

    state_end_stages = list(range(N_horizon + 1)) if param_interface == "stagewise" else []
    ctrl_end_stages = list(range(N_horizon)) if param_interface == "stagewise" else []

    # Weight log-bounds per state component
    w_state_min_log = np.array([-1., -1., -1., -2., -2., -2., -1., -1., -1., -1., -1., -1.])
    w_state_max_log = np.array([2., 2., 2., 1., 1., 1., 2., 2., 2., 1., 1., 1.])
    w_ctrl_min_log = np.array([-1., -1., -1., -1.])
    w_ctrl_max_log = np.array([1., 1., 1., 1.])

    w_state_default_log = (w_state_min_log + w_state_max_log) / 2
    w_ctrl_default_log = (w_ctrl_min_log + w_ctrl_max_log) / 2

    # Reference bounds (physical)
    yref_state_low = np.array([
        -pos_offset_max, -pos_offset_max, 0.,
        -roll_pitch_max, -roll_pitch_max, -yaw_max,
        -5., -5., -5., -10., -10., -10.
    ])
    yref_state_high = np.array([
        pos_offset_max, pos_offset_max, 4.5,
        roll_pitch_max, roll_pitch_max, yaw_max,
        5., 5., 5., 10., 10., 10.
    ])

    return [
        AcadosParameter(
            name="w_state",
            default=np.power(10., w_state_default_log),
            space=gym.spaces.Box(
                low=np.power(10., w_state_min_log),
                high=np.power(10., w_state_max_log),
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=state_end_stages,
        ),
        AcadosParameter(
            name="w_control",
            default=np.power(10., w_ctrl_default_log),
            space=gym.spaces.Box(
                low=np.power(10., w_ctrl_min_log),
                high=np.power(10., w_ctrl_max_log),
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=ctrl_end_stages,
        ),
        AcadosParameter(
            name="yref_state",
            default=np.zeros(NX),
            space=gym.spaces.Box(low=yref_state_low, high=yref_state_high, dtype=np.float64),
            interface="learnable",
            end_stages=state_end_stages,
        ),
        AcadosParameter(
            name="yref_control",
            default=np.array([0., 0., 0., hover_thrust]),
            space=gym.spaces.Box(
                low=np.array([-roll_pitch_max, -roll_pitch_max, -yaw_max, thrust_min]),
                high=np.array([roll_pitch_max, roll_pitch_max, yaw_max, thrust_max]),
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=ctrl_end_stages,
        ),
    ]


def get_drone_learnable_param_dim(
    N_horizon: int,
    param_interface: QuadrotorAcadosParamInterface,
) -> int:
    """Total dimension of learnable parameters for the drone OCP."""
    if param_interface == "global":
        return NX + NU + NX + NU  # 32
    n_state_stages = N_horizon + 1
    n_ctrl_stages = N_horizon
    return (NX + NX) * n_state_stages + (NU + NU) * n_ctrl_stages


def export_drone_ocp_linear_ls(
    param_manager: AcadosParameterManager,
    name: str = "drone_so_rpy_euler_linear_ls",
    N_horizon: int = 2,
    T_horizon: float = 0.02,
    dt: float = 0.01,
    drone_model: str = "cf2x_T350",
    velocity_max: float | None = None,
    roll_pitch_max: float = 0.5,
    yaw_max: float = 0.5,
    thrust_min: float | None = None,
    thrust_max: float | None = None,
    mass: float | None = None,
    gravity: float | None = None,
) -> AcadosOcp:
    """Export the drone OCP for LEAP-C using LINEAR_LS cost structure.

    Args:
        param_manager: Manager containing learnable parameters.
        name: Acados model name for code generation.
        N_horizon: MPC horizon steps.
        T_horizon: Total horizon time [s].
        dt: Integration timestep [s].
        drone_model: Drone model identifier.
        velocity_max: Max velocity per axis [m/s]. None to disable.
        roll_pitch_max: Max roll/pitch command [rad].
        yaw_max: Max yaw command [rad].
        thrust_min: Min collective thrust [N]. None to load from drone_model.
        thrust_max: Max collective thrust [N]. None to load from drone_model.
        mass: Drone mass [kg]. None to load from drone_model.
        gravity: Gravitational acceleration [m/s²]. None to load from drone_model.

    Returns:
        Configured AcadosOcp.
    """
    ocp = AcadosOcp()
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    param_manager.assign_to_ocp(ocp)

    ocp.model.name = name
    ocp.dims.nx = NX
    ocp.dims.nu = NU

    x_next, x, u = define_so_rpy_euler_dynamics(dt, drone_model, mass=mass, gravity=gravity)
    ocp.model.x = x
    ocp.model.u = u
    ocp.model.disc_dyn_expr = x_next

    # EXTERNAL cost with LINEAR_LS structure
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    w_state = param_manager.get("w_state")
    w_control = param_manager.get("w_control")
    yref_state = param_manager.get("yref_state")
    yref_control = param_manager.get("yref_control")

    y = ca.vertcat(x, u)
    y_ref = ca.vertcat(yref_state, yref_control)
    W = ca.diag(ca.vertcat(w_state, w_control))
    W_e = ca.diag(w_state)

    y_res = y - y_ref
    y_res_e = x - yref_state

    ocp.model.cost_expr_ext_cost = 0.5 * (y_res.T @ W @ y_res)
    ocp.model.cost_expr_ext_cost_e = 0.5 * (y_res_e.T @ W_e @ y_res_e)

    ocp.constraints.x0 = np.zeros(NX)

    # Load thrust limits if not provided
    if thrust_min is None or thrust_max is None:
        drone_params = load_params("so_rpy", drone_model)
        if thrust_min is None:
            thrust_min = float(drone_params["thrust_min"]) * 4
        if thrust_max is None:
            thrust_max = float(drone_params["thrust_max"]) * 4

    ocp.constraints.lbu = np.array([-roll_pitch_max, -roll_pitch_max, -yaw_max, thrust_min])
    ocp.constraints.ubu = np.array([roll_pitch_max, roll_pitch_max, yaw_max, thrust_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    if velocity_max is not None:
        ocp.constraints.lbx = np.array([-velocity_max, -velocity_max, -velocity_max])
        ocp.constraints.ubx = np.array([velocity_max, velocity_max, velocity_max])
        ocp.constraints.idxbx = np.array([6, 7, 8])
        ocp.constraints.lbx_e = np.array([-velocity_max, -velocity_max, -velocity_max])
        ocp.constraints.ubx_e = np.array([velocity_max, velocity_max, velocity_max])
        ocp.constraints.idxbx_e = np.array([6, 7, 8])

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.print_level = 0
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.qp_solver_cond_N = N_horizon
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.qp_tol = 1e-6
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    return ocp
