"""Acados OCP for the Crazyflie drone attitude control with LINEAR_LS cost.

Supports two state representations:
- Euler (12D): [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]
- Quaternion (13D): [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]

Control: [roll_cmd, pitch_cmd, yaw_cmd, thrust] (4D)

Cost: J = 0.5 * (y - y_ref)' W (y - y_ref),  y = [x; u]

The neural network outputs:
  1. W (weights)  — log-scaled diagonal entries
  2. y_ref        — linearly scaled to physical bounds
"""

from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp

from drone_models.core import load_params
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

# State / control dimensions
NX_EULER = 12  # [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]
NX_QUAT = 13   # [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
NY_QUAT = 12   # Cost residual dim for quat: [pos(3), quat_err_xyz(3), vel(3), ang_vel(3)]
NX = NX_EULER   # Default (backward compat)
NU = 4          # [roll_cmd, pitch_cmd, yaw_cmd, thrust]

StateType = Literal["euler", "quat"]
IntegratorType = Literal["rk4", "euler"]
QuadrotorAcadosParamInterface = Literal["global", "stagewise"]


# ---------------------------------------------------------------------------
# Parameter creation
# ---------------------------------------------------------------------------

def create_drone_params_linear_ls(
    N_horizon: int = 2,
    param_interface: QuadrotorAcadosParamInterface = "global",
    drone_model: str = "cf2x_T350",
    state_type: StateType = "euler",
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
        state_type: "euler" (12D) or "quat" (13D).
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

    nx = NX_QUAT if state_type == "quat" else NX_EULER
    ny = NY_QUAT if state_type == "quat" else NX_EULER  # cost residual dim

    state_end_stages = list(range(N_horizon + 1)) if param_interface == "stagewise" else []
    ctrl_end_stages = list(range(N_horizon)) if param_interface == "stagewise" else []

    # Weight log-bounds per cost residual component
    if state_type == "quat":
        # [pos(3), quat_err_xyz(3), vel(3), ang_vel(3)] = 12D
        w_state_min_log = np.array([-1., -1., -1., -2., -2., -2., -1., -1., -1., -1., -1., -1.])
        w_state_max_log = np.array([2., 2., 2., 1., 1., 1., 2., 2., 2., 1., 1., 1.])
    else:
        # [pos(3), rpy(3), vel(3), drpy(3)] = 12D
        w_state_min_log = np.array([-1., -1., -1., -2., -2., -2., -1., -1., -1., -1., -1., -1.])
        w_state_max_log = np.array([2., 2., 2., 1., 1., 1., 2., 2., 2., 1., 1., 1.])
    w_ctrl_min_log = np.array([-1., -1., -1., -1.])
    w_ctrl_max_log = np.array([1., 1., 1., 1.])

    w_state_default_log = (w_state_min_log + w_state_max_log) / 2
    w_ctrl_default_log = (w_ctrl_min_log + w_ctrl_max_log) / 2

    # Reference bounds (physical)
    if state_type == "quat":
        # [pos(3), quat(4:xyzw), vel(3), ang_vel(3)] = 13D
        yref_state_low = np.array([
            -pos_offset_max, -pos_offset_max, 0.,
            -1., -1., -1., -1.,
            -5., -5., -5., -10., -10., -10.
        ])
        yref_state_high = np.array([
            pos_offset_max, pos_offset_max, 4.5,
            1., 1., 1., 1.,
            5., 5., 5., 10., 10., 10.
        ])
        yref_state_default = np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
    else:
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
        yref_state_default = np.zeros(nx)

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
            default=yref_state_default,
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


# ---------------------------------------------------------------------------
# Learnable param dimension
# ---------------------------------------------------------------------------

def get_drone_learnable_param_dim(
    N_horizon: int,
    param_interface: QuadrotorAcadosParamInterface,
    state_type: StateType = "euler",
) -> int:
    """Total dimension of learnable parameters for the drone OCP."""
    nx = NX_QUAT if state_type == "quat" else NX_EULER
    ny = NY_QUAT if state_type == "quat" else NX_EULER  # w_state dim
    if param_interface == "global":
        return ny + NU + nx + NU  # w_state(ny) + w_ctrl(NU) + yref_state(nx) + yref_ctrl(NU)
    n_state_stages = N_horizon + 1
    n_ctrl_stages = N_horizon
    return (ny + nx) * n_state_stages + (NU + NU) * n_ctrl_stages


# ---------------------------------------------------------------------------
# OCP export
# ---------------------------------------------------------------------------

def export_drone_ocp_linear_ls(
    param_manager: AcadosParameterManager,
    name: str = "drone_so_rpy_linear_ls",
    N_horizon: int = 2,
    T_horizon: float = 0.02,
    dt: float = 0.01,
    drone_model: str = "cf2x_T350",
    state_type: StateType = "euler",
    integrator: IntegratorType = "rk4",
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
        state_type: "euler" (12D) or "quat" (13D).
        integrator: "rk4" or "euler" (forward Euler).
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
    from .so_rpy_dynamics_sx import (
        integrate_euler_sx,
        integrate_rk4_sx,
        symbolic_dynamics_euler_sx,
        symbolic_dynamics_sx,
    )

    nx = NX_QUAT if state_type == "quat" else NX_EULER

    ocp = AcadosOcp()
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    param_manager.assign_to_ocp(ocp)

    ocp.model.name = name
    ocp.dims.nx = nx
    ocp.dims.nu = NU

    # Load drone parameters and build SX dynamics
    params = load_params("so_rpy", drone_model)
    common_kwargs = dict(
        model_rotor_vel=False,
        mass=float(params["mass"]) if mass is None else mass,
        gravity_vec=params["gravity_vec"],
        J=params["J"],
        J_inv=params["J_inv"],
        acc_coef=params["acc_coef"],
        cmd_f_coef=params["cmd_f_coef"],
        rpy_coef=params["rpy_coef"],
        rpy_rates_coef=params["rpy_rates_coef"],
        cmd_rpy_coef=params["cmd_rpy_coef"],
    )

    if state_type == "quat":
        X_dot, X, U, _ = symbolic_dynamics_sx(**common_kwargs)
    else:
        X_dot, X, U, _ = symbolic_dynamics_euler_sx(**common_kwargs)

    # Discretize dynamics
    if integrator == "rk4":
        X_next = integrate_rk4_sx(X_dot, X, U, dt)
    else:
        X_next = integrate_euler_sx(X_dot, X, U, dt)
    ocp.model.x = X
    ocp.model.u = U
    ocp.model.disc_dyn_expr = X_next

    # EXTERNAL cost with LINEAR_LS structure
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    w_state = param_manager.get("w_state")
    w_control = param_manager.get("w_control")
    yref_state = param_manager.get("yref_state")
    yref_control = param_manager.get("yref_control")

    if state_type == "quat":
        # Quaternion error cost: residual uses error quaternion vector part (3D)
        # instead of component-wise q - q_ref (4D)
        # State: [pos(3), qx, qy, qz, qw, vel(3), ang_vel(3)]
        pos = X[:3]
        qx, qy, qz, qw = X[3], X[4], X[5], X[6]
        vel = X[7:10]
        ang_vel = X[10:13]

        pos_ref = yref_state[:3]
        qx_r, qy_r, qz_r, qw_r = yref_state[3], yref_state[4], yref_state[5], yref_state[6]
        vel_ref = yref_state[7:10]
        ang_vel_ref = yref_state[10:13]

        # Error quaternion: conj(q_ref) ⊗ q — vector part is zero when aligned
        # Naturally sign-invariant: both q and -q give zero vector part
        q_err_x = qw_r * qx - qx_r * qw - qy_r * qz + qz_r * qy
        q_err_y = qw_r * qy + qx_r * qz - qy_r * qw - qz_r * qx
        q_err_z = qw_r * qz - qx_r * qy + qy_r * qx - qz_r * qw

        # 12D state cost residual
        state_res = ca.vertcat(pos - pos_ref, q_err_x, q_err_y, q_err_z,
                               vel - vel_ref, ang_vel - ang_vel_ref)

        ctrl_res = U - yref_control
        y_res = ca.vertcat(state_res, ctrl_res)
        W = ca.diag(ca.vertcat(w_state, w_control))
        W_e = ca.diag(w_state)

        ocp.model.cost_expr_ext_cost = 0.5 * (y_res.T @ W @ y_res)
        ocp.model.cost_expr_ext_cost_e = 0.5 * (state_res.T @ W_e @ state_res)
    else:
        # Euler: simple component-wise residual
        y = ca.vertcat(X, U)
        y_ref = ca.vertcat(yref_state, yref_control)
        W = ca.diag(ca.vertcat(w_state, w_control))
        W_e = ca.diag(w_state)

        y_res = y - y_ref
        y_res_e = X - yref_state

        ocp.model.cost_expr_ext_cost = 0.5 * (y_res.T @ W @ y_res)
        ocp.model.cost_expr_ext_cost_e = 0.5 * (y_res_e.T @ W_e @ y_res_e)

    # Initial state constraint
    if state_type == "quat":
        ocp.constraints.x0 = np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
    else:
        ocp.constraints.x0 = np.zeros(nx)

    # Load thrust limits if not provided
    if thrust_min is None or thrust_max is None:
        if thrust_min is None:
            thrust_min = float(params["thrust_min"]) * 4
        if thrust_max is None:
            thrust_max = float(params["thrust_max"]) * 4

    ocp.constraints.lbu = np.array([-roll_pitch_max, -roll_pitch_max, -yaw_max, thrust_min])
    ocp.constraints.ubu = np.array([roll_pitch_max, roll_pitch_max, yaw_max, thrust_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # Velocity constraint indices depend on state type
    if velocity_max is not None:
        if state_type == "quat":
            vel_idx = np.array([7, 8, 9])
        else:
            vel_idx = np.array([6, 7, 8])
        ocp.constraints.lbx = np.array([-velocity_max, -velocity_max, -velocity_max])
        ocp.constraints.ubx = np.array([velocity_max, velocity_max, velocity_max])
        ocp.constraints.idxbx = vel_idx
        ocp.constraints.lbx_e = np.array([-velocity_max, -velocity_max, -velocity_max])
        ocp.constraints.ubx_e = np.array([velocity_max, velocity_max, velocity_max])
        ocp.constraints.idxbx_e = vel_idx

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
