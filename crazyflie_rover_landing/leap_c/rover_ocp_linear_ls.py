"""Acados OCP for the unicycle ground rover with LINEAR_LS cost.

State:   [x, y, c, s, v]   (position x/y, cos(heading), sin(heading), speed)  (NX_ROVER = 5)
Control: [a, ω]              (longitudinal acceleration, yaw rate)              (NU_ROVER = 2)

Dynamics (polynomial — no trig):
  ẋ = v·c     ẏ = v·s     ċ = -ω·s     ṡ = ω·c     v̇ = a

Cost: J = 0.5 * (y - y_ref)' W (y - y_ref),  y = [x; u]

The neural network outputs:
  1. W (weights)  — log-scaled diagonal entries
  2. y_ref        — linearly scaled to physical bounds
"""

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp

from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

# Dimensions
NX_ROVER = 5   # [x, y, c, s, v]
NU_ROVER = 2   # [a, ω]
NY_ROVER = NX_ROVER + NU_ROVER  # 7

# Physical limits (must match rover_dynamics.py)
_MAX_SPEED = 1.5    # m/s
_MIN_SPEED = -0.5   # m/s
_MAX_OMEGA = 1.5708  # rad/s (≈π/2)
_MAX_ACCEL = 2.0    # m/s²


# ---------------------------------------------------------------------------
# CasADi unicycle dynamics helper (polynomial — no trig)
# ---------------------------------------------------------------------------

def _unicycle_ode(x: ca.SX, u: ca.SX) -> ca.SX:
    """Continuous-time unicycle ODE with cos/sin state.

    ẋ = v·c,  ẏ = v·s,  ċ = -ω·s,  ṡ = ω·c,  v̇ = a
    """
    x_pos, y_pos, c, s, v = x[0], x[1], x[2], x[3], x[4]
    a, omega = u[0], u[1]
    return ca.vertcat(
        v * c,
        v * s,
        -omega * s,
        omega * c,
        a,
    )


def _integrate_rk4_rover(x: ca.SX, u: ca.SX, dt: float) -> ca.SX:
    """RK4 integration of the unicycle ODE."""
    k1 = _unicycle_ode(x, u)
    k2 = _unicycle_ode(x + dt / 2 * k1, u)
    k3 = _unicycle_ode(x + dt / 2 * k2, u)
    k4 = _unicycle_ode(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def define_unicycle_dynamics(dt: float) -> tuple[ca.SX, ca.SX, ca.SX]:
    """Define discrete unicycle dynamics using RK4 integration.

    Args:
        dt: Integration timestep [s].

    Returns:
        (x_next, x, u) CasADi SX expressions.
    """
    x = ca.SX.sym("x", NX_ROVER)
    u = ca.SX.sym("u", NU_ROVER)
    x_next = _integrate_rk4_rover(x, u, dt)
    return x_next, x, u


def create_rover_params_linear_ls(
    N_horizon: int = 4,
    pos_offset_max: float = 2.0,
    max_speed: float = _MAX_SPEED,
    min_speed: float = _MIN_SPEED,
    max_omega: float = _MAX_OMEGA,
    max_accel: float = _MAX_ACCEL,
) -> list[AcadosParameter]:
    """Create learnable parameters for the rover MPC with LINEAR_LS cost.

    Args:
        N_horizon: MPC horizon steps.
        pos_offset_max: Max position reference offset [m].
        max_speed: Max rover speed [m/s].
        min_speed: Min rover speed [m/s] (negative for reverse).
        max_omega: Max yaw rate [rad/s].
        max_accel: Max acceleration [m/s²].

    Returns:
        List of AcadosParameter objects.
    """
    # Log-scale weight bounds
    # State: [x, y, c, s, v]
    w_state_min_log = np.array([-2., -2., -2., -2., -2.])
    w_state_max_log = np.array([2., 2., 1., 1., 1.])
    # Control: [a, ω]
    w_ctrl_min_log = np.array([-2., -2.])
    w_ctrl_max_log = np.array([1., 1.])

    w_state_default_log = (w_state_min_log + w_state_max_log) / 2
    w_ctrl_default_log = (w_ctrl_min_log + w_ctrl_max_log) / 2

    # Reference bounds (physical)
    # Position refs: relative offset from current rover position
    # cos/sin refs: [-1, 1] (any heading direction)
    # Speed ref: [min_speed, max_speed]
    yref_state_low = np.array([-pos_offset_max, -pos_offset_max, -1.0, -1.0, min_speed])
    yref_state_high = np.array([pos_offset_max, pos_offset_max, 1.0, 1.0, max_speed])
    yref_state_default = np.zeros(NX_ROVER)

    yref_ctrl_low = np.array([-max_accel, -max_omega])
    yref_ctrl_high = np.array([max_accel, max_omega])
    yref_ctrl_default = np.zeros(NU_ROVER)

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
            end_stages=[],
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
            end_stages=[],
        ),
        AcadosParameter(
            name="yref_state",
            default=yref_state_default,
            space=gym.spaces.Box(low=yref_state_low, high=yref_state_high, dtype=np.float64),
            interface="learnable",
            end_stages=[],
        ),
        AcadosParameter(
            name="yref_control",
            default=yref_ctrl_default,
            space=gym.spaces.Box(low=yref_ctrl_low, high=yref_ctrl_high, dtype=np.float64),
            interface="learnable",
            end_stages=[],
        ),
    ]


def get_rover_learnable_param_dim() -> int:
    """Total dimension of learnable parameters for the rover OCP (global interface)."""
    # w_state(5) + w_ctrl(2) + yref_state(5) + yref_ctrl(2) = 14
    return NX_ROVER + NU_ROVER + NX_ROVER + NU_ROVER


def export_rover_ocp_linear_ls(
    param_manager: AcadosParameterManager,
    name: str = "rover_unicycle_linear_ls",
    N_horizon: int = 4,
    T_horizon: float = 0.4,
    dt: float = 0.1,
    max_speed: float = _MAX_SPEED,
    min_speed: float = _MIN_SPEED,
    max_omega: float = _MAX_OMEGA,
    max_accel: float = _MAX_ACCEL,
) -> AcadosOcp:
    """Export the rover OCP for LEAP-C using LINEAR_LS cost structure.

    Args:
        param_manager: Manager containing learnable parameters.
        name: Acados model name for code generation.
        N_horizon: MPC horizon steps (default 4 → T=0.4s at dt=0.1s).
        T_horizon: Total horizon time [s].
        dt: Integration timestep [s].
        max_speed: Max rover forward speed [m/s].
        min_speed: Max rover reverse speed [m/s].
        max_omega: Max yaw rate [rad/s].
        max_accel: Max acceleration [m/s²].

    Returns:
        Configured AcadosOcp.
    """
    ocp = AcadosOcp()
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    param_manager.assign_to_ocp(ocp)

    ocp.model.name = name
    ocp.dims.nx = NX_ROVER
    ocp.dims.nu = NU_ROVER

    x_next, x, u = define_unicycle_dynamics(dt)
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

    ocp.constraints.x0 = np.zeros(NX_ROVER)

    # Control box constraints
    ocp.constraints.lbu = np.array([-max_accel, -max_omega])
    ocp.constraints.ubu = np.array([max_accel, max_omega])
    ocp.constraints.idxbu = np.array([0, 1])

    # State box constraints: cos/sin in [-1, 1], speed in [min_speed, max_speed]
    ocp.constraints.lbx = np.array([-1.0, -1.0, min_speed])
    ocp.constraints.ubx = np.array([1.0, 1.0, max_speed])
    ocp.constraints.idxbx = np.array([2, 3, 4])
    ocp.constraints.lbx_e = np.array([-1.0, -1.0, min_speed])
    ocp.constraints.ubx_e = np.array([1.0, 1.0, max_speed])
    ocp.constraints.idxbx_e = np.array([2, 3, 4])

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
