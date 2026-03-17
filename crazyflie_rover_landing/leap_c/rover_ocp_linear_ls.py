"""Acados OCP for the TurtleBot3 Burger differential-drive rover (LINEAR_LS cost).

State:   [x, y, c, s, v_L, v_R]  (NX_ROVER = 6)
  x, y  : world-frame position
  c, s  : cos(θ), sin(θ)
  v_L   : left-wheel linear velocity  [m/s]
  v_R   : right-wheel linear velocity [m/s]

Control: [ω_L_cmd, ω_R_cmd]  (NU_ROVER = 2)  — wheel angular velocity commands [rad/s]

Dynamics (continuous-time, RK4 integrated):
  v_body = (v_L+v_R)/2,  ω_body = (v_R−v_L)/L
  ẋ = v_body·c,  ẏ = v_body·s,  ċ = −ω_body·s,  ṡ = ω_body·c
  v̇_L = _DRIVE·ω_L_cmd − _DECAY·v_L
  v̇_R = _DRIVE·ω_R_cmd − _DECAY·v_R

Cost: J = 0.5·(y − y_ref)'·W·(y − y_ref),  y = [x; u]
"""

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp

from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

# Dimensions
NX_ROVER = 6   # [x, y, c, s, v_L, v_R]
NU_ROVER = 2   # [ω_L_cmd, ω_R_cmd]
NY_ROVER = NX_ROVER + NU_ROVER  # 8

# Physical constants (TurtleBot3 Burger)
_R: float    = 0.033        # wheel radius [m]
_L: float    = 0.160        # wheelbase [m]
_I_EFF: float = 0.01        # effective wheel inertia [kg·m²]
_KV: float   = 0.1          # velocity gain [N·m·s/rad]
_B: float    = 0.1          # viscous friction [N·m]
_DRIVE: float = _KV * _R / _I_EFF   # ≈ 0.33
_DECAY: float = (_KV + _B) / _I_EFF  # 20.0

# Actuator limits
_WHEEL_VEL_MAX: float     = 6.67                  # rad/s  (ctrlrange in XML)
_WHEEL_LIN_VEL_MAX: float = _R * _WHEEL_VEL_MAX   # ≈ 0.220 m/s


# ---------------------------------------------------------------------------
# CasADi differential-drive dynamics
# ---------------------------------------------------------------------------

def _diff_drive_ode(x: ca.SX, u: ca.SX) -> ca.SX:
    """Continuous-time differential-drive ODE."""
    c, s, v_L, v_R = x[2], x[3], x[4], x[5]
    v_body  = (v_L + v_R) / 2
    omega_b = (v_R - v_L) / _L
    return ca.vertcat(
        v_body * c,
        v_body * s,
        -omega_b * s,
         omega_b * c,
        _DRIVE * u[0] - _DECAY * v_L,
        _DRIVE * u[1] - _DECAY * v_R,
    )


def _integrate_rk4_rover(x: ca.SX, u: ca.SX, dt: float) -> ca.SX:
    """RK4 integration of the differential-drive ODE."""
    k1 = _diff_drive_ode(x, u)
    k2 = _diff_drive_ode(x + dt / 2 * k1, u)
    k3 = _diff_drive_ode(x + dt / 2 * k2, u)
    k4 = _diff_drive_ode(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def define_diff_drive_dynamics(dt: float) -> tuple[ca.SX, ca.SX, ca.SX]:
    """Define discrete differential-drive dynamics using RK4."""
    x = ca.SX.sym("x", NX_ROVER)
    u = ca.SX.sym("u", NU_ROVER)
    x_next = _integrate_rk4_rover(x, u, dt)
    return x_next, x, u


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_rover_params_linear_ls(
    N_horizon: int = 4,
    pos_offset_max: float = 2.0,
    wheel_vel_max: float = _WHEEL_VEL_MAX,
    wheel_lin_vel_max: float = _WHEEL_LIN_VEL_MAX,
) -> list[AcadosParameter]:
    """Create learnable parameters for the rover MPC with LINEAR_LS cost."""
    # Log-scale weight bounds
    # State: [x, y, c, s, v_L, v_R]
    w_state_min_log = np.array([-2., -2., -2., -2., -2., -2.])
    w_state_max_log = np.array([ 2.,  2.,  1.,  1.,  1.,  1.])
    # Control: [ω_L_cmd, ω_R_cmd]
    w_ctrl_min_log = np.array([-2., -2.])
    w_ctrl_max_log = np.array([ 1.,  1.])

    w_state_default_log = (w_state_min_log + w_state_max_log) / 2
    w_ctrl_default_log  = (w_ctrl_min_log  + w_ctrl_max_log)  / 2

    # Reference bounds
    yref_state_low  = np.array([-pos_offset_max, -pos_offset_max, -1., -1.,
                                 -wheel_lin_vel_max, -wheel_lin_vel_max])
    yref_state_high = np.array([ pos_offset_max,  pos_offset_max,  1.,  1.,
                                  wheel_lin_vel_max,  wheel_lin_vel_max])
    yref_state_default = np.zeros(NX_ROVER)

    yref_ctrl_low    = np.array([-wheel_vel_max, -wheel_vel_max])
    yref_ctrl_high   = np.array([ wheel_vel_max,  wheel_vel_max])
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
    """Total dimension of learnable parameters (global interface).
    w_state(6) + w_ctrl(2) + yref_state(6) + yref_ctrl(2) = 16
    """
    return NX_ROVER + NU_ROVER + NX_ROVER + NU_ROVER


def export_rover_ocp_linear_ls(
    param_manager: AcadosParameterManager,
    name: str = "rover_diff_drive_linear_ls",
    N_horizon: int = 4,
    T_horizon: float = 0.4,
    dt: float = 0.1,
    wheel_vel_max: float = _WHEEL_VEL_MAX,
    wheel_lin_vel_max: float = _WHEEL_LIN_VEL_MAX,
) -> AcadosOcp:
    """Export the rover OCP for LEAP-C using LINEAR_LS cost structure."""
    ocp = AcadosOcp()
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    param_manager.assign_to_ocp(ocp)

    ocp.model.name = name
    ocp.dims.nx = NX_ROVER
    ocp.dims.nu = NU_ROVER

    x_next, x, u = define_diff_drive_dynamics(dt)
    ocp.model.x = x
    ocp.model.u = u
    ocp.model.disc_dyn_expr = x_next

    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    w_state   = param_manager.get("w_state")
    w_control = param_manager.get("w_control")
    yref_state   = param_manager.get("yref_state")
    yref_control = param_manager.get("yref_control")

    y     = ca.vertcat(x, u)
    y_ref = ca.vertcat(yref_state, yref_control)
    W     = ca.diag(ca.vertcat(w_state, w_control))
    W_e   = ca.diag(w_state)

    y_res   = y - y_ref
    y_res_e = x - yref_state

    ocp.model.cost_expr_ext_cost   = 0.5 * (y_res.T   @ W   @ y_res)
    ocp.model.cost_expr_ext_cost_e = 0.5 * (y_res_e.T @ W_e @ y_res_e)

    ocp.constraints.x0 = np.zeros(NX_ROVER)

    # Control box constraints: wheel angular velocity commands [rad/s]
    ocp.constraints.lbu    = np.array([-wheel_vel_max, -wheel_vel_max])
    ocp.constraints.ubu    = np.array([ wheel_vel_max,  wheel_vel_max])
    ocp.constraints.idxbu  = np.array([0, 1])

    # State box constraints: cos/sin ∈ [-1,1], wheel linear vel ∈ [-v_max, v_max]
    ocp.constraints.lbx    = np.array([-1., -1., -wheel_lin_vel_max, -wheel_lin_vel_max])
    ocp.constraints.ubx    = np.array([ 1.,  1.,  wheel_lin_vel_max,  wheel_lin_vel_max])
    ocp.constraints.idxbx  = np.array([2, 3, 4, 5])
    ocp.constraints.lbx_e  = np.array([-1., -1., -wheel_lin_vel_max, -wheel_lin_vel_max])
    ocp.constraints.ubx_e  = np.array([ 1.,  1.,  wheel_lin_vel_max,  wheel_lin_vel_max])
    ocp.constraints.idxbx_e = np.array([2, 3, 4, 5])

    ocp.solver_options.qp_solver            = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx       = "EXACT"
    ocp.solver_options.integrator_type      = "DISCRETE"
    ocp.solver_options.nlp_solver_type      = "SQP"
    ocp.solver_options.print_level          = 0
    ocp.solver_options.qp_solver_ric_alg    = 1
    ocp.solver_options.qp_solver_cond_N     = N_horizon
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.tol                  = 1e-6
    ocp.solver_options.qp_tol              = 1e-6
    ocp.solver_options.qp_solver_iter_max  = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    return ocp
