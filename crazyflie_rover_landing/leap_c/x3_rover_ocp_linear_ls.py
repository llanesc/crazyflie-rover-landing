"""Acados OCP for the Yahboom RosMaster X3 mecanum rover (LINEAR_LS cost).

State:   [x, y, c, s, vx, vy, ωz]  (NX_ROVER = 7)
  x, y  : world-frame position
  c, s  : cos(θ), sin(θ)
  vx    : body-frame forward velocity [m/s]
  vy    : body-frame lateral velocity [m/s]
  ωz    : yaw rate [rad/s]

Control: [vx_cmd, vy_cmd, ωz_cmd]  (NU_ROVER = 3)
  Body-frame velocity commands matching the real /cmd_vel interface.

Dynamics (continuous-time, RK4 integrated) with per-wheel saturation:
  1. Inverse kinematics: body cmd → 4 wheel angular velocity targets
  2. Clip each wheel to [-ω_max, ω_max]
  3. Inverse kinematics: current body vel → 4 current wheel speeds
  4. Per-wheel first-order lag: ω̇_i = (target_i − current_i) / τ
  5. Forward kinematics: wheel derivatives → body velocity derivatives
  6. Pose update in world frame

This matches the sim dynamics in mecanum_dynamics.py, where the STM32
PID runs independently per wheel with motor saturation.

Cost: J = 0.5·(y − y_ref)'·W·(y − y_ref),  y = [x; u]
"""

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp

from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

# Dimensions
NX_ROVER = 7   # [x, y, c, s, vx, vy, ωz]
NU_ROVER = 3   # [vx_cmd, vy_cmd, ωz_cmd]
NY_ROVER = NX_ROVER + NU_ROVER  # 10

# Physical constants (RosMaster X3)
_TAU: float = 0.1    # s — first-order PID time constant
_WHEEL_RADIUS: float = 0.0325    # m (65 mm mecanum wheel)
_HALF_WHEELBASE: float = 0.08    # m (160 mm / 2, front-to-rear)
_HALF_TRACK: float = 0.0845      # m (169 mm / 2, left-to-right)
_K: float = _HALF_WHEELBASE + _HALF_TRACK  # 0.1645 m
_WHEEL_VEL_MAX: float = 34.9     # rad/s (333 RPM motor, 1:30 gear ratio)

# Velocity command limits (ROS /cmd_vel)
_VX_MAX: float = 1.0    # m/s
_VY_MAX: float = 1.0    # m/s
_WZ_MAX: float = 5.0    # rad/s


# ---------------------------------------------------------------------------
# CasADi mecanum dynamics
# ---------------------------------------------------------------------------

def _inv_kinematics_ca(vx: ca.SX, vy: ca.SX, wz: ca.SX) -> ca.SX:
    """Body-frame velocities → 4 wheel angular velocities (CasADi)."""
    r_inv = 1.0 / _WHEEL_RADIUS
    return ca.vertcat(
        r_inv * (vx - vy - _K * wz),   # front-left
        r_inv * (vx + vy + _K * wz),   # front-right
        r_inv * (vx + vy - _K * wz),   # back-left
        r_inv * (vx - vy + _K * wz),   # back-right
    )


def _fwd_kinematics_ca(wheels: ca.SX) -> tuple[ca.SX, ca.SX, ca.SX]:
    """4 wheel angular velocities → body-frame velocities (CasADi)."""
    r4 = _WHEEL_RADIUS / 4.0
    vx = r4 * (wheels[0] + wheels[1] + wheels[2] + wheels[3])
    vy = r4 * (-wheels[0] + wheels[1] + wheels[2] - wheels[3])
    wz = r4 / _K * (-wheels[0] + wheels[1] - wheels[2] + wheels[3])
    return vx, vy, wz


def _mecanum_ode_simple(x: ca.SX, u: ca.SX) -> ca.SX:
    """Continuous-time mecanum ODE with first-order body-velocity tracking.

    No wheel-level kinematics or saturation — assumes body velocity commands
    are directly achievable.
    """
    c, s = x[2], x[3]
    vx, vy, wz = x[4], x[5], x[6]

    return ca.vertcat(
        vx * c - vy * s,         # ẋ
        vx * s + vy * c,         # ẏ
        -wz * s,                 # ċ
         wz * c,                 # ṡ
        (u[0] - vx) / _TAU,     # v̇x
        (u[1] - vy) / _TAU,     # v̇y
        (u[2] - wz) / _TAU,     # ω̇z
    )


def _mecanum_ode_wheel(x: ca.SX, u: ca.SX, wheel_vel_max: float) -> ca.SX:
    """Continuous-time mecanum ODE with per-wheel saturation.

    Routes through inverse/forward kinematics with per-motor clipping,
    matching the sim dynamics in mecanum_dynamics.py.
    """
    c, s = x[2], x[3]
    vx, vy, wz = x[4], x[5], x[6]

    # Commanded wheel targets (from body velocity command), clipped per motor
    wheels_target = _inv_kinematics_ca(u[0], u[1], u[2])
    wheels_target = ca.fmin(ca.fmax(wheels_target, -wheel_vel_max), wheel_vel_max)

    # Current wheel speeds (from current body velocity)
    wheels_current = _inv_kinematics_ca(vx, vy, wz)

    # Per-wheel first-order lag
    wheels_dot = (wheels_target - wheels_current) / _TAU

    # Forward kinematics of wheel derivatives → body velocity derivatives
    dvx, dvy, dwz = _fwd_kinematics_ca(wheels_dot)

    return ca.vertcat(
        vx * c - vy * s,    # ẋ
        vx * s + vy * c,    # ẏ
        -wz * s,            # ċ
         wz * c,            # ṡ
        dvx,                # v̇x
        dvy,                # v̇y
        dwz,                # ω̇z
    )


def _integrate_rk4_rover(
    x: ca.SX, u: ca.SX, dt: float,
    wheel_vel_max: float | None,
) -> ca.SX:
    """RK4 integration of the mecanum ODE."""
    def ode(x_, u_):
        if wheel_vel_max is not None:
            return _mecanum_ode_wheel(x_, u_, wheel_vel_max)
        return _mecanum_ode_simple(x_, u_)

    k1 = ode(x, u)
    k2 = ode(x + dt / 2 * k1, u)
    k3 = ode(x + dt / 2 * k2, u)
    k4 = ode(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def define_mecanum_dynamics(
    dt: float,
    wheel_vel_max: float | None = _WHEEL_VEL_MAX,
) -> tuple[ca.SX, ca.SX, ca.SX]:
    """Define discrete mecanum dynamics using RK4.

    Args:
        dt: Integration timestep [s].
        wheel_vel_max: Max wheel angular velocity [rad/s]. If None, uses
            the simple body-velocity model (no wheel-level clipping).
    """
    x = ca.SX.sym("x", NX_ROVER)
    u = ca.SX.sym("u", NU_ROVER)
    x_next = _integrate_rk4_rover(x, u, dt, wheel_vel_max)
    return x_next, x, u


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_x3_rover_params_linear_ls(
    N_horizon: int = 4,
    pos_offset_max: float = 2.0,
    vx_max: float = _VX_MAX,
    vy_max: float = _VY_MAX,
    wz_max: float = _WZ_MAX,
) -> list[AcadosParameter]:
    """Create learnable parameters for the X3 rover MPC with LINEAR_LS cost."""
    # Log-scale weight bounds
    # State: [x, y, c, s, vx, vy, ωz]
    w_state_min_log = np.array([-2., -2., -2., -2., -2., -2., -2.])
    w_state_max_log = np.array([ 2.,  2.,  1.,  1.,  1.,  1.,  1.])
    # Control: [vx_cmd, vy_cmd, ωz_cmd]
    w_ctrl_min_log = np.array([-2., -2., -2.])
    w_ctrl_max_log = np.array([ 1.,  1.,  1.])

    w_state_default_log = (w_state_min_log + w_state_max_log) / 2
    w_ctrl_default_log  = (w_ctrl_min_log  + w_ctrl_max_log)  / 2

    # Reference bounds
    yref_state_low  = np.array([-pos_offset_max, -pos_offset_max, -1., -1.,
                                 -vx_max, -vy_max, -wz_max])
    yref_state_high = np.array([ pos_offset_max,  pos_offset_max,  1.,  1.,
                                  vx_max,  vy_max,  wz_max])
    yref_state_default = np.zeros(NX_ROVER)

    yref_ctrl_low    = np.array([-vx_max, -vy_max, -wz_max])
    yref_ctrl_high   = np.array([ vx_max,  vy_max,  wz_max])
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


def get_x3_rover_learnable_param_dim() -> int:
    """Total dimension of learnable parameters.
    w_state(7) + w_ctrl(3) + yref_state(7) + yref_ctrl(3) = 20
    """
    return NX_ROVER + NU_ROVER + NX_ROVER + NU_ROVER


def export_x3_rover_ocp_linear_ls(
    param_manager: AcadosParameterManager,
    name: str = "rover_mecanum_linear_ls",
    N_horizon: int = 4,
    T_horizon: float = 0.4,
    dt: float = 0.1,
    vx_max: float = _VX_MAX,
    vy_max: float = _VY_MAX,
    wz_max: float = _WZ_MAX,
    wheel_vel_max: float | None = _WHEEL_VEL_MAX,
) -> AcadosOcp:
    """Export the X3 rover OCP for LEAP-C using LINEAR_LS cost structure."""
    ocp = AcadosOcp()
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    param_manager.assign_to_ocp(ocp)

    ocp.model.name = name
    ocp.dims.nx = NX_ROVER
    ocp.dims.nu = NU_ROVER

    x_next, x, u = define_mecanum_dynamics(dt, wheel_vel_max=wheel_vel_max)
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

    # Control box constraints: body velocity commands
    ocp.constraints.lbu    = np.array([-vx_max, -vy_max, -wz_max])
    ocp.constraints.ubu    = np.array([ vx_max,  vy_max,  wz_max])
    ocp.constraints.idxbu  = np.array([0, 1, 2])

    # State box constraints: cos/sin ∈ [-1,1], body velocities bounded
    ocp.constraints.lbx    = np.array([-1., -1., -vx_max, -vy_max, -wz_max])
    ocp.constraints.ubx    = np.array([ 1.,  1.,  vx_max,  vy_max,  wz_max])
    ocp.constraints.idxbx  = np.array([2, 3, 4, 5, 6])
    ocp.constraints.lbx_e  = np.array([-1., -1., -vx_max, -vy_max, -wz_max])
    ocp.constraints.ubx_e  = np.array([ 1.,  1.,  vx_max,  vy_max,  wz_max])
    ocp.constraints.idxbx_e = np.array([2, 3, 4, 5, 6])

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
