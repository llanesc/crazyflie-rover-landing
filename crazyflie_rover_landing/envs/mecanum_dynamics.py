"""JAX mecanum-drive dynamics for the Yahboom RosMaster X3.

State:   [x, y, c, s, vx, vy, ωz]   (NX_ROVER = 7)
  x, y  : world-frame position [m]
  c, s  : cos(θ), sin(θ)  heading
  vx    : body-frame forward velocity [m/s]
  vy    : body-frame lateral velocity [m/s]
  ωz    : yaw rate [rad/s]

Control: [vx_cmd, vy_cmd, ωz_cmd]   (NU_ROVER = 3)
  Body-frame velocity commands matching the real /cmd_vel interface.

Physical parameters (RosMaster X3, from URDF + ROS2 controller config):
  r     = 0.0325  m     wheel radius (65 mm mecanum wheel)
  l     = 0.08    m     half-wheelbase  (160 mm / 2, front-to-rear axle)
  d     = 0.0845  m     half-track width (169 mm / 2, left-to-right wheel center)
  K     = l + d = 0.1645 m   combined kinematic parameter

Motor: JGB37-520 DC gear motor, 1:30 gear ratio, ~333 RPM no-load.
  ω_max ≈ 34.9 rad/s per wheel at output shaft.

Velocity command limits (from ROS /cmd_vel bridge):
  vx_cmd_max = 1.0  m/s
  vy_cmd_max = 1.0  m/s
  ωz_cmd_max = 5.0  rad/s

Control pipeline (matches real robot architecture):
  1. Clip body velocity commands to /cmd_vel limits
  2. Inverse kinematics: body cmds → 4 wheel angular velocities
  3. Clip each wheel to [-ω_max, ω_max]  (physical motor limit)
  4. Forward kinematics: clipped wheel speeds → achievable body velocities
  5. First-order motor response tracking achievable velocities
  6. Pose update in world frame via rotation

The first-order time constant models the STM32 PID loop response.

Integrated with RK4.  (c, s) re-normalized after each step.
"""

import jax
import jax.numpy as jnp

# ── Physical constants (RosMaster X3) ────────────────────────────────────────
WHEEL_RADIUS: float   = 0.0325       # m   (65 mm mecanum wheel)
HALF_WHEELBASE: float  = 0.08         # m   (160 mm / 2, front-to-rear)
HALF_TRACK: float      = 0.0845       # m   (169 mm / 2, left-to-right)
_K: float              = HALF_WHEELBASE + HALF_TRACK  # 0.1645 m

# Motor response (effective first-order time constant from STM32 PID)
_TAU_MOTOR: float = 0.1              # s

# Wheel speed limit (333 RPM motor with 1:30 gear ratio ≈ 34.9 rad/s)
WHEEL_VEL_MAX: float = 34.9          # rad/s

# Body velocity command limits (ROS /cmd_vel)
VX_CMD_MAX: float = 1.0              # m/s
VY_CMD_MAX: float = 1.0              # m/s
WZ_CMD_MAX: float = 5.0              # rad/s

NX_ROVER = 7  # [x, y, c, s, vx, vy, ωz]
NU_ROVER = 3  # [vx_cmd, vy_cmd, ωz_cmd]


# ── Mecanum kinematics ───────────────────────────────────────────────────────

def _inv_kinematics(vx: float, vy: float, wz: float) -> jnp.ndarray:
    """Body-frame velocities → 4 wheel angular velocities [rad/s].

    Convention (standard mecanum, rollers at 45°):
      FL = (1/r) * (vx - vy - K*ωz)
      FR = (1/r) * (vx + vy + K*ωz)
      BL = (1/r) * (vx + vy - K*ωz)
      BR = (1/r) * (vx - vy + K*ωz)
    """
    r_inv = 1.0 / WHEEL_RADIUS
    return jnp.array([
        r_inv * (vx - vy - _K * wz),   # front-left
        r_inv * (vx + vy + _K * wz),   # front-right
        r_inv * (vx + vy - _K * wz),   # back-left
        r_inv * (vx - vy + _K * wz),   # back-right
    ])


def _fwd_kinematics(wheels: jnp.ndarray) -> tuple[float, float, float]:
    """4 wheel angular velocities → body-frame velocities (vx, vy, ωz).

    Pseudo-inverse of the inverse kinematics (exact for ideal rollers).
    """
    r4 = WHEEL_RADIUS / 4.0
    w_fl, w_fr, w_bl, w_br = wheels[0], wheels[1], wheels[2], wheels[3]
    vx = r4 * (w_fl + w_fr + w_bl + w_br)
    vy = r4 * (-w_fl + w_fr + w_bl - w_br)
    wz = r4 / _K * (-w_fl + w_fr - w_bl + w_br)
    return vx, vy, wz


# ── ODE ──────────────────────────────────────────────────────────────────────

def _ode(x: jnp.ndarray, u: jnp.ndarray, wheel_vel_max: float) -> jnp.ndarray:
    """Continuous-time mecanum ODE with per-wheel first-order motor lag.

    The STM32 PID runs independently on each wheel, so the first-order
    response is modeled per-wheel:
      1. Inverse kinematics: body cmd → 4 wheel targets (clipped)
      2. Inverse kinematics: current body vel → 4 current wheel speeds
      3. Per-wheel lag: ω̇_i = (ω_target_i − ω_i) / τ
      4. Forward kinematics: wheel speed derivatives → body vel derivatives

    Args:
        x: [x, y, c, s, vx, vy, ωz], shape (7,).
        u: [vx_cmd, vy_cmd, ωz_cmd] (already clipped to cmd limits).
        wheel_vel_max: Max wheel angular velocity [rad/s].

    Returns:
        dx/dt, shape (7,).
    """
    c, s = x[2], x[3]
    vx, vy, wz = x[4], x[5], x[6]

    # Commanded wheel targets (from body velocity command)
    wheels_target = _inv_kinematics(u[0], u[1], u[2])
    wheels_target = jnp.clip(wheels_target, -wheel_vel_max, wheel_vel_max)

    # Current wheel speeds (from current body velocity)
    wheels_current = _inv_kinematics(vx, vy, wz)

    # Per-wheel first-order lag: ω̇_i = (target_i − current_i) / τ
    wheels_dot = (wheels_target - wheels_current) / _TAU_MOTOR

    # Forward kinematics of wheel *derivatives* → body velocity derivatives
    dvx, dvy, dwz = _fwd_kinematics(wheels_dot)

    # Pose update in world frame
    dx = vx * c - vy * s
    dy = vx * s + vy * c
    dc = -wz * s
    ds = wz * c

    return jnp.array([dx, dy, dc, ds, dvx, dvy, dwz])


# ── Integrator ───────────────────────────────────────────────────────────────

def mecanum_step(
    state: jnp.ndarray,
    control: jnp.ndarray,
    dt: float,
    wheel_vel_max: float = WHEEL_VEL_MAX,
) -> jnp.ndarray:
    """RK4-integrate the mecanum model for one timestep.

    Args:
        state:         [x, y, c, s, vx, vy, ωz], shape (7,).
        control:       [vx_cmd, vy_cmd, ωz_cmd] in m/s and rad/s, shape (3,).
        dt:            Timestep [s].
        wheel_vel_max: Max wheel angular velocity [rad/s].

    Returns:
        Next state [x, y, c, s, vx, vy, ωz], shape (7,).
    """
    # Clip body velocity commands to /cmd_vel limits
    u = jnp.array([
        jnp.clip(control[0], -VX_CMD_MAX, VX_CMD_MAX),
        jnp.clip(control[1], -VY_CMD_MAX, VY_CMD_MAX),
        jnp.clip(control[2], -WZ_CMD_MAX, WZ_CMD_MAX),
    ])

    k1 = _ode(state, u, wheel_vel_max)
    k2 = _ode(state + dt / 2 * k1, u, wheel_vel_max)
    k3 = _ode(state + dt / 2 * k2, u, wheel_vel_max)
    k4 = _ode(state + dt * k3, u, wheel_vel_max)
    nxt = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Re-normalize (c, s)
    norm = jnp.sqrt(nxt[2] ** 2 + nxt[3] ** 2)
    nxt = nxt.at[2].set(nxt[2] / norm)
    nxt = nxt.at[3].set(nxt[3] / norm)
    return nxt


# Vectorized over the (N_worlds,) batch dimension
mecanum_step_batched = jax.jit(
    jax.vmap(mecanum_step, in_axes=(0, 0, None, None))
)
