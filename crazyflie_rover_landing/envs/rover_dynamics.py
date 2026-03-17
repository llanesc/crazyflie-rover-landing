"""JAX differential-drive dynamics for the TurtleBot3 Burger ground rover.

State:   [x, y, c, s, v_L, v_R]   (NX_ROVER = 6)
  x, y  : world-frame position [m]
  c, s  : cos(θ), sin(θ)  heading
  v_L   : left-wheel linear velocity  [m/s]  (= r · ω_L)
  v_R   : right-wheel linear velocity [m/s]  (= r · ω_R)

Control: [ω_L_cmd, ω_R_cmd]   (NU_ROVER = 2)  — wheel angular velocity commands [rad/s]

Physical parameters (TurtleBot3 Burger MuJoCo XML):
  r     = 0.033  m     wheel radius
  L     = 0.160  m     wheelbase  (2 × 0.080 m half-wheelbase in XML)
  I_eff = 0.01   kg·m² armature + wheel inertia  (armature=0.01 >> I_wheel≈1.5e-5)
  kv    = 0.1    N·m·s/rad  velocity actuator gain
  b     = 0.1    N·m   viscous friction (approximates frictionloss for differentiability)

First-order wheel dynamics (each wheel i):
  v̇_i = (kv·r·ω_cmd_i − (kv + b)·v_i) / I_eff

Kinematics:
  ẋ = (v_L + v_R)/2 · c    ẏ = (v_L + v_R)/2 · s
  ċ = −(v_R − v_L)/L · s   ṡ =  (v_R − v_L)/L · c

Integrated with RK4. (c, s) re-normalized after each step.
"""

import jax
import jax.numpy as jnp

# ── Physical constants (TurtleBot3 Burger) ──────────────────────────────────
WHEEL_RADIUS: float = 0.033        # m
WHEELBASE: float    = 0.160        # m  (2 × 0.080 m from XML)
_I_EFF: float       = 0.01         # kg·m²  armature=0.01, I_wheel≈1.5e-5 (negligible)
_KV: float          = 0.1          # N·m·s/rad  (kv in XML)
_B: float           = 0.1          # N·m        (frictionloss, treated as viscous)

# Precomputed coefficients
_DRIVE_COEF: float = _KV * WHEEL_RADIUS / _I_EFF   # m·s⁻¹ per rad/s command  ≈ 0.33
_DECAY_COEF: float = (_KV + _B) / _I_EFF            # s⁻¹  ≈ 20.0

# MuJoCo actuator limits  (ctrlrange="-6.67 6.67" rad/s)
WHEEL_VEL_MAX: float     = 6.67
WHEEL_LIN_VEL_MAX: float = WHEEL_RADIUS * WHEEL_VEL_MAX   # ≈ 0.220 m/s

NX_ROVER = 6  # [x, y, c, s, v_L, v_R]
NU_ROVER = 2  # [ω_L_cmd, ω_R_cmd]


def _ode(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Continuous-time differential-drive ODE."""
    c, s, v_L, v_R = x[2], x[3], x[4], x[5]
    v_body  = (v_L + v_R) * 0.5
    omega_b = (v_R - v_L) / WHEELBASE
    return jnp.array([
        v_body * c,
        v_body * s,
        -omega_b * s,
         omega_b * c,
        _DRIVE_COEF * u[0] - _DECAY_COEF * v_L,
        _DRIVE_COEF * u[1] - _DECAY_COEF * v_R,
    ])


def rover_step(
    state: jnp.ndarray,
    control: jnp.ndarray,
    dt: float,
    wheel_vel_max: float = WHEEL_VEL_MAX,
) -> jnp.ndarray:
    """RK4-integrate the differential-drive model for one timestep.

    Args:
        state:         [x, y, c, s, v_L, v_R], shape (6,).
        control:       [ω_L_cmd, ω_R_cmd] in rad/s, shape (2,).
        dt:            Timestep [s].
        wheel_vel_max: Max wheel angular velocity command [rad/s].

    Returns:
        Next state [x, y, c, s, v_L, v_R], shape (6,).
    """
    u = jnp.clip(control, -wheel_vel_max, wheel_vel_max)

    k1 = _ode(state, u)
    k2 = _ode(state + dt / 2 * k1, u)
    k3 = _ode(state + dt / 2 * k2, u)
    k4 = _ode(state + dt * k3, u)
    nxt = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Re-normalize (c, s)
    norm = jnp.sqrt(nxt[2] ** 2 + nxt[3] ** 2)
    nxt = nxt.at[2].set(nxt[2] / norm)
    nxt = nxt.at[3].set(nxt[3] / norm)
    return nxt


# Vectorized over the (N_worlds,) batch dimension
rover_step_batched = jax.jit(
    jax.vmap(rover_step, in_axes=(0, 0, None, None))
)
