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
  kv    = 0.1    N·m·s/rad  velocity actuator gain  (XML: kv="0.1")
  fc    = 0.1    N·m        Coulomb friction loss   (XML: frictionloss="0.1")

MuJoCo equation of motion per wheel (angular):
  I_eff · ω̇ = kv · (ω_cmd − ω) − fc · sign(ω)

In linear wheel velocity v = r·ω:
  v̇_i = (kv·r / I_eff) · ω_cmd_i − (kv / I_eff) · v_i − (fc·r / I_eff) · sign(v_i)

Coulomb sign() smoothed via tanh(v/ε) for differentiability in JAX and the OCP.

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
_I_W: float         = 0.01         # kg·m²  armature=0.01, I_wheel≈1.5e-5 (negligible)
_KV: float          = 0.1          # N·m·s/rad  velocity gain  (XML: kv="0.1")
_FC: float          = 0.1          # N·m        Coulomb friction  (XML: frictionloss="0.1")
_BODY_MASS: float   = 0.957        # kg   total robot mass (base + wheels from XML)
_BODY_IZ: float     = 0.003387     # kg·m²  yaw inertia about base z-axis (from XML)
_COULOMB_EPS: float = 0.005        # m/s    tanh smoothing width

# ── Coupled body-wheel dynamics ─────────────────────────────────────────────
# Each wheel is coupled through the body via the no-slip constraint.
# Deriving from Newton's laws with no-slip:
#   (m + 2·I_w/r²)·v̇_body = (τ_L + τ_R)/r
#   (I_z + I_w·L²/(2r²))·ω̇_body = (τ_R − τ_L)·L/(2r)
# Converting back to wheel velocities v̇_L = v̇ − ω̇·L/2, v̇_R = v̇ + ω̇·L/2:
#   v̇_L = (α+β)·τ_L + (α−β)·τ_R
#   v̇_R = (α−β)·τ_L + (α+β)·τ_R
_M_LIN: float = _BODY_MASS + 2 * _I_W / WHEEL_RADIUS**2           # effective translation mass
_M_ROT: float = _BODY_IZ + _I_W * WHEELBASE**2 / (2 * WHEEL_RADIUS**2)  # effective yaw inertia
_ALPHA: float = 1.0 / (WHEEL_RADIUS * _M_LIN)                     # translation coupling
_BETA: float  = WHEELBASE**2 / (4.0 * WHEEL_RADIUS * _M_ROT)      # yaw coupling
_SELF: float  = _ALPHA + _BETA    # self-coupling  (≈ 3.17, was r/I_w = 3.30)
_CROSS: float = _ALPHA - _BETA    # cross-coupling (≈ −0.036, was 0)

# Legacy coefficients (for reference / OCP linearization)
_DRIVE_COEF: float   = _KV * WHEEL_RADIUS / _I_W          # ≈ 0.33
_DECAY_COEF: float   = _KV / _I_W                          # = 10.0
_COULOMB_COEF: float = _FC * WHEEL_RADIUS / _I_W           # ≈ 0.33

# MuJoCo actuator limits  (ctrlrange="-6.67 6.67" rad/s)
WHEEL_VEL_MAX: float     = 6.67
WHEEL_LIN_VEL_MAX: float = WHEEL_RADIUS * WHEEL_VEL_MAX   # ≈ 0.220 m/s

NX_ROVER = 6  # [x, y, c, s, v_L, v_R]
NU_ROVER = 2  # [ω_L_cmd, ω_R_cmd]


def _coulomb(v: jnp.ndarray) -> jnp.ndarray:
    """Smooth Coulomb friction: tanh approximation of sign(v)."""
    return jnp.tanh(v / _COULOMB_EPS)


def _ode(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Continuous-time differential-drive ODE with body-wheel coupling.

    Motor torques:  τ_i = kv·(ω_cmd_i − v_i/r) − fc·tanh(v_i/ε)
    Coupled accel:  v̇_L = _SELF·τ_L + _CROSS·τ_R
                    v̇_R = _CROSS·τ_L + _SELF·τ_R

    At steady state (v̇=0) the coupling drops out, so v_ss is unchanged.
    The coupling only affects transient response (~5% slower acceleration
    due to body mass, ~3% slower yaw due to body I_z).
    """
    c, s, v_L, v_R = x[2], x[3], x[4], x[5]
    v_body  = (v_L + v_R) * 0.5
    omega_b = (v_R - v_L) / WHEELBASE

    # Motor torques (in joint-torque domain, N·m)
    tau_L = _KV * (u[0] - v_L / WHEEL_RADIUS) - _FC * _coulomb(v_L)
    tau_R = _KV * (u[1] - v_R / WHEEL_RADIUS) - _FC * _coulomb(v_R)

    return jnp.array([
        v_body * c,
        v_body * s,
        -omega_b * s,
         omega_b * c,
        _SELF * tau_L + _CROSS * tau_R,
        _CROSS * tau_L + _SELF * tau_R,
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
