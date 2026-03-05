"""JAX unicycle kinematic model for the ground rover.

State:   [x, y, c, s, v]       (position x, position y, cos(heading), sin(heading), speed)
Control: [a, ω]                 (acceleration, yaw rate)

Dynamics (polynomial — no trig):
  ẋ = v * c
  ẏ = v * s
  ċ = -ω * s
  ṡ =  ω * c
  v̇ = a

Integrated with forward Euler at the control frequency.
After each step, (c, s) is re-normalized to the unit circle.
"""

import jax
import jax.numpy as jnp

# Physical limits (defaults; can be overridden per-config)
MAX_SPEED: float = 1.5       # m/s, max forward speed
MIN_SPEED: float = -0.5      # m/s, allow slow reverse
MAX_OMEGA: float = 1.5708    # rad/s (~π/2)
MAX_ACCEL: float = 2.0       # m/s²

NX_ROVER = 5  # [x, y, c, s, v]
NU_ROVER = 2  # [a, ω]


def rover_step(
    state: jnp.ndarray,
    control: jnp.ndarray,
    dt: float,
    max_speed: float = MAX_SPEED,
    min_speed: float = MIN_SPEED,
    max_omega: float = MAX_OMEGA,
    max_accel: float = MAX_ACCEL,
) -> jnp.ndarray:
    """Forward-Euler integrate the unicycle model for one timestep.

    Args:
        state: Rover state [x, y, c, s, v], shape (5,).
        control: Rover control [a, ω], shape (2,).
        dt: Integration timestep [s].
        max_speed: Maximum forward speed [m/s].
        min_speed: Minimum (reverse) speed [m/s].
        max_omega: Maximum yaw rate magnitude [rad/s].
        max_accel: Maximum acceleration magnitude [m/s²].

    Returns:
        Next rover state [x, y, c, s, v], shape (5,).
    """
    x, y, c, s, v = state[0], state[1], state[2], state[3], state[4]
    a, omega = control[0], control[1]

    # Clamp control inputs to physical limits
    a = jnp.clip(a, -max_accel, max_accel)
    omega = jnp.clip(omega, -max_omega, max_omega)

    # Euler integration
    v_new = jnp.clip(v + a * dt, min_speed, max_speed)
    x_new = x + v * c * dt
    y_new = y + v * s * dt
    c_new = c - omega * s * dt
    s_new = s + omega * c * dt

    # Re-normalize (c, s) to the unit circle
    norm = jnp.sqrt(c_new ** 2 + s_new ** 2)
    c_new = c_new / norm
    s_new = s_new / norm

    return jnp.stack([x_new, y_new, c_new, s_new, v_new])


# Vectorized over the (N_worlds,) batch dimension
rover_step_batched = jax.jit(
    jax.vmap(rover_step, in_axes=(0, 0, None, None, None, None, None))
)
