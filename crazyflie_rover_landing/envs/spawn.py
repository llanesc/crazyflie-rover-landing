"""JIT-compiled spawn functions for the drone-rover landing environment.

All spawn functions return:
  - drone_pos: (N, 3)        initial drone positions [x, y, z]
  - rover_state: (N, 5)      initial rover state [x, y, cos(θ), sin(θ), v]

The signature for a SpawnFn is: (key, N) -> (drone_pos, rover_state).
"""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp


# Type alias for spawn functions used by LandingEnv
SpawnFn = Callable[[jax.Array, int], tuple[jnp.ndarray, jnp.ndarray]]
"""(key, N) -> (drone_pos (N,3), rover_state (N,5))"""


# =============================================================================
# Pure implementations (called inside JIT'd functions)
# =============================================================================

def _drone_box_random_impl(
    key: jax.Array,
    N: int,
    x_half: float,
    y_half: float,
    z_min: float,
    z_max: float,
) -> jnp.ndarray:
    """Sample drone positions uniformly within a box.

    Returns:
        drone_pos: (N, 3) array of [x, y, z] positions.
    """
    key, xk, yk, zk = jax.random.split(key, 4)
    x = jax.random.uniform(xk, (N,), minval=-x_half, maxval=x_half)
    y = jax.random.uniform(yk, (N,), minval=-y_half, maxval=y_half)
    z = jax.random.uniform(zk, (N,), minval=z_min, maxval=z_max)
    return jnp.stack([x, y, z], axis=-1)


def _rover_uniform_impl(
    key: jax.Array,
    N: int,
    x_half: float,
    y_half: float,
    min_speed: float,
    max_speed: float,
) -> jnp.ndarray:
    """Sample rover initial state uniformly within the arena.

    State: [x, y, cos(θ), sin(θ), v]
      - Position uniformly in ±x_half × ±y_half.
      - Heading uniformly in [0, 2π).
      - Speed uniformly in [min_speed, max_speed].

    Returns:
        rover_state: (N, 5) array.
    """
    key, xk, yk, tk, vk = jax.random.split(key, 5)
    x = jax.random.uniform(xk, (N,), minval=-x_half, maxval=x_half)
    y = jax.random.uniform(yk, (N,), minval=-y_half, maxval=y_half)
    theta = jax.random.uniform(tk, (N,), minval=0.0, maxval=2.0 * jnp.pi)
    v = jax.random.uniform(vk, (N,), minval=min_speed, maxval=max_speed)
    return jnp.stack([x, y, jnp.cos(theta), jnp.sin(theta), v], axis=-1)


def _rover_stationary_impl(
    key: jax.Array,
    N: int,
    x_half: float,
    y_half: float,
) -> jnp.ndarray:
    """Spawn rover at a random XY position with zero velocity.

    Returns:
        rover_state: (N, 5) with v=0.
    """
    key, xk, yk, tk = jax.random.split(key, 4)
    x = jax.random.uniform(xk, (N,), minval=-x_half, maxval=x_half)
    y = jax.random.uniform(yk, (N,), minval=-y_half, maxval=y_half)
    theta = jax.random.uniform(tk, (N,), minval=0.0, maxval=2.0 * jnp.pi)
    v = jnp.zeros((N,))
    return jnp.stack([x, y, jnp.cos(theta), jnp.sin(theta), v], axis=-1)


# =============================================================================
# JIT-compiled spawn function factory
# =============================================================================

def create_spawn_fn_from_config(spawn_config: dict) -> SpawnFn:
    """Create a JIT-compiled spawn function from a configuration dict.

    Expected keys in spawn_config:
        drone:
            x_half: float     (default 2.5)
            y_half: float     (default 2.5)
            z_min:  float     (default 0.5)
            z_max:  float     (default 3.0)
        rover:
            stationary: bool  (default false) — if true, spawn with v=0
            x_half: float     (default 2.5)
            y_half: float     (default 2.5)
            min_speed: float  (default -1.5) — ignored when stationary=True
            max_speed: float  (default 1.5) — ignored when stationary=True

    Args:
        spawn_config: Configuration dict with "drone" and "rover" sub-dicts.

    Returns:
        JIT-compiled spawn function (key, N) -> (drone_pos, rover_state).
    """
    drone_cfg = spawn_config.get("drone", {})
    rover_cfg = spawn_config.get("rover", {})

    # Drone parameters
    d_x_half = float(drone_cfg.get("x_half", 2.5))
    d_y_half = float(drone_cfg.get("y_half", 2.5))
    d_z_min = float(drone_cfg.get("z_min", 0.5))
    d_z_max = float(drone_cfg.get("z_max", 3.0))

    # Rover parameters
    r_stationary = bool(rover_cfg.get("stationary", False))
    r_x_half = float(rover_cfg.get("x_half", 2.5))
    r_y_half = float(rover_cfg.get("y_half", 2.5))
    r_max_speed = float(rover_cfg.get("max_speed", 1.5))
    r_min_speed = float(rover_cfg.get("min_speed", -r_max_speed))

    if r_stationary:
        @partial(jax.jit, static_argnames=["N"])
        def spawn_fn(key: jax.Array, N: int) -> tuple[jnp.ndarray, jnp.ndarray]:
            key, dk, rk = jax.random.split(key, 3)
            drone_pos = _drone_box_random_impl(dk, N, d_x_half, d_y_half, d_z_min, d_z_max)
            rover_state = _rover_stationary_impl(rk, N, r_x_half, r_y_half)
            return drone_pos, rover_state
    else:
        @partial(jax.jit, static_argnames=["N"])
        def spawn_fn(key: jax.Array, N: int) -> tuple[jnp.ndarray, jnp.ndarray]:
            key, dk, rk = jax.random.split(key, 3)
            drone_pos = _drone_box_random_impl(dk, N, d_x_half, d_y_half, d_z_min, d_z_max)
            rover_state = _rover_uniform_impl(rk, N, r_x_half, r_y_half, r_min_speed, r_max_speed)
            return drone_pos, rover_state

    return spawn_fn


def create_default_spawn_fn(
    drone_x_half: float = 2.5,
    drone_y_half: float = 2.5,
    drone_z_min: float = 0.5,
    drone_z_max: float = 3.0,
    rover_x_half: float = 2.5,
    rover_y_half: float = 2.5,
    rover_min_speed: float = -1.5,
    rover_max_speed: float = 1.5,
    rover_stationary: bool = False,
) -> SpawnFn:
    """Create a spawn function from explicit keyword arguments."""
    cfg = {
        "drone": {
            "x_half": drone_x_half,
            "y_half": drone_y_half,
            "z_min": drone_z_min,
            "z_max": drone_z_max,
        },
        "rover": {
            "stationary": rover_stationary,
            "x_half": rover_x_half,
            "y_half": rover_y_half,
            "min_speed": rover_min_speed,
            "max_speed": rover_max_speed,
        },
    }
    return create_spawn_fn_from_config(cfg)
