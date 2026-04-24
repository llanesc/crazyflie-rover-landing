"""Wind disturbance model: constant wind + OU gusts + Dryden turbulence.

Produces a 3D wind velocity in world frame. Since the physics model already
computes drag from drone velocity, the wind correction is -drag_matrix @ wind_body
(the additional drag due to wind alone).

Based on the CrazySim WindModel implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

_TURBULENCE_SIGMA = {
    "none": 0.0,
    "light": 0.5,
    "moderate": 1.5,
    "severe": 3.0,
}


class WindModel:
    """Vectorized wind model for N parallel worlds."""

    def __init__(
        self,
        n_worlds: int,
        wind_speed: float = 0.0,
        wind_direction: float = 0.0,
        gust_intensity: float = 0.0,
        gust_correlation_time: float = 4.0,
        turbulence_level: str = "none",
        turbulence_time_constant: float = 5.0,
        dt: float = 0.01,
    ):
        self.n_worlds = n_worlds
        self.dt = dt

        dir_rad = np.radians(wind_direction)
        self.constant_wind = jnp.array(
            [wind_speed * np.cos(dir_rad),
             wind_speed * np.sin(dir_rad),
             0.0]
        )

        self.gust_intensity = gust_intensity
        alpha = dt / gust_correlation_time
        self.gust_decay = 1.0 - alpha
        self.gust_noise_scale = gust_intensity * np.sqrt(2.0 * alpha)

        turb_sigma = _TURBULENCE_SIGMA.get(turbulence_level, 0.0)
        self.turb_sigma = turb_sigma
        turb_alpha = dt / turbulence_time_constant
        self.turb_decay = 1.0 - turb_alpha
        self.turb_noise_scale = turb_sigma * np.sqrt(2.0 * turb_alpha)

        # State: (n_worlds, 1, 3) to broadcast with drone dims
        self._gust_state = jnp.zeros((n_worlds, 1, 3))
        self._turb_state = jnp.zeros((n_worlds, 1, 3))

    def reset(self):
        self._gust_state = jnp.zeros((self.n_worlds, 1, 3))
        self._turb_state = jnp.zeros((self.n_worlds, 1, 3))

    def step(self, rng_key: jax.Array) -> jnp.ndarray:
        """Advance one control step. Returns wind velocity (n_worlds, 1, 3)."""
        shape = (self.n_worlds, 1, 3)

        wind = jnp.broadcast_to(self.constant_wind, shape)

        if self.gust_intensity > 0:
            rng_key, k = jax.random.split(rng_key)
            noise = jax.random.normal(k, shape)
            self._gust_state = (
                self._gust_state * self.gust_decay + noise * self.gust_noise_scale
            )
            wind = wind + self._gust_state

        if self.turb_sigma > 0:
            rng_key, k = jax.random.split(rng_key)
            noise = jax.random.normal(k, shape)
            self._turb_state = (
                self._turb_state * self.turb_decay + noise * self.turb_noise_scale
            )
            wind = wind + self._turb_state

        return wind


def compute_wind_drag_force(
    quat: jnp.ndarray,
    wind_vel: jnp.ndarray,
    drag_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the wind-induced drag correction force in world frame.

    The physics model already applies drag from drone velocity:
        F_physics = drag_matrix @ vel_body
    With wind, the correct total drag should use relative airspeed:
        F_correct = drag_matrix @ (vel_body - wind_body)
    So the correction we inject is the difference:
        F_correction = -drag_matrix @ wind_body

    Args:
        quat: Drone quaternion wxyz (..., 4)
        wind_vel: Wind velocity in world frame (..., 3)
        drag_matrix: 3x3 drag matrix in body frame (N*s/m)

    Returns:
        Wind drag correction force in world frame (..., 3)
    """
    # Quaternion to rotation matrix (wxyz convention)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    # R: body-to-world rotation matrix
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    # R^T @ wind_vel  (world to body)
    w_body_x = r00 * wind_vel[..., 0] + r10 * wind_vel[..., 1] + r20 * wind_vel[..., 2]
    w_body_y = r01 * wind_vel[..., 0] + r11 * wind_vel[..., 1] + r21 * wind_vel[..., 2]
    w_body_z = r02 * wind_vel[..., 0] + r12 * wind_vel[..., 1] + r22 * wind_vel[..., 2]

    # Correction: -drag_matrix @ wind_body (drag_matrix has negative entries, so this pushes drone downwind)
    f_body_x = -drag_matrix[0, 0] * w_body_x
    f_body_y = -drag_matrix[1, 1] * w_body_y
    f_body_z = -drag_matrix[2, 2] * w_body_z

    # R @ f_body (body to world)
    f_world_x = r00 * f_body_x + r01 * f_body_y + r02 * f_body_z
    f_world_y = r10 * f_body_x + r11 * f_body_y + r12 * f_body_z
    f_world_z = r20 * f_body_x + r21 * f_body_y + r22 * f_body_z

    return jnp.stack([f_world_x, f_world_y, f_world_z], axis=-1)
