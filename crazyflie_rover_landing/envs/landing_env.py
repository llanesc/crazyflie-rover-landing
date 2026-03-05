"""Cooperative drone-rover landing environment using Crazyflow + JAX.

Two cooperative agents:
  - "drone": Crazyflie CF2X_T350, controlled via attitude commands from ACMPC.
  - "rover": Unicycle ground vehicle, controlled via [a, ω] from ACMPC.

The drone earns a bonus when it lands on the rover with low velocity.
Both agents receive the same cooperative reward at each step.

N parallel worlds are simulated simultaneously using JAX vectorization.
"""

from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.physics import Physics
from crazyflow.control.control import Control
from crazyflow.randomize import randomize_mass, randomize_inertia
from crazyflow.utils import leaf_replace
from drone_models.core import load_params

from crazyflie_rover_landing.envs.landing_config import LandingEnvConfig
from crazyflie_rover_landing.envs.rover_dynamics import rover_step_batched
from crazyflie_rover_landing.envs.spawn import SpawnFn, create_default_spawn_fn


# =============================================================================
# JIT-compiled helpers
# =============================================================================

@jax.jit
def _jit_quat_to_rpy(quat: jnp.ndarray) -> jnp.ndarray:
    """Convert quaternion (xyzw) to roll-pitch-yaw.

    Args:
        quat: (..., 4) in xyzw order.

    Returns:
        rpy: (..., 3) roll-pitch-yaw in radians.
    """
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = jnp.arcsin(jnp.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return jnp.stack([roll, pitch, yaw], axis=-1)


@jax.jit
def _jit_quat_to_matrix(quat: jnp.ndarray) -> jnp.ndarray:
    """Convert quaternion (xyzw) to 3×3 rotation matrix.

    Args:
        quat: (..., 4) in xyzw order.

    Returns:
        R: (..., 3, 3)
    """
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    R00 = 1 - 2 * (y * y + z * z)
    R01 = 2 * (x * y - w * z)
    R02 = 2 * (x * z + w * y)
    R10 = 2 * (x * y + w * z)
    R11 = 1 - 2 * (x * x + z * z)
    R12 = 2 * (y * z - w * x)
    R20 = 2 * (x * z - w * y)
    R21 = 2 * (y * z + w * x)
    R22 = 1 - 2 * (x * x + y * y)
    row0 = jnp.stack([R00, R01, R02], axis=-1)
    row1 = jnp.stack([R10, R11, R12], axis=-1)
    row2 = jnp.stack([R20, R21, R22], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


@jax.jit
def _jit_ang_vel_to_rpy_rates(
    quat: jnp.ndarray,
    ang_vel: jnp.ndarray,
) -> jnp.ndarray:
    """Convert body angular velocity to RPY rates via Euler angle Jacobian.

    Args:
        quat: (..., 4) xyzw quaternion.
        ang_vel: (..., 3) body angular velocity [wx, wy, wz].

    Returns:
        rpy_rates: (..., 3) [roll_rate, pitch_rate, yaw_rate].
    """
    rpy = _jit_quat_to_rpy(quat)
    roll, pitch = rpy[..., 0], rpy[..., 1]
    cp = jnp.cos(pitch)
    sp = jnp.sin(pitch)
    cr = jnp.cos(roll)
    sr = jnp.sin(roll)
    # Jacobian: rpy_dot = J_inv @ ang_vel  (body rates)
    roll_rate = ang_vel[..., 0] + sp / cp * (sr * ang_vel[..., 1] + cr * ang_vel[..., 2])
    pitch_rate = cr * ang_vel[..., 1] - sr * ang_vel[..., 2]
    yaw_rate = (sr * ang_vel[..., 1] + cr * ang_vel[..., 2]) / (cp + 1e-6)
    return jnp.stack([roll_rate, pitch_rate, yaw_rate], axis=-1)


@jax.jit
def _jit_check_landing(
    drone_pos: jnp.ndarray,
    drone_vel: jnp.ndarray,
    drone_rpy: jnp.ndarray,
    rover_state: jnp.ndarray,
    rover_height: float,
    platform_radius: float,
    z_tol: float,
    vel_xy_tol: float,
    vel_z_tol: float,
    attitude_tol: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Check landing and crash conditions for each world.

    Landing is successful when:
      - horizontal distance < platform_radius (drone is on the pad)
      - drone z is within [rover_height - 0.05, rover_height + z_tol]
      - relative XY speed (drone minus rover) < vel_xy_tol
      - relative Z velocity is downward (<=0) and |rel_vz| < vel_z_tol
      - |roll| < attitude_tol AND |pitch| < attitude_tol

    Crash = drone hits the ground anywhere (z below rover_height + z_tol)
    without satisfying landing conditions.

    Args:
        drone_pos: (N, 3) drone positions.
        drone_vel: (N, 3) drone velocities (world frame).
        drone_rpy: (N, 3) drone roll-pitch-yaw [rad].
        rover_state: (N, 4) rover [x, y, θ, v].
        rover_height: Height of landing pad surface above ground [m].
        platform_radius: Rover landing pad radius [m].
        z_tol: Max height above rover pad for success [m].
        vel_xy_tol: Max relative XY speed at touchdown [m/s].
        vel_z_tol: Max descent speed at touchdown [m/s].
        attitude_tol: Max |roll| and |pitch| at touchdown [rad].

    Returns:
        landed: (N,) bool — successful soft landing.
        crashed: (N,) bool — ground contact that isn't a landing.
    """
    rover_xy = rover_state[:, :2]

    # Rover velocity in world frame from unicycle state [x, y, c, s, v]
    c = rover_state[:, 2]
    s = rover_state[:, 3]
    v = rover_state[:, 4]
    rover_vx = v * c
    rover_vy = v * s

    # Relative velocity (drone minus rover)
    rel_vx = drone_vel[:, 0] - rover_vx
    rel_vy = drone_vel[:, 1] - rover_vy
    rel_vz = drone_vel[:, 2]  # rover has no vertical velocity

    rel_speed_xy = jnp.sqrt(rel_vx ** 2 + rel_vy ** 2)

    horiz_dist = jnp.linalg.norm(drone_pos[:, :2] - rover_xy, axis=-1)

    # Drone is near ground level (at or below pad surface + tolerance)
    near_ground = drone_pos[:, 2] < rover_height + z_tol

    on_pad = (horiz_dist < platform_radius) & near_ground
    # Only allow downward or zero vertical velocity (rel_vz <= 0), not upward
    low_speed = (
        (rel_speed_xy < vel_xy_tol)
        & (rel_vz <= 0.0)
        & (jnp.abs(rel_vz) < vel_z_tol)
    )
    level_attitude = (jnp.abs(drone_rpy[:, 0]) < attitude_tol) & (jnp.abs(drone_rpy[:, 1]) < attitude_tol)

    landed = on_pad & low_speed & level_attitude
    crashed = near_ground & ~landed

    return landed, crashed


@jax.jit
def _jit_check_oob(
    drone_pos: jnp.ndarray,
    map_half_x: float,
    map_half_y: float,
    z_max: float,
) -> jnp.ndarray:
    """Check if drone is out of the arena bounds (XY or altitude ceiling).

    Ground contact is handled separately by crash detection.

    Args:
        drone_pos: (N, 3) drone positions.
        map_half_x: Half-width in X [m].
        map_half_y: Half-width in Y [m].
        z_max: Maximum altitude [m].

    Returns:
        oob: (N,) bool.
    """
    return (
        (jnp.abs(drone_pos[:, 0]) > map_half_x)
        | (jnp.abs(drone_pos[:, 1]) > map_half_y)
        | (drone_pos[:, 2] > z_max)
    )


@jax.jit
def _jit_clamp_rover(
    rover_state: jnp.ndarray,
    map_half_x: float,
    map_half_y: float,
) -> jnp.ndarray:
    """Clamp rover position to arena boundaries.

    When the rover hits a boundary, its velocity is set to zero.

    Args:
        rover_state: (N, 5) [x, y, c, s, v].
        map_half_x: Half-width in X [m].
        map_half_y: Half-width in Y [m].

    Returns:
        Updated rover_state with clamped position and zeroed velocity on hit.
    """
    x = rover_state[:, 0]
    y = rover_state[:, 1]
    c = rover_state[:, 2]
    s = rover_state[:, 3]
    v = rover_state[:, 4]

    hit_x = (x < -map_half_x) | (x > map_half_x)
    hit_y = (y < -map_half_y) | (y > map_half_y)
    hit_any = hit_x | hit_y

    x_clamped = jnp.clip(x, -map_half_x, map_half_x)
    y_clamped = jnp.clip(y, -map_half_y, map_half_y)
    v_clamped = jnp.where(hit_any, 0.0, v)

    return jnp.stack([x_clamped, y_clamped, c, s, v_clamped], axis=-1)


@jax.jit
def _jit_compute_rewards(
    drone_pos: jnp.ndarray,
    drone_vel: jnp.ndarray,
    drone_rpy: jnp.ndarray,
    drone_cmd: jnp.ndarray,
    last_drone_cmd: jnp.ndarray,
    rover_state: jnp.ndarray,
    rover_cmd: jnp.ndarray,
    last_rover_cmd: jnp.ndarray,
    landed: jnp.ndarray,
    crashed: jnp.ndarray,
    oob: jnp.ndarray,
    rover_height: float,
    reward_landing: float,
    reward_crash: float,
    reward_boundary: float,
    reward_proximity_coef: float,
    reward_proximity_decay: float,
    reward_angle_coef: float,
    reward_action_smoothness_thrust: float,
    reward_action_smoothness_rpy: float,
    reward_action_smoothness_accel: float,
    reward_action_smoothness_yawrate: float,
    reward_landing_velocity_coef: float,
    corridor_radius: float,
    corridor_transition: float,
    max_descent_speed: float,
    reward_descent_speed_coef: float,
    reward_altitude_floor_coef: float,
    reward_time_penalty: float,
    reward_rover_stillness_coef: float,
    reward_rover_yawrate_coef: float,
    reward_drone_velocity_coef: float,
    reward_rover_boundary_coef: float,
    map_half_x: float,
    map_half_y: float,
) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
    """Compute per-agent rewards with phased navigation-then-landing structure.

    Rewards are split into team (shared) + agent-specific components.
    Drone gets team + drone-only penalties; rover gets team + rover-only penalties.

    Phase 1 — Navigation (outside corridor): XY proximity only, altitude floor.
    Phase 2 — Landing (inside corridor): XY + Z proximity, descent speed limit.
    Smooth sigmoid transition at corridor boundary prevents chattering.

    Returns:
        total_reward: (N,) float32.
        components: dict of individual reward components, each (N,).
    """
    rover_xy = rover_state[:, :2]
    c = rover_state[:, 2]
    s = rover_state[:, 3]
    v = rover_state[:, 4]

    # Distances
    horiz_dist = jnp.linalg.norm(drone_pos[:, :2] - rover_xy, axis=-1)
    vert_dist = jnp.abs(drone_pos[:, 2] - rover_height)

    # Corridor blend: sigmoid transition from navigation to landing phase
    # z_weight ≈ 0 when far from rover, ≈ 1 when inside corridor
    z_weight = jax.nn.sigmoid((corridor_radius - horiz_dist) / corridor_transition)

    # XY proximity — always active
    r_xy = reward_proximity_coef * jnp.exp(-reward_proximity_decay * horiz_dist)

    # Z proximity — only inside corridor (smoothly blended)
    r_z = reward_proximity_coef * z_weight * jnp.exp(-reward_proximity_decay * vert_dist)

    # Descent speed penalty — only inside corridor
    # Penalize descent speed exceeding max_descent_speed
    descent_speed = jnp.maximum(-drone_vel[:, 2], 0.0)  # positive when going down
    excess_speed = jnp.maximum(descent_speed - max_descent_speed, 0.0)
    r_descent = -reward_descent_speed_coef * z_weight * excess_speed ** 2

    # Altitude floor — only outside corridor (navigation phase)
    # Gently discourage premature descent
    nav_weight = 1.0 - z_weight
    altitude_floor = rover_height + 0.3
    below_floor = jnp.maximum(altitude_floor - drone_pos[:, 2], 0.0)
    r_altitude = -reward_altitude_floor_coef * nav_weight * below_floor

    # Rover stillness — penalize rover speed when drone is in landing corridor
    rover_speed = jnp.abs(rover_state[:, 4])
    r_rover_stillness = -reward_rover_stillness_coef * z_weight * rover_speed ** 2

    # Rover yaw rate penalty — penalize heading changes when drone is in landing corridor
    rover_omega = rover_cmd[:, 1]  # yaw rate command
    r_rover_yawrate = -reward_rover_yawrate_coef * z_weight * rover_omega ** 2

    # Rover boundary penalty: penalize rover being at arena edge
    rover_at_boundary = (
        (jnp.abs(rover_state[:, 0]) >= map_half_x - 0.01) |
        (jnp.abs(rover_state[:, 1]) >= map_half_y - 0.01)
    )
    r_rover_boundary = jnp.where(rover_at_boundary, -reward_rover_boundary_coef, 0.0)

    # Drone velocity penalty: penalize fast flight
    drone_speed_sq = jnp.sum(drone_vel ** 2, axis=-1)
    r_drone_velocity = -reward_drone_velocity_coef * drone_speed_sq

    # Angle penalty: penalize roll/pitch deviation
    r_angle = -reward_angle_coef * (drone_rpy[:, 0] ** 2 + drone_rpy[:, 1] ** 2)

    # Action smoothness (drone): per-actuator penalties
    delta_drone = drone_cmd - last_drone_cmd
    r_smooth_drone = (
        -reward_action_smoothness_rpy * jnp.sum(delta_drone[:, :3] ** 2, axis=-1)
        - reward_action_smoothness_thrust * delta_drone[:, 3] ** 2
    )

    # Action smoothness (rover): per-actuator penalties
    delta_rover = rover_cmd - last_rover_cmd
    r_smooth_rover = (
        -reward_action_smoothness_accel * delta_rover[:, 0] ** 2
        - reward_action_smoothness_yawrate * delta_rover[:, 1] ** 2
    )

    # Landing bonus + relative velocity penalty at touchdown
    rover_vx = v * c
    rover_vy = v * s
    rel_vel = jnp.stack([
        drone_vel[:, 0] - rover_vx,
        drone_vel[:, 1] - rover_vy,
        drone_vel[:, 2],
    ], axis=-1)
    landing_speed = jnp.linalg.norm(rel_vel, axis=-1)
    r_landing = jnp.where(
        landed,
        reward_landing - reward_landing_velocity_coef * landing_speed,
        0.0,
    )

    # Crash penalty
    r_crash = jnp.where(crashed, reward_crash, 0.0)

    # Boundary penalty
    r_boundary = jnp.where(oob, reward_boundary, 0.0)

    # Per-step time penalty to discourage hovering
    r_time = -reward_time_penalty * jnp.ones_like(r_xy)

    # Per-agent reward split: team + agent-specific
    team = r_xy + r_z + r_landing + r_time
    drone_only = (r_descent + r_altitude + r_drone_velocity + r_angle
                  + r_smooth_drone + r_crash + r_boundary)
    rover_only = r_rover_stillness + r_rover_yawrate + r_smooth_rover + r_rover_boundary

    drone_reward = team + drone_only
    rover_reward = team + rover_only

    components = {
        "proximity_xy": r_xy,
        "proximity_z": r_z,
        "descent_speed": r_descent,
        "altitude_floor": r_altitude,
        "angle": r_angle,
        "smooth_drone": r_smooth_drone,
        "smooth_rover": r_smooth_rover,
        "landing": r_landing,
        "crash": r_crash,
        "boundary": r_boundary,
        "time": r_time,
        "rover_stillness": r_rover_stillness,
        "rover_yawrate": r_rover_yawrate,
        "rover_boundary": r_rover_boundary,
        "drone_velocity": r_drone_velocity,
    }
    return drone_reward.astype(jnp.float32), rover_reward.astype(jnp.float32), components


# =============================================================================
# LandingEnv
# =============================================================================

class LandingEnv(gym.Env):
    """Cooperative drone-rover landing environment.

    Two agents: "drone" and "rover".
    N worlds run in parallel using JAX.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        cfg: LandingEnvConfig | None = None,
        spawn_fn: SpawnFn | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.cfg = cfg if cfg is not None else LandingEnvConfig()
        self.render_mode = render_mode

        if spawn_fn is None:
            spawn_fn = create_default_spawn_fn(
                drone_z_min=self.cfg.drone_z_min,
                drone_z_max=self.cfg.drone_z_max,
                drone_x_half=self.cfg.map_half_x,
                drone_y_half=self.cfg.map_half_y,
                rover_x_half=self.cfg.map_half_x,
                rover_y_half=self.cfg.map_half_y,
                rover_max_speed=self.cfg.rover_max_speed,
            )
        self._spawn_fn = spawn_fn

        self._init_simulator()

        # Agent identifiers
        self.possible_agents = ["drone", "rover"]
        self.agents = self.possible_agents.copy()
        self.num_agents = 2

        self._define_spaces()
        self._init_state_tensors()

        self.episode_steps = np.zeros(self.cfg.n_worlds, dtype=np.int32)
        self._max_episode_steps = self.cfg.max_episode_steps

        # Trajectory tracking for visualization
        self._trajectory_enabled = False
        self._drone_trajectory: list[np.ndarray] = []  # list of (3,) positions
        self._rover_trajectory: list[np.ndarray] = []  # list of (2,) positions
        self._trajectory_subsample = 5  # record every N steps
        self._trajectory_step_counter = 0
        self._trajectory_pending_clear = False

        # Pre-reset render state (for screenshots at episode end)
        self._pre_reset_render_state = None

    # -------------------------------------------------------------------------
    # Initialization helpers
    # -------------------------------------------------------------------------

    def _init_simulator(self):
        """Create Crazyflow simulator for the drone."""
        self.sim = Sim(
            n_worlds=self.cfg.n_worlds,
            n_drones=1,
            drone_model=self.cfg.drone_model,
            physics=Physics.first_principles,
            control=Control.attitude,
            freq=self.cfg.sim_freq,
            attitude_freq=self.cfg.mellinger_freq,
            device=self.cfg.device,
        )

        # Override mass if provided
        if self.cfg.mass is not None:
            params = self.sim.data.params.replace(
                mass=jnp.full_like(self.sim.data.params.mass, self.cfg.mass)
            )
            self.sim.data = self.sim.data.replace(params=params)
            self.sim.default_data = self.sim.default_data.replace(params=params)

        self._apply_domain_randomization()

        # Store base pipeline (without disturbance) for dynamic enable/disable
        self._base_step_pipeline = self.sim.step_pipeline
        self._disturbance_enabled = False

        # Inject disturbance function into simulation pipeline if enabled
        if self.cfg.enable_disturbance:
            self._enable_disturbance()

        # Hover RPM from first_principles parameters
        fp_params = load_params("first_principles", self.cfg.drone_model)
        rpm2thrust = fp_params["rpm2thrust"]
        a_c = rpm2thrust[2]
        b_c = rpm2thrust[1]
        mass = self.cfg.mass if self.cfg.mass is not None else float(fp_params["mass"])
        c_c = rpm2thrust[0] - mass * self.cfg.gravity / 4
        self.hover_rpm = float((-b_c + np.sqrt(b_c ** 2 - 4 * a_c * c_c)) / (2 * a_c))

    def _apply_domain_randomization(self, mask: jnp.ndarray | None = None):
        """Randomize mass and/or inertia for enabled worlds."""
        if not self.cfg.randomize_mass and not self.cfg.randomize_inertia:
            return

        if self.cfg.randomize_mass:
            key = jax.random.key(np.random.randint(0, 2 ** 31))
            mass_noise = jax.random.normal(
                key, (self.cfg.n_worlds, 1, 1)
            ) * self.cfg.mass_randomization_std
            base_mass = self.cfg.mass if self.cfg.mass is not None else float(
                self.sim.data.params.mass.mean()
            )
            randomize_mass(self.sim, base_mass + mass_noise, mask)

        if self.cfg.randomize_inertia:
            key = jax.random.key(np.random.randint(0, 2 ** 31))
            J_noise = jax.random.normal(
                key, (self.cfg.n_worlds, 1, 3, 3)
            ) * self.cfg.inertia_randomization_std
            J_rand = self.sim.data.params.J + J_noise
            randomize_inertia(self.sim, J_rand, mask)

    def _create_disturbance_fn(self):
        """Create a disturbance function for the simulation pipeline."""
        force_std = self.cfg.disturbance_force_std
        torque_std = self.cfg.disturbance_torque_std

        def disturbance_fn(data: SimData) -> SimData:
            key = data.core.rng_key
            key, force_key, torque_key = jax.random.split(key, 3)
            states = data.states
            disturbance_force = jax.random.normal(force_key, states.force.shape) * force_std
            disturbance_torque = jax.random.normal(torque_key, states.torque.shape) * torque_std
            states = states.replace(force=disturbance_force, torque=disturbance_torque)
            core = data.core.replace(rng_key=key)
            return data.replace(states=states, core=core)

        return disturbance_fn

    def _enable_disturbance(self):
        """Enable disturbance injection in simulation pipeline."""
        if self._disturbance_enabled:
            self._disable_disturbance()
        disturbance_fn = self._create_disturbance_fn()
        self.sim.step_pipeline = (
            self._base_step_pipeline[:2] + (disturbance_fn,) + self._base_step_pipeline[2:]
        )
        self.sim.build_step_fn()
        self._disturbance_enabled = True

    def _disable_disturbance(self):
        """Disable disturbance injection in simulation pipeline."""
        if not self._disturbance_enabled:
            return
        self.sim.step_pipeline = self._base_step_pipeline
        self.sim.build_step_fn()
        self._disturbance_enabled = False

    def _define_spaces(self):
        """Define per-agent observation and action spaces."""
        # Drone observation: 28D (see module docstring)
        #   own: pos(3) + vel(3) + rotmat_flat(9) + body_rates(3) = 18
        #   rover: xy(2) + vel_xy(2) + heading_sincos(2) + speed(1) = 7
        #   relative: pos(3) = 3
        # Total: 28
        self.drone_obs_dim = 28

        # Rover observation: 13D (body-frame)
        #   own: pos(2) + rot_matrix(2: cos,sin) + speed(1) = 5
        #   body-frame drone: rel_pos(3) + vel(3) = 6
        #   drone_speed(1) + dist(1) = 2
        # Total: 5 + 6 + 2 = 13
        self.rover_obs_dim = 13

        # MPC state dimensions
        self.drone_mpc_state_dim = 12  # [pos(3), rpy(3), vel(3), drpy(3)]
        self.rover_mpc_state_dim = 5   # [x, y, c, s, v]

        # Shared state for centralized critic: 22D
        #   drone: pos(3) + vel(3) + rpy(3) + body_rates(3) = 12
        #   rover: pos(2) + vel_xy(2) + heading_sincos(2) + speed(1) = 7
        #   relative: drone_pos - rover_xy (3) = 3
        self.shared_state_dim = 22

        self.observation_space = spaces.Dict({
            "drone": spaces.Box(-np.inf, np.inf, (self.drone_obs_dim,), dtype=np.float32),
            "rover": spaces.Box(-np.inf, np.inf, (self.rover_obs_dim,), dtype=np.float32),
        })
        self.observation_spaces = self.observation_space

        self.action_space = spaces.Dict({
            "drone": spaces.Box(
                low=np.array([-self.cfg.roll_pitch_max, -self.cfg.roll_pitch_max,
                               -self.cfg.yaw_max, self.cfg.thrust_min], dtype=np.float32),
                high=np.array([self.cfg.roll_pitch_max, self.cfg.roll_pitch_max,
                                self.cfg.yaw_max, self.cfg.thrust_max], dtype=np.float32),
            ),
            "rover": spaces.Box(
                low=np.array([-self.cfg.rover_max_accel, -self.cfg.rover_max_omega], dtype=np.float32),
                high=np.array([self.cfg.rover_max_accel, self.cfg.rover_max_omega], dtype=np.float32),
            ),
        })
        self.action_spaces = self.action_space

        self.shared_observation_space = spaces.Box(
            -np.inf, np.inf, (self.shared_state_dim,), dtype=np.float32
        )
        self.state_spaces = {
            agent: self.shared_observation_space for agent in self.possible_agents
        }

        # No binary dims for this env (no one-hot or alive flags)
        self._obs_binary_dims: list[int] = []
        self._state_binary_dims: list[int] = []

    @property
    def obs_binary_dims(self) -> list[int]:
        return self._obs_binary_dims

    @property
    def state_binary_dims(self) -> list[int]:
        return self._state_binary_dims

    def _init_state_tensors(self):
        """Initialize JAX state arrays."""
        N = self.cfg.n_worlds
        self.rover_state = jnp.zeros((N, 5))       # [x, y, c, s, v]
        self.drone_cmd = jnp.zeros((N, 4))         # attitude cmd
        self.last_drone_cmd = jnp.zeros((N, 4))
        self.rover_cmd = jnp.zeros((N, 2))         # [a, ω]
        self.last_rover_cmd = jnp.zeros((N, 2))
        self._cached_drone_rpy = None
        self._cached_drone_rpy_rates = None
        self._cached_drone_rotmat_flat = None

        # Landing success count for curriculum tracking
        self.landing_count = np.zeros(N, dtype=np.int32)
        self.last_termination_events: dict[str, float] = {
            "landing": 0.0,
            "crash": 0.0,
            "out_of_bounds": 0.0,
            "max_steps": 0.0,
        }

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self.cfg.n_worlds

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            self.sim.seed(seed)

        self.sim.reset()
        self._apply_domain_randomization()
        self.episode_steps = np.zeros(self.cfg.n_worlds, dtype=np.int32)
        self.landing_count = np.zeros(self.cfg.n_worlds, dtype=np.int32)

        self.last_drone_cmd = jnp.zeros((self.cfg.n_worlds, 4))
        self.last_rover_cmd = jnp.zeros((self.cfg.n_worlds, 2))

        self._spawn_agents()
        self.clear_trajectory()
        obs = self._get_observations()
        info = self._get_info()
        return obs, info

    def enable_trajectory(self, enabled: bool = True, subsample: int = 5):
        """Enable or disable trajectory tracking for visualization.

        Args:
            enabled: Whether to record positions each step.
            subsample: Record every N steps to limit marker count.
        """
        self._trajectory_enabled = enabled
        self._trajectory_subsample = subsample
        self.clear_trajectory()

    def clear_trajectory(self):
        """Clear stored trajectory data."""
        self._drone_trajectory = []
        self._rover_trajectory = []
        self._trajectory_step_counter = 0

    def set_render_resolution(self, width: int, height: int):
        """Set render resolution. Must be called before the first render().

        Args:
            width: Render width in pixels.
            height: Render height in pixels.
        """
        self._render_width = width
        self._render_height = height

    def set_camera(self, distance: float | None = None, azimuth: float | None = None,
                   elevation: float | None = None, lookat: list[float] | None = None):
        """Override default camera parameters. Must be called before the first render().

        Args:
            distance: Camera distance from lookat point.
            azimuth: Camera azimuth angle (degrees).
            elevation: Camera elevation angle (degrees).
            lookat: Camera lookat point [x, y, z].
        """
        if not hasattr(self, '_cam_overrides'):
            self._cam_overrides = {}
        if distance is not None:
            self._cam_overrides['distance'] = distance
        if azimuth is not None:
            self._cam_overrides['azimuth'] = azimuth
        if elevation is not None:
            self._cam_overrides['elevation'] = elevation
        if lookat is not None:
            self._cam_overrides['lookat'] = lookat

    def _spawn_agents(self):
        """Spawn drone and rover using the configured spawn function."""
        N = self.cfg.n_worlds
        key = self.sim.data.core.rng_key
        key, spawn_key = jax.random.split(key)
        self.sim.data = self.sim.data.replace(core=self.sim.data.core.replace(rng_key=key))

        drone_pos, rover_state = self._spawn_fn(spawn_key, N)  # (N,3), (N,4)
        self.rover_state = rover_state

        # Initialize drone state
        all_vel = jnp.zeros((N, 1, 3))
        all_ang_vel = jnp.zeros((N, 1, 3))
        identity_quat = jnp.array([0.0, 0.0, 0.0, 1.0])
        all_quat = jnp.broadcast_to(identity_quat, (N, 1, 4))
        all_rotor_vel = jnp.full((N, 1, 4), self.hover_rpm)
        all_pos = drone_pos[:, None, :]  # (N, 1, 3)

        states = self.sim.data.states.replace(
            pos=all_pos,
            vel=all_vel,
            quat=all_quat,
            ang_vel=all_ang_vel,
            rotor_vel=all_rotor_vel,
        )
        self.sim.data = self.sim.data.replace(states=states)

    def _reset_done_worlds(self, done_mask: np.ndarray):
        """Reset specific worlds that have finished."""
        N = self.cfg.n_worlds
        if not done_mask.any():
            return

        mask_jnp = jnp.array(done_mask)
        self.episode_steps[done_mask] = 0

        # Reset commands for done worlds
        self.last_drone_cmd = jnp.where(mask_jnp[:, None], 0.0, self.last_drone_cmd)
        self.last_rover_cmd = jnp.where(mask_jnp[:, None], 0.0, self.last_rover_cmd)

        # Re-spawn
        key = self.sim.data.core.rng_key
        key, spawn_key = jax.random.split(key)
        self.sim.data = self.sim.data.replace(core=self.sim.data.core.replace(rng_key=key))

        all_drone_pos, all_rover_state = self._spawn_fn(spawn_key, N)

        # Update rover state (vectorized where)
        self.rover_state = jnp.where(mask_jnp[:, None], all_rover_state, self.rover_state)

        # Reset Crazyflow state for done worlds
        self.sim.reset(mask=mask_jnp)
        hover_rpm = jnp.full((N, 1, 4), self.hover_rpm)
        states = leaf_replace(
            self.sim.data.states, mask=mask_jnp, pos=all_drone_pos[:, None, :], rotor_vel=hover_rpm
        )
        self.sim.data = self.sim.data.replace(states=states)
        self._apply_domain_randomization(mask=mask_jnp)

        # Defer trajectory clear so the final render still shows the full trajectory
        if self._trajectory_enabled and done_mask[0]:
            self._trajectory_pending_clear = True

    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[dict, dict, dict, dict, dict]:
        """Execute one environment step.

        Args:
            actions: {"drone": (N, 4), "rover": (N, 2)} in physical units.

        Returns:
            obs, rewards, terminated, truncated, info
        """
        # Save previous commands for smoothness penalty
        self.last_drone_cmd = self.drone_cmd
        self.last_rover_cmd = self.rover_cmd

        # Process and clip drone action
        N = self.cfg.n_worlds
        drone_action = np.array(actions.get("drone", np.zeros((N, 4))), copy=True).reshape(N, 4)
        drone_action[:, 0] = np.clip(drone_action[:, 0], -self.cfg.roll_pitch_max, self.cfg.roll_pitch_max)
        drone_action[:, 1] = np.clip(drone_action[:, 1], -self.cfg.roll_pitch_max, self.cfg.roll_pitch_max)
        drone_action[:, 2] = np.clip(drone_action[:, 2], -self.cfg.yaw_max, self.cfg.yaw_max)
        drone_action[:, 3] = np.clip(drone_action[:, 3], self.cfg.thrust_min, self.cfg.thrust_max)
        self.drone_cmd = jnp.array(drone_action)

        # Process and clip rover action
        rover_action = np.array(actions.get("rover", np.zeros((N, 2))), copy=True).reshape(N, 2)
        rover_action[:, 0] = np.clip(rover_action[:, 0], -self.cfg.rover_max_accel, self.cfg.rover_max_accel)
        rover_action[:, 1] = np.clip(rover_action[:, 1], -self.cfg.rover_max_omega, self.cfg.rover_max_omega)
        self.rover_cmd = jnp.array(rover_action)

        # Step rover dynamics
        self.rover_state = rover_step_batched(
            self.rover_state, self.rover_cmd, self.cfg.dt,
            self.cfg.rover_max_speed, self.cfg.rover_min_speed,
            self.cfg.rover_max_omega, self.cfg.rover_max_accel,
        )
        # Clamp rover to arena
        self.rover_state = _jit_clamp_rover(
            self.rover_state, self.cfg.map_half_x, self.cfg.map_half_y
        )

        # Step drone (Crazyflow)
        self.sim.attitude_control(self.drone_cmd[:, None, :])
        self.sim.step(n_steps=self.cfg.sim_steps_per_control)

        # Extract drone state
        states = self.sim.data.states
        drone_pos_jnp = states.pos[:, 0]   # (N, 3)
        drone_vel_jnp = states.vel[:, 0]   # (N, 3)
        drone_rpy = _jit_quat_to_rpy(states.quat[:, 0])  # (N, 3)

        # Record trajectory for visualization
        if self._trajectory_enabled:
            self._trajectory_step_counter += 1
            if self._trajectory_step_counter % self._trajectory_subsample == 0:
                self._drone_trajectory.append(np.asarray(drone_pos_jnp[0]).copy())
                self._rover_trajectory.append(np.asarray(self.rover_state[0, :2]).copy())

        # Check conditions
        landed, crashed = _jit_check_landing(
            drone_pos_jnp, drone_vel_jnp, drone_rpy, self.rover_state,
            self.cfg.rover_height,
            self.cfg.rover_platform_radius, self.cfg.landing_z_tol,
            self.cfg.landing_vel_xy_tol, self.cfg.landing_vel_z_tol,
            self.cfg.landing_attitude_tol,
        )
        oob = _jit_check_oob(
            drone_pos_jnp,
            self.cfg.map_half_x, self.cfg.map_half_y,
            self.cfg.drone_z_max,
        )

        drone_reward, rover_reward, components = _jit_compute_rewards(
            drone_pos_jnp, drone_vel_jnp, drone_rpy,
            self.drone_cmd, self.last_drone_cmd,
            self.rover_state, self.rover_cmd, self.last_rover_cmd,
            landed, crashed, oob,
            self.cfg.rover_height,
            self.cfg.reward_landing,
            self.cfg.reward_crash,
            self.cfg.reward_boundary,
            self.cfg.reward_proximity_coef,
            self.cfg.reward_proximity_decay,
            self.cfg.reward_angle_coef,
            self.cfg.reward_action_smoothness_thrust,
            self.cfg.reward_action_smoothness_rpy,
            self.cfg.reward_action_smoothness_accel,
            self.cfg.reward_action_smoothness_yawrate,
            self.cfg.reward_landing_velocity_coef,
            self.cfg.corridor_radius,
            self.cfg.corridor_transition,
            self.cfg.max_descent_speed,
            self.cfg.reward_descent_speed_coef,
            self.cfg.reward_altitude_floor_coef,
            self.cfg.reward_time_penalty,
            self.cfg.reward_rover_stillness_coef,
            self.cfg.reward_rover_yawrate_coef,
            self.cfg.reward_drone_velocity_coef,
            self.cfg.reward_rover_boundary_coef,
            self.cfg.map_half_x,
            self.cfg.map_half_y,
        )

        self.last_reward_components = {k: np.asarray(v) for k, v in components.items()}
        self.last_drone_reward = np.asarray(drone_reward)
        self.last_rover_reward = np.asarray(rover_reward)

        # Update episode steps
        self.episode_steps += 1

        # Termination / truncation
        done_np = np.asarray(landed | crashed | oob)
        truncated_np = self.episode_steps >= self._max_episode_steps
        done_mask = done_np | truncated_np

        # Track event rates
        N = self.cfg.n_worlds
        landed_np = np.asarray(landed)
        crashed_np = np.asarray(crashed)
        oob_np = np.asarray(oob)
        self.last_termination_events = {
            "landing": float(landed_np.sum()) / N,
            "crash": float(crashed_np.sum()) / N,
            "out_of_bounds": float(oob_np.sum()) / N,
            "max_steps": float((truncated_np & ~done_np).sum()) / N,
        }

        # Build reward dicts (per-agent rewards: team + agent-specific)
        rewards = {"drone": self.last_drone_reward, "rover": self.last_rover_reward}

        terminated = {"drone": done_np, "rover": done_np}
        truncated = {"drone": truncated_np, "rover": truncated_np}

        # Save pre-reset render state for screenshot/video (before auto-reset)
        if done_mask.any() and done_mask[0] and self.render_mode is not None:
            self._pre_reset_render_state = {
                'drone_pos': np.asarray(self.sim.data.states.pos[0, 0]).copy(),
                'drone_quat': np.asarray(self.sim.data.states.quat[0, 0]).copy(),
                'rover_state': np.asarray(self.rover_state[0]).copy(),
            }

        # Auto-reset done worlds (after saving pre-reset state for info)
        pre_reset_landed = landed_np.copy()
        if done_mask.any():
            self._reset_done_worlds(done_mask)

        obs = self._get_observations()
        info = self._get_info(
            rewards=rewards,
            landed=pre_reset_landed,
            episode_terminated=done_np,
            episode_truncated=truncated_np,
        )

        return obs, rewards, terminated, truncated, info

    # -------------------------------------------------------------------------
    # Observations
    # -------------------------------------------------------------------------

    def _get_observations(self) -> dict[str, np.ndarray]:
        """Build per-agent observations."""
        N = self.cfg.n_worlds
        states = self.sim.data.states

        # Convert to numpy in one batch
        drone_pos = np.asarray(states.pos[:, 0])    # (N, 3)
        drone_vel = np.asarray(states.vel[:, 0])    # (N, 3)
        drone_quat = states.quat[:, 0]              # (N, 4)

        drone_rotmat = np.asarray(_jit_quat_to_matrix(drone_quat))   # (N, 3, 3)
        drone_rotmat_flat = drone_rotmat.reshape(N, 9)
        drone_ang_vel = np.asarray(states.ang_vel[:, 0])              # (N, 3)

        drone_rpy = np.asarray(_jit_quat_to_rpy(drone_quat))          # (N, 3)
        drone_rpy_rates = np.asarray(_jit_ang_vel_to_rpy_rates(drone_quat, states.ang_vel[:, 0]))  # (N, 3)

        # Cache for _get_info
        self._cached_drone_rpy = drone_rpy
        self._cached_drone_rpy_rates = drone_rpy_rates
        self._cached_drone_rotmat_flat = drone_rotmat_flat

        rover = np.asarray(self.rover_state)  # (N, 5): [x, y, c, s, v]
        rover_xy = rover[:, :2]
        rover_c = rover[:, 2]
        rover_s = rover[:, 3]
        rover_v = rover[:, 4]
        rover_vel_xy = rover_v[:, None] * np.stack([rover_c, rover_s], axis=-1)

        # ---- Drone observation (28D) ----
        rel_pos = drone_pos - np.concatenate([rover_xy, np.zeros((N, 1))], axis=-1)  # (N, 3)
        drone_obs = np.concatenate([
            drone_pos,                          # 3
            drone_vel,                          # 3
            drone_rotmat_flat,                  # 9
            drone_ang_vel,                      # 3
            rover_xy,                           # 2
            rover_vel_xy,                       # 2
            rover_s[:, None],                   # 1
            rover_c[:, None],                   # 1
            rover_v[:, None],                   # 1
            rel_pos,                            # 3
        ], axis=-1).astype(np.float32)  # 28D

        # ---- Rover observation (13D) — world-frame ----
        drone_speed = np.linalg.norm(drone_vel, axis=-1, keepdims=True)  # (N, 1)
        dist = np.linalg.norm(rel_pos, axis=-1, keepdims=True)           # (N, 1)

        rover_obs = np.concatenate([
            rover_xy,                                   # 2  world position (boundary awareness)
            rover_c[:, None],                           # 1  cos(heading)
            rover_s[:, None],                           # 1  sin(heading)
            rover_v[:, None],                           # 1  → 5  (self)
            rel_pos,                                    # 3  → 8  drone - rover (world frame)
            drone_vel,                                  # 3  → 11  drone velocity (world frame)
            drone_speed,                                # 1  → 12  drone speed scalar
            dist,                                       # 1  → 13  Euclidean distance
        ], axis=-1).astype(np.float32)  # 13D

        return {"drone": drone_obs, "rover": rover_obs}

    def _get_shared_state(self) -> np.ndarray:
        """Get shared state for centralized critic (22D)."""
        N = self.cfg.n_worlds
        states = self.sim.data.states

        drone_pos = np.asarray(states.pos[:, 0])
        drone_vel = np.asarray(states.vel[:, 0])
        rover = np.asarray(self.rover_state)
        rover_xy = rover[:, :2]
        rover_c = rover[:, 2]
        rover_s = rover[:, 3]
        rover_v = rover[:, 4]
        rover_vel_xy = rover_v[:, None] * np.stack([rover_c, rover_s], axis=-1)

        if self._cached_drone_rpy is not None:
            drone_rpy = self._cached_drone_rpy
        else:
            drone_rpy = np.asarray(_jit_quat_to_rpy(states.quat[:, 0]))

        if self._cached_drone_rpy_rates is not None:
            drone_body_rates = self._cached_drone_rpy_rates
        else:
            drone_body_rates = np.asarray(states.ang_vel[:, 0])

        rel_pos = drone_pos - np.concatenate([rover_xy, np.zeros((N, 1))], axis=-1)

        shared = np.concatenate([
            drone_pos,                              # 3
            drone_vel,                              # 3
            drone_rpy,                              # 3
            drone_body_rates,                       # 3  → 12
            rover_xy,                               # 2
            rover_vel_xy,                           # 2
            rover_s[:, None],                       # 1
            rover_c[:, None],                       # 1
            rover_v[:, None],                       # 1  → 19
            rel_pos,                                # 3  → 22
        ], axis=-1).astype(np.float32)

        return shared

    def state(self) -> np.ndarray:
        """Global/shared state for centralized critic (SKRL interface)."""
        return self._get_shared_state()

    # -------------------------------------------------------------------------
    # Info
    # -------------------------------------------------------------------------

    def _get_info(
        self,
        rewards: dict[str, np.ndarray] | None = None,
        landed: np.ndarray | None = None,
        episode_terminated: np.ndarray | None = None,
        episode_truncated: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Build info dictionary including MPC states for each agent."""
        info: dict[str, Any] = {}

        if episode_terminated is not None:
            info["episode_terminated"] = episode_terminated
        if episode_truncated is not None:
            info["episode_truncated"] = episode_truncated

        if rewards is not None:
            info["reward/mean"] = float(rewards["drone"].mean())

        if landed is not None:
            info["landing_rate"] = float(landed.mean())

        # MPC states
        states = self.sim.data.states
        drone_pos = np.asarray(states.pos[:, 0])
        drone_vel = np.asarray(states.vel[:, 0])

        if self._cached_drone_rpy is not None and self._cached_drone_rpy_rates is not None:
            drone_rpy = self._cached_drone_rpy
            drone_drpy = self._cached_drone_rpy_rates
            drone_mpc_state = np.concatenate([drone_pos, drone_rpy, drone_vel, drone_drpy], axis=-1)
        else:
            drone_quat = states.quat[:, 0]
            drone_rpy = np.asarray(_jit_quat_to_rpy(drone_quat))
            drone_drpy = np.asarray(_jit_ang_vel_to_rpy_rates(drone_quat, states.ang_vel[:, 0]))
            drone_mpc_state = np.concatenate([drone_pos, drone_rpy, drone_vel, drone_drpy], axis=-1)

        info["mpc_state"] = {
            "drone": drone_mpc_state.astype(np.float32),
            "rover": np.asarray(self.rover_state).astype(np.float32),
        }

        return info

    # -------------------------------------------------------------------------
    # Misc
    # -------------------------------------------------------------------------

    @staticmethod
    def _smooth_trajectory(points: list[np.ndarray], window: int = 5) -> list[np.ndarray]:
        """Smooth trajectory points with a moving average, keeping endpoints."""
        n = len(points)
        if n < 3:
            return points
        arr = np.array(points)
        smoothed = np.copy(arr)
        half = window // 2
        for i in range(1, n - 1):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            smoothed[i] = arr[lo:hi].mean(axis=0)
        return [smoothed[i] for i in range(n)]

    @staticmethod
    def _draw_line_segment(viewer, mujoco, p0: np.ndarray, p1: np.ndarray,
                           width: float, rgba: np.ndarray):
        """Draw an unlit line segment between two 3D points.

        Appends a special marker dict that _add_marker_to_scene handles
        using mjv_initGeom + mjv_connector for flat (unlit) rendering.
        """
        viewer._markers.append({
            "_line": True,
            "from": p0.astype(np.float64),
            "to": p1.astype(np.float64),
            "width": float(width),
            "rgba": rgba.astype(np.float32),
        })

    def render(self, world: int = 0) -> np.ndarray | None:
        """Render the environment using MuJoCo with the rover drawn as markers.

        The drone is rendered via Crazyflow's MuJoCo model.

        Args:
            world: Which parallel world to render (default 0).
        """
        if self.render_mode is None:
            return None

        import mujoco
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        # Initialize viewer if needed (same pattern as MAPE)
        if self.sim.viewer is None:
            w = getattr(self, '_render_width', 1920)
            h = getattr(self, '_render_height', 1080)
            self.sim.mj_model.vis.global_.offwidth = w
            self.sim.mj_model.vis.global_.offheight = h
            cam_config = {
                "distance": 5.0,
                "azimuth": 90.0,
                "elevation": -30.0,
                "lookat": [0.0, 0.0, 0.5],
            }
            # Apply user camera overrides
            overrides = getattr(self, '_cam_overrides', {})
            cam_config.update(overrides)
            self.sim.viewer = MujocoRenderer(
                self.sim.mj_model,
                self.sim.mj_data,
                max_geom=self.sim.max_visual_geom,
                default_cam_config=cam_config,
                height=h,
                width=w,
                camera_id=-1,
            )
            # Disable floor reflection and shadows
            floor_id = mujoco.mj_name2id(self.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            if floor_id >= 0:
                mat_id = self.sim.mj_model.geom_matid[floor_id]
                if mat_id >= 0:
                    self.sim.mj_model.mat_reflectance[mat_id] = 0.0
            self.sim.mj_model.vis.quality.shadowsize = 0

        # Clear trajectory if pending (deferred from auto-reset so the final
        # render of the previous episode still shows the full trajectory)
        if self._trajectory_pending_clear and self._pre_reset_render_state is None:
            self.clear_trajectory()
            self._trajectory_pending_clear = False

        # Use pre-reset state if available (first render after auto-reset
        # shows drone/rover at final episode positions, not new spawn positions)
        pre = self._pre_reset_render_state
        if pre is not None:
            drone_pos = pre['drone_pos']
            drone_quat = pre['drone_quat']
            rover_for_render = pre['rover_state']
            self._pre_reset_render_state = None  # consume once
        else:
            states = self.sim.data.states
            drone_pos = np.asarray(states.pos[world, 0])    # (3,)
            drone_quat = np.asarray(states.quat[world, 0])  # (4,) xyzw
            rover_for_render = None  # use current state

        qpos = np.zeros(7)
        qpos[:3] = drone_pos
        # Crazyflow uses xyzw internally but MuJoCo qpos wants wxyz
        qpos[3] = drone_quat[3]   # w
        qpos[4:7] = drone_quat[:3]  # xyz
        self.sim.mj_data.qpos[:] = qpos
        self.sim.mj_data.mocap_pos[:] = np.asarray(
            self.sim.mjx_data.mocap_pos[world, :]
        )
        self.sim.mj_data.mocap_quat[:] = np.asarray(
            self.sim.mjx_data.mocap_quat[world, :]
        )

        # Forward dynamics to update rendering state
        mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)

        # Ensure inner viewer is created and patch for mujoco 3.5+ compat
        self.sim.viewer._get_viewer(self.render_mode)
        viewer = self.sim.viewer.viewer
        if hasattr(viewer, '_hide_menu') and self.render_mode == "rgb_array":
            viewer._hide_menu = True
        if not getattr(viewer, '_marker_patched', False):
            def _add_marker_to_scene(marker):
                if viewer.scn.ngeom >= viewer.scn.maxgeom:
                    return
                g = viewer.scn.geoms[viewer.scn.ngeom]

                # Line segments: use mjv_initGeom + mjv_connector for unlit rendering
                if marker.get("_line"):
                    mujoco.mjv_initGeom(
                        g,
                        type=mujoco.mjtGeom.mjGEOM_LINE,
                        size=np.zeros(3),
                        pos=np.zeros(3),
                        mat=np.eye(3).flatten(),
                        rgba=marker["rgba"],
                    )
                    mujoco.mjv_connector(
                        g,
                        type=mujoco.mjtGeom.mjGEOM_LINE,
                        width=marker["width"],
                        from_=marker["from"],
                        to=marker["to"],
                    )
                    g.category = mujoco.mjtCatBit.mjCAT_DECOR
                    viewer.scn.ngeom += 1
                    return

                g.dataid = -1
                g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
                g.objid = -1
                g.category = mujoco.mjtCatBit.mjCAT_DECOR
                g.emission = 0
                g.specular = 0.5
                g.shininess = 0.5
                g.reflectance = 0
                g.type = mujoco.mjtGeom.mjGEOM_BOX
                g.size[:] = np.ones(3) * 0.1
                g.mat[:] = np.eye(3)
                g.rgba[:] = np.ones(4)
                for key, value in marker.items():
                    if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                        setattr(g, key, value)
                    elif isinstance(value, np.ndarray):
                        attr = getattr(g, key)
                        attr[:] = value.reshape(attr.shape)
                    elif isinstance(value, (str, bytes)):
                        if isinstance(value, str):
                            value = value.encode()
                        setattr(g, key, value)
                viewer.scn.ngeom += 1
            viewer._add_marker_to_scene = _add_marker_to_scene
            viewer._marker_patched = True

            # Patch overlay to show camera info (only for live window viewer)
            if hasattr(viewer, '_create_overlay'):
                _orig_create_overlay = viewer._create_overlay
                def _create_overlay_with_camera():
                    _orig_create_overlay()
                    bottomright = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT
                    cam = viewer.cam
                    viewer.add_overlay(bottomright, "Distance", f"{cam.distance:.2f}")
                    viewer.add_overlay(bottomright, "Azimuth", f"{cam.azimuth:.1f}")
                    viewer.add_overlay(bottomright, "Elevation", f"{cam.elevation:.1f}")
                    viewer.add_overlay(
                        bottomright, "Lookat",
                        f"[{cam.lookat[0]:.2f}, {cam.lookat[1]:.2f}, {cam.lookat[2]:.2f}]",
                    )
                viewer._create_overlay = _create_overlay_with_camera

        # Draw rover markers (use pre-reset state if available)
        rover = rover_for_render if rover_for_render is not None else np.asarray(self.rover_state[world])  # (5,) [x, y, c, s, v]
        rx, ry = float(rover[0]), float(rover[1])
        cth, sth = float(rover[2]), float(rover[3])
        pad_r = self.cfg.rover_platform_radius
        rh = self.cfg.rover_height
        rover_mat = np.array([
            cth, -sth, 0,
            sth,  cth, 0,
              0,    0, 1,
        ], dtype=np.float64)

        # Rover body
        body_half_z = rh / 2.0
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=np.array([pad_r * 1.2, pad_r * 0.8, body_half_z]),
            pos=np.array([rx, ry, body_half_z]),
            mat=rover_mat,
            rgba=np.array([0.3, 0.3, 0.3, 1.0]),
            label="",
        )

        # Landing pad on top
        pad_thickness = 0.005
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=np.array([pad_r, pad_r, pad_thickness]),
            pos=np.array([rx, ry, rh + pad_thickness]),
            mat=np.eye(3).flatten(),
            rgba=np.array([0.1, 0.8, 0.1, 0.9]),
            label="",
        )

        # Heading indicator
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([0.008, 0.008, pad_r * 0.3]),
            pos=np.array([rx + cth * pad_r, ry + sth * pad_r, rh + pad_thickness + 0.01]),
            mat=rover_mat,
            rgba=np.array([1.0, 0.2, 0.2, 1.0]),
            label="",
        )

        # Draw trajectory lines (thin capsules between consecutive points)
        if self._trajectory_enabled and len(self._drone_trajectory) > 1:
            drone_rgba = np.array([0.9, 0.0, 0.0, 1.0])   # red (like MuJoCo X-axis)
            rover_rgba = np.array([0.0, 0.9, 0.0, 1.0])   # green (like MuJoCo Y-axis)
            line_width = 3.0  # pixels

            drone_pts = self._smooth_trajectory(self._drone_trajectory)
            for i in range(1, len(drone_pts)):
                self._draw_line_segment(viewer, mujoco, drone_pts[i - 1], drone_pts[i],
                                        line_width, drone_rgba)

            rover_pts_2d = self._smooth_trajectory(self._rover_trajectory)
            for i in range(1, len(rover_pts_2d)):
                p0 = np.array([rover_pts_2d[i - 1][0], rover_pts_2d[i - 1][1], 0.01])
                p1 = np.array([rover_pts_2d[i][0], rover_pts_2d[i][1], 0.01])
                self._draw_line_segment(viewer, mujoco, p0, p1, line_width, rover_rgba)

        # Render (viewer.render() calls mjv_updateScene, then our markers, then mjr_render)
        return self.sim.viewer.render(self.render_mode)

    def close(self):
        self.sim.close()

    def update_curriculum_params(
        self,
        spawn_fn: SpawnFn | None = None,
        **params,
    ):
        """Update environment parameters for curriculum learning.

        Args:
            spawn_fn: New spawn function. If None, current spawn function is kept.
            **params: Additional config parameters to override.
        """
        if spawn_fn is not None:
            self._spawn_fn = spawn_fn

        disturbance_changed = False
        old_enable_disturbance = self.cfg.enable_disturbance

        for key, value in params.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)
            if key in ("enable_disturbance", "disturbance_force_std", "disturbance_torque_std"):
                disturbance_changed = True

        if disturbance_changed:
            if self.cfg.enable_disturbance:
                self._enable_disturbance()
            elif old_enable_disturbance and not self.cfg.enable_disturbance:
                self._disable_disturbance()
