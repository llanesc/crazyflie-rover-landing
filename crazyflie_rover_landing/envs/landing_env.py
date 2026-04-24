"""Cooperative drone-rover landing environment using Crazyflow + JAX.

Two cooperative agents:
  - "drone": Crazyflie CF2X_T350, controlled via attitude commands from ACMPC.
  - "rover": Yahboom RosMaster X3 mecanum rover, controlled via [vx, vy, wz].

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
from crazyflie_rover_landing.envs.mecanum_dynamics import mecanum_step_batched, WHEEL_VEL_MAX
from crazyflie_rover_landing.envs.wind import WindModel, compute_wind_drag_force
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
    rover_xy: jnp.ndarray,
    rover_vel_xy: jnp.ndarray,
    rover_height: float,
    platform_radius: float,
    landing_zone_radius: float,
    z_tol: float,
    vel_xy_tol: float,
    vel_z_tol: float,
    attitude_tol: float,
    pad_contact: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Check landing and crash conditions for each world.

    Uses MJX contact detection when pad_contact is provided. The drone must
    physically touch the pad (contact), be within the safe landing zone
    (smaller than the pad), and satisfy soft-landing criteria.

    Landing on the pad edge (contact but outside landing_zone_radius) = crash.

    Args:
        drone_pos: (N, 3) drone positions.
        drone_vel: (N, 3) drone velocities (world frame).
        drone_rpy: (N, 3) drone roll-pitch-yaw [rad].
        rover_xy: (N, 2) rover XY position.
        rover_vel_xy: (N, 2) rover world-frame XY velocity.
        rover_height: Height of landing pad surface above ground [m].
        platform_radius: Rover landing pad radius [m].
        landing_zone_radius: Safe landing zone radius [m] (< platform_radius).
        z_tol: Max height above rover pad for success [m].
        vel_xy_tol: Max relative XY speed at touchdown [m/s].
        vel_z_tol: Max descent speed at touchdown [m/s].
        attitude_tol: Max |roll| and |pitch| at touchdown [rad].
        pad_contact: (N,) bool from MJX contact detection, or None for legacy mode.

    Returns:
        landed: (N,) bool — successful soft landing.
        crashed: (N,) bool — ground contact that isn't a landing.
    """
    # Relative velocity (drone minus rover)
    rel_vx = drone_vel[:, 0] - rover_vel_xy[:, 0]
    rel_vy = drone_vel[:, 1] - rover_vel_xy[:, 1]
    rel_vz = drone_vel[:, 2]  # rover has no vertical velocity

    rel_speed_xy = jnp.sqrt(rel_vx ** 2 + rel_vy ** 2)
    horiz_dist = jnp.linalg.norm(drone_pos[:, :2] - rover_xy, axis=-1)

    # Soft-landing criteria: low speed + level attitude
    low_speed = (
        (rel_speed_xy < vel_xy_tol)
        & (rel_vz <= 0.0)
        & (jnp.abs(rel_vz) < vel_z_tol)
    )
    level_attitude = (jnp.abs(drone_rpy[:, 0]) < attitude_tol) & (jnp.abs(drone_rpy[:, 1]) < attitude_tol)

    if pad_contact is not None:
        # Contact-based detection
        touching_pad = pad_contact
        in_safe_zone = horiz_dist < landing_zone_radius
        # Ground crash: drone below safe altitude but not on pad
        ground_hit = drone_pos[:, 2] < rover_height * 0.5
        # Successful landing: contact + in safe zone + soft + level
        landed = touching_pad & in_safe_zone & low_speed & level_attitude
        # Crash: any contact that isn't a good landing, or ground hit
        crashed = (touching_pad & ~landed) | ground_hit
    else:
        # Legacy position-threshold mode
        near_ground = drone_pos[:, 2] < rover_height + z_tol
        on_pad = (horiz_dist < landing_zone_radius) & near_ground
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
def _jit_clamp_rover_x3(
    rover_state: jnp.ndarray,
    map_half_x: float,
    map_half_y: float,
) -> jnp.ndarray:
    """Clamp X3 rover (7-state) position to arena boundaries."""
    x, y = rover_state[:, 0], rover_state[:, 1]
    hit_any = (x < -map_half_x) | (x > map_half_x) | (y < -map_half_y) | (y > map_half_y)
    return jnp.stack([
        jnp.clip(x, -map_half_x, map_half_x),
        jnp.clip(y, -map_half_y, map_half_y),
        rover_state[:, 2], rover_state[:, 3],
        jnp.where(hit_any, 0.0, rover_state[:, 4]),
        jnp.where(hit_any, 0.0, rover_state[:, 5]),
        jnp.where(hit_any, 0.0, rover_state[:, 6]),
    ], axis=-1)


@jax.jit
def _jit_compute_rewards(
    drone_pos: jnp.ndarray,
    drone_vel: jnp.ndarray,
    drone_rpy: jnp.ndarray,
    drone_cmd: jnp.ndarray,
    last_drone_cmd: jnp.ndarray,
    rover_xy: jnp.ndarray,
    rover_vel_xy: jnp.ndarray,
    rover_speed: jnp.ndarray,
    rover_omega: jnp.ndarray,
    rover_vx_body: jnp.ndarray,
    rover_vy_body: jnp.ndarray,
    rover_cmd: jnp.ndarray,
    last_rover_cmd: jnp.ndarray,
    landed: jnp.ndarray,
    crashed: jnp.ndarray,
    oob: jnp.ndarray,
    rover_height: float,
    reward_landing: float,
    reward_crash: float,
    reward_boundary: float,
    reward_progress_coef: float,
    last_horiz_dist: jnp.ndarray,
    last_vert_dist: jnp.ndarray,
    reward_angle_coef: float,
    reward_action_smoothness_thrust: float,
    reward_action_smoothness_rpy: float,
    reward_action_smoothness_wheel: float,
    rover_smoothness_weights: jnp.ndarray,
    reward_landing_velocity_coef: float,
    corridor_radius: float,
    corridor_transition: float,
    max_descent_speed: float,
    reward_descent_speed_coef: float,
    reward_altitude_hold_coef: float,
    cruise_altitude: float,
    reward_time_penalty: float,
    reward_rover_stillness_coef: float,
    reward_rover_yawrate_coef: float,
    reward_rover_lateral_coef: float,
    reward_drone_velocity_coef: float,
    reward_drone_xy_corridor_coef: float,
    max_drone_speed: float,
    reward_rover_boundary_coef: float,
    reward_landing_precision_coef: float,
    landing_zone_radius: float,
    map_half_x: float,
    map_half_y: float,
) -> tuple[jnp.ndarray, jnp.ndarray, dict, jnp.ndarray, jnp.ndarray]:
    """Compute per-agent rewards with phased navigation-then-landing structure.

    Rewards are split into team (shared) + agent-specific components.
    Drone gets team + drone-only penalties; rover gets team + rover-only penalties.

    Phase 1 — Navigation (outside corridor): XY progress only, altitude floor.
    Phase 2 — Landing (inside corridor): XY + Z progress, descent speed limit.
    Smooth sigmoid transition at corridor boundary prevents chattering.

    Returns:
        total_reward: (N,) float32.
        components: dict of individual reward components, each (N,).
    """
    # Distances
    horiz_dist = jnp.linalg.norm(drone_pos[:, :2] - rover_xy, axis=-1)
    vert_dist = jnp.abs(drone_pos[:, 2] - rover_height)

    # Corridor blend: sigmoid transition from navigation to landing phase
    # z_weight ≈ 0 when far from rover, ≈ 1 when inside corridor
    z_weight = jax.nn.sigmoid((corridor_radius - horiz_dist) / corridor_transition)

    # XY progress — reward for getting closer (positive = closing in)
    r_xy = reward_progress_coef * (last_horiz_dist - horiz_dist)

    # Z progress — only inside corridor (smoothly blended)
    r_z = reward_progress_coef * z_weight * (last_vert_dist - vert_dist)

    # Descent speed penalty — only inside corridor
    # Penalize descent speed exceeding max_descent_speed
    descent_speed = jnp.maximum(-drone_vel[:, 2], 0.0)  # positive when going down
    excess_speed = jnp.maximum(descent_speed - max_descent_speed, 0.0)
    r_descent = -reward_descent_speed_coef * z_weight * excess_speed ** 2

    # Altitude hold — only outside corridor (navigation phase)
    # Penalize dropping below cruise altitude to prevent premature descent
    nav_weight = 1.0 - z_weight
    altitude_error = cruise_altitude - drone_pos[:, 2]
    r_altitude = -reward_altitude_hold_coef * nav_weight * altitude_error ** 2

    # Rover stillness — penalize body speed when drone is in landing corridor
    r_rover_stillness = -reward_rover_stillness_coef * z_weight * rover_speed ** 2

    # Rover yaw rate penalty — penalize heading changes when drone is in landing corridor
    r_rover_yawrate = -reward_rover_yawrate_coef * z_weight * rover_omega ** 2

    # Rover lateral/backward penalty — camera faces forward, so penalize
    # lateral speed (vy²) and backward speed (negative vx)
    backward_speed = jnp.maximum(-rover_vx_body, 0.0)  # only when vx < 0
    r_rover_lateral = -reward_rover_lateral_coef * (rover_vy_body ** 2 + backward_speed ** 2)

    # Rover boundary penalty: penalize rover being at arena edge
    rover_at_boundary = (
        (jnp.abs(rover_xy[:, 0]) >= map_half_x - 0.01) |
        (jnp.abs(rover_xy[:, 1]) >= map_half_y - 0.01)
    )
    r_rover_boundary = jnp.where(rover_at_boundary, -reward_rover_boundary_coef, 0.0)

    # Drone velocity penalty: only penalize speed above max_drone_speed (dead-zone)
    drone_speed = jnp.linalg.norm(drone_vel, axis=-1)
    excess_drone_speed = jnp.maximum(drone_speed - max_drone_speed, 0.0)
    r_drone_velocity = -reward_drone_velocity_coef * excess_drone_speed ** 2

    # Drone XY speed penalty in corridor — force hover-then-descend (not glide)
    drone_speed_xy = jnp.linalg.norm(drone_vel[:, :2], axis=-1)
    r_drone_xy_corridor = -reward_drone_xy_corridor_coef * z_weight * drone_speed_xy ** 2

    # Angle penalty: penalize roll/pitch deviation
    r_angle = -reward_angle_coef * (drone_rpy[:, 0] ** 2 + drone_rpy[:, 1] ** 2)

    # Action smoothness (drone): per-actuator penalties
    delta_drone = drone_cmd - last_drone_cmd
    r_smooth_drone = (
        -reward_action_smoothness_rpy * jnp.sum(delta_drone[:, :3] ** 2, axis=-1)
        - reward_action_smoothness_thrust * delta_drone[:, 3] ** 2
    )

    # Action smoothness (rover): per-axis penalty
    delta_rover = rover_cmd - last_rover_cmd
    r_smooth_rover = -jnp.sum(rover_smoothness_weights * delta_rover ** 2, axis=-1)

    # Landing bonus + relative velocity penalty at touchdown
    rel_vel = jnp.stack([
        drone_vel[:, 0] - rover_vel_xy[:, 0],
        drone_vel[:, 1] - rover_vel_xy[:, 1],
        drone_vel[:, 2],
    ], axis=-1)
    landing_speed = jnp.linalg.norm(rel_vel, axis=-1)
    r_landing = jnp.where(
        landed,
        reward_landing - reward_landing_velocity_coef * landing_speed,
        0.0,
    )

    # Landing precision bonus: reward inversely proportional to distance from pad center
    # precision = 1.0 at center, 0.0 at landing_zone_radius edge
    precision = jnp.where(
        landed,
        reward_landing_precision_coef * jnp.maximum(1.0 - horiz_dist / landing_zone_radius, 0.0),
        0.0,
    )
    r_landing = r_landing + precision

    # Crash penalty
    r_crash = jnp.where(crashed, reward_crash, 0.0)

    # Boundary penalty
    r_boundary = jnp.where(oob, reward_boundary, 0.0)

    # Per-step time penalty to discourage hovering
    r_time = -reward_time_penalty * jnp.ones_like(r_xy)

    # Per-agent reward split: team + agent-specific
    team = r_xy + r_z + r_landing + r_time
    drone_only = (r_descent + r_altitude + r_drone_velocity + r_drone_xy_corridor
                  + r_angle + r_smooth_drone + r_crash + r_boundary)
    rover_only = r_rover_stillness + r_rover_yawrate + r_rover_lateral + r_smooth_rover + r_rover_boundary

    drone_reward = team + drone_only
    rover_reward = team + rover_only

    components = {
        "progress_xy": r_xy,
        "progress_z": r_z,
        "descent_speed": r_descent,
        "altitude_hold": r_altitude,
        "angle": r_angle,
        "smooth_drone": r_smooth_drone,
        "smooth_rover": r_smooth_rover,
        "landing": r_landing,
        "crash": r_crash,
        "boundary": r_boundary,
        "time": r_time,
        "rover_stillness": r_rover_stillness,
        "rover_yawrate": r_rover_yawrate,
        "rover_lateral": r_rover_lateral,
        "rover_boundary": r_rover_boundary,
        "drone_velocity": r_drone_velocity,
        "drone_xy_corridor": r_drone_xy_corridor,
    }
    return drone_reward.astype(jnp.float32), rover_reward.astype(jnp.float32), components, horiz_dist, vert_dist


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
        self._rover_type = self.cfg.rover_type
        self.render_mode = render_mode

        # Per-axis rover action smoothness weights (X3: vx, vy, wz)
        self._rover_smoothness_weights = jnp.array([
            self.cfg.reward_action_smoothness_vx,
            self.cfg.reward_action_smoothness_vy,
            self.cfg.reward_action_smoothness_wz,
        ])

        if spawn_fn is None:
            spawn_fn = create_default_spawn_fn(
                drone_z_min=self.cfg.drone_z_min,
                drone_z_max=self.cfg.drone_z_max,
                drone_x_half=self.cfg.map_half_x,
                drone_y_half=self.cfg.map_half_y,
                rover_x_half=self.cfg.map_half_x,
                rover_y_half=self.cfg.map_half_y,
                rover_max_speed=self.cfg.rover_max_speed,
                rover_nx=self.cfg.rover_nx,
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

        # Add rover landing pad as a mocap body for contact-based landing detection
        self._add_landing_pad_to_scene()

        # Override sim mass (sim_mass overrides only the simulator, not MPC)
        override_mass = self.cfg.sim_mass if self.cfg.sim_mass is not None else self.cfg.mass
        if override_mass is not None:
            params = self.sim.data.params.replace(
                mass=jnp.full_like(self.sim.data.params.mass, override_mass)
            )
            self.sim.data = self.sim.data.replace(params=params)
            self.sim.default_data = self.sim.default_data.replace(params=params)

        self._apply_domain_randomization()

        # Store the core pipeline from Crazyflow (no injections)
        self._core_step_pipeline = self.sim.step_pipeline
        self._disturbance_enabled = False

        # OU disturbance state: (N_worlds, 1_drone, 3) for force and torque
        N = self.cfg.n_worlds
        self._ou_force = jnp.zeros((N, 1, 3))
        self._ou_torque = jnp.zeros((N, 1, 3))

        # Pre-create ground effect function if configured (always-on, not toggled)
        self._ge_fn = self._create_ground_effect_fn() if self.cfg.enable_ground_effect else None

        # Wind model
        self._wind_model = None
        self._wind_drag_matrix = None
        self._wind_enabled = False
        if self.cfg.enable_wind:
            self._init_wind_model()

        # Build the full pipeline with current disturbance/GE settings
        self._rebuild_step_pipeline()

        # Hover RPM from first_principles parameters
        fp_params = load_params("first_principles", self.cfg.drone_model)
        rpm2thrust = fp_params["rpm2thrust"]
        a_c = rpm2thrust[2]
        b_c = rpm2thrust[1]
        mass = self.cfg.mass if self.cfg.mass is not None else float(fp_params["mass"])
        c_c = rpm2thrust[0] - mass * self.cfg.gravity / 4
        self.hover_rpm = float((-b_c + np.sqrt(b_c ** 2 - 4 * a_c * c_c)) / (2 * a_c))

    def _add_landing_pad_to_scene(self):
        """Add a mocap body representing the rover landing pad for contact detection.

        The pad is a flat box at rover_height that moves with the rover each step.
        Contact between the drone's collision geoms and this pad triggers landing detection.
        """
        import mujoco

        spec = self.sim.spec
        pad_body = spec.worldbody.add_body(
            name="landing_pad",
            mocap=True,
            pos=[0.0, 0.0, self.cfg.rover_height],
        )
        # Pad thickness = 0.003m half-height (6mm total)
        pad_body.add_geom(
            name="pad_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.cfg.rover_platform_radius, self.cfg.rover_platform_radius, 0.003],
            contype=1,
            conaffinity=1,
        )

        # Switch drone collision from sphere to box (better for flat landing)
        # Shift col_box down so its bottom aligns with the legs/guards.
        # In the drone XML: body at z=0.05 above freejoint, col_box at pos=[0,0,0]
        # with half-height 0.02. Box bottom = 0.05 - 0.02 = 0.03 above freejoint.
        # Legs touch ground at freejoint origin (z=0), so shift box down by 0.03.
        for geom in spec.geoms:
            if geom.name == "col_sphere:0":
                geom.contype = 0
                geom.conaffinity = 0
            elif geom.name == "col_box:0":
                geom.contype = 1
                geom.conaffinity = 1
                geom.pos = [0.0, 0.0, -0.03]  # shift down to align with leg tips

        # Rebuild MJX model with the pad and updated collision geoms
        self.sim.build_mjx()

        # Store the mocap body index for fast updates
        self._pad_mocap_id = self.sim.mj_model.body("landing_pad").mocapid[0]

    def _sync_pad_to_rover(self):
        """Update the landing pad mocap position to match the current rover state."""
        rover = np.asarray(self.rover_state)
        rover_xy = rover[:, :2]
        pad_z = self.cfg.rover_height
        # Build (N, 1, 3) mocap_pos — 1 mocap body per world
        pad_pos = np.zeros((self.cfg.n_worlds, 1, 3))
        pad_pos[:, 0, :2] = rover_xy
        pad_pos[:, 0, 2] = pad_z
        self.sim.mjx_data = self.sim.mjx_data.replace(
            mocap_pos=self.sim.mjx_data.mocap_pos.at[:, self._pad_mocap_id, :].set(
                jnp.array(pad_pos[:, 0, :])
            )
        )
        # Invalidate mjx sync flag so next contacts() call re-syncs
        self.sim.data = self.sim.data.replace(
            core=self.sim.data.core.replace(mjx_synced=False)
        )

    def _check_pad_contact(self) -> np.ndarray:
        """Check if the drone is in contact with the landing pad.

        Returns:
            (N,) bool array — True if drone is touching the pad in each world.
        """
        # contacts() triggers sync_sim2mjx (kinematics + collision) if needed
        contact_flags = self.sim.contacts("drone:0")
        # contact_flags shape: (N, max_contacts) — True where dist < 0 for drone geoms
        # Reduce to per-world: any contact for the drone
        return np.asarray(contact_flags.any(axis=-1))

    def set_sim_mass(self, mass: float):
        """Override the Crazyflow simulator mass without recreating the env."""
        self.cfg.sim_mass = mass
        params = self.sim.data.params.replace(
            mass=jnp.full_like(self.sim.data.params.mass, mass)
        )
        self.sim.data = self.sim.data.replace(params=params)
        self.sim.default_data = self.sim.default_data.replace(params=params)

    def set_disturbance(self, enabled: bool,
                        force_std: float | None = None,
                        torque_std: float | None = None):
        """Enable/disable disturbance and update magnitudes without recreating the env."""
        if force_std is not None:
            self.cfg.disturbance_force_std = force_std
        if torque_std is not None:
            self.cfg.disturbance_torque_std = torque_std
        self._disturbance_enabled = enabled
        self.cfg.enable_disturbance = enabled
        self._rebuild_step_pipeline()

    def _init_wind_model(self):
        """Initialize the wind model and drag matrix."""
        fp_params = load_params("first_principles", self.cfg.drone_model)
        self._wind_drag_matrix = jnp.array(fp_params["drag_matrix"])
        self._wind_model = WindModel(
            n_worlds=self.cfg.n_worlds,
            wind_speed=self.cfg.wind_speed,
            wind_direction=self.cfg.wind_direction,
            gust_intensity=self.cfg.gust_intensity,
            gust_correlation_time=self.cfg.gust_correlation_time,
            turbulence_level=self.cfg.turbulence_level,
            turbulence_time_constant=self.cfg.turbulence_time_constant,
            dt=self.cfg.dt,
        )
        self._wind_velocity = jnp.zeros((self.cfg.n_worlds, 1, 3))
        self._wind_enabled = True

    def set_wind(self, enabled: bool = True,
                 wind_speed: float | None = None,
                 wind_direction: float | None = None,
                 gust_intensity: float | None = None,
                 turbulence_level: str | None = None):
        """Configure wind disturbance without recreating the env."""
        if wind_speed is not None:
            self.cfg.wind_speed = wind_speed
        if wind_direction is not None:
            self.cfg.wind_direction = wind_direction
        if gust_intensity is not None:
            self.cfg.gust_intensity = gust_intensity
        if turbulence_level is not None:
            self.cfg.turbulence_level = turbulence_level
        self.cfg.enable_wind = enabled
        if enabled:
            self._init_wind_model()
        else:
            self._wind_model = None
            self._wind_enabled = False
        self._rebuild_step_pipeline()

    def _create_wind_fn(self):
        """Create a wind drag function for the simulation pipeline.

        Wind velocity is computed outside JIT in _update_wind_state() and
        stored in self._wind_velocity. The pipeline function reads it via
        closure (same pattern as OU disturbance).
        """
        drag_matrix = self._wind_drag_matrix

        def wind_fn(data: SimData) -> SimData:
            drag_force = compute_wind_drag_force(
                data.states.quat, self._wind_velocity, drag_matrix
            )
            states = data.states.replace(
                force=data.states.force + drag_force,
            )
            return data.replace(states=states)

        return wind_fn

    def _update_wind_state(self):
        """Advance wind model state by one control step. Called outside JIT."""
        if not self._wind_enabled or self._wind_model is None:
            return
        key = self.sim.data.core.rng_key
        key, wind_key = jax.random.split(key)
        self.sim.data = self.sim.data.replace(
            core=self.sim.data.core.replace(rng_key=key))
        self._wind_velocity = self._wind_model.step(wind_key)

    def _apply_domain_randomization(self, mask: jnp.ndarray | None = None):
        """Randomize mass and/or inertia for enabled worlds."""
        if not self.cfg.randomize_mass and not self.cfg.randomize_inertia:
            return

        if self.cfg.randomize_mass:
            key = jax.random.key(np.random.randint(0, 2 ** 31))
            mass_noise = jax.random.normal(
                key, (self.cfg.n_worlds, 1, 1)
            ) * self.cfg.mass_randomization_std
            base_mass = self.cfg.sim_mass if self.cfg.sim_mass is not None else (
                self.cfg.mass if self.cfg.mass is not None else float(
                    self.sim.data.params.mass.mean()
                )
            )
            randomize_mass(self.sim, base_mass + mass_noise, mask)

        if self.cfg.randomize_inertia:
            key = jax.random.key(np.random.randint(0, 2 ** 31))
            J_noise = jax.random.normal(
                key, (self.cfg.n_worlds, 1, 3, 3)
            ) * self.cfg.inertia_randomization_std
            J_rand = self.sim.data.params.J + J_noise
            randomize_inertia(self.sim, J_rand, mask)

    def _create_ground_effect_fn(self):
        """Create a ground effect function for the simulation pipeline.

        Ground effect model: each rotor's thrust is multiplied by
            k_ge = 1 / (1 - (R / (4 * z_eff))^2)
        where z_eff is the rotor's height above the nearest surface (ground or
        rover pad). The extra thrust per rotor is converted to an additive CoM
        force and torque in the world frame.

        The function captures a reference to self.rover_state so it always uses
        the current rover position.
        """
        R = self.cfg.ground_effect_rotor_radius
        ge_scale = self.cfg.ground_effect_scale
        rover_height = self.cfg.rover_height
        pad_radius = self.cfg.rover_platform_radius
        gravity = self.cfg.gravity
        # Motor site positions in body frame (X-config): (4, 3)
        L = float(load_params("first_principles", self.cfg.drone_model)["L"])
        motor_pos_body = jnp.array([
            [L, -L, 0.0],   # motor0
            [-L, -L, 0.0],  # motor1
            [-L, L, 0.0],   # motor2
            [L, L, 0.0],    # motor3
        ])
        # We need rpm2thrust to convert rotor_vel → per-motor thrust
        rpm2thrust = jnp.array(load_params("first_principles", self.cfg.drone_model)["rpm2thrust"])
        # Mixing matrix for torque: (3, 4) maps per-motor thrust to body torques
        # We use motor_pos_body cross body_z to get torque arms
        env_ref = self  # capture reference for rover state access

        def ground_effect_fn(data: SimData) -> SimData:
            states = data.states
            pos = states.pos           # (N, 1, 3)
            quat = states.quat         # (N, 1, 4) xyzw
            rotor_vel = states.rotor_vel  # (N, 1, 4) RPM

            # Per-motor thrust from RPM: T = a*rpm^2 + b*rpm + c
            rpm = rotor_vel[:, 0, :]  # (N, 4)
            per_motor_thrust = rpm2thrust[2] * rpm ** 2 + rpm2thrust[1] * rpm + rpm2thrust[0]  # (N, 4)
            per_motor_thrust = jnp.maximum(per_motor_thrust, 0.0)

            # Rotation matrix from body to world (from quat xyzw)
            # Using the same quat convention as Crazyflow
            x, y, z, w = quat[:, 0, 0], quat[:, 0, 1], quat[:, 0, 2], quat[:, 0, 3]
            R_00 = 1 - 2*(y*y + z*z); R_01 = 2*(x*y - z*w); R_02 = 2*(x*z + y*w)
            R_10 = 2*(x*y + z*w); R_11 = 1 - 2*(x*x + z*z); R_12 = 2*(y*z - x*w)
            R_20 = 2*(x*z - y*w); R_21 = 2*(y*z + x*w); R_22 = 1 - 2*(x*x + y*y)
            # R: (N, 3, 3)
            rot = jnp.stack([
                jnp.stack([R_00, R_01, R_02], axis=-1),
                jnp.stack([R_10, R_11, R_12], axis=-1),
                jnp.stack([R_20, R_21, R_22], axis=-1),
            ], axis=-2)

            # Motor world positions: (N, 4, 3)
            drone_pos = pos[:, 0, :]  # (N, 3)
            motor_world = drone_pos[:, None, :] + jnp.einsum('nij,kj->nki', rot, motor_pos_body)

            # Get rover XY — use the captured env reference
            rover_xy = jnp.array(env_ref.rover_state[:, :2])  # (N, 2)

            # Per-motor effective height above nearest surface
            motor_xy = motor_world[:, :, :2]  # (N, 4, 2)
            motor_z = motor_world[:, :, 2]    # (N, 4)

            # Check if each motor is over the pad
            motor_dist_to_rover = jnp.linalg.norm(
                motor_xy - rover_xy[:, None, :], axis=-1
            )  # (N, 4)
            over_pad = motor_dist_to_rover < pad_radius

            # Effective height: over pad → z - rover_height, otherwise → z (above ground)
            z_eff = jnp.where(over_pad, motor_z - rover_height, motor_z)
            z_eff = jnp.maximum(z_eff, 0.01)  # clamp to avoid div by zero

            # Ground effect multiplier per motor
            ratio = R / (4.0 * z_eff)
            ratio = jnp.minimum(ratio, 0.9)  # clamp to prevent singularity
            k_ge = 1.0 / (1.0 - ratio ** 2)

            # Extra thrust per motor (in body z direction)
            extra_thrust = per_motor_thrust * (k_ge - 1.0) * ge_scale  # (N, 4)

            # Convert to world-frame force: extra thrust along body z-axis
            body_z_world = rot[:, :, 2]  # (N, 3) — body z in world frame
            net_extra_force = jnp.sum(extra_thrust, axis=-1, keepdims=True) * body_z_world  # (N, 3)

            # Convert to torque: cross product of motor arm × extra force (in body frame)
            # Torque per motor in body frame = r_motor × (0, 0, extra_T)
            # = (r_y * extra_T, -r_x * extra_T, 0)
            torque_body_x = jnp.sum(motor_pos_body[None, :, 1] * extra_thrust, axis=-1)  # (N,)
            torque_body_y = jnp.sum(-motor_pos_body[None, :, 0] * extra_thrust, axis=-1)  # (N,)
            torque_body_z = jnp.zeros_like(torque_body_x)
            torque_body = jnp.stack([torque_body_x, torque_body_y, torque_body_z], axis=-1)  # (N, 3)

            # Rotate body torque to world frame
            net_extra_torque = jnp.einsum('nij,nj->ni', rot, torque_body)  # (N, 3)

            # Add ground effect on top of existing forces (e.g. disturbance noise)
            new_force = states.force + net_extra_force[:, None, :]   # (N, 1, 3)
            new_torque = states.torque + net_extra_torque[:, None, :]  # (N, 1, 3)

            states = states.replace(force=new_force, torque=new_torque)
            return data.replace(states=states)

        return ground_effect_fn

    def _create_disturbance_fn(self):
        """Create a disturbance function for the simulation pipeline.

        Supports two modes:
          - "gaussian": i.i.d. white noise each substep
          - "ou": Ornstein-Uhlenbeck process — injects current OU state (updated
                  externally in _update_ou_state per substep to avoid JAX JIT issues)
        """
        force_std = self.cfg.disturbance_force_std
        torque_std = self.cfg.disturbance_torque_std
        dist_type = self.cfg.disturbance_type

        if dist_type == "ou":
            # OU disturbance: inject self._ou_force/torque (updated outside JIT)
            def disturbance_fn(data: SimData) -> SimData:
                states = data.states
                states = states.replace(
                    force=states.force + self._ou_force,
                    torque=states.torque + self._ou_torque,
                )
                return data.replace(states=states)
        else:
            # Original Gaussian white noise
            def disturbance_fn(data: SimData) -> SimData:
                key = data.core.rng_key
                key, force_key, torque_key = jax.random.split(key, 3)
                states = data.states
                disturbance_force = jax.random.normal(force_key, states.force.shape) * force_std
                disturbance_torque = jax.random.normal(torque_key, states.torque.shape) * torque_std
                states = states.replace(
                    force=states.force + disturbance_force,
                    torque=states.torque + disturbance_torque,
                )
                core = data.core.replace(rng_key=key)
                return data.replace(states=states, core=core)

        return disturbance_fn

    def _update_ou_state(self):
        """Advance OU disturbance state by one control step. Called outside JIT."""
        if self.cfg.disturbance_type != "ou" or not self._disturbance_enabled:
            return
        theta = self.cfg.disturbance_ou_theta
        sim_dt = 1.0 / self.cfg.control_freq  # control step dt (not substep)
        force_std = self.cfg.disturbance_force_std
        torque_std = self.cfg.disturbance_torque_std

        key = self.sim.data.core.rng_key
        key, fk, tk = jax.random.split(key, 3)
        self.sim.data = self.sim.data.replace(
            core=self.sim.data.core.replace(rng_key=key))

        noise_f = jax.random.normal(fk, self._ou_force.shape)
        noise_t = jax.random.normal(tk, self._ou_torque.shape)
        self._ou_force = (
            self._ou_force
            - theta * self._ou_force * sim_dt
            + force_std * jnp.sqrt(sim_dt) * noise_f
        )
        self._ou_torque = (
            self._ou_torque
            - theta * self._ou_torque * sim_dt
            + torque_std * jnp.sqrt(sim_dt) * noise_t
        )

    def _rebuild_step_pipeline(self):
        """Rebuild the simulation pipeline with current disturbance/GE settings.

        Pipeline order (injected between force_torque_ctrl and integration):
          1. reset_external_forces — zeros states.force/torque (prevents accumulation)
          2. disturbance_fn — adds random noise (if enabled)
          3. ground_effect_fn — adds per-rotor GE force/torque (if enabled)
        """
        extra_fns = []
        need_reset = self._disturbance_enabled or self._ge_fn is not None or self._wind_enabled

        if need_reset:
            def reset_external_forces(data: SimData) -> SimData:
                states = data.states
                zeros = jnp.zeros_like(states.force)
                return data.replace(states=states.replace(force=zeros, torque=zeros))
            extra_fns.append(reset_external_forces)

        if self._disturbance_enabled:
            extra_fns.append(self._create_disturbance_fn())

        if self._wind_enabled:
            extra_fns.append(self._create_wind_fn())

        if self._ge_fn is not None:
            extra_fns.append(self._ge_fn)

        self.sim.step_pipeline = (
            self._core_step_pipeline[:2] + tuple(extra_fns) + self._core_step_pipeline[2:]
        )
        self.sim.build_step_fn()

    def _enable_disturbance(self):
        """Enable disturbance injection in simulation pipeline."""
        self._disturbance_enabled = True
        self._rebuild_step_pipeline()

    def _disable_disturbance(self):
        """Disable disturbance injection in simulation pipeline."""
        if not self._disturbance_enabled:
            return
        self._disturbance_enabled = False
        self._rebuild_step_pipeline()

    def _define_spaces(self):
        """Define per-agent observation and action spaces."""
        # Drone observation:
        #   own: pos(3) + vel(3) + rotmat_flat(9) + body_rates(3) = 18
        #   rover: xy(2) + vel_xy(2) + heading_sincos(2) + speed(1) + lateral(1) = 8
        #   relative: pos(3) = 3
        self.drone_obs_dim = 29

        # Rover observation (X3):
        #   pos(2)+cs(2)+vx(1)+vy(1)+wz(1) + rel_pos(3)+vel(3)+speed(1)+dist(1) = 15
        self.rover_obs_dim = 15

        # MPC state dimensions
        self._drone_state_type = self.cfg.drone_state_type
        self.drone_mpc_state_dim = 13 if self._drone_state_type == "quat" else 12
        self.rover_mpc_state_dim = self.cfg.rover_nx  # 7 (x3)

        # Shared state for centralized critic:
        #   drone: pos(3) + vel(3) + rpy(3) + body_rates(3) = 12
        #   rover: pos(2) + vel_xy(2) + heading_sincos(2) + vx(1) + vy(1) + ωz(1) = 9
        #   relative: drone_pos - rover_xy (3) = 3
        self.shared_state_dim = 24

        self.observation_space = spaces.Dict({
            "drone": spaces.Box(-np.inf, np.inf, (self.drone_obs_dim,), dtype=np.float32),
            "rover": spaces.Box(-np.inf, np.inf, (self.rover_obs_dim,), dtype=np.float32),
        })
        self.observation_spaces = self.observation_space

        # Rover action bounds (X3 body velocity commands)
        rover_action_low = np.array(
            [-self.cfg.rover_vx_max, -self.cfg.rover_vy_max, -self.cfg.rover_wz_max],
            dtype=np.float32,
        )
        rover_action_high = np.array(
            [self.cfg.rover_vx_max, self.cfg.rover_vy_max, self.cfg.rover_wz_max],
            dtype=np.float32,
        )

        self.action_space = spaces.Dict({
            "drone": spaces.Box(
                low=np.array([-self.cfg.roll_pitch_max, -self.cfg.roll_pitch_max,
                               -self.cfg.yaw_max, self.cfg.thrust_min], dtype=np.float32),
                high=np.array([self.cfg.roll_pitch_max, self.cfg.roll_pitch_max,
                                self.cfg.yaw_max, self.cfg.thrust_max], dtype=np.float32),
            ),
            "rover": spaces.Box(low=rover_action_low, high=rover_action_high),
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
        nx = self.cfg.rover_nx
        nu = self.cfg.rover_nu
        self.rover_state = jnp.zeros((N, nx))
        self.drone_cmd = jnp.zeros((N, 4))         # attitude cmd
        self.last_drone_cmd = jnp.zeros((N, 4))
        self.rover_cmd = jnp.zeros((N, nu))
        self.last_rover_cmd = jnp.zeros((N, nu))
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
        self.last_rover_cmd = jnp.zeros((self.cfg.n_worlds, self.cfg.rover_nu))

        # Reset OU disturbance state
        N = self.cfg.n_worlds
        self._ou_force = jnp.zeros((N, 1, 3))
        self._ou_torque = jnp.zeros((N, 1, 3))

        # Reset wind model state
        if self._wind_model is not None:
            self._wind_model.reset()

        self._spawn_agents()
        self._init_last_dist()
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

        # Random initial yaw: quaternion [0, 0, sin(ψ/2), cos(ψ/2)] (xyzw)
        if self.cfg.drone_init_yaw_max > 0:
            key, yaw_key = jax.random.split(key)
            self.sim.data = self.sim.data.replace(core=self.sim.data.core.replace(rng_key=key))
            yaw = jax.random.uniform(
                yaw_key, shape=(N,),
                minval=-self.cfg.drone_init_yaw_max,
                maxval=self.cfg.drone_init_yaw_max,
            )
            half_yaw = yaw / 2.0
            all_quat = jnp.stack([
                jnp.zeros(N), jnp.zeros(N),
                jnp.sin(half_yaw), jnp.cos(half_yaw),
            ], axis=-1)[:, None, :]  # (N, 1, 4)
        else:
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

    def _init_last_dist(self):
        """Initialize last distances for progress reward computation."""
        drone_pos = self.sim.data.states.pos[:, 0]  # (N, 3)
        rover_xy = self.rover_state[:, :2]
        self.last_horiz_dist = jnp.linalg.norm(drone_pos[:, :2] - rover_xy, axis=-1)
        self.last_vert_dist = jnp.abs(drone_pos[:, 2] - self.cfg.rover_height)

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

        # Reset OU disturbance state for done worlds
        self._ou_force = jnp.where(mask_jnp[:, None, None], 0.0, self._ou_force)
        self._ou_torque = jnp.where(mask_jnp[:, None, None], 0.0, self._ou_torque)

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

        # Random initial yaw for reset worlds
        if self.cfg.drone_init_yaw_max > 0:
            key, yaw_key = jax.random.split(key)
            self.sim.data = self.sim.data.replace(core=self.sim.data.core.replace(rng_key=key))
            yaw = jax.random.uniform(
                yaw_key, shape=(N,),
                minval=-self.cfg.drone_init_yaw_max,
                maxval=self.cfg.drone_init_yaw_max,
            )
            half_yaw = yaw / 2.0
            reset_quat = jnp.stack([
                jnp.zeros(N), jnp.zeros(N),
                jnp.sin(half_yaw), jnp.cos(half_yaw),
            ], axis=-1)[:, None, :]  # (N, 1, 4)
        else:
            reset_quat = None

        replace_kwargs = dict(pos=all_drone_pos[:, None, :], rotor_vel=hover_rpm)
        if reset_quat is not None:
            replace_kwargs["quat"] = reset_quat
        states = leaf_replace(
            self.sim.data.states, mask=mask_jnp, **replace_kwargs
        )
        self.sim.data = self.sim.data.replace(states=states)
        self._apply_domain_randomization(mask=mask_jnp)

        # Reset last distances for progress reward in re-spawned worlds
        new_horiz = jnp.linalg.norm(all_drone_pos[:, :2] - all_rover_state[:, :2], axis=-1)
        new_vert = jnp.abs(all_drone_pos[:, 2] - self.cfg.rover_height)
        self.last_horiz_dist = jnp.where(mask_jnp, new_horiz, self.last_horiz_dist)
        self.last_vert_dist = jnp.where(mask_jnp, new_vert, self.last_vert_dist)

        # Defer trajectory clear so the final render still shows the full trajectory
        if self._trajectory_enabled and done_mask[0]:
            self._trajectory_pending_clear = True

    # -------------------------------------------------------------------------
    # Rover helpers (rover-type-agnostic extraction)
    # -------------------------------------------------------------------------

    def _extract_rover_values(self, rover_state: jnp.ndarray) -> dict:
        """Extract rover values from X3 rover state.

        Returns dict with: xy, c, s, vel_xy (world), speed, omega, vx_body, vy_body, wz.
        """
        xy = rover_state[:, :2]
        c = rover_state[:, 2]
        s = rover_state[:, 3]
        vx_body = rover_state[:, 4]
        vy_body = rover_state[:, 5]
        wz = rover_state[:, 6]
        vel_x_world = vx_body * c - vy_body * s
        vel_y_world = vx_body * s + vy_body * c
        speed = jnp.sqrt(vx_body ** 2 + vy_body ** 2)

        vel_xy = jnp.stack([vel_x_world, vel_y_world], axis=-1)
        return {
            "xy": xy, "c": c, "s": s,
            "vel_xy": vel_xy, "speed": speed, "omega": wz,
            "vx_body": vx_body, "vy_body": vy_body, "wz": wz,
        }

    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[dict, dict, dict, dict, dict]:
        """Execute one environment step.

        Args:
            actions: {"drone": (N, 4), "rover": (N, nu)} in physical units.

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

        # Process and clip rover action (X3 body velocity commands)
        nu = self.cfg.rover_nu
        rover_action = np.array(actions.get("rover", np.zeros((N, nu))), copy=True).reshape(N, nu)
        rover_action[:, 0] = np.clip(rover_action[:, 0], -self.cfg.rover_vx_max, self.cfg.rover_vx_max)
        rover_action[:, 1] = np.clip(rover_action[:, 1], -self.cfg.rover_vy_max, self.cfg.rover_vy_max)
        rover_action[:, 2] = np.clip(rover_action[:, 2], -self.cfg.rover_wz_max, self.cfg.rover_wz_max)
        self.rover_cmd = jnp.array(rover_action)

        # Step rover dynamics (X3 mecanum)
        self.rover_state = mecanum_step_batched(
            self.rover_state, self.rover_cmd, self.cfg.dt, WHEEL_VEL_MAX,
        )
        self.rover_state = _jit_clamp_rover_x3(
            self.rover_state, self.cfg.map_half_x, self.cfg.map_half_y
        )

        # Step drone (Crazyflow)
        self.sim.attitude_control(self.drone_cmd[:, None, :])
        self._update_ou_state()  # Advance OU disturbance (once per control step)
        self._update_wind_state()  # Advance wind model (once per control step)
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

        # Extract rover values (rover-type-agnostic)
        rv = self._extract_rover_values(self.rover_state)

        # Sync pad position to rover and check contact-based landing
        self._sync_pad_to_rover()
        pad_contact = self._check_pad_contact()  # (N,) bool

        landed, crashed = _jit_check_landing(
            drone_pos_jnp, drone_vel_jnp, drone_rpy,
            rv["xy"], rv["vel_xy"],
            self.cfg.rover_height,
            self.cfg.rover_platform_radius, self.cfg.landing_zone_radius,
            self.cfg.landing_z_tol,
            self.cfg.landing_vel_xy_tol, self.cfg.landing_vel_z_tol,
            self.cfg.landing_attitude_tol,
            jnp.array(pad_contact),
        )
        oob = _jit_check_oob(
            drone_pos_jnp,
            self.cfg.map_half_x, self.cfg.map_half_y,
            self.cfg.drone_z_max,
        )

        drone_reward, rover_reward, components, horiz_dist, vert_dist = _jit_compute_rewards(
            drone_pos_jnp, drone_vel_jnp, drone_rpy,
            self.drone_cmd, self.last_drone_cmd,
            rv["xy"], rv["vel_xy"], rv["speed"], rv["omega"],
            rv["vx_body"], rv["vy_body"],
            self.rover_cmd, self.last_rover_cmd,
            landed, crashed, oob,
            self.cfg.rover_height,
            self.cfg.reward_landing,
            self.cfg.reward_crash,
            self.cfg.reward_boundary,
            self.cfg.reward_progress_coef,
            self.last_horiz_dist,
            self.last_vert_dist,
            self.cfg.reward_angle_coef,
            self.cfg.reward_action_smoothness_thrust,
            self.cfg.reward_action_smoothness_rpy,
            self.cfg.reward_action_smoothness_wheel,
            self._rover_smoothness_weights,
            self.cfg.reward_landing_velocity_coef,
            self.cfg.corridor_radius,
            self.cfg.corridor_transition,
            self.cfg.max_descent_speed,
            self.cfg.reward_descent_speed_coef,
            self.cfg.reward_altitude_hold_coef,
            self.cfg.cruise_altitude,
            self.cfg.reward_time_penalty,
            self.cfg.reward_rover_stillness_coef,
            self.cfg.reward_rover_yawrate_coef,
            self.cfg.reward_rover_lateral_coef,
            self.cfg.reward_drone_velocity_coef,
            self.cfg.reward_drone_xy_corridor_coef,
            self.cfg.max_drone_speed,
            self.cfg.reward_rover_boundary_coef,
            self.cfg.reward_landing_precision_coef,
            self.cfg.landing_zone_radius,
            self.cfg.map_half_x,
            self.cfg.map_half_y,
        )

        self.last_horiz_dist = horiz_dist
        self.last_vert_dist = vert_dist

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

        rover = np.asarray(self.rover_state)
        rover_xy = rover[:, :2]
        rover_c = rover[:, 2]
        rover_s = rover[:, 3]

        rover_vx_body = rover[:, 4]
        rover_vy_body = rover[:, 5]
        rover_wz = rover[:, 6]
        rover_speed = np.sqrt(rover_vx_body ** 2 + rover_vy_body ** 2)
        vel_x_world = rover_vx_body * rover_c - rover_vy_body * rover_s
        vel_y_world = rover_vx_body * rover_s + rover_vy_body * rover_c

        rover_vel_xy = np.stack([vel_x_world, vel_y_world], axis=-1)

        # ---- Drone observation ----
        rel_pos = drone_pos - np.concatenate([rover_xy, np.zeros((N, 1))], axis=-1)

        drone_obs_parts = [
            drone_pos,                          # 3
            drone_vel,                          # 3
            drone_rotmat_flat,                  # 9
            drone_ang_vel,                      # 3
            rover_xy,                           # 2
            rover_vel_xy,                       # 2
            rover_s[:, None],                   # 1
            rover_c[:, None],                   # 1
            rover_speed[:, None],               # 1
            rover_vy_body[:, None],             # 1 lateral speed
            rel_pos,                            # 3
        ]
        drone_obs = np.concatenate(drone_obs_parts, axis=-1).astype(np.float32)

        # ---- Rover observation ----
        drone_speed = np.linalg.norm(drone_vel, axis=-1, keepdims=True)
        dist = np.linalg.norm(rel_pos, axis=-1, keepdims=True)

        rover_obs_parts = [
            rover_xy,                                   # 2
            rover_c[:, None],                           # 1
            rover_s[:, None],                           # 1
            rover_vx_body[:, None],                     # 1
            rover_vy_body[:, None],                     # 1
            rover_wz[:, None],                          # 1
        ]
        rover_obs_parts += [
            rel_pos,                                    # 3
            drone_vel,                                  # 3
            drone_speed,                                # 1
            dist,                                       # 1
        ]
        rover_obs = np.concatenate(rover_obs_parts, axis=-1).astype(np.float32)

        return {"drone": drone_obs, "rover": rover_obs}

    def _get_shared_state(self) -> np.ndarray:
        """Get shared state for centralized critic."""
        N = self.cfg.n_worlds
        states = self.sim.data.states

        drone_pos = np.asarray(states.pos[:, 0])
        drone_vel = np.asarray(states.vel[:, 0])
        rover = np.asarray(self.rover_state)
        rover_xy = rover[:, :2]
        rover_c = rover[:, 2]
        rover_s = rover[:, 3]

        vx_body = rover[:, 4]
        vy_body = rover[:, 5]
        wz = rover[:, 6]
        vel_x_world = vx_body * rover_c - vy_body * rover_s
        vel_y_world = vx_body * rover_s + vy_body * rover_c

        rover_vel_xy = np.stack([vel_x_world, vel_y_world], axis=-1)

        if self._cached_drone_rpy is not None:
            drone_rpy = self._cached_drone_rpy
        else:
            drone_rpy = np.asarray(_jit_quat_to_rpy(states.quat[:, 0]))

        if self._cached_drone_rpy_rates is not None:
            drone_body_rates = self._cached_drone_rpy_rates
        else:
            drone_body_rates = np.asarray(states.ang_vel[:, 0])

        rel_pos = drone_pos - np.concatenate([rover_xy, np.zeros((N, 1))], axis=-1)

        parts = [
            drone_pos,                              # 3
            drone_vel,                              # 3
            drone_rpy,                              # 3
            drone_body_rates,                       # 3  -> 12
            rover_xy,                               # 2
            rover_vel_xy,                           # 2
            rover_s[:, None],                       # 1
            rover_c[:, None],                       # 1
            vx_body[:, None],                       # 1
            vy_body[:, None],                       # 1
            wz[:, None],                            # 1  -> 21
            rel_pos,                                # 3  -> 24
        ]

        shared = np.concatenate(parts, axis=-1).astype(np.float32)
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

        if self._drone_state_type == "quat":
            # Quaternion state: [pos(3), quat_xyzw(4), vel(3), body_rates(3)] = 13D
            drone_quat = np.asarray(states.quat[:, 0])  # (N, 4) xyzw
            drone_ang_vel = np.asarray(states.ang_vel[:, 0])  # (N, 3) body rates
            drone_mpc_state = np.concatenate([drone_pos, drone_quat, drone_vel, drone_ang_vel], axis=-1)
        elif self._cached_drone_rpy is not None and self._cached_drone_rpy_rates is not None:
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

    def _build_render_model(self):
        """Build a combined render-only MuJoCo model: drone scene + X3 rover.

        Replicates Crazyflow's build_mjx_spec() and attaches the rover model
        with prefix "rover_". The resulting model/data are used solely for
        visualization; physics remain in self.sim.mj_model / JAX.
        """
        import mujoco
        from pathlib import Path

        rover_xml = (
            Path(__file__).parents[2]
            / "crazyflie_rover_landing/meshes/rosmaster_x3.xml"
        )

        # Build fresh spec matching Crazyflow's build_mjx_spec
        spec = mujoco.MjSpec.from_file(str(self.sim._xml_path))
        spec.option.timestep = 1 / self.sim.freq
        spec.copy_during_attach = True
        drone_spec = mujoco.MjSpec.from_file(str(self.sim.drone_path))
        frame = spec.worldbody.add_frame(name="world")
        for i in range(self.sim.n_drones):
            drone_body = drone_spec.body("drone")
            drone = frame.attach_body(drone_body, "", f":{i}")
            drone.add_freejoint()

        # Attach rover with "rover_" prefix (rover has a free joint)
        rover_spec = mujoco.MjSpec.from_file(str(rover_xml))
        rover_frame = spec.worldbody.add_frame(name="rover_world")
        rover_base = rover_spec.body("base")
        rover_frame.attach_body(rover_base, "rover_", "")

        self._render_model = spec.compile()
        self._render_data = mujoco.MjData(self._render_model)

        # Patch rover landing pad visual to match config radius
        pad_id = mujoco.mj_name2id(
            self._render_model, mujoco.mjtObj.mjOBJ_GEOM, "rover_pad_geom"
        )
        if pad_id >= 0:
            self._render_model.geom_size[pad_id, 0] = self.cfg.rover_platform_radius

        # Cache rover free-joint qpos address
        rover_joint_id = mujoco.mj_name2id(
            self._render_model, mujoco.mjtObj.mjOBJ_JOINT, "rover_base_joint"
        )
        self._rover_qpos_addr = int(self._render_model.jnt_qposadr[rover_joint_id])

    def render(self, world: int = 0) -> np.ndarray | None:
        """Render the environment using MuJoCo with the X3 rover mesh.

        The drone is rendered via a combined render model (drone + X3 rover).

        Args:
            world: Which parallel world to render (default 0).
        """
        if self.render_mode is None:
            return None

        import mujoco
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        # Initialize viewer if needed
        if self.sim.viewer is None:
            self._build_render_model()
            w = getattr(self, '_render_width', 1920)
            h = getattr(self, '_render_height', 1080)
            self._render_model.vis.global_.offwidth = w
            self._render_model.vis.global_.offheight = h
            cam_config = {
                "distance": 12.0,
                "azimuth": 90.0,
                "elevation": -45.0,
                "lookat": [0.0, 0.0, 0.3],
            }
            # Apply user camera overrides
            overrides = getattr(self, '_cam_overrides', {})
            cam_config.update(overrides)
            self.sim.viewer = MujocoRenderer(
                self._render_model,
                self._render_data,
                max_geom=self.sim.max_visual_geom,
                default_cam_config=cam_config,
                height=h,
                width=w,
                camera_id=-1,
            )
            # Disable floor reflection and shadows
            floor_id = mujoco.mj_name2id(self._render_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            if floor_id >= 0:
                mat_id = self._render_model.geom_matid[floor_id]
                if mat_id >= 0:
                    self._render_model.mat_reflectance[mat_id] = 0.0
            self._render_model.vis.quality.shadowsize = 0

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

        # Set drone pose in render data (Crazyflow xyzw → MuJoCo qpos wxyz)
        qpos = np.zeros(7)
        qpos[:3] = drone_pos
        qpos[3] = drone_quat[3]   # w
        qpos[4:7] = drone_quat[:3]  # xyz
        self._render_data.qpos[:7] = qpos

        # Set rover pose in render data from rover state [x, y, c, s, ...]
        rover = rover_for_render if rover_for_render is not None else np.asarray(self.rover_state[world])
        rx, ry, cth, sth = float(rover[0]), float(rover[1]), float(rover[2]), float(rover[3])
        half_th = np.arctan2(sth, cth) / 2.0
        # Freejoint z = 2*wheel_radius so wheel bottoms touch ground
        # (base body default pos already has +0.0325, but freejoint overrides it)
        rover_z = 0.065  # X3 wheel radius offset
        addr = self._rover_qpos_addr
        self._render_data.qpos[addr:addr + 7] = [
            rx, ry, rover_z, np.cos(half_th), 0.0, 0.0, np.sin(half_th)
        ]

        # Forward kinematics to update body positions for rendering
        mujoco.mj_forward(self._render_model, self._render_data)

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

        # Clear previous frame's markers so they don't accumulate
        viewer._markers.clear()

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

        # Render main view
        main_frame = self.sim.viewer.render(self.render_mode)

        # Overlay: drone body-frame camera pointing -Z (downward)
        if self.render_mode == "rgb_array" and main_frame is not None:
            main_frame = self._composite_drone_cam(main_frame, drone_pos, drone_quat)

        return main_frame

    def _composite_drone_cam(
        self, main_frame: np.ndarray, drone_pos: np.ndarray, drone_quat_xyzw: np.ndarray
    ) -> np.ndarray:
        """Render a small drone body-frame downward camera and composite onto main frame."""
        import mujoco

        viewer = self.sim.viewer.viewer
        model = self._render_model
        data = self._render_data

        # Overlay size: 1/4 of main frame
        oh = main_frame.shape[0] // 4
        ow = main_frame.shape[1] // 4

        # Save original camera state
        orig_type = viewer.cam.type
        orig_fixedcamid = viewer.cam.fixedcamid
        orig_distance = viewer.cam.distance
        orig_azimuth = viewer.cam.azimuth
        orig_elevation = viewer.cam.elevation
        orig_lookat = viewer.cam.lookat.copy()

        # Camera just below drone, looking straight down at the ground
        # Lookat = ground point directly below the drone
        lookat = np.array([drone_pos[0], drone_pos[1], 0.0])
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.fixedcamid = -1
        viewer.cam.lookat[:] = lookat
        viewer.cam.distance = drone_pos[2] + 0.5  # 0.5m above drone, looking down
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -90.0  # straight down

        # Render to small viewport
        old_viewport = mujoco.MjrRect(viewer.viewport.left, viewer.viewport.bottom,
                                       viewer.viewport.width, viewer.viewport.height)
        viewer.viewport.width = ow
        viewer.viewport.height = oh

        mujoco.mjv_updateScene(
            model, data, viewer.vopt, viewer.pert, viewer.cam,
            mujoco.mjtCatBit.mjCAT_ALL, viewer.scn,
        )
        # Re-add markers for the overlay render
        for marker_params in viewer._markers:
            viewer._add_marker_to_scene(marker_params)
        mujoco.mjr_render(viewer.viewport, viewer.scn, viewer.con)

        rgb_arr = np.zeros(3 * ow * oh, dtype=np.uint8)
        depth_arr = np.zeros(ow * oh, dtype=np.float32)
        mujoco.mjr_readPixels(rgb_arr, depth_arr, viewer.viewport, viewer.con)
        overlay_frame = rgb_arr.reshape((oh, ow, 3))[::-1, :]

        # Restore camera and viewport
        viewer.cam.type = orig_type
        viewer.cam.fixedcamid = orig_fixedcamid
        viewer.cam.distance = orig_distance
        viewer.cam.azimuth = orig_azimuth
        viewer.cam.elevation = orig_elevation
        viewer.cam.lookat[:] = orig_lookat
        viewer.viewport.left = old_viewport.left
        viewer.viewport.bottom = old_viewport.bottom
        viewer.viewport.width = old_viewport.width
        viewer.viewport.height = old_viewport.height

        # Composite: bottom-left with 1px border
        margin = 10
        border = 2
        result = main_frame.copy()
        y_start = main_frame.shape[0] - oh - margin
        x_start = margin
        # Draw dark border
        result[y_start - border:y_start + oh + border,
               x_start - border:x_start + ow + border] = 40
        # Paste overlay
        result[y_start:y_start + oh, x_start:x_start + ow] = overlay_frame

        return result

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
