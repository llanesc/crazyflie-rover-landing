"""Configuration dataclass for the drone-rover cooperative landing environment."""

from dataclasses import dataclass, field

import numpy as np
from drone_models.core import load_params


@dataclass
class LandingEnvConfig:
    """Configuration for the Crazyflie-rover cooperative landing environment.

    Two heterogeneous cooperative agents: a Crazyflie drone and a TurtleBot3 Burger
    differential-drive rover. Both agents collaborate to land the drone on the rover with
    minimal touchdown velocity.

    Attributes:
        n_worlds: Number of parallel environment instances.
        sim_freq: Physics simulation frequency [Hz] (Crazyflow internal).
        mellinger_freq: Mellinger attitude control frequency [Hz].
        control_freq: MPC policy frequency [Hz] (env step frequency).
        episode_length_s: Maximum episode duration [s].

        drone_model: Drone model identifier for Crazyflow and drone-models.
        dynamics: Crazyflow physics type ("first_principles").
        mass: Drone mass [kg]. None to load from drone_model.
        randomize_mass: Randomize mass each reset.
        randomize_inertia: Randomize inertia each reset.
        mass_randomization_std: Std of mass noise [kg].
        inertia_randomization_std: Std of inertia noise [kg*m^2].
        enable_disturbance: Enable random disturbance forces and torques.
        disturbance_force_std: Std of force disturbance [N].
        disturbance_torque_std: Std of torque disturbance [Nm].

        rover_wheel_vel_max: Max wheel angular velocity command [rad/s] (matches MuJoCo ctrlrange).
        rover_platform_radius: Landing pad radius on rover [m].
        rover_height: Height of the landing pad surface above ground [m].

        map_size_x: Total arena width in X [m] (drone stays within ±map_size_x/2).
        map_size_y: Total arena width in Y [m].
        drone_z_min: Minimum drone altitude [m].
        drone_z_max: Maximum drone altitude [m].

        landing_z_tol: Max vertical offset above rover pad for landing [m].
        landing_vel_xy_tol: Max relative XY speed at touchdown [m/s].
        landing_vel_z_tol: Max relative Z (descent) speed at touchdown [m/s].
        landing_attitude_tol: Max roll/pitch magnitude at touchdown [rad].

        roll_pitch_max: Max roll/pitch command [rad].
        yaw_max: Max yaw command [rad].

        reward_landing: Bonus reward for successful landing.
        reward_crash: Penalty for drone crash / hard collision.
        reward_boundary: Penalty for drone leaving the arena.
        reward_proximity_coef: Scale of proximity shaping reward.
        reward_proximity_decay: Exponential decay constant for proximity reward.
        reward_angle_coef: Penalty coefficient for drone roll/pitch deviation.
        reward_action_smoothness_thrust: Smoothness penalty for drone thrust changes.
        reward_action_smoothness_rpy: Smoothness penalty for drone roll/pitch/yaw changes.
        reward_action_smoothness_accel: Smoothness penalty for rover acceleration changes.
        reward_action_smoothness_yawrate: Smoothness penalty for rover yaw rate changes.
        reward_landing_velocity_coef: Extra penalty per (m/s) at touchdown.
        reward_rover_stillness_coef: Penalty for rover speed when drone is in landing corridor.
        reward_drone_velocity_coef: Penalty for drone speed (discourages fast flight).

        device: JAX/Torch device string ("cpu" or "cuda").
    """

    # Simulation
    n_worlds: int = 256
    sim_freq: int = 500
    mellinger_freq: int = 500
    control_freq: int = 100
    episode_length_s: float = 15.0

    # Drone model
    drone_model: str = "cf2x_T350"
    dynamics: str = "first_principles"
    mass: float | None = None
    randomize_mass: bool = False
    randomize_inertia: bool = False
    mass_randomization_std: float = 2e-3
    inertia_randomization_std: float = 3e-6

    # Disturbance forces/torques
    enable_disturbance: bool = False
    disturbance_force_std: float = 0.01
    disturbance_torque_std: float = 1e-4

    # Rover physical limits
    rover_wheel_vel_max: float = 6.67   # rad/s  (MuJoCo ctrlrange)
    rover_platform_radius: float = 0.10  # Landing pad radius [m]
    rover_height: float = 0.152          # Height of landing pad surface above ground [m]

    # Arena / boundary
    map_size_x: float = 5.0   # Total size; drone stays within ±2.5 m
    map_size_y: float = 5.0
    drone_z_min: float = 0.0
    drone_z_max: float = 3.0

    # Landing success thresholds
    landing_z_tol: float = 0.05            # m above rover pad — must be very close
    landing_vel_xy_tol: float = 0.1        # m/s — max relative XY speed at touchdown
    landing_vel_z_tol: float = 0.5         # m/s — max descent speed at touchdown (upward rejected)
    landing_attitude_tol: float = 0.05     # rad (~3°) — max roll/pitch magnitude at touchdown

    # Attitude limits (used for action clipping and OCP constraints)
    roll_pitch_max: float = 0.5
    yaw_max: float = 0.5

    # ---- Reward weights ----
    reward_landing: float = 100.0
    reward_crash: float = -20.0
    reward_boundary: float = -5.0
    reward_proximity_coef: float = 1.0
    reward_proximity_decay: float = 2.0
    reward_angle_coef: float = 0.05
    reward_action_smoothness_thrust: float = 5.0
    reward_action_smoothness_rpy: float = 1.0
    reward_action_smoothness_accel: float = 1.0
    reward_action_smoothness_yawrate: float = 0.5
    reward_landing_velocity_coef: float = 2.0
    reward_descent_speed_coef: float = 5.0    # Penalty for exceeding max descent speed in corridor
    reward_altitude_floor_coef: float = 0.5   # Penalty for being below altitude floor during navigation
    reward_time_penalty: float = 0.0          # Per-step cost to discourage hovering
    reward_rover_stillness_coef: float = 0.5 # Penalty for rover speed when drone is in landing corridor
    reward_rover_yawrate_coef: float = 0.5  # Penalty for rover yaw rate when drone is in landing corridor
    reward_drone_velocity_coef: float = 0.1  # Penalty for drone speed
    reward_rover_boundary_coef: float = 1.0  # Per-step penalty for rover at arena boundary

    # Landing corridor
    corridor_radius: float = 0.5       # Horizontal distance (m) to activate landing phase
    corridor_transition: float = 0.1   # Sigmoid transition width (m)
    max_descent_speed: float = 0.5     # Max allowed descent speed in corridor (m/s)

    # Device
    device: str = "cpu"

    # ---- Derived fields (set in __post_init__) ----
    gravity: float = field(init=False)
    thrust_min: float = field(init=False)
    thrust_max: float = field(init=False)

    def __post_init__(self) -> None:
        """Load physical parameters from drone-models and compute derived fields."""
        # Load from first_principles parameters (needed for hover RPM)
        fp_params = load_params("first_principles", self.drone_model)
        if self.mass is None:
            self.mass = float(fp_params["mass"])
        self.gravity = float(np.abs(fp_params["gravity_vec"][2]))

        # Thrust limits from so_rpy parameters
        so_rpy_params = load_params("so_rpy", self.drone_model)
        cmd_f_coef = float(so_rpy_params["cmd_f_coef"])
        # so_rpy thrust_min/max are per-motor values → multiply by 4
        self.thrust_min = float(so_rpy_params["thrust_min"]) * 4
        self.thrust_max = float(so_rpy_params["thrust_max"]) * 4

        # Store for convenience
        self._fp_params = fp_params

    @property
    def sim_steps_per_control(self) -> int:
        """Number of simulator sub-steps per environment control step."""
        return self.sim_freq // self.control_freq

    @property
    def max_episode_steps(self) -> int:
        """Maximum number of environment steps per episode."""
        return int(self.episode_length_s * self.control_freq)

    @property
    def dt(self) -> float:
        """Environment timestep [s] = 1 / control_freq."""
        return 1.0 / self.control_freq

    @property
    def map_half_x(self) -> float:
        """Half-width of the arena in X [m]."""
        return self.map_size_x / 2.0

    @property
    def map_half_y(self) -> float:
        """Half-width of the arena in Y [m]."""
        return self.map_size_y / 2.0

    @property
    def rover_max_speed(self) -> float:
        """Max body linear speed [m/s] = r × rover_wheel_vel_max."""
        from crazyflie_rover_landing.envs.rover_dynamics import WHEEL_RADIUS
        return WHEEL_RADIUS * self.rover_wheel_vel_max
