"""Configuration dataclass for the drone-rover cooperative landing environment."""

from dataclasses import dataclass, field

import numpy as np
from drone_models.core import load_params


@dataclass
class LandingEnvConfig:
    """Configuration for the Crazyflie-rover cooperative landing environment.

    Supports multiple rover types via the ``rover_type`` field:
      - "burger": TurtleBot3 Burger differential-drive (NX=6, NU=2)
      - "x3": Yahboom RosMaster X3 mecanum omnidirectional (NX=7, NU=3)

    Two heterogeneous cooperative agents: a Crazyflie drone and a ground rover.
    Both agents collaborate to land the drone on the rover with minimal
    touchdown velocity.

    Attributes:
        rover_type: Which rover model to use ("burger" or "x3").
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

        rover_wheel_vel_max: Max wheel angular velocity command [rad/s].
        rover_vx_max: (x3 only) Max body-frame forward velocity [m/s].
        rover_vy_max: (x3 only) Max body-frame lateral velocity [m/s].
        rover_wz_max: (x3 only) Max yaw rate command [rad/s].
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

        device: JAX/Torch device string ("cpu" or "cuda").
    """

    # Rover type: "burger" (differential-drive) or "x3" (mecanum)
    rover_type: str = "burger"

    # Drone MPC state type: "euler" (12D) or "quat" (13D)
    drone_state_type: str = "euler"

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

    # Ground effect
    enable_ground_effect: bool = False
    ground_effect_rotor_radius: float = 0.0275  # Prop radius [m] (55mm diameter / 2)
    ground_effect_scale: float = 1.0            # Multiplier for GE force (>1 overestimates for robustness)

    # Disturbance forces/torques
    enable_disturbance: bool = False
    disturbance_type: str = "gaussian"  # "gaussian" (white noise) or "ou" (Ornstein-Uhlenbeck)
    disturbance_force_std: float = 0.01
    disturbance_torque_std: float = 1e-4
    disturbance_ou_theta: float = 2.0   # OU mean-reversion rate [1/s] (lower = smoother)

    # Rover physical limits — burger (differential-drive)
    rover_wheel_vel_max: float = 6.67   # rad/s  (MuJoCo ctrlrange for burger, motor limit for x3)
    rover_platform_radius: float = 0.10  # Landing pad radius [m]
    landing_zone_radius: float = 0.07    # Safe landing zone radius [m] (smaller than pad — edge landings crash)
    rover_height: float = 0.152          # Height of landing pad surface above ground [m]

    # Rover physical limits — x3 (mecanum) body velocity commands
    rover_vx_max: float = 1.0           # m/s  (ROS /cmd_vel forward limit)
    rover_vy_max: float = 1.0           # m/s  (ROS /cmd_vel lateral limit)
    rover_wz_max: float = 5.0           # rad/s (ROS /cmd_vel yaw rate limit)

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

    # Initial yaw randomization at spawn [rad] (uniform ±drone_init_yaw_max)
    drone_init_yaw_max: float = 0.0

    # ---- Reward weights ----
    reward_landing: float = 100.0
    reward_crash: float = -20.0
    reward_boundary: float = -5.0
    reward_progress_coef: float = 10.0       # Scale for progress reward (dist reduction per step)
    reward_angle_coef: float = 0.05
    reward_action_smoothness_thrust: float = 5.0
    reward_action_smoothness_rpy: float = 1.0
    reward_action_smoothness_wheel: float = 0.03
    # X3 per-axis rover smoothness (overrides action_smoothness_wheel when rover_type=="x3")
    reward_action_smoothness_vx: float = 0.03
    reward_action_smoothness_vy: float = 0.03
    reward_action_smoothness_wz: float = 0.001
    reward_landing_velocity_coef: float = 2.0
    reward_landing_precision_coef: float = 10.0  # Bonus for landing near pad center (scales linearly)
    reward_descent_speed_coef: float = 5.0    # Penalty for exceeding max descent speed in corridor
    reward_altitude_floor_coef: float = 0.5   # Penalty for being below altitude floor during navigation
    reward_time_penalty: float = 0.0          # Per-step cost to discourage hovering
    reward_rover_stillness_coef: float = 0.5 # Penalty for rover speed when drone is in landing corridor
    reward_rover_yawrate_coef: float = 0.5  # Penalty for rover yaw rate when drone is in landing corridor
    reward_rover_lateral_coef: float = 0.0  # Penalty for rover body-frame lateral speed (vy²)
    reward_drone_velocity_coef: float = 0.1  # Penalty for drone speed above max_drone_speed
    reward_drone_xy_corridor_coef: float = 0.0  # Penalty for drone XY speed inside landing corridor
    reward_rover_boundary_coef: float = 1.0  # Per-step penalty for rover at arena boundary

    # Landing corridor
    corridor_radius: float = 0.5       # Horizontal distance (m) to activate landing phase
    corridor_transition: float = 0.1   # Sigmoid transition width (m)
    max_descent_speed: float = 0.5     # Max allowed descent speed in corridor (m/s)
    max_drone_speed: float = 0.5       # Max drone speed (m/s) before penalty kicks in

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
        """Max body linear speed [m/s]."""
        if self.rover_type == "x3":
            return self.rover_vx_max
        from crazyflie_rover_landing.envs.rover_dynamics import WHEEL_RADIUS
        return WHEEL_RADIUS * self.rover_wheel_vel_max

    @property
    def rover_nx(self) -> int:
        """Rover state dimension."""
        return 7 if self.rover_type == "x3" else 6

    @property
    def rover_nu(self) -> int:
        """Rover control dimension."""
        return 3 if self.rover_type == "x3" else 2
