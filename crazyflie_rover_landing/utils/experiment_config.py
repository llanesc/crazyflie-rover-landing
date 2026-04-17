"""Experiment configuration loading utilities for the drone-rover landing task.

Adapted from crazyflie-mape-crazyflow utils/experiment_config.py.
"""

from pathlib import Path

import yaml

from crazyflie_rover_landing.envs.landing_config import LandingEnvConfig
from crazyflie_rover_landing.envs.spawn import SpawnFn, create_spawn_fn_from_config


def load_experiment_config(experiment_path: Path) -> dict:
    """Load experiment configuration from YAML file.

    Args:
        experiment_path: Path to the experiment directory containing config.yaml.

    Returns:
        Dictionary containing the experiment configuration.

    Raises:
        FileNotFoundError: If config.yaml is not found.
    """
    config_path = experiment_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def config_to_env_config(config: dict, device: str | None = None) -> LandingEnvConfig:
    """Convert experiment config dictionary to LandingEnvConfig.

    Args:
        config: Experiment configuration dictionary.
        device: Override device if specified.

    Returns:
        LandingEnvConfig instance.
    """
    env_cfg = config.get("environment", {})
    training_cfg = config.get("training", {})
    rewards_cfg = config.get("rewards", {})

    kwargs = {}

    # Rover type
    if "rover_type" in env_cfg:
        kwargs["rover_type"] = env_cfg["rover_type"]

    # Environment settings
    if "drone_model" in env_cfg:
        kwargs["drone_model"] = env_cfg["drone_model"]
    if "dynamics" in env_cfg:
        kwargs["dynamics"] = env_cfg["dynamics"]

    # Drone control limits and state type (from policy.drone section)
    drone_cfg = config.get("policy", {}).get("drone", {})
    if "state_type" in drone_cfg:
        kwargs["drone_state_type"] = drone_cfg["state_type"]
    if "roll_pitch_max" in drone_cfg:
        kwargs["roll_pitch_max"] = drone_cfg["roll_pitch_max"]
    if "yaw_max" in drone_cfg:
        kwargs["yaw_max"] = drone_cfg["yaw_max"]

    # Drone initial yaw randomization
    if "drone_init_yaw_max" in env_cfg:
        kwargs["drone_init_yaw_max"] = env_cfg["drone_init_yaw_max"]

    # Simulation frequencies
    for key in ("sim_freq", "mellinger_freq", "control_freq", "episode_length_s"):
        if key in env_cfg:
            kwargs[key] = env_cfg[key]

    # Map
    for key in ("map_size_x", "map_size_y", "drone_z_min", "drone_z_max"):
        if key in env_cfg:
            kwargs[key] = env_cfg[key]

    # Rover params (common + type-specific)
    for key in ("rover_platform_radius", "landing_zone_radius", "rover_height",
                "rover_vx_max", "rover_vy_max", "rover_wz_max"):
        if key in env_cfg:
            kwargs[key] = env_cfg[key]

    # Landing thresholds
    for key in ("landing_z_tol",
                "landing_vel_xy_tol", "landing_vel_z_tol", "landing_attitude_tol"):
        if key in env_cfg:
            kwargs[key] = env_cfg[key]

    # Physical parameters
    if "mass" in env_cfg and env_cfg["mass"] is not None:
        kwargs["mass"] = env_cfg["mass"]

    # Ground effect
    for key in ("enable_ground_effect", "ground_effect_rotor_radius", "ground_effect_scale"):
        if key in env_cfg:
            kwargs[key] = env_cfg[key]

    # Domain randomization
    for key in ("randomize_mass", "randomize_inertia",
                "mass_randomization_std", "inertia_randomization_std",
                "enable_disturbance", "disturbance_force_std", "disturbance_torque_std",
                "disturbance_type", "disturbance_ou_theta"):
        if key in env_cfg:
            kwargs[key] = env_cfg[key]

    # Reward settings
    reward_mapping = {
        "landing": "reward_landing",
        "crash": "reward_crash",
        "boundary": "reward_boundary",
        "progress_coef": "reward_progress_coef",
        "angle_coef": "reward_angle_coef",
        "action_smoothness_thrust": "reward_action_smoothness_thrust",
        "action_smoothness_rpy": "reward_action_smoothness_rpy",
        "action_smoothness_wheel": "reward_action_smoothness_wheel",
        "action_smoothness_vx": "reward_action_smoothness_vx",
        "action_smoothness_vy": "reward_action_smoothness_vy",
        "action_smoothness_wz": "reward_action_smoothness_wz",
        "landing_velocity_coef": "reward_landing_velocity_coef",
        "landing_precision_coef": "reward_landing_precision_coef",
        "descent_speed_coef": "reward_descent_speed_coef",
        "altitude_hold_coef": "reward_altitude_hold_coef",
        "time_penalty": "reward_time_penalty",
        "rover_stillness_coef": "reward_rover_stillness_coef",
        "rover_yawrate_coef": "reward_rover_yawrate_coef",
        "rover_lateral_coef": "reward_rover_lateral_coef",
        "drone_velocity_coef": "reward_drone_velocity_coef",
        "drone_xy_corridor_coef": "reward_drone_xy_corridor_coef",
        "rover_boundary_coef": "reward_rover_boundary_coef",
    }
    for yaml_key, cfg_key in reward_mapping.items():
        if yaml_key in rewards_cfg:
            kwargs[cfg_key] = rewards_cfg[yaml_key]

    # Landing corridor settings
    for key in ("corridor_radius", "corridor_transition", "max_descent_speed",
                "max_drone_speed", "cruise_altitude"):
        if key in env_cfg:
            kwargs[key] = env_cfg[key]

    # Training settings
    if "n_worlds" in training_cfg:
        kwargs["n_worlds"] = training_cfg["n_worlds"]

    # Device: command line override > config > default
    if device is not None:
        kwargs["device"] = device
    elif "device" in training_cfg:
        kwargs["device"] = training_cfg["device"]

    return LandingEnvConfig(**kwargs)


def get_spawn_fn_from_config(config: dict, rover_nx: int | None = None) -> SpawnFn:
    """Create spawn function from experiment configuration.

    Uses curriculum level 0 spawn if curriculum is defined; otherwise falls
    back to the environment.spawn section.

    Args:
        config: Experiment configuration dictionary.

    Returns:
        Spawn function with signature (key, N) -> (drone_pos, rover_state).
    """
    # Infer rover state dimension from rover_type if not provided
    if rover_nx is None:
        rover_type = config.get("environment", {}).get("rover_type", "burger")
        rover_nx = 7 if rover_type == "x3" else 6

    # Check if curriculum defines an initial spawn
    curriculum_cfg = config.get("curriculum", {})
    if curriculum_cfg.get("enabled", False):
        levels = curriculum_cfg.get("levels", [])
        if levels:
            level0 = levels[0]
            spawn_cfg = {}
            if "drone_spawn" in level0:
                spawn_cfg["drone"] = level0["drone_spawn"]
            if "rover_spawn" in level0:
                spawn_cfg["rover"] = level0["rover_spawn"]
            if spawn_cfg:
                return create_spawn_fn_from_config(spawn_cfg, rover_nx=rover_nx)

    spawn_cfg = config.get("environment", {}).get("spawn", {})
    return create_spawn_fn_from_config(spawn_cfg, rover_nx=rover_nx)


def get_training_config(config: dict) -> dict:
    """Extract training configuration from experiment config.

    Args:
        config: Experiment configuration dictionary.

    Returns:
        Dictionary with training parameters.
    """
    training_cfg = config.get("training", {})

    return {
        "timesteps": training_cfg.get("timesteps", 2_000_000),
        "n_worlds": training_cfg.get("n_worlds", 256),
        "device": training_cfg.get("device", "cpu"),
        "rollouts": training_cfg.get("rollouts", 512),
        "learning_epochs": training_cfg.get("learning_epochs", 8),
        "mini_batches": training_cfg.get("mini_batches", 4),
        "learning_rate": training_cfg.get("learning_rate", 3e-4),
        "gamma": training_cfg.get("gamma", 0.99),
        "gae_lambda": training_cfg.get("gae_lambda", 0.95),
        "grad_norm_clip": training_cfg.get("grad_norm_clip", 0.5),
        "entropy_loss_scale": training_cfg.get("entropy_loss_scale", 0.01),
        "value_loss_scale": training_cfg.get("value_loss_scale", 2.0),
        "ratio_clip": training_cfg.get("ratio_clip", 0.2),
        "value_clip": training_cfg.get("value_clip", 0.2),
        "kl_threshold": training_cfg.get("kl_threshold", 0.0),
        "observation_preprocessor": training_cfg.get("observation_preprocessor", None),
        "state_preprocessor": training_cfg.get("state_preprocessor", None),
        "value_preprocessor": training_cfg.get("value_preprocessor", None),
        "learning_rate_scheduler": training_cfg.get("learning_rate_scheduler", None),
        "learning_rate_scheduler_kwargs": training_cfg.get("learning_rate_scheduler_kwargs", {}),
    }


def get_policy_config(config: dict) -> dict:
    """Extract per-agent policy configuration from experiment config.

    Args:
        config: Experiment configuration dictionary.

    Returns:
        Dictionary with per-agent policy parameters.
        Keys: "drone" and "rover", each containing:
            mpc_horizon, mpc_dt, cost_net_sizes, initial_log_std, pos_offset_max, ...
        Also includes "value_net_sizes" at the top level.
    """
    policy_cfg = config.get("policy", {})
    drone_cfg = policy_cfg.get("drone", {})
    rover_cfg = policy_cfg.get("rover", {})

    # Shared fallbacks (per-agent overrides take priority)
    shared_log_std = policy_cfg.get("initial_log_std", -1.2)
    shared_cost_activation = policy_cfg.get("cost_net_activation", "relu")

    return {
        "drone": {
            "mpc_horizon": drone_cfg.get("mpc_horizon", 2),
            "mpc_dt": drone_cfg.get("mpc_dt", 0.01),
            "cost_net_sizes": drone_cfg.get("cost_net_sizes", [256, 256]),
            "state_type": drone_cfg.get("state_type", "euler"),
            "integrator": drone_cfg.get("integrator", "rk4"),
            "roll_pitch_max": drone_cfg.get("roll_pitch_max", 0.5),
            "yaw_max": drone_cfg.get("yaw_max", 0.5),
            "pos_offset_max": drone_cfg.get("pos_offset_max", 2.0),
            "initial_log_std": drone_cfg.get("initial_log_std", shared_log_std),
            "activation": drone_cfg.get("activation", shared_cost_activation),
        },
        "rover": {
            "mpc_horizon": rover_cfg.get("mpc_horizon", 4),
            "mpc_dt": rover_cfg.get("mpc_dt", 0.1),
            "cost_net_sizes": rover_cfg.get("cost_net_sizes", [256, 256]),
            "pos_offset_max": rover_cfg.get("pos_offset_max", 2.0),
            "initial_log_std": rover_cfg.get("initial_log_std", shared_log_std),
            "activation": rover_cfg.get("activation", shared_cost_activation),
        },
        "value_net_sizes": policy_cfg.get("value_net_sizes", [256, 256]),
        "value_activation": policy_cfg.get("value_activation", "relu"),
    }


def find_experiment_path(experiment_name: str, policy_type: str = "acmpc") -> Path:
    """Find the experiment directory path.

    Args:
        experiment_name: Name of the experiment (e.g., "default").
        policy_type: Policy type subdirectory (e.g., "acmpc", "mlp").

    Returns:
        Path to the experiment directory.

    Raises:
        FileNotFoundError: If experiment directory is not found.
    """
    project_root = Path(__file__).parent.parent.parent
    experiment_path = project_root / "results" / policy_type / experiment_name

    if not experiment_path.exists():
        policy_dir = project_root / "results" / policy_type
        available = (
            "\n".join(f"  - {d.name}" for d in policy_dir.iterdir() if d.is_dir())
            if policy_dir.exists() else "  (none)"
        )
        raise FileNotFoundError(
            f"Experiment not found: {experiment_path}\n"
            f"Available experiments in results/{policy_type}/:\n{available}"
        )

    return experiment_path
