"""Main entry point for cooperative drone-rover landing.

Launches mission manager, drone agent, and optionally rover agent
as separate processes (for GIL-free parallel execution).
"""

import argparse
import multiprocessing
import os
import signal
import sys
from pathlib import Path

import yaml


def run_mission_manager(config: dict):
    """Run mission manager in its own process."""
    import rclpy
    from cf_landing_drone.mission_manager import MissionManager

    rclpy.init()
    node = MissionManager(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run_drone_agent(config: dict, training_config: dict):
    """Run drone agent in its own process."""
    import rclpy
    from cf_landing_drone.drone_agent import DroneAgent

    rclpy.init()
    node = DroneAgent(config, training_config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run_rover_agent(config: dict, training_config: dict):
    """Run rover agent in its own process."""
    import rclpy
    from cf_landing_drone.rover_agent import RoverAgent

    rclpy.init()
    node = RoverAgent(config, training_config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Cooperative drone-rover landing")
    parser.add_argument("--drone-mode", choices=["sim", "hw"], default="sim",
                        help="Drone mode: sim (CrazySim) or hw (real Crazyflie)")
    parser.add_argument("--rover-mode", choices=["sim", "hw"], default="sim",
                        help="Rover mode: sim (CrazySim) or hw (real X3)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to landing_config.yaml")
    # Filter out ROS2 args (--ros-args and everything after)
    import sys
    filtered = []
    skip = False
    for a in sys.argv[1:]:
        if a == '--ros-args':
            break
        filtered.append(a)
    args = parser.parse_args(filtered)

    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        from ament_index_python.packages import get_package_share_directory
        config_path = Path(get_package_share_directory('cf_landing_drone')) / 'config' / 'landing_config.yaml'

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["drone_mode"] = args.drone_mode
    config["rover_mode"] = args.rover_mode

    # Load training config
    from cf_landing_drone.policy_loader import get_models_dir, load_env_config
    policy_type = config.get("policy_type", "mlp")
    training_config_path = config.get("training_config_path")
    if training_config_path:
        with open(training_config_path) as f:
            training_config = yaml.safe_load(f)
    else:
        models_dir = get_models_dir()
        config_yaml = models_dir / policy_type / "config.yaml"
        if config_yaml.is_file():
            with open(config_yaml) as f:
                training_config = yaml.safe_load(f)
        else:
            # Build training_config from environment_config.json
            env_config = load_env_config(models_dir, policy_type)
            training_config = {
                "environment": {
                    "drone_model": env_config.get("drone_model", "cf21B_500"),
                    "rover_vx_max": 1.0,
                    "rover_vy_max": 1.0,
                    "rover_wz_max": 5.0,
                    "rover_wheel_vel_max": 34.9,
                    "roll_pitch_max": 0.1,
                    "yaw_max": 0.001,
                    "map_size_x": env_config.get("map_size_x", 15.0),
                    "map_size_y": env_config.get("map_size_y", 15.0),
                    "drone_z_max": 3.0,
                    "rover_height": 0.213,
                    "landing_z_tol": env_config.get("landing_z_tol", 0.05),
                    "landing_vel_xy_tol": env_config.get("landing_vel_xy_tol", 0.1),
                    "landing_vel_z_tol": env_config.get("landing_vel_z_tol", 0.1),
                    "landing_attitude_tol": env_config.get("landing_attitude_tol", 0.05),
                    "landing_zone_radius": env_config.get("landing_zone_radius", 0.05),
                    "rover_platform_radius": env_config.get("rover_platform_radius", 0.127),
                },
                "policy": {
                    "drone": {
                        **env_config.get("drone_policy", {"hidden_sizes": [256, 256], "activation": "relu"}),
                        **{k: v for k, v in env_config.get("drone_mpc", {}).items()},
                        "cost_net_sizes": env_config.get("drone_mpc", {}).get("cost_net_sizes", [256, 256]),
                    },
                    "rover": {
                        **env_config.get("rover_policy", {"hidden_sizes": [256, 256], "activation": "relu"}),
                        **{k: v for k, v in env_config.get("rover_mpc", {}).items()},
                        "cost_net_sizes": env_config.get("rover_mpc", {}).get("cost_net_sizes", [256, 256]),
                    },
                    "cost_net_activation": "relu",
                },
            }

    # Merge relevant training config into landing config
    env_section = training_config.get("environment", {})
    for key in ["rover_height", "landing_z_tol", "landing_vel_xy_tol", "landing_vel_z_tol",
                "landing_attitude_tol", "landing_zone_radius", "map_size_x", "map_size_y",
                "drone_z_max"]:
        if key in env_section and key not in config:
            config[key] = env_section[key]

    multiprocessing.set_start_method('spawn', force=True)

    processes = []

    # Mission manager (always on PC)
    p_manager = multiprocessing.Process(
        target=run_mission_manager,
        args=(config,),
        name="mission_manager",
    )
    processes.append(p_manager)

    # Drone agent (always on PC)
    p_drone = multiprocessing.Process(
        target=run_drone_agent,
        args=(config, training_config),
        name="drone_agent",
    )
    processes.append(p_drone)

    # Rover agent (on PC only for sim mode — hw mode runs on X3 Docker)
    if args.rover_mode == "sim":
        p_rover = multiprocessing.Process(
            target=run_rover_agent,
            args=(config, training_config),
            name="rover_agent",
        )
        processes.append(p_rover)

    # Start all
    for p in processes:
        p.start()
        print(f"Started {p.name} (PID {p.pid})")

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\nShutting down...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait for all
    for p in processes:
        p.join()

    print("All processes finished.")


if __name__ == '__main__':
    main()
