#!/usr/bin/env python3
"""Evaluation script for the drone-rover landing task with MLP policy.

Loads a trained MAPPO MLP checkpoint and evaluates landing performance.

Usage:
    python scripts/eval_mappo_mlp.py --experiment default --run run_20260101120000
    python scripts/eval_mappo_mlp.py --experiment default --checkpoint path/to/checkpoint.pt
"""

import argparse
import json
import os
import time
from pathlib import Path

import yaml


# Force CPU — Crazyflow (JAX) does not support GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import numpy as np
import torch

from crazyflie_rover_landing.envs import LandingEnv, LandingEnvConfig, RescaleActionWrapper
from crazyflie_rover_landing.envs.spawn import create_spawn_fn_from_config
from crazyflie_rover_landing.policies import MLPGaussianPolicy, SharedCritic
from crazyflie_rover_landing.preprocessors import PartialRunningStandardScaler
from crazyflie_rover_landing.utils import (
    load_experiment_config,
    config_to_env_config,
    get_spawn_fn_from_config,
    get_training_config,
    find_experiment_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MAPPO MLP checkpoint on drone-rover landing task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name (e.g., 'default')")
    parser.add_argument("--run", type=str, default=None,
                        help="Run name within experiment (e.g., 'run_20260101120000')")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Direct path to checkpoint .pt file")
    parser.add_argument("--n-episodes", type=int, default=100,
                        help="Number of episodes to evaluate (default: 100)")
    parser.add_argument("--n-worlds", type=int, default=None,
                        help="Override number of parallel worlds for evaluation")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during evaluation")
    parser.add_argument("--level", type=int, default=None,
                        help="Curriculum level to evaluate (uses that level's spawn config)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic actions (mean of distribution)")
    parser.add_argument("--trajectory", action="store_true",
                        help="Draw trajectory trails (red=drone, green=rover)")
    parser.add_argument("--video", nargs="?", const="auto", default=None,
                        help="Save video. Optionally specify path (default: saved next to checkpoint).")
    parser.add_argument("--screenshot-episode", type=int, default=None,
                        help="Save a high-res PNG of the last frame of the given episode number (0-indexed)")
    parser.add_argument("--screenshot-resolution", type=int, nargs=2, default=[3840, 2160],
                        metavar=("W", "H"),
                        help="Screenshot resolution in pixels (default: 3840 2160)")
    parser.add_argument("--cam-distance", type=float, default=None,
                        help="Camera distance from lookat point")
    parser.add_argument("--cam-azimuth", type=float, default=None,
                        help="Camera azimuth angle (degrees)")
    parser.add_argument("--cam-elevation", type=float, default=None,
                        help="Camera elevation angle (degrees)")
    parser.add_argument("--cam-lookat", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Camera lookat point (3 floats)")
    parser.add_argument("--drone-pos", type=str, default=None,
                        help="Fixed drone initial position x,y (e.g. '2.0,0.5')")
    parser.add_argument("--rover-pos", type=str, default=None,
                        help="Fixed rover initial position x,y (e.g. '0,0')")
    parser.add_argument("--log-csv", type=str, default=None,
                        help="Save per-step obs/action data to CSV file")
    return parser.parse_args()


def find_checkpoint(experiment_path: Path, run_name: str | None) -> Path:
    """Find checkpoint file given experiment and optional run name."""
    if run_name is not None:
        run_dir = experiment_path / "results" / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # Prefer latest agent_*.pt by step number
        import re
        checkpoints_dir = run_dir / "checkpoints"
        if checkpoints_dir.exists():
            agent_files = []
            for f in checkpoints_dir.glob("agent_*.pt"):
                m = re.search(r"agent_(\d+)\.pt$", f.name)
                if m:
                    agent_files.append((f, int(m.group(1))))
            if agent_files:
                agent_files.sort(key=lambda x: x[1], reverse=True)
                return agent_files[0][0]

        # Fallback to final_checkpoint
        final = run_dir / "final_checkpoint.pt"
        if final.exists():
            return final

    raise FileNotFoundError(
        f"No checkpoint found. "
        f"Specify --run <run_name> or --checkpoint <path>"
    )


def get_mlp_policy_config(config: dict) -> dict:
    """Extract MLP policy configuration from experiment config."""
    policy_cfg = config.get("policy", {})
    drone_cfg = policy_cfg.get("drone", {})
    rover_cfg = policy_cfg.get("rover", {})

    shared_log_std = policy_cfg.get("initial_log_std", 0.0)
    shared_activation = policy_cfg.get("activation", "relu")

    return {
        "drone": {
            "hidden_sizes": drone_cfg.get("hidden_sizes", [256, 256]),
            "activation": drone_cfg.get("activation", shared_activation),
            "initial_log_std": drone_cfg.get("initial_log_std", shared_log_std),
            "roll_pitch_max": drone_cfg.get("roll_pitch_max", 0.5),
            "yaw_max": drone_cfg.get("yaw_max", 0.5),
        },
        "rover": {
            "hidden_sizes": rover_cfg.get("hidden_sizes", [256, 256]),
            "activation": rover_cfg.get("activation", shared_activation),
            "initial_log_std": rover_cfg.get("initial_log_std", shared_log_std),
        },
        "value_net_sizes": policy_cfg.get("value_net_sizes", [256, 256]),
        "value_activation": policy_cfg.get("value_activation", "relu"),
    }


def main():
    args = parse_args()

    experiment_path = find_experiment_path(args.experiment, policy_type="mlp")
    config = load_experiment_config(experiment_path)

    # Find checkpoint
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path = find_checkpoint(experiment_path, args.run)

    print(f"Loading checkpoint: {checkpoint_path}")

    training_cfg = get_training_config(config)
    policy_cfg = get_mlp_policy_config(config)

    device = torch.device("cpu")

    env_cfg = config_to_env_config(config, device="cpu")
    env_cfg.roll_pitch_max = policy_cfg["drone"]["roll_pitch_max"]
    env_cfg.yaw_max = policy_cfg["drone"]["yaw_max"]
    spawn_fn = get_spawn_fn_from_config(config)

    # Override spawn with specific curriculum level if requested
    if args.level is not None:
        curriculum_cfg = config.get("curriculum", {})
        levels = curriculum_cfg.get("levels", [])
        if args.level < 1 or args.level > len(levels):
            raise ValueError(
                f"--level {args.level} out of range. "
                f"Available levels: 1-{len(levels)}"
            )
        level = levels[args.level - 1]
        spawn_cfg = {}
        if "drone_spawn" in level:
            spawn_cfg["drone"] = level["drone_spawn"]
        if "rover_spawn" in level:
            spawn_cfg["rover"] = level["rover_spawn"]
        if spawn_cfg:
            # Force rover to spawn stationary for eval
            spawn_cfg.setdefault("rover", {})["stationary"] = True
            spawn_fn = create_spawn_fn_from_config(spawn_cfg, rover_nx=env_cfg.rover_nx)
        # Apply env overrides from the level
        if level.get("randomize_mass") is not None:
            env_cfg.randomize_mass = level["randomize_mass"]
        if level.get("randomize_inertia") is not None:
            env_cfg.randomize_inertia = level["randomize_inertia"]
        if level.get("enable_disturbance") is not None:
            env_cfg.enable_disturbance = level["enable_disturbance"]
        print(f"Using curriculum level {args.level}: '{level.get('name', 'unnamed')}'")
    else:
        # No level specified — rebuild default spawn with stationary rover
        curriculum_cfg = config.get("curriculum", {})
        levels = curriculum_cfg.get("levels", [])
        if curriculum_cfg.get("enabled", False) and levels:
            level0 = levels[0]
            spawn_cfg = {}
            if "drone_spawn" in level0:
                spawn_cfg["drone"] = level0["drone_spawn"]
            if "rover_spawn" in level0:
                spawn_cfg["rover"] = level0["rover_spawn"]
        else:
            spawn_cfg = config.get("environment", {}).get("spawn", {})
        spawn_cfg.setdefault("rover", {})["stationary"] = True
        spawn_fn = create_spawn_fn_from_config(spawn_cfg)

    # Override spawn with fixed initial positions if specified
    if args.drone_pos is not None or args.rover_pos is not None:
        drone_xy = [float(x) for x in args.drone_pos.split(',')] if args.drone_pos else [0, 0]
        rover_xy = [float(x) for x in args.rover_pos.split(',')] if args.rover_pos else [0, 0]
        rover_nx = env_cfg.rover_nx
        env_cfg.drone_init_yaw_max = 0.0  # No random yaw for fixed spawn
        def fixed_spawn_fn(key, N):
            import jax.numpy as jnp
            # Drone position: (N, 3)
            drone_pos = jnp.zeros((N, 3))
            drone_pos = drone_pos.at[:, 0].set(drone_xy[0])
            drone_pos = drone_pos.at[:, 1].set(drone_xy[1])
            drone_pos = drone_pos.at[:, 2].set(1.0)
            # Rover state: (N, rover_nx) — [x, y, cos(theta), sin(theta), ...]
            rover_state = jnp.zeros((N, rover_nx))
            rover_state = rover_state.at[:, 0].set(rover_xy[0])
            rover_state = rover_state.at[:, 1].set(rover_xy[1])
            rover_state = rover_state.at[:, 2].set(1.0)  # cos(0)
            return drone_pos, rover_state
        spawn_fn = fixed_spawn_fn
        print(f"Fixed spawn: drone=({drone_xy[0]}, {drone_xy[1]}, 1.0) rover=({rover_xy[0]}, {rover_xy[1]})")

    # Override n_worlds if specified
    if args.n_worlds is not None:
        env_cfg.n_worlds = args.n_worlds

    n_worlds = env_cfg.n_worlds

    # Create environment
    recording_video = args.video is not None
    screenshot_episode = args.screenshot_episode
    screenshot_rendering = screenshot_episode is not None
    if recording_video or screenshot_rendering:
        render_mode = "rgb_array"
    elif args.render:
        render_mode = "human"
    else:
        render_mode = None
    env = LandingEnv(cfg=env_cfg, spawn_fn=spawn_fn, render_mode=render_mode)
    raw_env = env

    # Apply camera overrides
    raw_env.set_camera(
        distance=args.cam_distance,
        azimuth=args.cam_azimuth,
        elevation=args.cam_elevation,
        lookat=args.cam_lookat,
    )

    if args.trajectory:
        raw_env.enable_trajectory(enabled=True, subsample=5)

    # Set high resolution for screenshots
    screenshot_path = None
    screenshot_saved = False
    if screenshot_rendering:
        sw, sh = args.screenshot_resolution
        raw_env.set_render_resolution(sw, sh)
        level_str = f"_level_{args.level}" if args.level is not None else ""
        # Save to run directory (go up from checkpoints/ if needed)
        run_dir = checkpoint_path.parent
        if run_dir.name == "checkpoints":
            run_dir = run_dir.parent
        screenshot_path = run_dir / f"screenshot_ep{screenshot_episode}{level_str}.png"
        print(f"Screenshot: episode {screenshot_episode} -> {screenshot_path}")

    # Set up video writer
    video_writer = None
    if recording_video:
        import imageio
        if args.video == "auto":
            video_path = checkpoint_path.parent / "eval_video.mp4"
        else:
            video_path = Path(args.video)
        video_writer = imageio.get_writer(
            str(video_path), fps=env_cfg.control_freq, codec="libx264",
            quality=8,
        )
        print(f"Recording video to: {video_path} (FPS={env_cfg.control_freq})")

    print(f"Environment: {n_worlds} parallel worlds, {env_cfg.episode_length_s}s episodes")

    # Wrap
    env = RescaleActionWrapper(env)

    from skrl.envs.wrappers.torch import wrap_env
    from skrl.memories.torch import RandomMemory
    from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_CFG

    env_skrl = wrap_env(env, wrapper="pettingzoo")
    possible_agents = env_skrl.possible_agents

    # Memories (rollouts=1 for eval — we only care about act, not training)
    memories = {
        a: RandomMemory(memory_size=1, num_envs=n_worlds, device=device)
        for a in possible_agents
    }

    d_cfg = policy_cfg["drone"]
    r_cfg = policy_cfg["rover"]

    drone_policy = MLPGaussianPolicy(
        observation_space=env_skrl.observation_space("drone"),
        action_space=env_skrl.action_space("drone"),
        device=device,
        hidden_sizes=tuple(d_cfg["hidden_sizes"]),
        activation=d_cfg["activation"],
        initial_log_std=d_cfg["initial_log_std"],
    )

    rover_policy = MLPGaussianPolicy(
        observation_space=env_skrl.observation_space("rover"),
        action_space=env_skrl.action_space("rover"),
        device=device,
        hidden_sizes=tuple(r_cfg["hidden_sizes"]),
        activation=r_cfg["activation"],
        initial_log_std=r_cfg["initial_log_std"],
    )

    drone_critic = SharedCritic(
        observation_space=raw_env.shared_observation_space,
        action_space=env_skrl.action_space("drone"),
        device=device,
        value_net_sizes=policy_cfg.get("value_net_sizes", [256, 256]),
        activation=policy_cfg.get("value_activation", "relu"),
    )
    rover_critic = SharedCritic(
        observation_space=raw_env.shared_observation_space,
        action_space=env_skrl.action_space("rover"),
        device=device,
        value_net_sizes=policy_cfg.get("value_net_sizes", [256, 256]),
        activation=policy_cfg.get("value_activation", "relu"),
    )

    models = {
        "drone": {"policy": drone_policy, "value": drone_critic},
        "rover": {"policy": rover_policy, "value": rover_critic},
    }

    # Set up preprocessors matching training config
    from skrl.resources.preprocessors.torch import RunningStandardScaler

    obs_preprocessor_class = None
    obs_preprocessor_kwargs = {a: {} for a in possible_agents}
    if training_cfg.get("observation_preprocessor") == "RunningStandardScaler":
        obs_preprocessor_class = PartialRunningStandardScaler
        obs_preprocessor_kwargs = {
            "drone": {"size": raw_env.drone_obs_dim,
                      "skip_dims": raw_env.obs_binary_dims, "device": device},
            "rover": {"size": raw_env.rover_obs_dim,
                      "skip_dims": raw_env.obs_binary_dims, "device": device},
        }

    state_preprocessor_class = None
    state_preprocessor_kwargs = {a: {} for a in possible_agents}
    if training_cfg.get("state_preprocessor") == "RunningStandardScaler":
        state_preprocessor_class = PartialRunningStandardScaler
        base_kwargs = {"size": raw_env.shared_state_dim,
                       "skip_dims": raw_env.state_binary_dims, "device": device}
        state_preprocessor_kwargs = {a: base_kwargs.copy() for a in possible_agents}

    value_preprocessor_class = None
    value_preprocessor_kwargs = {a: {} for a in possible_agents}
    if training_cfg.get("value_preprocessor") == "RunningStandardScaler":
        value_preprocessor_class = RunningStandardScaler
        base_kwargs = {"size": 1, "device": device}
        value_preprocessor_kwargs = {a: base_kwargs.copy() for a in possible_agents}

    mappo_cfg = MAPPO_CFG(
        rollouts=1,
        learning_rate_scheduler_kwargs={a: {} for a in possible_agents},
        observation_preprocessor=obs_preprocessor_class,
        observation_preprocessor_kwargs=obs_preprocessor_kwargs,
        state_preprocessor=state_preprocessor_class,
        state_preprocessor_kwargs=state_preprocessor_kwargs,
        value_preprocessor=value_preprocessor_class,
        value_preprocessor_kwargs=value_preprocessor_kwargs,
        experiment={"directory": "", "experiment_name": "eval", "write_interval": 0},
    )

    agent = MAPPO(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        cfg=mappo_cfg,
        observation_spaces={a: env_skrl.observation_space(a) for a in possible_agents},
        action_spaces={a: env_skrl.action_space(a) for a in possible_agents},
        state_spaces={a: raw_env.shared_observation_space for a in possible_agents},
        device=device,
    )

    # Load checkpoint
    agent.load(str(checkpoint_path))
    if args.deterministic:
        agent.enable_training_mode(False, apply_to_models=True)
        print("Checkpoint loaded successfully (deterministic mode)")
    else:
        agent.enable_training_mode(True, apply_to_models=True)
        print("Checkpoint loaded successfully (stochastic mode)")

    # -----------------------------------------------------------------------
    # CSV logging setup
    # -----------------------------------------------------------------------
    csv_file = None
    csv_writer = None
    if args.log_csv:
        import csv
        csv_file = open(args.log_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        # Header: step, drone obs (29), drone action (4), rover obs, rover action
        drone_obs_cols = [f'drone_obs_{i}' for i in range(raw_env.drone_obs_dim)]
        drone_act_cols = ['drone_roll', 'drone_pitch', 'drone_yaw', 'drone_thrust']
        rover_obs_cols = [f'rover_obs_{i}' for i in range(raw_env.rover_obs_dim)]
        rover_act_cols = ['rover_vx', 'rover_vy', 'rover_wz']
        csv_writer.writerow(['step', 'time'] + drone_obs_cols + drone_act_cols + rover_obs_cols + rover_act_cols)
        print(f"Logging to CSV: {args.log_csv}")

    # -----------------------------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------------------------
    n_episodes = args.n_episodes
    landings = 0
    crashes = 0
    oob = 0
    timeouts = 0

    episodes_done = 0

    obs, infos = env_skrl.reset()

    step = 0
    screenshot_frame = None

    print(f"Evaluating {n_episodes} episodes across {n_worlds} worlds...")

    while episodes_done < n_episodes:
        # Log obs BEFORE action (what the policy sees)
        if csv_writer is not None and episodes_done == 0:
            drone_obs_pre = obs["drone"][0].cpu().numpy().tolist()
            rover_obs_pre = obs["rover"][0].cpu().numpy().tolist()

        # Get actions (deterministic: use mean of distribution)
        states = {a: infos.get("state", {}).get(a) for a in possible_agents}
        with torch.no_grad():
            actions, _ = agent.act(obs, states, timestep=10**9, timesteps=10**9)

        obs, rewards, terminated, truncated, infos = env_skrl.step(actions)

        # Log to CSV: obs before action + action applied
        if csv_writer is not None and episodes_done == 0:
            drone_act = raw_env.drone_cmd[0].tolist() if hasattr(raw_env, 'drone_cmd') else actions["drone"][0].cpu().numpy().tolist()
            rover_act = raw_env.last_rover_cmd[0].tolist() if hasattr(raw_env, 'last_rover_cmd') else actions["rover"][0].cpu().numpy().tolist()
            t = step / env_cfg.control_freq
            csv_writer.writerow([step, f'{t:.4f}'] + drone_obs_pre + drone_act + rover_obs_pre + rover_act)

        # Count events from raw env
        term_events = raw_env.last_termination_events
        n_landing = int(round(term_events.get("landing", 0.0) * n_worlds))
        n_crash = int(round(term_events.get("crash", 0.0) * n_worlds))
        n_oob = int(round(term_events.get("out_of_bounds", 0.0) * n_worlds))
        n_timeout = int(round(term_events.get("max_steps", 0.0) * n_worlds))

        n_done_this_step = n_landing + n_crash + n_oob + n_timeout
        landings += n_landing
        crashes += n_crash
        oob += n_oob
        timeouts += n_timeout
        episodes_done += n_done_this_step

        # Screenshot: render right after target episode ends (pre-reset state
        # is saved and will be consumed by this render call)
        if (screenshot_rendering and not screenshot_saved
                and n_done_this_step > 0
                and episodes_done > screenshot_episode):
            screenshot_frame = raw_env.render()
            if screenshot_frame is not None and screenshot_path:
                from PIL import Image
                img = Image.fromarray(screenshot_frame)
                img.save(screenshot_path)
                print(f"\nScreenshot saved: {screenshot_path} "
                      f"({screenshot_frame.shape[1]}x{screenshot_frame.shape[0]})")
                screenshot_saved = True

        # Regular rendering (for live view or video)
        if args.render:
            t_start = time.monotonic()
            raw_env.render()
            elapsed = time.monotonic() - t_start
            sleep_time = env_cfg.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        elif recording_video:
            frame = raw_env.render()
            if frame is not None:
                video_writer.append_data(frame)

        step += 1

    total = landings + crashes + oob + timeouts
    landing_rate = landings / total if total > 0 else 0.0
    crash_rate = crashes / total if total > 0 else 0.0
    oob_rate = oob / total if total > 0 else 0.0
    timeout_rate = timeouts / total if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Evaluation Results ({total} episodes):")
    print(f"{'='*60}")
    print(f"  Landing rate:   {landing_rate:.1%} ({landings}/{total})")
    print(f"  Crash rate:     {crash_rate:.1%} ({crashes}/{total})")
    print(f"  OOB rate:       {oob_rate:.1%} ({oob}/{total})")
    print(f"  Timeout rate:   {timeout_rate:.1%} ({timeouts}/{total})")
    print(f"{'='*60}")

    results = {
        "checkpoint": str(checkpoint_path),
        "n_episodes": total,
        "n_worlds": n_worlds,
        "landing_rate": landing_rate,
        "crash_rate": crash_rate,
        "oob_rate": oob_rate,
        "timeout_rate": timeout_rate,
        "landings": landings,
        "crashes": crashes,
        "oob": oob,
        "timeouts": timeouts,
    }

    results_path = checkpoint_path.parent / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    if video_writer is not None:
        video_writer.close()
        print(f"Video saved to: {video_path}")

    env_skrl.close()

    if csv_file is not None:
        csv_file.close()
        print(f"CSV data saved to: {args.log_csv}")


if __name__ == "__main__":
    main()
