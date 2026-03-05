#!/usr/bin/env python3
"""Evaluation script for the drone-rover landing task.

Loads a trained MAPPO ACMPC checkpoint and evaluates landing performance.

Usage:
    python scripts/eval_mappo_acmpc.py --experiment default --run run_20260101120000
    python scripts/eval_mappo_acmpc.py --experiment default --checkpoint path/to/checkpoint.pt
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

import gymnasium
import numpy as np
import torch

from crazyflie_rover_landing.agents import MAPPO_MPC
from crazyflie_rover_landing.envs import LandingEnv, LandingEnvConfig, RescaleActionWrapper
from crazyflie_rover_landing.envs.spawn import create_spawn_fn_from_config
from crazyflie_rover_landing.policies import (
    DroneACMPCGaussianPolicy,
    RoverACMPCGaussianPolicy,
)
from crazyflie_rover_landing.policies.shared_critic import (
    DualHeadCriticBackbone,
    CriticHead,
)
from crazyflie_rover_landing.preprocessors import PartialRunningStandardScaler
from crazyflie_rover_landing.utils import (
    load_experiment_config,
    config_to_env_config,
    get_spawn_fn_from_config,
    get_training_config,
    get_policy_config,
    find_experiment_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MAPPO ACMPC checkpoint on drone-rover landing task",
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
    return parser.parse_args()


def find_checkpoint(experiment_path: Path, run_name: str | None) -> Path:
    """Find checkpoint file given experiment and optional run name."""
    if run_name is not None:
        run_dir = experiment_path / "results" / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # Prefer final_checkpoint
        final = run_dir / "final_checkpoint.pt"
        if final.exists():
            return final

        # Try best_agent
        best_agents = sorted(run_dir.glob("best_agent_*.pt"))
        if best_agents:
            return best_agents[-1]

        # Try checkpoints subdir
        checkpoints_dir = run_dir / "checkpoints"
        if checkpoints_dir.exists():
            checkpoints = []
            for f in checkpoints_dir.glob("agent_*.pt"):
                try:
                    step = int(f.stem.split("_")[1])
                    checkpoints.append((f, step))
                except (IndexError, ValueError):
                    continue
            if checkpoints:
                checkpoints.sort(key=lambda x: x[1], reverse=True)
                return checkpoints[0][0]

    raise FileNotFoundError(
        f"No checkpoint found. "
        f"Specify --run <run_name> or --checkpoint <path>"
    )


def main():
    args = parse_args()

    experiment_path = find_experiment_path(args.experiment)
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
    policy_cfg = get_policy_config(config)

    device = torch.device("cpu")

    env_cfg = config_to_env_config(config, device="cpu")
    spawn_fn = get_spawn_fn_from_config(config)

    # Override spawn with specific curriculum level if requested
    if args.level is not None:
        curriculum_cfg = config.get("curriculum", {})
        levels = curriculum_cfg.get("levels", [])
        if args.level < 0 or args.level >= len(levels):
            raise ValueError(
                f"--level {args.level} out of range. "
                f"Available levels: 0-{len(levels) - 1}"
            )
        level = levels[args.level]
        spawn_cfg = {}
        if "drone_spawn" in level:
            spawn_cfg["drone"] = level["drone_spawn"]
        if "rover_spawn" in level:
            spawn_cfg["rover"] = level["rover_spawn"]
        if spawn_cfg:
            # Force rover to spawn stationary for eval
            spawn_cfg.setdefault("rover", {})["stationary"] = True
            spawn_fn = create_spawn_fn_from_config(spawn_cfg)
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
    from skrl.multi_agents.torch.mappo import MAPPO_CFG

    env_skrl = wrap_env(env, wrapper="pettingzoo")
    possible_agents = env_skrl.possible_agents

    # Memories (rollouts=1 for eval — we only care about act, not training)
    memories = {
        a: RandomMemory(memory_size=1, num_envs=n_worlds, device=device)
        for a in possible_agents
    }

    # Compute batch max for eval (just n_worlds)
    n_batch_max = n_worlds

    d_cfg = policy_cfg["drone"]
    r_cfg = policy_cfg["rover"]

    drone_policy = DroneACMPCGaussianPolicy(
        observation_space=env_skrl.observation_space("drone"),
        action_space=env_skrl.action_space("drone"),
        device=device,
        mpc_horizon=d_cfg["mpc_horizon"],
        mpc_dt=d_cfg["mpc_dt"],
        cost_net_sizes=d_cfg["cost_net_sizes"],
        roll_pitch_max=env_cfg.roll_pitch_max,
        yaw_max=env_cfg.yaw_max,
        thrust_min=env_cfg.thrust_min,
        thrust_max=env_cfg.thrust_max,
        mass=env_cfg.mass,
        gravity=env_cfg.gravity,
        drone_model=env_cfg.drone_model,
        n_batch_max=n_batch_max,
        initial_log_std=d_cfg["initial_log_std"],
        activation=d_cfg["activation"],
        pos_offset_max=d_cfg["pos_offset_max"],
    )

    rover_policy = RoverACMPCGaussianPolicy(
        observation_space=env_skrl.observation_space("rover"),
        action_space=env_skrl.action_space("rover"),
        device=device,
        mpc_horizon=r_cfg["mpc_horizon"],
        mpc_dt=r_cfg["mpc_dt"],
        cost_net_sizes=r_cfg["cost_net_sizes"],
        max_speed=env_cfg.rover_max_speed,
        min_speed=env_cfg.rover_min_speed,
        max_omega=env_cfg.rover_max_omega,
        max_accel=env_cfg.rover_max_accel,
        n_batch_max=n_batch_max,
        initial_log_std=r_cfg["initial_log_std"],
        activation=r_cfg["activation"],
        pos_offset_max=r_cfg["pos_offset_max"],
    )

    critic_backbone = DualHeadCriticBackbone(
        obs_dim=gymnasium.spaces.flatdim(raw_env.shared_observation_space),
        hidden_sizes=policy_cfg.get("value_net_sizes", [256, 256]),
        activation=policy_cfg.get("value_activation", "relu"),
    ).to(device)
    drone_critic = CriticHead(
        observation_space=raw_env.shared_observation_space,
        action_space=env_skrl.action_space("drone"),
        backbone=critic_backbone,
        head_index=0,
        device=device,
    )
    rover_critic = CriticHead(
        observation_space=raw_env.shared_observation_space,
        action_space=env_skrl.action_space("rover"),
        backbone=critic_backbone,
        head_index=1,
        device=device,
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

    agent = MAPPO_MPC(
        mpc_state_sizes={"drone": 12, "rover": 5},
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
    # Evaluation loop
    # -----------------------------------------------------------------------
    n_episodes = args.n_episodes
    landings = 0
    crashes = 0
    oob = 0
    timeouts = 0
    landing_velocities = []
    episode_lengths = []

    episodes_done = 0

    obs, infos = env_skrl.reset()

    # Seed mpc_state cache
    if "mpc_state" in infos:
        for uid in possible_agents:
            if uid in infos["mpc_state"]:
                agent._current_mpc_state[uid] = torch.as_tensor(
                    infos["mpc_state"][uid], dtype=torch.float32, device=device
                )

    step = 0
    screenshot_frame = None
    max_steps_per_episode = raw_env.cfg.max_episode_steps

    print(f"Evaluating {n_episodes} episodes across {n_worlds} worlds...")

    while episodes_done < n_episodes:
        # Get actions (deterministic: use mean of distribution)
        states = {a: infos.get("state", {}).get(a) for a in possible_agents}
        with torch.no_grad():
            actions, _ = agent.act(obs, states, timestep=10**9, timesteps=10**9)

        obs, rewards, terminated, truncated, infos = env_skrl.step(actions)

        # Update mpc_state cache
        if "mpc_state" in infos:
            for uid in possible_agents:
                if uid in infos["mpc_state"] and agent._mpc_state_sizes.get(uid, 0) > 0:
                    agent._current_mpc_state[uid] = torch.as_tensor(
                        infos["mpc_state"][uid], dtype=torch.float32, device=device
                    )

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


if __name__ == "__main__":
    main()
