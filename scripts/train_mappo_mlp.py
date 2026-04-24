#!/usr/bin/env python3
"""Training script for drone-rover landing using MAPPO with MLP (FFN) policies.

Two cooperative agents (drone + rover), each with a simple feedforward neural
network policy, trained using centralized MAPPO with a shared critic.

Usage:
    python scripts/train_mappo_mlp.py --experiment default
    python scripts/train_mappo_mlp.py --experiment default --resume-run run_20260101120000
"""

import argparse
import os
from pathlib import Path

import yaml


# Force CPU — Crazyflow (JAX) does not support GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import wrap_env
from torch.optim.lr_scheduler import LinearLR, StepLR

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
    load_curriculum_config,
    CurriculumManager,
)
from crazyflie_rover_landing.utils.curriculum import CurriculumLevel


class TerminationLoggingWrapper:
    """Wrapper that logs termination events, reward components, and episode stats.

    Tracks:
      - landing: successful soft landing (drone on rover with low velocity)
      - crash: drone hit ground / hard contact
      - out_of_bounds: drone left the arena
      - max_steps: episode timeout

    Also handles curriculum by tracking landing rate and calling update_curriculum_params
    on the raw environment when the level changes.
    """

    def __init__(
        self,
        env,
        raw_env: LandingEnv,
        log_interval: int = 5000,
        curriculum_manager: CurriculumManager | None = None,
        experiment_dir: Path | None = None,
        initial_timestep: int = 0,
    ):
        object.__setattr__(self, '_env', env)
        object.__setattr__(self, '_raw_env', raw_env)
        object.__setattr__(self, '_agent', None)
        object.__setattr__(self, '_log_interval', log_interval)
        object.__setattr__(self, '_termination_counts', defaultdict(float))
        object.__setattr__(self, '_step_count', initial_timestep)
        object.__setattr__(self, '_total_episodes', 0)
        object.__setattr__(self, '_curriculum_manager', curriculum_manager)
        object.__setattr__(self, '_n_worlds', raw_env.cfg.n_worlds)
        object.__setattr__(self, '_experiment_dir', experiment_dir)
        object.__setattr__(self, '_cumulative_components', None)
        object.__setattr__(self, '_component_names', None)
        object.__setattr__(self, '_completed_episode_components', defaultdict(list))

        # Track per-agent rewards per world (episode total return)
        object.__setattr__(self, '_cumulative_drone_reward', np.zeros(raw_env.cfg.n_worlds))
        object.__setattr__(self, '_cumulative_rover_reward', np.zeros(raw_env.cfg.n_worlds))
        object.__setattr__(self, '_completed_episode_drone_rewards', [])
        object.__setattr__(self, '_completed_episode_rover_rewards', [])

        # Track episode lengths per world (in control steps)
        object.__setattr__(self, '_episode_step_counts', np.zeros(raw_env.cfg.n_worlds, dtype=np.int32))
        object.__setattr__(self, '_completed_episode_lengths', [])

        if curriculum_manager is not None:
            curriculum_manager.on_level_change(self._on_curriculum_level_change)

    def _on_curriculum_level_change(self, level_idx: int, level_config: CurriculumLevel):
        """Handle curriculum level change by updating environment params.

        Resets toggleable params to defaults first so levels are not cumulative.
        """
        print(f"[Curriculum] Level change → {level_idx}: {level_config.name}")
        spawn_fn = None
        if level_config.spawn:
            spawn_fn = create_spawn_fn_from_config(level_config.spawn, rover_nx=self._raw_env.cfg.rover_nx)
        # Reset toggleable params to defaults before applying level overrides
        defaults = {
            "randomize_mass": False,
            "randomize_inertia": False,
            "enable_disturbance": False,
        }
        merged = {**defaults, **level_config.params}
        self._raw_env.update_curriculum_params(spawn_fn=spawn_fn, **merged)

    def set_agent(self, agent):
        object.__setattr__(self, '_agent', agent)

    def set_initial_timestep(self, timestep: int):
        object.__setattr__(self, '_step_count', timestep)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self._env.step(actions)

        term_events = self._raw_env.last_termination_events
        n_worlds = self._n_worlds

        for key in ("landing", "crash", "out_of_bounds", "max_steps"):
            self._termination_counts[f"termination/{key}"] += term_events.get(key, 0.0) * n_worlds

        n_episodes_ended = sum(
            int(round(term_events.get(k, 0.0) * n_worlds))
            for k in ("landing", "crash", "out_of_bounds", "max_steps")
        )
        if n_episodes_ended > 0:
            object.__setattr__(self, '_total_episodes', self._total_episodes + n_episodes_ended)

        # Increment per-world episode step counters
        self._episode_step_counts += 1

        # Track reward components per world
        if hasattr(self._raw_env, 'last_reward_components'):
            components = self._raw_env.last_reward_components
            if self._cumulative_components is None:
                names = list(components.keys())
                object.__setattr__(self, '_component_names', names)
                object.__setattr__(self, '_cumulative_components', {
                    name: np.zeros(n_worlds) for name in names
                })
            for name in self._component_names:
                self._cumulative_components[name] += components[name]

        # Track per-agent rewards per world
        if hasattr(self._raw_env, 'last_drone_reward'):
            self._cumulative_drone_reward += np.asarray(self._raw_env.last_drone_reward)
            self._cumulative_rover_reward += np.asarray(self._raw_env.last_rover_reward)

        # Detect which worlds ended an episode this step
        episode_terminated = info.get("episode_terminated", np.zeros(n_worlds, dtype=bool))
        episode_truncated = info.get("episode_truncated", np.zeros(n_worlds, dtype=bool))
        done_mask = np.asarray(episode_terminated).flatten() | np.asarray(episode_truncated).flatten()

        if done_mask.any():
            # Flush reward components
            if self._component_names is not None:
                for name in self._component_names:
                    completed = self._cumulative_components[name][done_mask]
                    self._completed_episode_components[name].extend(completed.tolist())
                    self._cumulative_components[name][done_mask] = 0.0

            # Flush per-agent rewards
            self._completed_episode_drone_rewards.extend(
                self._cumulative_drone_reward[done_mask].tolist()
            )
            self._cumulative_drone_reward[done_mask] = 0.0
            self._completed_episode_rover_rewards.extend(
                self._cumulative_rover_reward[done_mask].tolist()
            )
            self._cumulative_rover_reward[done_mask] = 0.0

            # Flush episode lengths
            self._completed_episode_lengths.extend(
                self._episode_step_counts[done_mask].tolist()
            )
            self._episode_step_counts[done_mask] = 0

        self._step_count += 1

        if self._agent is not None and self._step_count % self._log_interval == 0:
            self._log_events()

        return obs, rewards, terminated, truncated, info

    def _log_events(self):
        """Log accumulated events to TensorBoard via SKRL agent.track_data()."""
        total_terminations = sum(self._termination_counts.values())

        n_landings = self._termination_counts["termination/landing"]
        landing_rate = n_landings / total_terminations if total_terminations > 0 else 0.0

        print(
            f"[Step {self._step_count}] "
            f"landing={n_landings:.0f} "
            f"crash={self._termination_counts['termination/crash']:.0f} "
            f"oob={self._termination_counts['termination/out_of_bounds']:.0f} "
            f"timeout={self._termination_counts['termination/max_steps']:.0f} "
            f"total={total_terminations:.0f} "
            f"landing_rate={landing_rate:.2%}"
        )

        for key, count in self._termination_counts.items():
            self._agent.track_data(key, count)
            rate_key = key.replace("termination/", "termination_rate/")
            rate = count / total_terminations if total_terminations > 0 else 0.0
            self._agent.track_data(rate_key, rate)

        self._agent.track_data("episode/total", self._total_episodes)

        # Mean episode length
        if len(self._completed_episode_lengths) > 0:
            self._agent.track_data("episode/mean_length", np.mean(self._completed_episode_lengths))

        # Per-component episode returns
        for name, returns in self._completed_episode_components.items():
            if len(returns) > 0:
                self._agent.track_data(f"reward/{name}", np.mean(returns))

        # Per-agent episode returns
        if len(self._completed_episode_drone_rewards) > 0:
            drone_mean = np.mean(self._completed_episode_drone_rewards)
            rover_mean = np.mean(self._completed_episode_rover_rewards)
            self._agent.track_data("reward/drone_total", drone_mean)
            self._agent.track_data("reward/rover_total", rover_mean)
            self._agent.track_data("reward/total", (drone_mean + rover_mean) / 2)

        # Curriculum advancement
        if self._curriculum_manager is not None:
            advanced = self._curriculum_manager.check_advancement(landing_rate)
            if advanced:
                lvl = self._curriculum_manager.current_level
                lvl_name = self._curriculum_manager.current_level_config.name
                print(f"[Curriculum] Advanced to level {lvl} ({lvl_name}) "
                      f"at landing rate {landing_rate:.2%}")

            self._agent.track_data("curriculum/level", self._curriculum_manager.current_level)
            self._agent.track_data("curriculum/landing_rate", landing_rate)
            self._agent.track_data("curriculum/total_episodes", self._total_episodes)

        # Reset counters
        object.__setattr__(self, '_termination_counts', defaultdict(float))
        object.__setattr__(self, '_completed_episode_components', defaultdict(list))
        object.__setattr__(self, '_completed_episode_drone_rewards', [])
        object.__setattr__(self, '_completed_episode_rover_rewards', [])
        object.__setattr__(self, '_completed_episode_lengths', [])

    def close(self):
        return self._env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MAPPO MLP on drone-rover landing task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/train_mappo_mlp.py --experiment default
    python scripts/train_mappo_mlp.py --experiment default --resume-run run_20260101120000
    python scripts/train_mappo_mlp.py --experiment default --resume-run run_20260101120000 --curriculum-level 2
        """,
    )
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name (e.g., 'default')")
    parser.add_argument("--resume-run", type=str, default=None,
                        help="Run name or full path to resume from")
    parser.add_argument("--curriculum-level", type=int, default=None,
                        help="Curriculum level to start at when resuming")
    parser.add_argument("--render", action="store_true",
                        help="Enable periodic rendering during training")
    parser.add_argument("--render-interval", type=int, default=10000,
                        help="Render every N timesteps (default: 10000)")
    return parser.parse_args()


def find_latest_checkpoint(run_dir: Path) -> tuple[Path, int]:
    """Find the latest checkpoint in a run directory."""
    best_agents = list(run_dir.glob("best_agent_*.pt"))
    if best_agents:
        best_with_steps = []
        for f in best_agents:
            try:
                step = int(f.stem.split("_")[-1])
                best_with_steps.append((f, step))
            except (IndexError, ValueError):
                continue
        if best_with_steps:
            best_with_steps.sort(key=lambda x: x[1], reverse=True)
            return best_with_steps[0]

    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    checkpoint_files = []
    for f in checkpoints_dir.glob("agent_*.pt"):
        try:
            step = int(f.stem.split("_")[1])
            checkpoint_files.append((f, step))
        except (IndexError, ValueError):
            continue

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {run_dir}")

    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    return checkpoint_files[0]


def generate_run_id() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


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
    print(f"Loading experiment config from: {experiment_path / 'config.yaml'}")

    training_cfg = get_training_config(config)
    policy_cfg = get_mlp_policy_config(config)

    timesteps = training_cfg["timesteps"]
    n_worlds = training_cfg["n_worlds"]
    device = torch.device("cpu")

    # Set seed for reproducibility
    seed = training_cfg.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"[Seed] Set torch/numpy seed to {seed}")

    # Create env config and spawn function
    env_cfg = config_to_env_config(config, device="cpu")
    env_cfg.roll_pitch_max = policy_cfg["drone"]["roll_pitch_max"]
    env_cfg.yaw_max = policy_cfg["drone"]["yaw_max"]
    spawn_fn = get_spawn_fn_from_config(config)

    # Curriculum
    curriculum_cfg = load_curriculum_config(config)
    curriculum_manager = None
    if curriculum_cfg is not None:
        curriculum_manager = CurriculumManager(curriculum_cfg)
        print(f"[Curriculum] Enabled with {len(curriculum_cfg.levels)} levels")
        print(f"[Curriculum] Advance threshold: {curriculum_cfg.advance_threshold}")
        print(f"[Curriculum] Starting at level 1: {curriculum_cfg.levels[0].name}")

        initial_params = curriculum_manager.get_env_params()
        for param_name, param_value in initial_params.items():
            if param_name != "spawn" and hasattr(env_cfg, param_name):
                setattr(env_cfg, param_name, param_value)

        if "spawn" in initial_params and initial_params["spawn"]:
            spawn_fn = create_spawn_fn_from_config(initial_params["spawn"], rover_nx=env_cfg.rover_nx)

    # Set up results directory
    if args.resume_run is not None:
        resume_run_path = Path(args.resume_run)
        if not resume_run_path.exists():
            resume_run_path = experiment_path / "results" / args.resume_run
        if not resume_run_path.exists():
            raise FileNotFoundError(
                f"Run directory not found: {args.resume_run}\n"
                f"Tried: {args.resume_run} and {experiment_path / 'results' / args.resume_run}"
            )
        experiment_dir = resume_run_path
        results_dir = experiment_dir.parent
        run_name = experiment_dir.name
        print(f"Resuming training in existing run directory: {experiment_dir}")
    else:
        results_dir = experiment_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"run_{generate_run_id()}"
        experiment_dir = results_dir / run_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"Experiment: {args.experiment}")
    print(f"Results directory: {experiment_dir}")

    # Create environment
    render_mode = "human" if args.render else None
    env = LandingEnv(cfg=env_cfg, spawn_fn=spawn_fn, render_mode=render_mode)
    raw_env = env

    # Save environment config
    d_cfg = policy_cfg["drone"]
    r_cfg = policy_cfg["rover"]
    env_config = {
        "policy_type": "mlp",
        "experiment_name": args.experiment,
        "rover_type": env_cfg.rover_type,
        "drone_model": env_cfg.drone_model,
        "n_worlds": n_worlds,
        "sim_freq": env_cfg.sim_freq,
        "control_freq": env_cfg.control_freq,
        "episode_length_s": env_cfg.episode_length_s,
        "map_size_x": env_cfg.map_size_x,
        "map_size_y": env_cfg.map_size_y,
        "rover_max_speed": env_cfg.rover_max_speed,
        "rover_vx_max": env_cfg.rover_vx_max,
        "rover_vy_max": env_cfg.rover_vy_max,
        "rover_wz_max": env_cfg.rover_wz_max,
        "rover_platform_radius": env_cfg.rover_platform_radius,
        "rover_height": env_cfg.rover_height,
        "landing_z_tol": env_cfg.landing_z_tol,
        "landing_vel_xy_tol": env_cfg.landing_vel_xy_tol,
        "landing_vel_z_tol": env_cfg.landing_vel_z_tol,
        "landing_attitude_tol": env_cfg.landing_attitude_tol,
        "disturbance_type": env_cfg.disturbance_type,
        "disturbance_ou_theta": env_cfg.disturbance_ou_theta,
        "drone_obs_dim": raw_env.drone_obs_dim,
        "rover_obs_dim": raw_env.rover_obs_dim,
        "shared_state_dim": raw_env.shared_state_dim,
        "drone_policy": {
            "hidden_sizes": d_cfg["hidden_sizes"],
            "activation": d_cfg["activation"],
            "initial_log_std": d_cfg.get("initial_log_std", -1.2),
            "roll_pitch_max": d_cfg.get("roll_pitch_max", 0.5),
            "yaw_max": d_cfg.get("yaw_max", 0.5),
        },
        "rover_policy": {
            "hidden_sizes": r_cfg["hidden_sizes"],
            "activation": r_cfg["activation"],
            "initial_log_std": r_cfg.get("initial_log_std", -1.2),
        },
        "value_net_sizes": policy_cfg.get("value_net_sizes", [256, 256]),
        "value_activation": policy_cfg.get("value_activation", "relu"),
    }
    with open(experiment_dir / "environment_config.json", "w") as f:
        json.dump(env_config, f, indent=2)
    print(f"Environment config saved to: {experiment_dir / 'environment_config.json'}")

    # Print training summary
    print(f"Training configuration:")
    print(f"  - n_worlds: {n_worlds}")
    print(f"  - device: cpu")
    print(f"  - timesteps: {timesteps}")
    print(f"  - rollouts: {training_cfg['rollouts']}")
    print(f"  - drone obs_dim: {raw_env.drone_obs_dim}, rover obs_dim: {raw_env.rover_obs_dim}")
    print(f"  - shared_state_dim: {raw_env.shared_state_dim}")
    print(f"  - drone hidden_sizes: {d_cfg['hidden_sizes']}")
    print(f"  - rover hidden_sizes: {r_cfg['hidden_sizes']}")

    # Wrap environment
    env = RescaleActionWrapper(env)
    env = wrap_env(env, wrapper="pettingzoo")
    env = TerminationLoggingWrapper(
        env, raw_env,
        log_interval=5000,
        curriculum_manager=curriculum_manager,
        experiment_dir=experiment_dir,
    )

    possible_agents = env.possible_agents  # ["drone", "rover"]

    # Create memories
    memories = {
        agent_name: RandomMemory(
            memory_size=training_cfg["rollouts"],
            num_envs=n_worlds,
            device=device,
        )
        for agent_name in possible_agents
    }

    # Create per-agent MLP policies
    drone_obs_space = env.observation_space("drone")
    drone_act_space = env.action_space("drone")
    rover_obs_space = env.observation_space("rover")
    rover_act_space = env.action_space("rover")

    drone_policy = MLPGaussianPolicy(
        observation_space=drone_obs_space,
        action_space=drone_act_space,
        device=device,
        hidden_sizes=tuple(d_cfg["hidden_sizes"]),
        activation=d_cfg["activation"],
        initial_log_std=d_cfg["initial_log_std"],
    )

    rover_policy = MLPGaussianPolicy(
        observation_space=rover_obs_space,
        action_space=rover_act_space,
        device=device,
        hidden_sizes=tuple(r_cfg["hidden_sizes"]),
        activation=r_cfg["activation"],
        initial_log_std=r_cfg["initial_log_std"],
    )

    drone_critic = SharedCritic(
        observation_space=raw_env.shared_observation_space,
        action_space=drone_act_space,  # dummy (critic doesn't use action_space)
        device=device,
        value_net_sizes=policy_cfg.get("value_net_sizes", [256, 256]),
        activation=policy_cfg.get("value_activation", "relu"),
    )
    rover_critic = SharedCritic(
        observation_space=raw_env.shared_observation_space,
        action_space=rover_act_space,  # dummy (critic doesn't use action_space)
        device=device,
        value_net_sizes=policy_cfg.get("value_net_sizes", [256, 256]),
        activation=policy_cfg.get("value_activation", "relu"),
    )

    print(f"Drone policy params: {sum(p.numel() for p in drone_policy.parameters() if p.requires_grad)}")
    print(f"Rover policy params: {sum(p.numel() for p in rover_policy.parameters() if p.requires_grad)}")
    print(f"Drone critic params: {sum(p.numel() for p in drone_critic.parameters() if p.requires_grad)}")
    print(f"Rover critic params: {sum(p.numel() for p in rover_critic.parameters() if p.requires_grad)}")

    # Models dict: each agent has its own policy and value network
    models = {
        "drone": {"policy": drone_policy, "value": drone_critic},
        "rover": {"policy": rover_policy, "value": rover_critic},
    }

    # Optional schedulers and preprocessors
    lr_scheduler_class = None
    lr_scheduler_kwargs = {agent: {} for agent in possible_agents}
    lr_scheduler = training_cfg["learning_rate_scheduler"]
    if lr_scheduler == "KLAdaptiveLR":
        lr_scheduler_class = KLAdaptiveLR
        base_kwargs = training_cfg["learning_rate_scheduler_kwargs"]
        lr_scheduler_kwargs = {agent: base_kwargs.copy() for agent in possible_agents}
    elif lr_scheduler == "StepLR":
        lr_scheduler_class = StepLR
        base_kwargs = training_cfg["learning_rate_scheduler_kwargs"]
        lr_scheduler_kwargs = {agent: base_kwargs.copy() for agent in possible_agents}
    elif lr_scheduler == "LinearLR":
        lr_scheduler_class = LinearLR
        base_kwargs = training_cfg["learning_rate_scheduler_kwargs"]
        lr_scheduler_kwargs = {agent: base_kwargs.copy() for agent in possible_agents}

    obs_preprocessor_class = None
    obs_preprocessor_kwargs = {agent: {} for agent in possible_agents}
    if training_cfg.get("observation_preprocessor") == "RunningStandardScaler":
        obs_preprocessor_class = PartialRunningStandardScaler
        obs_preprocessor_kwargs = {
            "drone": {"size": raw_env.drone_obs_dim,
                      "skip_dims": raw_env.obs_binary_dims, "device": device},
            "rover": {"size": raw_env.rover_obs_dim,
                      "skip_dims": raw_env.obs_binary_dims, "device": device},
        }
        print(f"  - observation_preprocessor: PartialRunningStandardScaler")

    state_preprocessor_class = None
    state_preprocessor_kwargs = {agent: {} for agent in possible_agents}
    if training_cfg.get("state_preprocessor") == "RunningStandardScaler":
        state_preprocessor_class = PartialRunningStandardScaler
        base_kwargs = {"size": raw_env.shared_state_dim,
                       "skip_dims": raw_env.state_binary_dims, "device": device}
        state_preprocessor_kwargs = {agent: base_kwargs.copy() for agent in possible_agents}
        print(f"  - state_preprocessor: PartialRunningStandardScaler")

    value_preprocessor_class = None
    value_preprocessor_kwargs = {agent: {} for agent in possible_agents}
    if training_cfg.get("value_preprocessor") == "RunningStandardScaler":
        value_preprocessor_class = RunningStandardScaler
        base_kwargs = {"size": 1, "device": device}
        value_preprocessor_kwargs = {agent: base_kwargs.copy() for agent in possible_agents}
        print(f"  - value_preprocessor: RunningStandardScaler")

    mappo_cfg = MAPPO_CFG(
        rollouts=training_cfg["rollouts"],
        learning_epochs=training_cfg["learning_epochs"],
        mini_batches=training_cfg["mini_batches"],
        discount_factor=training_cfg["gamma"],
        lambda_=training_cfg["gae_lambda"],
        learning_rate=training_cfg["learning_rate"],
        learning_rate_scheduler=lr_scheduler_class,
        learning_rate_scheduler_kwargs=lr_scheduler_kwargs,
        observation_preprocessor=obs_preprocessor_class,
        observation_preprocessor_kwargs=obs_preprocessor_kwargs,
        state_preprocessor=state_preprocessor_class,
        state_preprocessor_kwargs=state_preprocessor_kwargs,
        grad_norm_clip=training_cfg["grad_norm_clip"],
        entropy_loss_scale=training_cfg["entropy_loss_scale"],
        value_loss_scale=training_cfg["value_loss_scale"],
        ratio_clip=training_cfg["ratio_clip"],
        value_clip=training_cfg["value_clip"],
        kl_threshold=training_cfg["kl_threshold"],
        value_preprocessor=value_preprocessor_class,
        value_preprocessor_kwargs=value_preprocessor_kwargs,
        experiment={
            "directory": str(results_dir),
            "experiment_name": run_name,
            "write_interval": 100,
            "checkpoint_interval": 5000,
        },
    )

    # Create standard MAPPO agent (no MPC state handling needed)
    agent = MAPPO(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        cfg=mappo_cfg,
        observation_spaces={a: env.observation_space(a) for a in possible_agents},
        action_spaces={a: env.action_space(a) for a in possible_agents},
        state_spaces={a: raw_env.shared_observation_space for a in possible_agents},
        device=device,
    )

    env.set_agent(agent)
    print("Agent created successfully")

    # Load checkpoint if resuming
    initial_timestep = 0
    if args.resume_run is not None:
        checkpoint_path, resume_step = find_latest_checkpoint(experiment_dir)
        print(f"Loading checkpoint: {checkpoint_path} (step {resume_step})")
        agent.load(str(checkpoint_path))
        initial_timestep = resume_step
        env.set_initial_timestep(initial_timestep)
        print(f"Resuming from step {initial_timestep}")

    # Set curriculum level if specified
    if args.curriculum_level is not None:
        if curriculum_manager is None:
            print(f"Warning: --curriculum-level specified but curriculum is not enabled")
        else:
            curriculum_manager.set_level(args.curriculum_level)
    elif args.resume_run is not None and curriculum_manager is not None:
        print(f"[Curriculum] Warning: Resuming but no --curriculum-level specified. Starting at level 1.")

    # Save learning config
    learning_config = {
        "policy_type": "mlp",
        "timesteps": timesteps,
        "n_worlds": n_worlds,
        "hyperparameters": {k: training_cfg[k] for k in (
            "rollouts", "learning_epochs", "mini_batches", "learning_rate",
            "gamma", "gae_lambda", "grad_norm_clip", "entropy_loss_scale",
            "value_loss_scale", "ratio_clip", "value_clip", "kl_threshold",
        )},
        "preprocessors": {
            "observation": training_cfg.get("observation_preprocessor"),
            "state": training_cfg.get("state_preprocessor"),
            "value": training_cfg.get("value_preprocessor"),
        },
        "rewards": config.get("rewards", {}),
        "curriculum": {
            "enabled": curriculum_cfg is not None,
            "levels": (
                [{"name": lvl.name, "params": lvl.params, "spawn": lvl.spawn}
                 for lvl in curriculum_cfg.levels]
                if curriculum_cfg is not None else []
            ),
            "advance_threshold": curriculum_cfg.advance_threshold if curriculum_cfg else None,
            "regression_threshold": curriculum_cfg.regression_threshold if curriculum_cfg else None,
            "window_size": curriculum_cfg.window_size if curriculum_cfg else None,
            "allow_regression": curriculum_cfg.allow_regression if curriculum_cfg else None,
        },
        "start_time": start_time_str,
        "resumed_from_run": args.resume_run,
        "initial_curriculum_level": args.curriculum_level,
    }
    learning_config_path = experiment_dir / "learning_config.json"
    with open(learning_config_path, "w") as f:
        json.dump(learning_config, f, indent=2)

    # Train
    total_timesteps = timesteps + initial_timestep
    trainer_cfg = {
        "timesteps": total_timesteps,
        "initial_timestep": initial_timestep,
        "headless": not args.render,
        "render_interval": args.render_interval,
    }

    trainer = SequentialTrainer(env=env, agents=agent, cfg=trainer_cfg)

    if initial_timestep > 0:
        print(f"Training from step {initial_timestep} to {total_timesteps}")
    print("Starting training...")
    trainer.train()

    end_time = datetime.now()
    duration_str = str(end_time - start_time).split(".")[0]
    learning_config["end_time"] = end_time.strftime("%Y-%m-%d %H:%M:%S")
    learning_config["duration"] = duration_str
    with open(learning_config_path, "w") as f:
        json.dump(learning_config, f, indent=2)

    agent.save(str(experiment_dir / "final_checkpoint.pt"))
    print(f"Training complete in {duration_str}. Checkpoint saved to {experiment_dir / 'final_checkpoint.pt'}")

    env.close()


if __name__ == "__main__":
    main()
