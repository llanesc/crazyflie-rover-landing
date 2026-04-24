#!/usr/bin/env python3
"""In-process parameter sweep — loads env/policy once, loops over values.

Much faster than sweep_eval.py for parameters that only affect the simulator
(e.g., sim_mass) since it avoids JAX re-init and MPC recompilation per combo.

Usage:
    python scripts/sweep_eval_inprocess.py sweeps/mass_sweep_acmpc.yaml
    python scripts/sweep_eval_inprocess.py sweeps/mass_sweep_acmpc.yaml --dry-run
"""

import argparse
import csv
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
_ACADOS_ROOT = os.path.join(_REPO_ROOT, "external", "leap-c", "external", "acados")
os.environ.setdefault("ACADOS_SOURCE_DIR", _ACADOS_ROOT)
os.environ["LD_LIBRARY_PATH"] = os.path.join(_ACADOS_ROOT, "lib") + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import torch

from crazyflie_rover_landing.envs import LandingEnv, LandingEnvConfig, RescaleActionWrapper
from crazyflie_rover_landing.envs.spawn import create_spawn_fn_from_config
from crazyflie_rover_landing.preprocessors import PartialRunningStandardScaler
from crazyflie_rover_landing.utils import (
    load_experiment_config,
    config_to_env_config,
    get_spawn_fn_from_config,
    get_training_config,
    find_experiment_path,
    apply_overrides,
)

METRIC_COLS = [
    "landing_rate", "crash_rate", "oob_rate", "timeout_rate",
    "landings", "crashes", "oob", "timeouts", "n_episodes",
    "avg_reward", "avg_time",
]

SIM_MASS_PARAMS = {"sim_mass"}


def build_param_grid(sweep_cfg: dict) -> tuple[list[str], list[tuple]]:
    param_names = sorted(sweep_cfg.keys())
    param_values = []
    for name in param_names:
        spec = sweep_cfg[name]
        if "values" in spec:
            param_values.append([v for v in spec["values"]])
        else:
            vals = np.arange(spec["min"], spec["max"] + spec["step"] / 2, spec["step"])
            param_values.append(np.round(vals, 10).tolist())
    return param_names, list(itertools.product(*param_values))


def load_completed(csv_path: Path, param_names: list[str]) -> set[tuple]:
    completed = set()
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    key = tuple(round(float(row[p]), 10) for p in param_names)
                    completed.add(key)
                except (KeyError, ValueError):
                    continue
    return completed


DISTURBANCE_PARAMS = {"disturbance_force_std", "disturbance_torque_std", "enable_disturbance"}
WIND_PARAMS = {"wind_speed", "wind_direction", "gust_intensity", "turbulence_level",
               "enable_wind", "gust_correlation_time", "turbulence_time_constant"}


def apply_sweep_params(raw_env: LandingEnv, env_cfg: LandingEnvConfig,
                       param_names: list[str], combo: tuple):
    """Apply sweep parameter values to the live env."""
    rebuild_disturbance = False
    rebuild_wind = False
    for name, val in zip(param_names, combo):
        if name == "sim_mass":
            raw_env.set_sim_mass(val)
        elif name in DISTURBANCE_PARAMS:
            setattr(env_cfg, name, val)
            rebuild_disturbance = True
        elif name in WIND_PARAMS:
            setattr(env_cfg, name, val)
            rebuild_wind = True
        else:
            setattr(env_cfg, name, val)
    if rebuild_disturbance:
        raw_env.set_disturbance(
            enabled=env_cfg.enable_disturbance,
            force_std=env_cfg.disturbance_force_std,
            torque_std=env_cfg.disturbance_torque_std,
        )
    if rebuild_wind:
        wind_active = (env_cfg.wind_speed > 0 or env_cfg.gust_intensity > 0
                       or env_cfg.turbulence_level != "none")
        raw_env.set_wind(
            enabled=wind_active,
            wind_speed=env_cfg.wind_speed,
            wind_direction=env_cfg.wind_direction,
            gust_intensity=env_cfg.gust_intensity,
            turbulence_level=env_cfg.turbulence_level,
        )


def run_eval_loop(env_skrl, raw_env, agent, n_episodes: int, n_worlds: int,
                  possible_agents: list[str], device: torch.device,
                  seed: int | None = None) -> dict:
    """Run evaluation episodes and return results dict."""
    landings = 0
    crashes = 0
    oob = 0
    timeouts = 0
    episodes_done = 0
    has_mpc_state = hasattr(agent, "_current_mpc_state")
    dt = raw_env.cfg.dt

    # Per-world accumulators for reward and episode time
    world_reward_sum = np.zeros(n_worlds)
    world_steps = np.zeros(n_worlds, dtype=np.int32)
    episode_rewards = []
    episode_times = []

    if seed is not None:
        env_skrl._seed = seed
    obs, infos = env_skrl.reset()

    if has_mpc_state and "mpc_state" in infos:
        for uid in possible_agents:
            if uid in infos["mpc_state"]:
                agent._current_mpc_state[uid] = torch.as_tensor(
                    infos["mpc_state"][uid], dtype=torch.float32, device=device
                )

    while episodes_done < n_episodes:
        states = {a: infos.get("state", {}).get(a) for a in possible_agents}
        with torch.no_grad():
            actions, _ = agent.act(obs, states, timestep=10**9, timesteps=10**9)

        obs, rewards, terminated, truncated, infos = env_skrl.step(actions)

        if has_mpc_state and "mpc_state" in infos:
            for uid in possible_agents:
                if uid in infos["mpc_state"] and agent._mpc_state_sizes.get(uid, 0) > 0:
                    agent._current_mpc_state[uid] = torch.as_tensor(
                        infos["mpc_state"][uid], dtype=torch.float32, device=device
                    )

        # Accumulate per-world reward and steps
        drone_r = rewards["drone"].cpu().numpy() if isinstance(rewards["drone"], torch.Tensor) else np.asarray(rewards["drone"])
        rover_r = rewards["rover"].cpu().numpy() if isinstance(rewards["rover"], torch.Tensor) else np.asarray(rewards["rover"])
        world_reward_sum += (drone_r.squeeze() + rover_r.squeeze())
        world_steps += 1

        term_events = raw_env.last_termination_events
        n_landing = int(round(term_events.get("landing", 0.0) * n_worlds))
        n_crash = int(round(term_events.get("crash", 0.0) * n_worlds))
        n_oob = int(round(term_events.get("out_of_bounds", 0.0) * n_worlds))
        n_timeout = int(round(term_events.get("max_steps", 0.0) * n_worlds))

        n_done = n_landing + n_crash + n_oob + n_timeout
        if n_done > 0:
            done_d = terminated["drone"]
            if isinstance(done_d, torch.Tensor):
                done_d = done_d.cpu().numpy()
            done_t = truncated["drone"]
            if isinstance(done_t, torch.Tensor):
                done_t = done_t.cpu().numpy()
            done_mask = done_d.squeeze() | done_t.squeeze()
            for w in range(n_worlds):
                if done_mask[w]:
                    episode_rewards.append(world_reward_sum[w])
                    episode_times.append(world_steps[w] * dt)
            world_reward_sum[done_mask] = 0.0
            world_steps[done_mask] = 0

        landings += n_landing
        crashes += n_crash
        oob += n_oob
        timeouts += n_timeout
        episodes_done += n_done

    total = landings + crashes + oob + timeouts
    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    avg_time = float(np.mean(episode_times)) if episode_times else 0.0
    return {
        "n_episodes": total,
        "landing_rate": landings / total if total > 0 else 0.0,
        "crash_rate": crashes / total if total > 0 else 0.0,
        "oob_rate": oob / total if total > 0 else 0.0,
        "timeout_rate": timeouts / total if total > 0 else 0.0,
        "landings": landings,
        "crashes": crashes,
        "oob": oob,
        "timeouts": timeouts,
        "avg_reward": avg_reward,
        "avg_time": avg_time,
    }


def find_checkpoint(experiment_path: Path, run_name: str | None) -> Path:
    if run_name is not None:
        run_dir = experiment_path / "results" / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
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
        final = run_dir / "final_checkpoint.pt"
        if final.exists():
            return final
    raise FileNotFoundError(f"No checkpoint found for run: {run_name}")


def load_run_configs(run_dir: Path) -> tuple[dict, dict]:
    env_config_path = run_dir / "environment_config.json"
    learning_config_path = run_dir / "learning_config.json"
    if not env_config_path.exists():
        raise FileNotFoundError(f"environment_config.json not found in {run_dir}")
    with open(env_config_path) as f:
        env_config = json.load(f)
    learning_config = {}
    if learning_config_path.exists():
        with open(learning_config_path) as f:
            learning_config = json.load(f)
    return env_config, learning_config


def setup_acmpc(cfg, experiment_path, checkpoint_path, run_dir, env_cfg, spawn_fn, n_worlds, device):
    from crazyflie_rover_landing.agents import MAPPO_MPC
    from crazyflie_rover_landing.policies import DroneACMPCGaussianPolicy, SharedCritic
    from crazyflie_rover_landing.leap_c.x3_rover_policy_linear_ls import X3RoverACMPCGaussianPolicy

    env_run_config, learning_config = load_run_configs(run_dir)

    drone_mpc = env_run_config.get("drone_mpc", {})
    rover_mpc = env_run_config.get("rover_mpc", {})
    policy_cfg = {
        "drone": {
            "mpc_horizon": drone_mpc["mpc_horizon"],
            "mpc_dt": drone_mpc["mpc_dt"],
            "cost_net_sizes": drone_mpc.get("cost_net_sizes", [256, 256]),
            "state_type": drone_mpc.get("state_type", "euler"),
            "integrator": drone_mpc.get("integrator", "rk4"),
            "roll_pitch_max": drone_mpc.get("roll_pitch_max", 0.5),
            "yaw_max": drone_mpc.get("yaw_max", 0.5),
            "pos_offset_max": drone_mpc.get("pos_offset_max", 2.0),
            "initial_log_std": drone_mpc.get("initial_log_std", -1.2),
            "activation": drone_mpc.get("activation", "relu"),
        },
        "rover": {
            "mpc_horizon": rover_mpc["mpc_horizon"],
            "mpc_dt": rover_mpc["mpc_dt"],
            "cost_net_sizes": rover_mpc.get("cost_net_sizes", [256, 256]),
            "pos_offset_max": rover_mpc.get("pos_offset_max", 2.0),
            "initial_log_std": rover_mpc.get("initial_log_std", -1.2),
            "activation": rover_mpc.get("activation", "relu"),
            "wheel_dynamics": rover_mpc.get("wheel_dynamics", False),
        },
        "value_net_sizes": env_run_config.get("value_net_sizes", [256, 256]),
        "value_activation": env_run_config.get("value_activation", "relu"),
    }

    if "rover_type" in env_run_config:
        env_cfg.rover_type = env_run_config["rover_type"]
    if "drone_model" in env_run_config:
        env_cfg.drone_model = env_run_config["drone_model"]
    if "roll_pitch_max" in drone_mpc:
        env_cfg.roll_pitch_max = drone_mpc["roll_pitch_max"]
    if "yaw_max" in drone_mpc:
        env_cfg.yaw_max = drone_mpc["yaw_max"]
    if "state_type" in drone_mpc:
        env_cfg.drone_state_type = drone_mpc["state_type"]

    # Curriculum spawn
    curriculum_data = learning_config.get("curriculum", {})
    curriculum_levels = curriculum_data.get("levels", [])
    level_num = cfg.get("level")

    if level_num is not None:
        if not curriculum_levels or level_num < 1 or level_num > len(curriculum_levels):
            raise ValueError(f"--level {level_num} out of range")
        level = curriculum_levels[level_num - 1]
        spawn_cfg_l = level.get("spawn", {})
        if spawn_cfg_l:
            spawn_cfg_l.setdefault("rover", {})["stationary"] = True
            spawn_fn = create_spawn_fn_from_config(spawn_cfg_l, rover_nx=env_cfg.rover_nx)
        level_params = level.get("params", {})
        for param_name, param_value in level_params.items():
            if hasattr(env_cfg, param_name):
                setattr(env_cfg, param_name, param_value)
    else:
        if curriculum_levels:
            spawn_cfg_l = curriculum_levels[0].get("spawn", {})
        else:
            spawn_cfg_l = {}
        spawn_cfg_l.setdefault("rover", {})["stationary"] = True
        spawn_fn = create_spawn_fn_from_config(spawn_cfg_l, rover_nx=env_cfg.rover_nx)

    if cfg.get("no_domain_rand"):
        env_cfg.randomize_mass = False
        env_cfg.randomize_inertia = False
    if cfg.get("no_disturbance"):
        env_cfg.enable_disturbance = False

    env_cfg.n_worlds = n_worlds

    env = LandingEnv(cfg=env_cfg, spawn_fn=spawn_fn)
    raw_env = env
    env = RescaleActionWrapper(env)

    from skrl.envs.wrappers.torch import wrap_env
    from skrl.memories.torch import RandomMemory
    from skrl.multi_agents.torch.mappo import MAPPO_CFG

    env_skrl = wrap_env(env, wrapper="pettingzoo")
    possible_agents = env_skrl.possible_agents

    memories = {a: RandomMemory(memory_size=1, num_envs=n_worlds, device=device)
                for a in possible_agents}

    d_cfg = policy_cfg["drone"]
    r_cfg = policy_cfg["rover"]

    drone_policy = DroneACMPCGaussianPolicy(
        observation_space=env_skrl.observation_space("drone"),
        action_space=env_skrl.action_space("drone"),
        device=device,
        mpc_horizon=d_cfg["mpc_horizon"],
        mpc_dt=d_cfg["mpc_dt"],
        cost_net_sizes=d_cfg["cost_net_sizes"],
        state_type=d_cfg.get("state_type", "euler"),
        integrator=d_cfg.get("integrator", "rk4"),
        roll_pitch_max=env_cfg.roll_pitch_max,
        yaw_max=env_cfg.yaw_max,
        thrust_min=env_cfg.thrust_min,
        thrust_max=env_cfg.thrust_max,
        mass=env_cfg.mass,
        gravity=env_cfg.gravity,
        drone_model=env_cfg.drone_model,
        n_batch_max=n_worlds,
        initial_log_std=d_cfg["initial_log_std"],
        activation=d_cfg["activation"],
        pos_offset_max=d_cfg["pos_offset_max"],
    )

    rover_policy_kwargs = dict(
        observation_space=env_skrl.observation_space("rover"),
        action_space=env_skrl.action_space("rover"),
        device=device,
        mpc_horizon=r_cfg["mpc_horizon"],
        mpc_dt=r_cfg["mpc_dt"],
        cost_net_sizes=r_cfg["cost_net_sizes"],
        n_batch_max=n_worlds,
        initial_log_std=r_cfg["initial_log_std"],
        activation=r_cfg["activation"],
        pos_offset_max=r_cfg["pos_offset_max"],
    )
    if env_cfg.rover_type == "x3":
        rover_policy_kwargs.update(
            vx_max=env_cfg.rover_vx_max,
            vy_max=env_cfg.rover_vy_max,
            wz_max=env_cfg.rover_wz_max,
            wheel_vel_max=env_cfg.rover_wheel_vel_max if r_cfg.get("wheel_dynamics", False) else None,
        )
    rover_policy = X3RoverACMPCGaussianPolicy(**rover_policy_kwargs)

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

    from skrl.resources.preprocessors.torch import RunningStandardScaler
    preprocessors = learning_config.get("preprocessors", {})

    obs_pp_class = None
    obs_pp_kwargs = {a: {} for a in possible_agents}
    if preprocessors.get("observation") == "RunningStandardScaler":
        obs_pp_class = PartialRunningStandardScaler
        obs_pp_kwargs = {
            "drone": {"size": raw_env.drone_obs_dim, "skip_dims": raw_env.obs_binary_dims, "device": device},
            "rover": {"size": raw_env.rover_obs_dim, "skip_dims": raw_env.obs_binary_dims, "device": device},
        }

    state_pp_class = None
    state_pp_kwargs = {a: {} for a in possible_agents}
    if preprocessors.get("state") == "RunningStandardScaler":
        state_pp_class = PartialRunningStandardScaler
        base = {"size": raw_env.shared_state_dim, "skip_dims": raw_env.state_binary_dims, "device": device}
        state_pp_kwargs = {a: base.copy() for a in possible_agents}

    value_pp_class = None
    value_pp_kwargs = {a: {} for a in possible_agents}
    if preprocessors.get("value") == "RunningStandardScaler":
        value_pp_class = RunningStandardScaler
        base = {"size": 1, "device": device}
        value_pp_kwargs = {a: base.copy() for a in possible_agents}

    mappo_cfg = MAPPO_CFG(
        rollouts=1,
        learning_rate_scheduler_kwargs={a: {} for a in possible_agents},
        observation_preprocessor=obs_pp_class,
        observation_preprocessor_kwargs=obs_pp_kwargs,
        state_preprocessor=state_pp_class,
        state_preprocessor_kwargs=state_pp_kwargs,
        value_preprocessor=value_pp_class,
        value_preprocessor_kwargs=value_pp_kwargs,
        experiment={"directory": "", "experiment_name": "eval", "write_interval": 0},
    )

    agent = MAPPO_MPC(
        mpc_state_sizes={"drone": raw_env.drone_mpc_state_dim, "rover": raw_env.rover_mpc_state_dim},
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        cfg=mappo_cfg,
        observation_spaces={a: env_skrl.observation_space(a) for a in possible_agents},
        action_spaces={a: env_skrl.action_space(a) for a in possible_agents},
        state_spaces={a: raw_env.shared_observation_space for a in possible_agents},
        device=device,
    )

    agent.load(str(checkpoint_path))
    agent.enable_training_mode(False, apply_to_models=True)

    return env_skrl, raw_env, agent, possible_agents


def setup_mlp(cfg, experiment_path, checkpoint_path, run_dir, env_cfg, spawn_fn, n_worlds, device):
    from crazyflie_rover_landing.policies import MLPGaussianPolicy, SharedCritic

    config = load_experiment_config(experiment_path)
    training_cfg = get_training_config(config)

    policy_section = config.get("policy", {})
    drone_sec = policy_section.get("drone", {})
    rover_sec = policy_section.get("rover", {})
    shared_log_std = policy_section.get("initial_log_std", 0.0)
    shared_activation = policy_section.get("activation", "relu")
    policy_cfg = {
        "drone": {
            "hidden_sizes": drone_sec.get("hidden_sizes", [256, 256]),
            "activation": drone_sec.get("activation", shared_activation),
            "initial_log_std": drone_sec.get("initial_log_std", shared_log_std),
            "roll_pitch_max": drone_sec.get("roll_pitch_max", 0.5),
            "yaw_max": drone_sec.get("yaw_max", 0.5),
        },
        "rover": {
            "hidden_sizes": rover_sec.get("hidden_sizes", [256, 256]),
            "activation": rover_sec.get("activation", shared_activation),
            "initial_log_std": rover_sec.get("initial_log_std", shared_log_std),
        },
        "value_net_sizes": policy_section.get("value_net_sizes", [256, 256]),
        "value_activation": policy_section.get("value_activation", "relu"),
    }

    env_cfg.roll_pitch_max = policy_cfg["drone"]["roll_pitch_max"]
    env_cfg.yaw_max = policy_cfg["drone"]["yaw_max"]

    # Curriculum spawn
    curriculum_cfg = config.get("curriculum", {})
    levels = curriculum_cfg.get("levels", [])
    level_num = cfg.get("level")

    if level_num is not None:
        if not levels or level_num < 1 or level_num > len(levels):
            raise ValueError(f"--level {level_num} out of range")
        level = levels[level_num - 1]
        spawn_cfg_l = {}
        if "drone_spawn" in level:
            spawn_cfg_l["drone"] = level["drone_spawn"]
        if "rover_spawn" in level:
            spawn_cfg_l["rover"] = level["rover_spawn"]
        if spawn_cfg_l:
            spawn_cfg_l.setdefault("rover", {})["stationary"] = True
            spawn_fn = create_spawn_fn_from_config(spawn_cfg_l, rover_nx=env_cfg.rover_nx)
        skip_keys = {"name", "level", "drone_spawn", "rover_spawn"}
        for param_name, param_value in level.items():
            if param_name not in skip_keys and hasattr(env_cfg, param_name):
                setattr(env_cfg, param_name, param_value)
    else:
        if curriculum_cfg.get("enabled", False) and levels:
            level0 = levels[0]
            spawn_cfg_l = {}
            if "drone_spawn" in level0:
                spawn_cfg_l["drone"] = level0["drone_spawn"]
            if "rover_spawn" in level0:
                spawn_cfg_l["rover"] = level0["rover_spawn"]
        else:
            spawn_cfg_l = config.get("environment", {}).get("spawn", {})
        spawn_cfg_l.setdefault("rover", {})["stationary"] = True
        spawn_fn = create_spawn_fn_from_config(spawn_cfg_l)

    if cfg.get("no_domain_rand"):
        env_cfg.randomize_mass = False
        env_cfg.randomize_inertia = False
    if cfg.get("no_disturbance"):
        env_cfg.enable_disturbance = False

    env_cfg.n_worlds = n_worlds

    env = LandingEnv(cfg=env_cfg, spawn_fn=spawn_fn)
    raw_env = env
    env = RescaleActionWrapper(env)

    from skrl.envs.wrappers.torch import wrap_env
    from skrl.memories.torch import RandomMemory
    from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_CFG

    env_skrl = wrap_env(env, wrapper="pettingzoo")
    possible_agents = env_skrl.possible_agents

    memories = {a: RandomMemory(memory_size=1, num_envs=n_worlds, device=device)
                for a in possible_agents}

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

    from skrl.resources.preprocessors.torch import RunningStandardScaler

    obs_pp_class = None
    obs_pp_kwargs = {a: {} for a in possible_agents}
    if training_cfg.get("observation_preprocessor") == "RunningStandardScaler":
        obs_pp_class = PartialRunningStandardScaler
        obs_pp_kwargs = {
            "drone": {"size": raw_env.drone_obs_dim, "skip_dims": raw_env.obs_binary_dims, "device": device},
            "rover": {"size": raw_env.rover_obs_dim, "skip_dims": raw_env.obs_binary_dims, "device": device},
        }

    state_pp_class = None
    state_pp_kwargs = {a: {} for a in possible_agents}
    if training_cfg.get("state_preprocessor") == "RunningStandardScaler":
        state_pp_class = PartialRunningStandardScaler
        base = {"size": raw_env.shared_state_dim, "skip_dims": raw_env.state_binary_dims, "device": device}
        state_pp_kwargs = {a: base.copy() for a in possible_agents}

    value_pp_class = None
    value_pp_kwargs = {a: {} for a in possible_agents}
    if training_cfg.get("value_preprocessor") == "RunningStandardScaler":
        value_pp_class = RunningStandardScaler
        base = {"size": 1, "device": device}
        value_pp_kwargs = {a: base.copy() for a in possible_agents}

    mappo_cfg = MAPPO_CFG(
        rollouts=1,
        learning_rate_scheduler_kwargs={a: {} for a in possible_agents},
        observation_preprocessor=obs_pp_class,
        observation_preprocessor_kwargs=obs_pp_kwargs,
        state_preprocessor=state_pp_class,
        state_preprocessor_kwargs=state_pp_kwargs,
        value_preprocessor=value_pp_class,
        value_preprocessor_kwargs=value_pp_kwargs,
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

    agent.load(str(checkpoint_path))
    agent.enable_training_mode(False, apply_to_models=True)

    return env_skrl, raw_env, agent, possible_agents


def main():
    parser = argparse.ArgumentParser(
        description="In-process parameter sweep (no subprocess overhead)",
    )
    parser.add_argument("config", type=str, help="Path to sweep YAML config")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for CSV output (default: next to config)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the parameter grid without running")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sweep_cfg = cfg["sweep"]
    param_names, grid = build_param_grid(sweep_cfg)

    print(f"Sweep parameters: {param_names}")
    print(f"Grid size: {len(grid)} combinations")

    if args.dry_run:
        for i, combo in enumerate(grid):
            labels = [f"{n}={v}" for n, v in zip(param_names, combo)]
            print(f"  [{i + 1}/{len(grid)}] {', '.join(labels)}")
        return

    # Determine output CSV
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.config).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_name = Path(args.config).stem + "_results.csv"
    csv_path = output_dir / csv_name

    completed = load_completed(csv_path, param_names)
    if completed:
        print(f"Resuming: {len(completed)}/{len(grid)} already completed")

    # Determine policy type
    eval_script = cfg["eval_script"]
    is_acmpc = "acmpc" in eval_script
    policy_type = "acmpc" if is_acmpc else "mlp"

    experiment_name = cfg["experiment"]
    experiment_path = find_experiment_path(experiment_name, policy_type=policy_type)
    config_data = load_experiment_config(experiment_path)
    device = torch.device("cpu")
    env_cfg = config_to_env_config(config_data, device="cpu")
    spawn_fn = get_spawn_fn_from_config(config_data)
    n_worlds = cfg.get("n_worlds", 100)
    n_episodes = cfg.get("n_episodes", 100)
    seed = cfg.get("seed", 42)

    # Find checkpoint
    run_name = cfg.get("run")
    checkpoint_str = cfg.get("checkpoint")
    if checkpoint_str:
        checkpoint_path = Path(checkpoint_str)
        run_dir = checkpoint_path.parent.parent
    else:
        checkpoint_path = find_checkpoint(experiment_path, run_name)
        run_dir = checkpoint_path.parent.parent

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Policy type: {policy_type}")

    # One-time setup
    t_setup = time.time()
    if is_acmpc:
        env_skrl, raw_env, agent, possible_agents = setup_acmpc(
            cfg, experiment_path, checkpoint_path, run_dir, env_cfg, spawn_fn, n_worlds, device
        )
    else:
        env_skrl, raw_env, agent, possible_agents = setup_mlp(
            cfg, experiment_path, checkpoint_path, run_dir, env_cfg, spawn_fn, n_worlds, device
        )
    print(f"Setup complete ({time.time() - t_setup:.1f}s)")

    # Sweep loop
    header = param_names + METRIC_COLS
    write_header = not csv_path.exists() or len(completed) == 0

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)

        for i, combo in enumerate(grid):
            key = tuple(round(v, 10) for v in combo)
            if key in completed:
                continue

            labels = ", ".join(f"{n}={v}" for n, v in zip(param_names, combo))
            print(f"\n[{i + 1}/{len(grid)}] {labels}")

            apply_sweep_params(raw_env, env_cfg, param_names, combo)

            t0 = time.time()
            metrics = run_eval_loop(
                env_skrl, raw_env, agent, n_episodes, n_worlds, possible_agents, device,
                seed=seed,
            )
            elapsed = time.time() - t0

            row = [f"{v:.6g}" for v in combo]
            row += [str(metrics.get(col, "")) for col in METRIC_COLS]
            writer.writerow(row)
            csvfile.flush()

            lr = metrics["landing_rate"]
            cr = metrics["crash_rate"]
            tr = metrics["timeout_rate"]
            ar = metrics["avg_reward"]
            at = metrics["avg_time"]
            print(f"  landing={lr * 100:.1f}% crash={cr * 100:.1f}% timeout={tr * 100:.1f}% "
                  f"reward={ar:.1f} time={at:.2f}s ({elapsed:.1f}s)")

    print(f"\nSweep complete. Results saved to {csv_path}")
    env_skrl.close()


if __name__ == "__main__":
    main()
