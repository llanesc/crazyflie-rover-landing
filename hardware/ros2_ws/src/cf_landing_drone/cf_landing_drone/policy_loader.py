"""Policy loader for drone and rover checkpoints (ACMPC and MLP).

Loads SKRL-trained policies for use in hardware experiments.
"""

import json
import os
from pathlib import Path
from typing import Optional

import gymnasium
import numpy as np
import torch


def get_models_dir() -> Path:
    """Find the models directory.

    Search order:
    1. CF_LANDING_MODELS_DIR environment variable
    2. Auto-detect from install path (../../../src/cf_landing_drone/models/)
    3. Relative to this file (../../models/)
    """
    env_dir = os.environ.get("CF_LANDING_MODELS_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir():
            return p

    this_file = Path(__file__).resolve()

    # Check if running from install (colcon workspace)
    if "/install/" in str(this_file):
        src_models = this_file
        while src_models.name != "install":
            src_models = src_models.parent
        src_models = src_models.parent / "src" / "cf_landing_drone" / "models"
        if src_models.is_dir():
            return src_models

    # Relative to source
    rel = this_file.parent.parent / "models"
    if rel.is_dir():
        return rel

    raise FileNotFoundError(
        "Cannot find models directory. Set CF_LANDING_MODELS_DIR or ensure "
        "models/ is in the package."
    )


def find_checkpoint(models_dir: Path, policy_type: str = "acmpc") -> Path:
    """Find a .pt checkpoint file in models/{policy_type}/ directory.

    Args:
        models_dir: Path to models/ directory.
        policy_type: "acmpc" or "mlp".

    Priority: agent_STEP.pt (highest step) > final_checkpoint.pt > any .pt
    """
    policy_dir = models_dir / policy_type
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"No {policy_type}/ directory in {models_dir}")

    # agent_STEP.pt (highest step number)
    import re
    agent_files = []
    for f in policy_dir.glob("agent_*.pt"):
        m = re.search(r"agent_(\d+)\.pt$", f.name)
        if m:
            agent_files.append((f, int(m.group(1))))
    if agent_files:
        agent_files.sort(key=lambda x: x[1], reverse=True)
        return agent_files[0][0]

    # final_checkpoint.pt
    final = policy_dir / "final_checkpoint.pt"
    if final.is_file():
        return final

    # Any .pt
    pt_files = list(policy_dir.glob("*.pt"))
    if pt_files:
        return pt_files[0]

    raise FileNotFoundError(f"No .pt checkpoint found in {policy_dir}")


def load_env_config(models_dir: Path, policy_type: str = "acmpc") -> dict:
    """Load environment_config.json from models/{policy_type}/."""
    config_path = models_dir / policy_type / "environment_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {config_path}")
    with open(config_path) as f:
        return json.load(f)


def load_drone_policy(
    checkpoint_path: Path,
    env_config: dict,
    training_config: dict,
    device: str = "cpu",
):
    """Load DroneACMPCGaussianPolicy from training checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        env_config: environment_config.json dict.
        training_config: The full training YAML config dict (results/acmpc/X3/config.yaml).
        device: Torch device.

    Returns:
        (policy, preprocessor) tuple. preprocessor may be None.
    """
    from crazyflie_rover_landing.policies.drone_policy_linear_ls import DroneACMPCGaussianPolicy
    from crazyflie_rover_landing.envs.landing_config import LandingEnvConfig

    obs_dim = env_config["drone_obs_dim"]
    d_cfg = training_config["policy"]["drone"]
    env_section = training_config["environment"]

    observation_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )

    # Build LandingEnvConfig to get physical params (thrust_min/max, mass, gravity)
    from crazyflie_rover_landing.utils.experiment_config import config_to_env_config
    landing_cfg = config_to_env_config(training_config)

    roll_pitch_max = d_cfg.get("roll_pitch_max", landing_cfg.roll_pitch_max)
    yaw_max = d_cfg.get("yaw_max", landing_cfg.yaw_max)

    action_space = gymnasium.spaces.Box(
        low=np.array([-roll_pitch_max, -roll_pitch_max, -yaw_max, landing_cfg.thrust_min]),
        high=np.array([roll_pitch_max, roll_pitch_max, yaw_max, landing_cfg.thrust_max]),
        dtype=np.float32,
    )

    policy = DroneACMPCGaussianPolicy(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        mpc_horizon=d_cfg["mpc_horizon"],
        mpc_dt=d_cfg["mpc_dt"],
        cost_net_sizes=d_cfg["cost_net_sizes"],
        state_type=d_cfg.get("state_type", "euler"),
        integrator=d_cfg.get("integrator", "rk4"),
        roll_pitch_max=roll_pitch_max,
        yaw_max=yaw_max,
        thrust_min=landing_cfg.thrust_min,
        thrust_max=landing_cfg.thrust_max,
        mass=landing_cfg.mass,
        gravity=landing_cfg.gravity,
        drone_model=landing_cfg.drone_model,
        n_batch_max=1,
        activation=d_cfg.get("activation", training_config["policy"].get("cost_net_activation", "relu")),
        pos_offset_max=d_cfg.get("pos_offset_max", 2.0),
    )

    # Load state dict from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _load_agent_state(policy, checkpoint, agent_key="drone", component="policy")

    # Load preprocessor
    preprocessor = _load_preprocessor(checkpoint, agent_key="drone", obs_dim=obs_dim, device=device)

    policy.eval()
    return policy, preprocessor


def load_rover_policy(
    checkpoint_path: Path,
    env_config: dict,
    training_config: dict,
    device: str = "cpu",
):
    """Load X3RoverACMPCGaussianPolicy from training checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        env_config: environment_config.json dict.
        training_config: The full training YAML config dict.
        device: Torch device.

    Returns:
        (policy, preprocessor) tuple.
    """
    from crazyflie_rover_landing.leap_c.x3_rover_policy_linear_ls import X3RoverACMPCGaussianPolicy

    obs_dim = env_config["rover_obs_dim"]
    r_cfg = training_config["policy"]["rover"]
    env_section = training_config["environment"]

    observation_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )

    vx_max = env_section.get("rover_vx_max", 1.0)
    vy_max = env_section.get("rover_vy_max", 1.0)
    wz_max = env_section.get("rover_wz_max", 5.0)

    action_space = gymnasium.spaces.Box(
        low=np.array([-vx_max, -vy_max, -wz_max]),
        high=np.array([vx_max, vy_max, wz_max]),
        dtype=np.float32,
    )

    wheel_vel_max = env_section.get("rover_wheel_vel_max", 34.9) if r_cfg.get("wheel_dynamics", False) else None

    policy = X3RoverACMPCGaussianPolicy(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        mpc_horizon=r_cfg["mpc_horizon"],
        mpc_dt=r_cfg["mpc_dt"],
        cost_net_sizes=r_cfg["cost_net_sizes"],
        vx_max=vx_max,
        vy_max=vy_max,
        wz_max=wz_max,
        wheel_vel_max=wheel_vel_max,
        n_batch_max=1,
        activation=r_cfg.get("activation", training_config["policy"].get("cost_net_activation", "relu")),
        pos_offset_max=r_cfg.get("pos_offset_max", 2.0),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _load_agent_state(policy, checkpoint, agent_key="rover", component="policy")

    preprocessor = _load_preprocessor(checkpoint, agent_key="rover", obs_dim=obs_dim, device=device)

    policy.eval()
    return policy, preprocessor


def _load_agent_state(model, checkpoint: dict, agent_key: str, component: str = "policy"):
    """Load state dict from SKRL multi-agent checkpoint.

    SKRL MAPPO saves checkpoints as nested dicts:
      checkpoint[agent_key][component] = state_dict
    """
    if agent_key in checkpoint and component in checkpoint[agent_key]:
        state_dict = checkpoint[agent_key][component]
    elif component in checkpoint:
        state_dict = checkpoint[component]
    else:
        raise KeyError(
            f"Cannot find '{agent_key}/{component}' in checkpoint. "
            f"Keys: {list(checkpoint.keys())}"
        )

    model.load_state_dict(state_dict)


def _load_preprocessor(
    checkpoint: dict,
    agent_key: str,
    obs_dim: int,
    device: str = "cpu",
):
    """Load observation preprocessor from checkpoint.

    Returns RunningStandardScaler or PartialRunningStandardScaler, or None.
    """
    preprocessor_key = "observation_preprocessor"

    state_dict = None
    if agent_key in checkpoint and preprocessor_key in checkpoint[agent_key]:
        state_dict = checkpoint[agent_key][preprocessor_key]
    elif preprocessor_key in checkpoint:
        state_dict = checkpoint[preprocessor_key]

    if state_dict is None:
        return None

    # Determine preprocessor type from state dict keys
    if "skip_dims" in state_dict or "skip_mask" in state_dict or "_skip_indices" in state_dict:
        from crazyflie_rover_landing.preprocessors import PartialRunningStandardScaler
        preprocessor = PartialRunningStandardScaler(size=obs_dim, device=device)
    else:
        from skrl.resources.preprocessors.torch import RunningStandardScaler
        preprocessor = RunningStandardScaler(size=obs_dim, device=device)

    preprocessor.load_state_dict(state_dict)
    return preprocessor


# ─── MLP Policy Loading ──────────────────────────────────────────────────────


def load_drone_mlp_policy(
    checkpoint_path: Path,
    env_config: dict,
    training_config: dict,
    device: str = "cpu",
):
    """Load MLPGaussianPolicy for drone from training checkpoint.

    Returns:
        (policy, preprocessor) tuple.
    """
    from crazyflie_rover_landing.policies.mlp_policy import MLPGaussianPolicy
    from drone_models.core import load_params

    obs_dim = env_config["drone_obs_dim"]
    d_cfg = training_config["policy"]["drone"]
    env_section = training_config.get("environment", {})

    # Get physical params for thrust bounds
    drone_model = env_section.get("drone_model", env_config.get("drone_model", "cf21B_500"))
    so_rpy_params = load_params("so_rpy", drone_model)
    thrust_min = float(so_rpy_params["thrust_min"]) * 4
    thrust_max = float(so_rpy_params["thrust_max"]) * 4

    roll_pitch_max = d_cfg.get("roll_pitch_max", env_section.get("roll_pitch_max", 0.1))
    yaw_max = d_cfg.get("yaw_max", env_section.get("yaw_max", 0.001))

    observation_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )

    action_space = gymnasium.spaces.Box(
        low=np.array([-roll_pitch_max, -roll_pitch_max, -yaw_max, thrust_min]),
        high=np.array([roll_pitch_max, roll_pitch_max, yaw_max, thrust_max]),
        dtype=np.float32,
    )

    hidden_sizes = tuple(d_cfg.get("hidden_sizes", [256, 256]))
    activation = d_cfg.get("activation", training_config["policy"].get("cost_net_activation", "relu"))

    policy = MLPGaussianPolicy(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        hidden_sizes=hidden_sizes,
        activation=activation,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _load_agent_state(policy, checkpoint, agent_key="drone", component="policy")
    preprocessor = _load_preprocessor(checkpoint, agent_key="drone", obs_dim=obs_dim, device=device)

    policy.eval()
    return policy, preprocessor


def load_rover_mlp_policy(
    checkpoint_path: Path,
    env_config: dict,
    training_config: dict,
    device: str = "cpu",
):
    """Load MLPGaussianPolicy for rover from training checkpoint.

    Returns:
        (policy, preprocessor) tuple.
    """
    from crazyflie_rover_landing.policies.mlp_policy import MLPGaussianPolicy

    obs_dim = env_config["rover_obs_dim"]
    r_cfg = training_config["policy"]["rover"]
    env_section = training_config["environment"]

    observation_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )

    vx_max = env_section.get("rover_vx_max", 1.0)
    vy_max = env_section.get("rover_vy_max", 1.0)
    wz_max = env_section.get("rover_wz_max", 5.0)

    action_space = gymnasium.spaces.Box(
        low=np.array([-vx_max, -vy_max, -wz_max]),
        high=np.array([vx_max, vy_max, wz_max]),
        dtype=np.float32,
    )

    hidden_sizes = tuple(r_cfg.get("hidden_sizes", [256, 256]))
    activation = r_cfg.get("activation", training_config["policy"].get("cost_net_activation", "relu"))

    policy = MLPGaussianPolicy(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        hidden_sizes=hidden_sizes,
        activation=activation,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _load_agent_state(policy, checkpoint, agent_key="rover", component="policy")
    preprocessor = _load_preprocessor(checkpoint, agent_key="rover", obs_dim=obs_dim, device=device)

    policy.eval()
    return policy, preprocessor


# ─── Generic Inference ───────────────────────────────────────────────────────


@torch.no_grad()
def infer_drone_action(
    policy,
    preprocessor,
    observation: np.ndarray,
    mpc_state: np.ndarray = None,
    device: str = "cpu",
    deterministic: bool = True,
) -> np.ndarray:
    """Run drone policy inference.

    Works for both ACMPC (needs mpc_state) and MLP (ignores mpc_state).

    Args:
        policy: DroneACMPCGaussianPolicy or MLPGaussianPolicy (eval mode).
        preprocessor: Observation preprocessor (or None).
        observation: (29,) drone observation.
        mpc_state: (13,) drone MPC state. Required for ACMPC, ignored for MLP.
        device: Torch device.
        deterministic: If True, return mean actions; if False, sample from distribution.

    Returns:
        (4,) actions in [-1, 1] (normalized).
    """
    obs_t = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    if preprocessor is not None:
        obs_t = preprocessor(obs_t)

    inputs = {"observations": obs_t}
    if mpc_state is not None:
        inputs["mpc_state"] = torch.tensor(mpc_state, dtype=torch.float32, device=device).unsqueeze(0)

    if deterministic:
        mean_actions, _ = policy.compute(inputs, role="policy")
        return mean_actions.squeeze(0).cpu().numpy()
    else:
        actions, _, _ = policy.act(inputs, role="policy")
        return actions.squeeze(0).cpu().numpy()


@torch.no_grad()
def infer_rover_action(
    policy,
    preprocessor,
    observation: np.ndarray,
    mpc_state: np.ndarray = None,
    device: str = "cpu",
    deterministic: bool = True,
) -> np.ndarray:
    """Run rover policy inference.

    Works for both ACMPC (needs mpc_state) and MLP (ignores mpc_state).

    Args:
        policy: X3RoverACMPCGaussianPolicy or MLPGaussianPolicy (eval mode).
        preprocessor: Observation preprocessor (or None).
        observation: (15,) rover observation.
        mpc_state: (7,) rover MPC state. Required for ACMPC, ignored for MLP.
        device: Torch device.
        deterministic: If True, return mean actions; if False, sample from distribution.

    Returns:
        (3,) actions in [-1, 1] (normalized).
    """
    obs_t = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    if preprocessor is not None:
        obs_t = preprocessor(obs_t)

    inputs = {"observations": obs_t}
    if mpc_state is not None:
        inputs["mpc_state"] = torch.tensor(mpc_state, dtype=torch.float32, device=device).unsqueeze(0)

    if deterministic:
        mean_actions, _ = policy.compute(inputs, role="policy")
        return mean_actions.squeeze(0).cpu().numpy()
    else:
        actions, _, _ = policy.act(inputs, role="policy")
        return actions.squeeze(0).cpu().numpy()
