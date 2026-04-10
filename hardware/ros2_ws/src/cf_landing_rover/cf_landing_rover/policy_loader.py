"""Rover-specific policy loader for X3 Docker deployment.

Loads the rover policy (ACMPC or MLP) from a training checkpoint.
"""

import json
import os
from pathlib import Path

import gymnasium
import numpy as np
import torch


def get_models_dir() -> Path:
    """Find the models directory."""
    env_dir = os.environ.get("CF_LANDING_MODELS_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir():
            return p

    this_file = Path(__file__).resolve()

    if "/install/" in str(this_file):
        src_models = this_file
        while src_models.name != "install":
            src_models = src_models.parent
        src_models = src_models.parent / "src" / "cf_landing_rover" / "models"
        if src_models.is_dir():
            return src_models

    rel = this_file.parent.parent / "models"
    if rel.is_dir():
        return rel

    raise FileNotFoundError("Cannot find models directory.")


def find_checkpoint(models_dir: Path, policy_type: str = "acmpc") -> Path:
    """Find a .pt checkpoint in models/{policy_type}/."""
    policy_dir = models_dir / policy_type
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"No {policy_type}/ directory in {models_dir}")

    for name in ["best_agent.pt", "final_checkpoint.pt"]:
        p = policy_dir / name
        if p.is_file():
            return p

    best_agents = sorted(policy_dir.glob("best_agent_*.pt"), reverse=True)
    if best_agents:
        return best_agents[0]

    pt_files = list(policy_dir.glob("*.pt"))
    if pt_files:
        return pt_files[0]

    raise FileNotFoundError(f"No .pt checkpoint found in {policy_dir}")


def load_env_config(models_dir: Path, policy_type: str = "acmpc") -> dict:
    config_path = models_dir / policy_type / "environment_config.json"
    with open(config_path) as f:
        return json.load(f)


def _load_preprocessor(checkpoint: dict, obs_dim: int, device: str = "cpu"):
    """Load observation preprocessor from checkpoint."""
    preprocessor_key = "observation_preprocessor"
    state_dict = None
    if "rover" in checkpoint and preprocessor_key in checkpoint["rover"]:
        state_dict = checkpoint["rover"][preprocessor_key]
    elif preprocessor_key in checkpoint:
        state_dict = checkpoint[preprocessor_key]

    if state_dict is None:
        return None

    if "skip_dims" in state_dict or "skip_mask" in state_dict:
        from crazyflie_rover_landing.preprocessors import PartialRunningStandardScaler
        preprocessor = PartialRunningStandardScaler(size=obs_dim, device=device)
    else:
        from skrl.resources.preprocessors.torch import RunningStandardScaler
        preprocessor = RunningStandardScaler(size=obs_dim, device=device)
    preprocessor.load_state_dict(state_dict)
    return preprocessor


def _load_policy_state(policy, checkpoint: dict):
    """Load rover policy state dict from checkpoint."""
    if "rover" in checkpoint and "policy" in checkpoint["rover"]:
        policy.load_state_dict(checkpoint["rover"]["policy"])
    elif "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    else:
        raise KeyError(f"Cannot find rover policy in checkpoint. Keys: {list(checkpoint.keys())}")


def load_rover_policy(checkpoint_path: Path, env_config: dict, training_config: dict,
                      policy_type: str = "acmpc", device: str = "cpu"):
    """Load rover policy from checkpoint.

    Args:
        policy_type: "acmpc" or "mlp".
    """
    if policy_type == "acmpc":
        return _load_rover_acmpc(checkpoint_path, env_config, training_config, device)
    elif policy_type == "mlp":
        return _load_rover_mlp(checkpoint_path, env_config, training_config, device)
    else:
        raise ValueError(f"Unknown policy_type: {policy_type}")


def _load_rover_acmpc(checkpoint_path, env_config, training_config, device):
    from crazyflie_rover_landing.leap_c.x3_rover_policy_linear_ls import X3RoverACMPCGaussianPolicy

    obs_dim = env_config["rover_obs_dim"]
    r_cfg = training_config["policy"]["rover"]
    env_section = training_config["environment"]

    observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    vx_max = env_section.get("rover_vx_max", 1.0)
    vy_max = env_section.get("rover_vy_max", 1.0)
    wz_max = env_section.get("rover_wz_max", 5.0)
    action_space = gymnasium.spaces.Box(
        low=np.array([-vx_max, -vy_max, -wz_max]),
        high=np.array([vx_max, vy_max, wz_max]),
        dtype=np.float32,
    )

    policy = X3RoverACMPCGaussianPolicy(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        mpc_horizon=r_cfg["mpc_horizon"],
        mpc_dt=r_cfg["mpc_dt"],
        cost_net_sizes=r_cfg["cost_net_sizes"],
        vx_max=vx_max, vy_max=vy_max, wz_max=wz_max,
        n_batch_max=1,
        activation=r_cfg.get("activation", training_config["policy"].get("cost_net_activation", "relu")),
        pos_offset_max=r_cfg.get("pos_offset_max", 2.0),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _load_policy_state(policy, checkpoint)
    preprocessor = _load_preprocessor(checkpoint, obs_dim, device)
    policy.eval()
    return policy, preprocessor


def _load_rover_mlp(checkpoint_path, env_config, training_config, device):
    from crazyflie_rover_landing.policies.mlp_policy import MLPGaussianPolicy

    obs_dim = env_config["rover_obs_dim"]
    r_cfg = training_config["policy"]["rover"]
    env_section = training_config["environment"]

    observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
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
    _load_policy_state(policy, checkpoint)
    preprocessor = _load_preprocessor(checkpoint, obs_dim, device)
    policy.eval()
    return policy, preprocessor


@torch.no_grad()
def infer_rover_action(policy, preprocessor, observation: np.ndarray,
                       mpc_state: np.ndarray = None, device: str = "cpu") -> np.ndarray:
    """Run rover policy inference (deterministic). Works for both ACMPC and MLP."""
    obs_t = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    if preprocessor is not None:
        obs_t = preprocessor(obs_t)

    inputs = {"observations": obs_t}
    if mpc_state is not None:
        inputs["mpc_state"] = torch.tensor(mpc_state, dtype=torch.float32, device=device).unsqueeze(0)

    mean_actions, _ = policy.act(inputs, role="policy")
    return mean_actions.squeeze(0).cpu().numpy()
