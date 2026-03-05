"""Simple MLP (feedforward) Gaussian policy for SKRL MAPPO.

Maps observations directly to actions via a feedforward neural network,
without any MPC planning layer. Used as a baseline comparison against ACMPC.

Action output is bounded to [-1, 1] via Tanh, then rescaled by
RescaleActionWrapper to physical bounds.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple, Union

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model


def _get_activation(name: str) -> type[nn.Module]:
    activations = {
        "relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU,
    }
    if name.lower() not in activations:
        raise ValueError(
            f"Unknown activation '{name}'. Available: {list(activations.keys())}"
        )
    return activations[name.lower()]


class MLPGaussianPolicy(GaussianMixin, Model):
    """SKRL Gaussian policy using a feedforward neural network.

    Architecture:
        obs → [Linear → act → ... → Linear → Tanh] → mean_actions ∈ [-1, 1]

    A learnable diagonal log_std parameter controls exploration noise.
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        device: Union[str, torch.device] = "cpu",
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: Union[float, Sequence[float]] = 0.0,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
        **kwargs,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        obs_dim = gymnasium.spaces.flatdim(observation_space)
        action_dim = gymnasium.spaces.flatdim(action_space)
        act_cls = _get_activation(activation)

        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        layers.append(nn.Tanh())

        self.policy_net = nn.Sequential(*layers)

        # Log std — scalar broadcast or per-action-dim array
        if isinstance(initial_log_std, (list, tuple, np.ndarray)):
            init_tensor = torch.tensor(
                initial_log_std, dtype=torch.float32, device=device
            )
            if init_tensor.shape[0] != action_dim:
                raise ValueError(
                    f"initial_log_std has size {init_tensor.shape[0]}, "
                    f"expected {action_dim}"
                )
        else:
            init_tensor = torch.full(
                (action_dim,), initial_log_std, dtype=torch.float32, device=device
            )
        self.log_std_parameter = nn.Parameter(init_tensor)

    def compute(
        self,
        inputs: Mapping[str, torch.Tensor],
        role: str = "",
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        obs = inputs["observations"]
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        mean_actions = self.policy_net(obs)
        return mean_actions, {"log_std": self.log_std_parameter}
