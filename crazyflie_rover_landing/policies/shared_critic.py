"""Centralized value network for MAPPO (CTDE).

Ported from crazyflie-mape-crazyflow shared_critic.py without changes.
"""

from typing import Mapping, Sequence, Tuple, Union

import gymnasium
import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, Model


def _get_activation(name: str) -> type[nn.Module]:
    activations = {
        "relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU,
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation '{name}'.")
    return activations[name.lower()]


def _build_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    activation: str = "relu",
) -> nn.Sequential:
    act_cls = _get_activation(activation)
    layers: list[nn.Module] = []
    prev = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(prev, size))
        layers.append(act_cls())
        prev = size
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class SharedCritic(DeterministicMixin, Model):
    """SKRL centralized value network for MAPPO.

    Input: shared state (drone + rover full state, 22D).
    Output: scalar state value.
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        device: Union[str, torch.device] = "cpu",
        clip_actions: bool = False,
        value_net_sizes: Sequence[int] = (256, 256),
        activation: str = "relu",
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        obs_dim = gymnasium.spaces.flatdim(observation_space)

        self.value_net = _build_mlp(obs_dim, value_net_sizes, 1, activation=activation)

    def compute(
        self,
        inputs: Mapping[str, torch.Tensor],
        role: str = "",
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        if "states" in inputs:
            state = inputs["states"]
        else:
            state = inputs.get("shared_states", inputs.get("observations", None))

        if state is None:
            raise ValueError("No state found in inputs for SharedCritic.")

        return self.value_net(state), {}
