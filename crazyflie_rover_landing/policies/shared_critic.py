"""Centralized value network for MAPPO (CTDE).

Single shared trunk with per-agent output heads. Both agents share the
same feature extractor; each gets its own scalar value output.
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


class DualHeadCriticBackbone(nn.Module):
    """Shared trunk with 2 value heads (drone, rover).

    Architecture:
        shared_state (22D) → trunk (hidden layers) → features
        features → drone_head → scalar
        features → rover_head → scalar
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: str = "relu",
    ):
        super().__init__()
        act_cls = _get_activation(activation)

        # Build trunk: obs_dim → hidden layers → last hidden dim
        layers: list[nn.Module] = []
        prev = obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(act_cls())
            prev = size
        self.trunk = nn.Sequential(*layers)

        # Per-agent output heads
        self.drone_head = nn.Linear(prev, 1)
        self.rover_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor, head_index: int) -> torch.Tensor:
        """Run trunk and select one head.

        Args:
            x: (B, obs_dim) shared state.
            head_index: 0 for drone, 1 for rover.

        Returns:
            (B, 1) scalar value.
        """
        features = self.trunk(x)
        if head_index == 0:
            return self.drone_head(features)
        else:
            return self.rover_head(features)


class CriticHead(DeterministicMixin, Model):
    """SKRL-compatible wrapper that selects one output from a shared backbone.

    Input: shared state (drone + rover full state, 22D).
    Output: scalar state value for the assigned agent.
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        backbone: DualHeadCriticBackbone,
        head_index: int,
        device: Union[str, torch.device] = "cpu",
        clip_actions: bool = False,
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.backbone = backbone
        self.head_index = head_index

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
            raise ValueError("No state found in inputs for CriticHead.")

        return self.backbone(state, self.head_index), {}
