"""Action-rescaling wrappers for the drone-rover landing environment."""

from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces


@jax.jit
def _jit_rescale_actions(
    actions: jnp.ndarray,
    scale: jnp.ndarray,
    mean: jnp.ndarray,
) -> jnp.ndarray:
    """Rescale actions from [-1, 1] to physical bounds.

    Args:
        actions: Normalized actions, shape (N, action_dim).
        scale: (high - low) / 2 for each action dimension.
        mean:  (high + low) / 2 for each action dimension.

    Returns:
        Rescaled actions in physical units.
    """
    clipped = jnp.clip(actions, -1.0, 1.0)
    return clipped * scale + mean


class RescaleActionWrapper(gym.Wrapper):
    """Rescale per-agent actions from [-1, 1] to the environment's physical space.

    Handles heterogeneous action spaces: the drone and rover have different action
    dimensions and physical bounds. Each agent's actions are rescaled independently.

    Example:
        env = LandingEnv(cfg)
        env = RescaleActionWrapper(env)
        # env.action_space now has [-1, 1] bounds for all agents
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        # Build per-agent scale/mean and normalized action space
        normalized_spaces: dict[str, spaces.Box] = {}
        self._scale: dict[str, jnp.ndarray] = {}
        self._mean: dict[str, jnp.ndarray] = {}

        for agent, space in env.action_space.items():
            if not isinstance(space, spaces.Box):
                raise ValueError(
                    f"RescaleActionWrapper only supports Box action spaces, got {type(space)}"
                )
            normalized_spaces[agent] = spaces.Box(
                low=-np.ones_like(space.low),
                high=np.ones_like(space.high),
                dtype=space.dtype,
            )
            self._scale[agent] = jnp.array((space.high - space.low) / 2.0)
            self._mean[agent] = jnp.array((space.high + space.low) / 2.0)

        self.action_space = spaces.Dict(normalized_spaces)
        self.action_spaces = self.action_space  # PettingZoo alias

        # Forward multi-agent attributes
        self.possible_agents = env.possible_agents
        self.agents = env.agents
        self.num_agents = env.num_agents
        self.cfg = env.cfg

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[dict, dict, dict, dict, dict]:
        """Step the environment with per-agent rescaled actions."""
        rescaled = {
            agent: np.asarray(
                _jit_rescale_actions(
                    jnp.asarray(actions[agent]),
                    self._scale[agent],
                    self._mean[agent],
                )
            )
            for agent in self.possible_agents
            if agent in actions
        }
        return self.env.step(rescaled)

    def state(self) -> np.ndarray:
        """Forward global state for centralized critic (SKRL interface)."""
        return self.env.state()
