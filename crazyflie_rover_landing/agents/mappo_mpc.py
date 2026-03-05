"""MAPPO_MPC: MAPPO with per-agent MPC state passthrough.

Extended from crazyflie-mape-crazyflow MAPPO_MPC to support heterogeneous
agents with different MPC state dimensions.

The environment provides:
    info["mpc_state"] = {"drone": ndarray(N, 12), "rover": ndarray(N, 5)}

Each agent's raw MPC state is stored in memory and injected into the policy
via inputs["mpc_state"], bypassing all observation preprocessors.
"""

from __future__ import annotations

import itertools
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.multi_agents.torch.mappo import MAPPO
from skrl.multi_agents.torch.mappo.mappo import compute_gae
from skrl.resources.schedulers.torch import KLAdaptiveLR


class MAPPO_MPC(MAPPO):
    """MAPPO with heterogeneous per-agent MPC state support.

    Each agent can have a different MPC state dimension, specified via the
    `mpc_state_sizes` dict. Raw MPC states are stored in memory and injected
    into the policy forward pass without normalization.

    Args:
        mpc_state_sizes: Dict mapping agent UID to its MPC state dimension.
            e.g. {"drone": 12, "rover": 5}.
            If None or empty, behaves identically to vanilla MAPPO.
        **kwargs: All other arguments forwarded to MAPPO.__init__().
    """

    def __init__(
        self,
        *,
        mpc_state_sizes: dict[str, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._mpc_state_sizes: dict[str, int] = mpc_state_sizes or {}
        self._current_mpc_state: dict[str, torch.Tensor] = {}

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize memories, adding per-agent mpc_state tensors."""
        super().init(trainer_cfg=trainer_cfg)

        if self.memories:
            for uid in self.possible_agents:
                size = self._mpc_state_sizes.get(uid, 0)
                if size > 0:
                    self.memories[uid].create_tensor(
                        name="mpc_state", size=size, dtype=torch.float32
                    )
            if any(self._mpc_state_sizes.get(uid, 0) > 0 for uid in self.possible_agents):
                self._tensors_names.append("mpc_state")

    def act(
        self,
        observations: dict[str, torch.Tensor],
        states: dict[str, torch.Tensor | None],
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Compute actions, injecting cached mpc_state into each agent's inputs."""
        actions = {}
        log_prob = {}
        outputs = {}

        for uid in self.possible_agents:
            inputs = {
                "observations": self._observation_preprocessor[uid](observations[uid]),
                "states": self._state_preprocessor[uid](states[uid]),
            }
            if self._mpc_state_sizes.get(uid, 0) > 0 and uid in self._current_mpc_state:
                inputs["mpc_state"] = self._current_mpc_state[uid]

            if timestep < self.cfg.random_timesteps:
                actions[uid], outputs[uid] = self.policies[uid].random_act(inputs, role="policy")
            else:
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    actions[uid], outputs[uid] = self.policies[uid].act(inputs, role="policy")
                    log_prob[uid] = outputs[uid]["log_prob"]

        self._current_log_prob = log_prob
        return actions, outputs

    def record_transition(
        self,
        *,
        observations: dict[str, torch.Tensor],
        states: dict[str, torch.Tensor | None],
        actions: dict[str, torch.Tensor],
        rewards: dict[str, torch.Tensor],
        next_observations: dict[str, torch.Tensor],
        next_states: dict[str, torch.Tensor],
        terminated: dict[str, torch.Tensor],
        truncated: dict[str, torch.Tensor],
        infos: dict[str, Any],
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record transition, storing per-agent mpc_state in memory."""
        super(MAPPO, self).record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        if self.memories:
            self._current_next_observations = next_observations
            self._current_next_states = next_states

            for uid in self.possible_agents:
                if self.cfg.rewards_shaper is not None:
                    rewards[uid] = self.cfg.rewards_shaper(rewards[uid], timestep, timesteps)

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor[uid](observations[uid]),
                        "states": self._state_preprocessor[uid](states[uid]),
                    }
                    values, _ = self.values[uid].act(inputs, role="value")
                    values = self._value_preprocessor[uid](values, inverse=True)

                if self.cfg.time_limit_bootstrap[uid]:
                    rewards[uid] += self.cfg.discount_factor[uid] * values * truncated[uid]

                samples = {
                    "observations": observations[uid],
                    "states": states[uid],
                    "actions": actions[uid],
                    "rewards": rewards[uid],
                    "terminated": terminated[uid],
                    "log_prob": self._current_log_prob[uid],
                    "values": values,
                }
                if self._mpc_state_sizes.get(uid, 0) > 0 and uid in self._current_mpc_state:
                    samples["mpc_state"] = self._current_mpc_state[uid]
                self.memories[uid].add_samples(**samples)

        # Update mpc_state cache from infos["mpc_state"]
        if "mpc_state" in infos:
            for uid in self.possible_agents:
                if uid in infos["mpc_state"] and self._mpc_state_sizes.get(uid, 0) > 0:
                    self._current_mpc_state[uid] = torch.as_tensor(
                        infos["mpc_state"][uid], dtype=torch.float32, device=self.device
                    )

    def update(self, *, timestep: int, timesteps: int, uid: str) -> None:
        """Policy and value update with mpc_state support in sampled batches."""
        policy = self.policies[uid]
        value = self.values[uid]
        memory = self.memories[uid]
        has_mpc_state = self._mpc_state_sizes.get(uid, 0) > 0

        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor[uid](self._current_next_observations[uid]),
                "states": self._state_preprocessor[uid](self._current_next_states[uid]),
            }
            value.enable_training_mode(False)
            last_values, _ = value.act(inputs, role="value")
            value.enable_training_mode(True)
            last_values = self._value_preprocessor[uid](last_values, inverse=True)

        values = memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=memory.get_tensor_by_name("rewards"),
            terminated=memory.get_tensor_by_name("terminated"),
            values=values,
            next_values=last_values,
            discount_factor=self.cfg.discount_factor[uid],
            lambda_coefficient=self.cfg.lambda_[uid],
        )

        memory.set_tensor_by_name("values", self._value_preprocessor[uid](values, train=True))
        memory.set_tensor_by_name("returns", self._value_preprocessor[uid](returns, train=True))
        memory.set_tensor_by_name("advantages", advantages)

        sampled_batches = memory.sample_all(names=self._tensors_names, mini_batches=self.cfg.mini_batches[uid])

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        for epoch in range(self.cfg.learning_epochs[uid]):
            kl_divergences = []

            for batch in sampled_batches:
                if has_mpc_state:
                    *base_tensors, sampled_mpc_state = batch
                else:
                    base_tensors = batch
                    sampled_mpc_state = None

                (
                    sampled_observations,
                    sampled_states,
                    sampled_actions,
                    sampled_log_prob,
                    sampled_values,
                    sampled_returns,
                    sampled_advantages,
                ) = base_tensors

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor[uid](sampled_observations, train=not epoch),
                        "states": self._state_preprocessor[uid](sampled_states, train=not epoch),
                    }
                    if sampled_mpc_state is not None:
                        inputs["mpc_state"] = sampled_mpc_state

                    _, outputs = policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    if self.cfg.kl_threshold[uid] and kl_divergence > self.cfg.kl_threshold[uid]:
                        break

                    if self.cfg.entropy_loss_scale[uid]:
                        entropy_loss = -self.cfg.entropy_loss_scale[uid] * policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip[uid], 1.0 + self.cfg.ratio_clip[uid]
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    predicted_values, _ = value.act(inputs, role="value")
                    if self.cfg.value_clip[uid] > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values,
                            min=-self.cfg.value_clip[uid],
                            max=self.cfg.value_clip[uid],
                        )
                    value_loss = self.cfg.value_loss_scale[uid] * F.mse_loss(sampled_returns, predicted_values)

                self.optimizers[uid].zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    policy.reduce_parameters()
                    if policy is not value:
                        value.reduce_parameters()

                if self.cfg.grad_norm_clip[uid] > 0:
                    self.scaler.unscale_(self.optimizers[uid])
                    if policy is value:
                        nn.utils.clip_grad_norm_(policy.parameters(), self.cfg.grad_norm_clip[uid])
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(policy.parameters(), value.parameters()),
                            self.cfg.grad_norm_clip[uid],
                        )

                self.scaler.step(self.optimizers[uid])
                self.scaler.update()

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale[uid]:
                    cumulative_entropy_loss += entropy_loss.item()

            if self.schedulers[uid]:
                if isinstance(self.schedulers[uid], KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.schedulers[uid].step(kl.item())
                else:
                    self.schedulers[uid].step()

        n_updates = self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]
        self.track_data(f"Loss / Policy loss ({uid})", cumulative_policy_loss / n_updates)
        self.track_data(f"Loss / Value loss ({uid})", cumulative_value_loss / n_updates)
        if self.cfg.entropy_loss_scale[uid]:
            self.track_data(f"Loss / Entropy loss ({uid})", cumulative_entropy_loss / n_updates)

        self.track_data(f"Policy / Standard deviation ({uid})",
                        policy.distribution(role="policy").stddev.mean().item())
        if self.schedulers[uid]:
            self.track_data(f"Learning / Learning rate ({uid})", self.schedulers[uid].get_last_lr()[0])
