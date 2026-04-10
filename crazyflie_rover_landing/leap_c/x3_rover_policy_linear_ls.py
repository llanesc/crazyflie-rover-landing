"""SKRL Gaussian policy for the RosMaster X3 mecanum rover using LEAP-C (LINEAR_LS).

obs (B, obs_dim) → W + y_ref → MPC → u0 (B, 3) → normalized in [-1, 1]

State [7D]:   [x, y, cos(θ), sin(θ), vx, vy, ωz]
Control [3D]: [vx_cmd, vy_cmd, ωz_cmd]  (body-frame velocity commands)
"""

from typing import Mapping, Optional, Sequence, Tuple, Union

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model

from crazyflie_rover_landing.leap_c.x3_rover_planner import X3RoverPlanner, X3RoverPlannerConfig
from crazyflie_rover_landing.leap_c.x3_rover_ocp_linear_ls import (
    NX_ROVER,
    NU_ROVER,
    _VX_MAX,
    _VY_MAX,
    _WZ_MAX,
)


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
    output_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    act_cls = _get_activation(activation)
    layers: list[nn.Module] = []
    prev = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(prev, size))
        layers.append(act_cls())
        prev = size
    layers.append(nn.Linear(prev, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class X3RoverMPCLayerLinearLS(nn.Module):
    """Neural network layer wrapping X3RoverPlanner with LINEAR_LS cost.

    obs (B, obs_dim) → W + y_ref → MPC → u0 (B, 3) → normalized in [-1, 1]
    """

    def __init__(
        self,
        observation_dim: int,
        mpc_horizon: int = 4,
        mpc_dt: float = 0.1,
        cost_net_sizes: Sequence[int] = (256, 256),
        device: Union[str, torch.device] = "cpu",
        vx_max: float = _VX_MAX,
        vy_max: float = _VY_MAX,
        wz_max: float = _WZ_MAX,
        n_batch_max: int = 128,
        num_threads: int = 8,
        activation: str = "relu",
        pos_offset_max: float = 2.0,
    ):
        super().__init__()
        self.device = device

        planner_cfg = X3RoverPlannerConfig(
            N_horizon=mpc_horizon,
            dt=mpc_dt,
            n_batch_max=n_batch_max,
            num_threads=num_threads,
            pos_offset_max=pos_offset_max,
            vx_max=vx_max,
            vy_max=vy_max,
            wz_max=wz_max,
        )
        self.planner = X3RoverPlanner(cfg=planner_cfg)

        # Action normalization: each axis scaled by its max
        self.register_buffer("action_mean",  torch.zeros(3, dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor([vx_max, vy_max, wz_max], dtype=torch.float32))

        # Weight log-scale bounds — state: [x, y, c, s, vx, vy, ωz]
        self.register_buffer("w_state_min_log", torch.tensor([-2., -2., -2., -2., -2., -2., -2.]))
        self.register_buffer("w_state_max_log", torch.tensor([ 2.,  2.,  1.,  1.,  1.,  1.,  1.]))
        # Control: [vx_cmd, vy_cmd, ωz_cmd]
        self.register_buffer("w_ctrl_min_log", torch.tensor([-2., -2., -2.]))
        self.register_buffer("w_ctrl_max_log", torch.tensor([ 1.,  1.,  1.]))

        # Reference bounds
        self.register_buffer("pos_offset_min_buf", torch.tensor([-pos_offset_max, -pos_offset_max]))
        self.register_buffer("pos_offset_max_buf", torch.tensor([ pos_offset_max,  pos_offset_max]))
        self.register_buffer("yref_cs_min", torch.tensor([-1.0, -1.0]))
        self.register_buffer("yref_cs_max", torch.tensor([ 1.0,  1.0]))
        self.register_buffer("yref_vel_min", torch.tensor([-vx_max, -vy_max, -wz_max]))
        self.register_buffer("yref_vel_max", torch.tensor([ vx_max,  vy_max,  wz_max]))
        self.register_buffer("yref_ctrl_min", torch.tensor([-vx_max, -vy_max, -wz_max]))
        self.register_buffer("yref_ctrl_max", torch.tensor([ vx_max,  vy_max,  wz_max]))

        # param_dim = w_state(7) + w_ctrl(3) + yref_state(7) + yref_ctrl(3) = 20
        self.param_dim = self.planner.get_learnable_param_dim()

        self.cost_net = _build_mlp(
            observation_dim, cost_net_sizes, self.param_dim,
            activation=activation, output_activation=nn.Sigmoid(),
        )

    def forward(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs:   (B, obs_dim) observations.
            state: (B, 7) raw rover MPC state [x, y, c, s, vx, vy, ωz].

        Returns:
            (B, 3) normalized action in [-1, 1].
        """
        raw = self.cost_net(obs)
        params = self._scale_params(raw, state)
        _, u0, _, _, _ = self.planner(obs=state, param=params)
        return (u0 - self.action_mean) / self.action_scale

    def _scale_params(self, net_out: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Scale [0,1] network output to X3 rover MPC parameters.

        Layout: w_state(7) + w_ctrl(3) + yref_state(7) + yref_ctrl(3) = 20
        """
        w_state_raw    = net_out[:, :NX_ROVER]
        w_ctrl_raw     = net_out[:, NX_ROVER:NX_ROVER + NU_ROVER]
        yref_state_raw = net_out[:, NX_ROVER + NU_ROVER:2 * NX_ROVER + NU_ROVER]
        yref_ctrl_raw  = net_out[:, 2 * NX_ROVER + NU_ROVER:]

        # Log-scale weights
        log_w_state = self.w_state_min_log + w_state_raw * (self.w_state_max_log - self.w_state_min_log)
        W_state = torch.pow(10., log_w_state)
        log_w_ctrl = self.w_ctrl_min_log + w_ctrl_raw * (self.w_ctrl_max_log - self.w_ctrl_min_log)
        W_ctrl = torch.pow(10., log_w_ctrl)

        # Position reference: relative offset from current rover (x, y)
        pos_offset = self.pos_offset_min_buf + yref_state_raw[:, :2] * (self.pos_offset_max_buf - self.pos_offset_min_buf)
        current_pos = state[:, :2]
        yref_pos = current_pos + pos_offset

        # Heading reference
        yref_cs = self.yref_cs_min + yref_state_raw[:, 2:4] * (self.yref_cs_max - self.yref_cs_min)

        # Body velocity references [vx, vy, ωz]
        yref_vel = self.yref_vel_min + yref_state_raw[:, 4:7] * (self.yref_vel_max - self.yref_vel_min)

        yref_state = torch.cat([yref_pos, yref_cs, yref_vel], dim=-1)

        # Control references
        yref_ctrl = self.yref_ctrl_min + yref_ctrl_raw * (self.yref_ctrl_max - self.yref_ctrl_min)

        return torch.cat([W_state, W_ctrl, yref_state, yref_ctrl], dim=-1)


class X3RoverACMPCGaussianPolicy(GaussianMixin, Model):
    """SKRL Gaussian policy for X3 rover ACMPC with LINEAR_LS cost."""

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        device: Union[str, torch.device] = "cpu",
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: Union[float, Sequence[float]] = -1.2,
        mpc_horizon: int = 4,
        mpc_dt: float = 0.1,
        cost_net_sizes: Sequence[int] = (256, 256),
        vx_max: float = _VX_MAX,
        vy_max: float = _VY_MAX,
        wz_max: float = _WZ_MAX,
        n_batch_max: int = 128,
        num_threads: int = 8,
        activation: str = "relu",
        pos_offset_max: float = 2.0,
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)

        self._g_clip_log_std = clip_log_std
        self._g_min_log_std  = min_log_std
        self._g_max_log_std  = max_log_std

        obs_dim    = gymnasium.spaces.flatdim(observation_space)
        action_dim = gymnasium.spaces.flatdim(action_space)

        self.mpc_layer = X3RoverMPCLayerLinearLS(
            observation_dim=obs_dim,
            mpc_horizon=mpc_horizon,
            mpc_dt=mpc_dt,
            cost_net_sizes=cost_net_sizes,
            device=device,
            vx_max=vx_max,
            vy_max=vy_max,
            wz_max=wz_max,
            n_batch_max=n_batch_max,
            num_threads=num_threads,
            activation=activation,
            pos_offset_max=pos_offset_max,
        )

        if isinstance(initial_log_std, (list, tuple, np.ndarray)):
            log_std_tensor = torch.tensor(initial_log_std, dtype=torch.float32, device=device)
        else:
            log_std_tensor = torch.full((action_dim,), initial_log_std, dtype=torch.float32, device=device)
        self.log_std_parameter = nn.Parameter(log_std_tensor)

    def compute(
        self,
        inputs: Mapping[str, torch.Tensor],
        role: str = "",
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        obs = inputs["observations"]
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        if "mpc_state" in inputs:
            state = inputs["mpc_state"]
        else:
            state = obs[:, :NX_ROVER]

        mean_actions = self.mpc_layer(obs, state)

        log_std = self.log_std_parameter
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_min_log_std, self._g_max_log_std)
        self._log_std = log_std
        self._num_samples = mean_actions.shape[0]

        return mean_actions, {"log_std": log_std}
