"""SKRL Gaussian policy for the Crazyflie drone using LEAP-C (LINEAR_LS).

Adapted from crazyflie-mape-crazyflow LeapCSharedGaussianPolicyLinearLS.
Default drone model changed to cf2x_T350 and planner updated to DronePlanner.
"""

from typing import Mapping, Optional, Sequence, Tuple, Union

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model

from crazyflie_rover_landing.leap_c.drone_planner import DronePlanner, DronePlannerConfig
from crazyflie_rover_landing.leap_c.drone_ocp_linear_ls import NX, NU


def _get_activation(name: str) -> type[nn.Module]:
    activations = {
        "relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU,
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(activations.keys())}")
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


class DroneMPCLayerLinearLS(nn.Module):
    """Neural network layer wrapping DronePlanner with LINEAR_LS cost.

    obs (B, obs_dim) → W + y_ref → MPC → u0 (B, 4) → normalized in [-1, 1]

    State [12D]: [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]
    Control [4D]: [roll_cmd, pitch_cmd, yaw_cmd, thrust]
    """

    def __init__(
        self,
        observation_dim: int,
        mpc_horizon: int = 2,
        mpc_dt: float = 0.01,
        cost_net_sizes: Sequence[int] = (256, 256),
        device: Union[str, torch.device] = "cpu",
        roll_pitch_max: float = 0.5,
        yaw_max: float = 0.5,
        thrust_min: float = 1.23,
        thrust_max: float = 3.68,
        mass: Optional[float] = None,
        gravity: Optional[float] = None,
        drone_model: str = "cf2x_T350",
        n_batch_max: int = 4096,
        num_threads: int = 8,
        velocity_max: Optional[float] = None,
        activation: str = "relu",
        pos_offset_max: float = 2.0,
    ):
        super().__init__()
        self.device = device
        self.mpc_horizon = mpc_horizon

        planner_cfg = DronePlannerConfig(
            N_horizon=mpc_horizon,
            dt=mpc_dt,
            param_interface="global",
            n_batch_max=n_batch_max,
            num_threads=num_threads,
            drone_model=drone_model,
            velocity_max=velocity_max,
            roll_pitch_max=roll_pitch_max,
            yaw_max=yaw_max,
            pos_offset_max=pos_offset_max,
            thrust_min=thrust_min,
            thrust_max=thrust_max,
            mass=mass,
            gravity=gravity,
        )
        self.planner = DronePlanner(cfg=planner_cfg)

        from drone_models.core import load_params as _lp
        drone_params = _lp("so_rpy", drone_model)
        self.mass = mass if mass is not None else float(drone_params["mass"])
        self.gravity = gravity if gravity is not None else float(np.abs(drone_params["gravity_vec"][2]))
        hover_thrust = self.mass * self.gravity

        # Action normalization buffers
        thrust_mean = (thrust_min + thrust_max) / 2.0
        thrust_scale = (thrust_max - thrust_min) / 2.0
        self.register_buffer("action_mean", torch.tensor([0., 0., 0., thrust_mean], dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor([roll_pitch_max, roll_pitch_max, yaw_max, thrust_scale], dtype=torch.float32))

        # Weight log-scale bounds
        self.register_buffer("w_state_min_log", torch.tensor([-1., -1., -1., -2., -2., -2., -1., -1., -1., -1., -1., -1.]))
        self.register_buffer("w_state_max_log", torch.tensor([2., 2., 2., 1., 1., 1., 2., 2., 2., 1., 1., 1.]))
        self.register_buffer("w_ctrl_min_log", torch.tensor([-1., -1., -1., -1.]))
        self.register_buffer("w_ctrl_max_log", torch.tensor([1., 1., 1., 1.]))

        # Reference linear-scale bounds
        self.register_buffer("pos_offset_min", torch.tensor([-pos_offset_max, -pos_offset_max, -pos_offset_max]))
        self.register_buffer("pos_offset_max_buf", torch.tensor([pos_offset_max, pos_offset_max, pos_offset_max]))
        self.register_buffer("yref_state_min", torch.tensor([0., 0., 0., -roll_pitch_max, -roll_pitch_max, -yaw_max, -5., -5., -5., -10., -10., -10.]))
        self.register_buffer("yref_state_max", torch.tensor([0., 0., 0., roll_pitch_max, roll_pitch_max, yaw_max, 5., 5., 5., 10., 10., 10.]))
        self.register_buffer("yref_ctrl_min", torch.tensor([-roll_pitch_max, -roll_pitch_max, -yaw_max, thrust_min]))
        self.register_buffer("yref_ctrl_max", torch.tensor([roll_pitch_max, roll_pitch_max, yaw_max, thrust_max]))
        self.register_buffer("hover_thrust_buf", torch.tensor(hover_thrust, dtype=torch.float32))
        self.register_buffer("thrust_min_buf", torch.tensor(thrust_min, dtype=torch.float32))
        self.register_buffer("thrust_max_buf", torch.tensor(thrust_max, dtype=torch.float32))

        # param_dim = w_state(12) + w_ctrl(4) + yref_state(12) + yref_ctrl(4) = 32
        self.param_dim = self.planner.get_learnable_param_dim()

        self.cost_net = _build_mlp(
            observation_dim, cost_net_sizes, self.param_dim,
            activation=activation, output_activation=nn.Sigmoid(),
        )

    def forward(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: (B, obs_dim) observations (potentially normalized).
            state: (B, 12) raw MPC state [pos, rpy, vel, drpy].

        Returns:
            (B, 4) normalized action in [-1, 1].
        """
        raw = self.cost_net(obs)
        params = self._scale_params(raw, obs.shape[0], state)
        _, u0, _, _, _ = self.planner(obs=state, param=params)
        return (u0 - self.action_mean) / self.action_scale

    def _scale_params(
        self, net_out: torch.Tensor, batch_size: int, state: torch.Tensor
    ) -> torch.Tensor:
        """Scale [0,1] network output to MPC parameters (global interface)."""
        # Global interface: single set of weights/refs shared across all stages
        # Layout: w_state(12) + w_ctrl(4) + yref_state(12) + yref_ctrl(4)
        w_state_raw = net_out[:, :NX]
        w_ctrl_raw = net_out[:, NX:NX + NU]
        yref_state_raw = net_out[:, NX + NU:NX + NU + NX]
        yref_ctrl_raw = net_out[:, NX + NU + NX:NX + NU + NX + NU]

        # Log-scale weights
        log_w_state = self.w_state_min_log + w_state_raw * (self.w_state_max_log - self.w_state_min_log)
        W_state = torch.pow(10., log_w_state)
        log_w_ctrl = self.w_ctrl_min_log + w_ctrl_raw * (self.w_ctrl_max_log - self.w_ctrl_min_log)
        W_ctrl = torch.pow(10., log_w_ctrl)

        # Position reference: relative offset from current position
        pos_offset = self.pos_offset_min + yref_state_raw[:, :3] * (self.pos_offset_max_buf - self.pos_offset_min)
        current_pos = state[:, :3]
        yref_pos = current_pos + pos_offset

        # Other state references (absolute)
        yref_other = self.yref_state_min[3:] + yref_state_raw[:, 3:] * (self.yref_state_max[3:] - self.yref_state_min[3:])
        yref_state = torch.cat([yref_pos, yref_other], dim=-1)

        # Control references
        yref_ctrl = self.yref_ctrl_min + yref_ctrl_raw * (self.yref_ctrl_max - self.yref_ctrl_min)
        # Thrust centered on hover (piecewise linear)
        thrust_raw = yref_ctrl_raw[:, 3]
        thrust_below = self.thrust_min_buf + 2.0 * thrust_raw * (self.hover_thrust_buf - self.thrust_min_buf)
        thrust_above = self.hover_thrust_buf + 2.0 * (thrust_raw - 0.5) * (self.thrust_max_buf - self.hover_thrust_buf)
        yref_ctrl = yref_ctrl.clone()
        yref_ctrl[:, 3] = torch.where(thrust_raw <= 0.5, thrust_below, thrust_above)

        return torch.cat([W_state, W_ctrl, yref_state, yref_ctrl], dim=-1)


class DroneACMPCGaussianPolicy(GaussianMixin, Model):
    """SKRL Gaussian policy for drone ACMPC with LINEAR_LS cost."""

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
        mpc_horizon: int = 2,
        mpc_dt: float = 0.01,
        cost_net_sizes: Sequence[int] = (256, 256),
        roll_pitch_max: float = 0.5,
        yaw_max: float = 0.5,
        thrust_min: float = 1.23,
        thrust_max: float = 3.68,
        mass: Optional[float] = None,
        gravity: Optional[float] = None,
        drone_model: str = "cf2x_T350",
        n_batch_max: int = 4096,
        num_threads: int = 8,
        velocity_max: Optional[float] = None,
        activation: str = "relu",
        pos_offset_max: float = 2.0,
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)

        self._g_clip_log_std = clip_log_std
        self._g_min_log_std = min_log_std
        self._g_max_log_std = max_log_std

        obs_dim = gymnasium.spaces.flatdim(observation_space)
        action_dim = gymnasium.spaces.flatdim(action_space)

        self.mpc_layer = DroneMPCLayerLinearLS(
            observation_dim=obs_dim,
            mpc_horizon=mpc_horizon,
            mpc_dt=mpc_dt,
            cost_net_sizes=cost_net_sizes,
            device=device,
            roll_pitch_max=roll_pitch_max,
            yaw_max=yaw_max,
            thrust_min=thrust_min,
            thrust_max=thrust_max,
            mass=mass,
            gravity=gravity,
            drone_model=drone_model,
            n_batch_max=n_batch_max,
            num_threads=num_threads,
            velocity_max=velocity_max,
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
            # Fallback: extract pos(3) + rpy(3) + vel(3) + drpy(3) from observation
            state = obs[:, :12]

        mean_actions = self.mpc_layer(obs, state)

        log_std = self.log_std_parameter
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_min_log_std, self._g_max_log_std)
        self._log_std = log_std
        self._num_samples = mean_actions.shape[0]

        return mean_actions, {"log_std": log_std}
