"""Differentiable MPC planner for the TurtleBot3 Burger differential-drive rover (LINEAR_LS).

State (6D):   [x, y, cos(θ), sin(θ), v_L, v_R]
Control (2D): [ω_L_cmd, ω_R_cmd]  — wheel angular velocity commands [rad/s]
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from acados_template.acados_ocp import AcadosOcp
from acados_template.acados_ocp_iterate import AcadosOcpFlattenedIterate
from leap_c.ocp.acados.data import AcadosOcpSolverInput
from leap_c.ocp.acados.initializer import AcadosDiffMpcInitializer
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch

from crazyflie_rover_landing.leap_c.rover_ocp_linear_ls import (
    NX_ROVER,
    NU_ROVER,
    create_rover_params_linear_ls,
    export_rover_ocp_linear_ls,
    get_rover_learnable_param_dim,
    _WHEEL_VEL_MAX,
    _WHEEL_LIN_VEL_MAX,
)


class RoverStationaryInitializer(AcadosDiffMpcInitializer):
    """Warm-start: state = x0 repeated, control = zero (wheels at rest)."""

    def __init__(self, ocp: AcadosOcp):
        self.default_iterate = ocp.create_default_initial_iterate().flatten()
        self.N  = ocp.solver_options.N_horizon
        self.nx = ocp.dims.nx
        self.nu = ocp.dims.nu
        self._zero_u_tiled = np.zeros(self.N * self.nu)

    def single_iterate(self, solver_input: AcadosOcpSolverInput) -> AcadosOcpFlattenedIterate:
        iterate = deepcopy(self.default_iterate)
        x0 = solver_input.x0.flatten()
        iterate.x = np.tile(x0, self.N + 1)
        iterate.u = self._zero_u_tiled.copy()
        return iterate

    def batch_iterate(self, solver_input: AcadosOcpSolverInput):
        from acados_template.acados_ocp_iterate import AcadosOcpFlattenedBatchIterate
        B = solver_input.batch_size
        x_batch = np.tile(solver_input.x0, (1, self.N + 1))
        u_batch = np.tile(self._zero_u_tiled, (B, 1))
        z_size  = self.default_iterate.z.size
        sl_size = self.default_iterate.sl.size
        su_size = self.default_iterate.su.size
        pi_size = self.default_iterate.pi.size
        lam_size = self.default_iterate.lam.size
        return AcadosOcpFlattenedBatchIterate(
            x=x_batch,
            u=u_batch,
            z=np.zeros((B, z_size))   if z_size   > 0 else np.zeros((B, 0)),
            sl=np.zeros((B, sl_size)) if sl_size   > 0 else np.zeros((B, 0)),
            su=np.zeros((B, su_size)) if su_size   > 0 else np.zeros((B, 0)),
            pi=np.zeros((B, pi_size)) if pi_size   > 0 else np.zeros((B, 0)),
            lam=np.zeros((B, lam_size)) if lam_size > 0 else np.zeros((B, 0)),
            N_batch=B,
        )


@dataclass(kw_only=True)
class RoverPlannerConfig:
    """Configuration for the differential-drive rover MPC planner."""

    N_horizon: int       = 4
    dt: float            = 0.1          # 10 Hz rover MPC
    T_horizon: float | None = None
    n_batch_max: int     = 4096
    num_threads: int     = 8
    pos_offset_max: float = 2.0
    wheel_vel_max: float  = _WHEEL_VEL_MAX
    wheel_lin_vel_max: float = _WHEEL_LIN_VEL_MAX
    dtype: torch.dtype   = torch.float32

    def __post_init__(self) -> None:
        if self.T_horizon is None:
            self.T_horizon = self.N_horizon * self.dt


class RoverPlanner(AcadosPlanner[AcadosDiffMpcCtx]):
    """Differentiable MPC planner for the TurtleBot3 Burger rover.

    Uses differential-drive dynamics and LINEAR_LS cost structure.
    State (6D): [x, y, cos(θ), sin(θ), v_L, v_R]
    Control (2D): [ω_L_cmd, ω_R_cmd]
    """

    def __init__(
        self,
        cfg: RoverPlannerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ):
        self.cfg = RoverPlannerConfig() if cfg is None else cfg

        if params is None:
            params = create_rover_params_linear_ls(
                N_horizon=self.cfg.N_horizon,
                pos_offset_max=self.cfg.pos_offset_max,
                wheel_vel_max=self.cfg.wheel_vel_max,
                wheel_lin_vel_max=self.cfg.wheel_lin_vel_max,
            )

        param_manager = AcadosParameterManager(
            parameters=params,
            N_horizon=self.cfg.N_horizon,
        )

        ocp = export_rover_ocp_linear_ls(
            param_manager=param_manager,
            name="rover_diff_drive_linear_ls",
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            dt=self.cfg.dt,
            wheel_vel_max=self.cfg.wheel_vel_max,
            wheel_lin_vel_max=self.cfg.wheel_lin_vel_max,
        )

        initializer = RoverStationaryInitializer(ocp)

        diff_mpc = AcadosDiffMpcTorch(
            ocp,
            initializer=initializer,
            export_directory=export_directory,
            n_batch_max=self.cfg.n_batch_max,
            num_threads_batch_solver=self.cfg.num_threads,
            dtype=self.cfg.dtype,
        )

        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        param: torch.Tensor | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve rover MPC.

        Args:
            obs:    Rover MPC state [x, y, c, s, v_L, v_R], shape (B, 6).
            action: Unused.
            param:  Learnable parameters, shape (B, n_learnable).
            ctx:    Optional warm-start context.

        Returns:
            (ctx, u0, x_traj, u_traj, value)
        """
        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            batch_size=obs.shape[0]
        )
        x0 = obs[:, :NX_ROVER]
        return self.diff_mpc(x0=x0, u0=action, p_global=param, p_stagewise=p_stagewise, ctx=ctx)

    def get_learnable_param_dim(self) -> int:
        return get_rover_learnable_param_dim()
