"""Differentiable MPC planner for the unicycle ground rover (LINEAR_LS).

State (5D):   [x, y, cos(θ), sin(θ), v]
Control (2D): [a, ω]
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
    _MAX_SPEED,
    _MIN_SPEED,
    _MAX_OMEGA,
    _MAX_ACCEL,
)


class RoverStationaryInitializer(AcadosDiffMpcInitializer):
    """Warm-start: state = x0 repeated, control = zero (stationary)."""

    def __init__(self, ocp: AcadosOcp):
        self.default_iterate = ocp.create_default_initial_iterate().flatten()
        self.N = ocp.solver_options.N_horizon
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
        z_size = self.default_iterate.z.size
        sl_size = self.default_iterate.sl.size
        su_size = self.default_iterate.su.size
        pi_size = self.default_iterate.pi.size
        lam_size = self.default_iterate.lam.size
        return AcadosOcpFlattenedBatchIterate(
            x=x_batch,
            u=u_batch,
            z=np.zeros((B, z_size)) if z_size > 0 else np.zeros((B, 0)),
            sl=np.zeros((B, sl_size)) if sl_size > 0 else np.zeros((B, 0)),
            su=np.zeros((B, su_size)) if su_size > 0 else np.zeros((B, 0)),
            pi=np.zeros((B, pi_size)) if pi_size > 0 else np.zeros((B, 0)),
            lam=np.zeros((B, lam_size)) if lam_size > 0 else np.zeros((B, 0)),
            N_batch=B,
        )


@dataclass(kw_only=True)
class RoverPlannerConfig:
    """Configuration for the rover MPC planner."""

    N_horizon: int = 4
    dt: float = 0.1           # 10 Hz rover MPC
    T_horizon: float | None = None
    n_batch_max: int = 4096
    num_threads: int = 8
    pos_offset_max: float = 2.0
    max_speed: float = _MAX_SPEED
    min_speed: float = _MIN_SPEED
    max_omega: float = _MAX_OMEGA
    max_accel: float = _MAX_ACCEL
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.T_horizon is None:
            self.T_horizon = self.N_horizon * self.dt


class RoverPlanner(AcadosPlanner[AcadosDiffMpcCtx]):
    """Differentiable MPC planner for the unicycle ground rover.

    Uses unicycle kinematics and LINEAR_LS cost structure.
    State (5D): [x, y, cos(θ), sin(θ), v]
    Control (2D): [a, ω]
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
                max_speed=self.cfg.max_speed,
                min_speed=self.cfg.min_speed,
                max_omega=self.cfg.max_omega,
                max_accel=self.cfg.max_accel,
            )

        param_manager = AcadosParameterManager(
            parameters=params,
            N_horizon=self.cfg.N_horizon,
        )

        ocp = export_rover_ocp_linear_ls(
            param_manager=param_manager,
            name="rover_unicycle_linear_ls",
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            dt=self.cfg.dt,
            max_speed=self.cfg.max_speed,
            min_speed=self.cfg.min_speed,
            max_omega=self.cfg.max_omega,
            max_accel=self.cfg.max_accel,
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
            obs: Rover MPC state [x, y, c, s, v], shape (B, 5).
            action: Unused.
            param: Learnable parameters, shape (B, n_learnable).
            ctx: Optional warm-start context.

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
