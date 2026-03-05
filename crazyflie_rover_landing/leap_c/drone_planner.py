"""Differentiable MPC planner for the Crazyflie drone (so_rpy, LINEAR_LS).

Adapted from crazyflie-mape-crazyflow QuadrotorPlanner.
Default model changed to cf2x_T350 and cost type fixed to linear_ls.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from acados_template.acados_ocp import AcadosOcp
from acados_template.acados_ocp_iterate import AcadosOcpFlattenedIterate
from drone_models.core import load_params
from leap_c.ocp.acados.data import AcadosOcpSolverInput
from leap_c.ocp.acados.initializer import AcadosDiffMpcInitializer
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch

from crazyflie_rover_landing.leap_c.drone_ocp_linear_ls import (
    NX,
    NU,
    create_drone_params_linear_ls,
    export_drone_ocp_linear_ls,
    get_drone_learnable_param_dim,
    QuadrotorAcadosParamInterface,
)


class DroneHoverInitializer(AcadosDiffMpcInitializer):
    """Warm-start: state = x0 repeated, control = hover thrust."""

    def __init__(self, ocp: AcadosOcp, mass: float, gravity: float, cmd_f_coef: float):
        self.default_iterate = ocp.create_default_initial_iterate().flatten()
        self.N = ocp.solver_options.N_horizon
        self.nx = ocp.dims.nx
        self.nu = ocp.dims.nu
        hover_thrust = (mass * gravity) / cmd_f_coef
        self.hover_u = np.zeros(self.nu)
        self.hover_u[-1] = hover_thrust
        self._hover_u_tiled = np.tile(self.hover_u, self.N)

    def single_iterate(self, solver_input: AcadosOcpSolverInput) -> AcadosOcpFlattenedIterate:
        iterate = deepcopy(self.default_iterate)
        x0 = solver_input.x0.flatten()
        iterate.x = np.tile(x0, self.N + 1)
        iterate.u = self._hover_u_tiled.copy()
        return iterate

    def batch_iterate(self, solver_input: AcadosOcpSolverInput):
        from acados_template.acados_ocp_iterate import AcadosOcpFlattenedBatchIterate
        B = solver_input.batch_size
        x_batch = np.tile(solver_input.x0, (1, self.N + 1))
        u_batch = np.tile(self._hover_u_tiled, (B, 1))
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
class DronePlannerConfig:
    """Configuration for the drone MPC planner."""

    N_horizon: int = 2
    dt: float = 0.01          # 100 Hz MPC
    T_horizon: float | None = None
    param_interface: QuadrotorAcadosParamInterface = "global"
    n_batch_max: int = 4096
    num_threads: int = 8
    drone_model: str = "cf2x_T350"
    velocity_max: float | None = None
    roll_pitch_max: float = 0.5
    yaw_max: float = 0.5
    pos_offset_max: float = 2.0
    thrust_min: float | None = None
    thrust_max: float | None = None
    mass: float | None = None
    gravity: float | None = None
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.T_horizon is None:
            self.T_horizon = self.N_horizon * self.dt


class DronePlanner(AcadosPlanner[AcadosDiffMpcCtx]):
    """Differentiable MPC planner for the Crazyflie drone.

    Uses so_rpy Euler dynamics and LINEAR_LS cost structure.
    State (12D): [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]
    Control (4D): [roll_cmd, pitch_cmd, yaw_cmd, thrust]
    """

    def __init__(
        self,
        cfg: DronePlannerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ):
        self.cfg = DronePlannerConfig() if cfg is None else cfg

        drone_params = load_params("so_rpy", self.cfg.drone_model)

        if params is None:
            params = create_drone_params_linear_ls(
                N_horizon=self.cfg.N_horizon,
                param_interface=self.cfg.param_interface,
                drone_model=self.cfg.drone_model,
                roll_pitch_max=self.cfg.roll_pitch_max,
                yaw_max=self.cfg.yaw_max,
                pos_offset_max=self.cfg.pos_offset_max,
                thrust_min=self.cfg.thrust_min,
                thrust_max=self.cfg.thrust_max,
                mass=self.cfg.mass,
                gravity=self.cfg.gravity,
            )

        param_manager = AcadosParameterManager(
            parameters=params,
            N_horizon=self.cfg.N_horizon,
        )

        ocp = export_drone_ocp_linear_ls(
            param_manager=param_manager,
            name="drone_so_rpy_euler_linear_ls",
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            dt=self.cfg.dt,
            drone_model=self.cfg.drone_model,
            velocity_max=self.cfg.velocity_max,
            roll_pitch_max=self.cfg.roll_pitch_max,
            yaw_max=self.cfg.yaw_max,
            thrust_min=self.cfg.thrust_min,
            thrust_max=self.cfg.thrust_max,
            mass=self.cfg.mass,
            gravity=self.cfg.gravity,
        )

        mass = self.cfg.mass if self.cfg.mass is not None else float(drone_params["mass"])
        gravity = (
            self.cfg.gravity
            if self.cfg.gravity is not None
            else float(np.abs(drone_params["gravity_vec"][2]))
        )
        cmd_f_coef = float(drone_params["cmd_f_coef"])

        initializer = DroneHoverInitializer(ocp, mass=mass, gravity=gravity, cmd_f_coef=cmd_f_coef)

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
        """Solve drone MPC.

        Args:
            obs: Drone MPC state [x,y,z,rpy,vel,drpy], shape (B, 12).
            action: Unused.
            param: Learnable parameters, shape (B, n_learnable).
            ctx: Optional warm-start context.

        Returns:
            (ctx, u0, x_traj, u_traj, value)
        """
        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            batch_size=obs.shape[0]
        )
        x0 = obs[:, :NX]
        return self.diff_mpc(x0=x0, u0=action, p_global=param, p_stagewise=p_stagewise, ctx=ctx)

    def get_learnable_param_dim(self) -> int:
        return get_drone_learnable_param_dim(self.cfg.N_horizon, self.cfg.param_interface)
