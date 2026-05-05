#!/usr/bin/env python3
"""Animate X3 rover MPC predicted trajectory vs actual sim trajectory.

Runs MPC in closed loop with JAX sim dynamics and creates an animation
showing the MPC's predicted horizon overlaid on the actual path.
A trapezoidal velocity profile planner generates straight-line position,
velocity, and heading references that the MPC tracks per-step.

Usage:
    python tests/animate_rover_mpc.py
    python tests/animate_rover_mpc.py --N 2 5 10 20
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from crazyflie_rover_landing.envs.mecanum_dynamics import (
    WHEEL_VEL_MAX,
    mecanum_step,
)
from crazyflie_rover_landing.leap_c.x3_rover_planner import (
    X3RoverPlanner,
    X3RoverPlannerConfig,
)
from crazyflie_rover_landing.leap_c.x3_rover_ocp_linear_ls import (
    NX_ROVER,
    NU_ROVER,
    _WHEEL_RADIUS,
    _K,
)

OUT_DIR = Path(__file__).parent
DT_CTRL = 0.02

START = np.array([0.0, 0.0, 0.0])

WAYPOINTS = [
    np.array([1.0, 0.0, 0.0]),
    np.array([1.0, 1.0, np.pi / 2]),
    np.array([0.0, 1.0, np.pi]),
    np.array([0.0, 0.0, 0.0]),
]


# ---------------------------------------------------------------------------
# Trapezoidal velocity profile planner
# ---------------------------------------------------------------------------

class TrapezoidalSegment:
    """Straight-line segment with trapezoidal velocity profile."""

    def __init__(self, p_start: np.ndarray, p_end: np.ndarray,
                 theta_start: float, theta_end: float,
                 v_max: float = 0.8, a_max: float = 2.0,
                 wz_max: float = 3.0, alpha_max: float = 10.0):
        self.p_start = p_start.copy()
        self.p_end = p_end.copy()
        self.theta_start = theta_start
        self.theta_end = theta_end

        # Translation profile
        self.dist = np.linalg.norm(p_end - p_start)
        if self.dist > 1e-6:
            self.direction = (p_end - p_start) / self.dist
        else:
            self.direction = np.array([0.0, 0.0])

        self.v_max = min(v_max, np.sqrt(a_max * self.dist)) if self.dist > 0 else 0
        self.a_max = a_max
        self._compute_translation_profile()

        # Rotation profile (trapezoidal in angular velocity)
        dtheta = theta_end - theta_start
        # Wrap to [-pi, pi]
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        self.dtheta = dtheta
        self.rot_sign = np.sign(dtheta) if abs(dtheta) > 1e-6 else 0
        self.abs_dtheta = abs(dtheta)
        self.wz_max_actual = min(wz_max, np.sqrt(alpha_max * self.abs_dtheta)) if self.abs_dtheta > 0 else 0
        self.alpha_max = alpha_max
        self._compute_rotation_profile()

        self.duration = max(self.t_trans_total, self.t_rot_total)

    def _compute_translation_profile(self):
        if self.dist < 1e-6:
            self.t_accel = 0
            self.t_cruise = 0
            self.t_trans_total = 0
            return
        t_accel = self.v_max / self.a_max
        d_accel = 0.5 * self.a_max * t_accel ** 2
        if 2 * d_accel >= self.dist:
            # Triangle profile
            t_accel = np.sqrt(self.dist / self.a_max)
            self.t_accel = t_accel
            self.t_cruise = 0
            self.v_max = self.a_max * t_accel
        else:
            self.t_accel = t_accel
            self.t_cruise = (self.dist - 2 * d_accel) / self.v_max
        self.t_trans_total = 2 * self.t_accel + self.t_cruise

    def _compute_rotation_profile(self):
        if self.abs_dtheta < 1e-6:
            self.t_rot_accel = 0
            self.t_rot_cruise = 0
            self.t_rot_total = 0
            return
        t_accel = self.wz_max_actual / self.alpha_max
        d_accel = 0.5 * self.alpha_max * t_accel ** 2
        if 2 * d_accel >= self.abs_dtheta:
            t_accel = np.sqrt(self.abs_dtheta / self.alpha_max)
            self.t_rot_accel = t_accel
            self.t_rot_cruise = 0
            self.wz_max_actual = self.alpha_max * t_accel
        else:
            self.t_rot_accel = t_accel
            self.t_rot_cruise = (self.abs_dtheta - 2 * d_accel) / self.wz_max_actual
        self.t_rot_total = 2 * self.t_rot_accel + self.t_rot_cruise

    def eval_translation(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Returns (position, velocity) in world frame at time t."""
        if self.dist < 1e-6 or t <= 0:
            return self.p_start.copy(), np.zeros(2)
        t = min(t, self.t_trans_total)

        if t <= self.t_accel:
            s = 0.5 * self.a_max * t ** 2
            v = self.a_max * t
        elif t <= self.t_accel + self.t_cruise:
            dt = t - self.t_accel
            s = 0.5 * self.a_max * self.t_accel ** 2 + self.v_max * dt
            v = self.v_max
        else:
            dt = t - self.t_accel - self.t_cruise
            s_accel = 0.5 * self.a_max * self.t_accel ** 2
            s_cruise = self.v_max * self.t_cruise
            s = s_accel + s_cruise + self.v_max * dt - 0.5 * self.a_max * dt ** 2
            v = self.v_max - self.a_max * dt

        s = min(s, self.dist)
        v = max(v, 0.0)
        pos = self.p_start + s * self.direction
        vel = v * self.direction
        return pos, vel

    def eval_rotation(self, t: float) -> tuple[float, float]:
        """Returns (theta, omega) at time t."""
        if self.abs_dtheta < 1e-6 or t <= 0:
            return self.theta_start, 0.0
        t = min(t, self.t_rot_total)

        if t <= self.t_rot_accel:
            angle = 0.5 * self.alpha_max * t ** 2
            wz = self.alpha_max * t
        elif t <= self.t_rot_accel + self.t_rot_cruise:
            dt = t - self.t_rot_accel
            angle = 0.5 * self.alpha_max * self.t_rot_accel ** 2 + self.wz_max_actual * dt
            wz = self.wz_max_actual
        else:
            dt = t - self.t_rot_accel - self.t_rot_cruise
            a_accel = 0.5 * self.alpha_max * self.t_rot_accel ** 2
            a_cruise = self.wz_max_actual * self.t_rot_cruise
            angle = a_accel + a_cruise + self.wz_max_actual * dt - 0.5 * self.alpha_max * dt ** 2
            wz = self.wz_max_actual - self.alpha_max * dt

        angle = min(angle, self.abs_dtheta)
        wz = max(wz, 0.0)
        theta = self.theta_start + self.rot_sign * angle
        omega = self.rot_sign * wz
        return theta, omega

    def eval(self, t: float) -> dict:
        pos, vel_world = self.eval_translation(t)
        theta, omega = self.eval_rotation(t)
        c, s = np.cos(theta), np.sin(theta)
        # World-frame vel → body-frame vel
        vx_body = vel_world[0] * c + vel_world[1] * s
        vy_body = -vel_world[0] * s + vel_world[1] * c
        return {
            "pos": pos, "theta": theta, "c": c, "s": s,
            "vx_body": vx_body, "vy_body": vy_body, "wz": omega,
        }


class StraightLinePlanner:
    """Sequence of straight-line segments with trapezoidal profiles."""

    def __init__(self, waypoints: list[np.ndarray], **kwargs):
        self.segments: list[TrapezoidalSegment] = []
        self.segment_starts: list[float] = []
        t = 0.0
        for i in range(len(waypoints)):
            j = (i + 1) % len(waypoints)
            if i == len(waypoints) - 1:
                break
            seg = TrapezoidalSegment(
                waypoints[i][:2], waypoints[j][:2],
                waypoints[i][2], waypoints[j][2],
                **kwargs,
            )
            self.segments.append(seg)
            self.segment_starts.append(t)
            t += seg.duration
        self.total_duration = t

    def eval(self, t: float) -> dict:
        t = np.clip(t, 0, self.total_duration - 1e-6)
        for i in range(len(self.segments) - 1, -1, -1):
            if t >= self.segment_starts[i]:
                local_t = t - self.segment_starts[i]
                return self.segments[i].eval(local_t)
        return self.segments[0].eval(0)


# ---------------------------------------------------------------------------
# MPC closed loop
# ---------------------------------------------------------------------------

def build_mpc_params(ref: dict, w_pos=50.0, w_heading=5.0, w_vel=2.0, w_ctrl=0.1):
    W_state = np.array([w_pos, w_pos, w_heading, w_heading, w_vel, w_vel, w_vel])
    W_ctrl = np.array([w_ctrl, w_ctrl, w_ctrl])
    yref_state = np.array([
        ref["pos"][0], ref["pos"][1], ref["c"], ref["s"],
        ref["vx_body"], ref["vy_body"], ref["wz"],
    ])
    yref_ctrl = np.array([ref["vx_body"], ref["vy_body"], ref["wz"]])
    params = np.concatenate([W_state, W_ctrl, yref_state, yref_ctrl])
    return torch.tensor(params, dtype=torch.float32).unsqueeze(0)


def run_closed_loop(N_horizon: int, mpc_dt: float = DT_CTRL):
    cfg = X3RoverPlannerConfig(
        N_horizon=N_horizon, dt=mpc_dt,
        n_batch_max=1, num_threads=1,
        wheel_vel_max=WHEEL_VEL_MAX,
    )
    planner = X3RoverPlanner(cfg=cfg)

    path_planner = StraightLinePlanner([START] + WAYPOINTS, v_max=0.8, a_max=2.0,
                                       wz_max=3.0, alpha_max=10.0)
    total_steps = int(np.ceil(path_planner.total_duration / DT_CTRL)) + 50

    state = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    states = [np.array(state)]
    controls = []
    predicted_trajs = []
    ref_positions = []

    for step in range(total_steps):
        t = step * DT_CTRL
        ref = path_planner.eval(t)
        ref_positions.append(ref["pos"].copy())

        state_np = np.array(state)
        obs = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        params = build_mpc_params(ref)

        with torch.no_grad():
            _, u0, x_traj, u_traj, _ = planner(obs=obs, param=params)

        x_pred = x_traj.squeeze(0).numpy().reshape(N_horizon + 1, NX_ROVER)
        predicted_trajs.append(x_pred)

        u = u0.squeeze(0).numpy()
        controls.append(u.copy())

        state = mecanum_step(state, jnp.array(u), DT_CTRL, WHEEL_VEL_MAX)
        states.append(np.array(state))

    return {
        "states": np.array(states),
        "controls": np.array(controls),
        "predicted_trajs": predicted_trajs,
        "ref_positions": np.array(ref_positions),
        "N_horizon": N_horizon,
        "mpc_dt": mpc_dt,
        "path_planner": path_planner,
    }


def animate_run(result: dict, out_path: Path, fps: int = 30, skip: int = 2):
    states = result["states"]
    predicted_trajs = result["predicted_trajs"]
    ref_positions = result["ref_positions"]
    N = result["N_horizon"]
    n_steps = len(predicted_trajs)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    margin = 0.3
    ax.set_xlim(-margin, 1.0 + margin)
    ax.set_ylim(-margin, 1.0 + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    mpc_dt = result["mpc_dt"]
    ax.set_title(f"MPC N={N}, dt={mpc_dt}s (horizon={N*mpc_dt:.2f}s)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # Plot planned straight-line path
    all_ref = ref_positions
    ax.plot(all_ref[:, 0], all_ref[:, 1], "g-", linewidth=1.0, alpha=0.4, label="planned path")

    for wp in WAYPOINTS:
        ax.plot(wp[0], wp[1], "rx", markersize=12, markeredgewidth=2.5, zorder=5)

    actual_line, = ax.plot([], [], "b-", linewidth=1.5, alpha=0.6, label="actual")
    pred_line, = ax.plot([], [], "r--", linewidth=2.0, alpha=0.8, label="MPC predicted")
    rover_dot, = ax.plot([], [], "bo", markersize=8, zorder=4)
    ref_dot, = ax.plot([], [], "g^", markersize=8, zorder=4, label="reference")
    heading_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                arrowprops=dict(arrowstyle="->", color="darkblue", lw=2))
    ax.legend(loc="upper right", fontsize=9)

    frames = list(range(0, n_steps, skip))

    def init():
        actual_line.set_data([], [])
        pred_line.set_data([], [])
        rover_dot.set_data([], [])
        ref_dot.set_data([], [])
        return actual_line, pred_line, rover_dot, ref_dot

    def update(frame_idx):
        i = frames[frame_idx]
        actual_line.set_data(states[:i+1, 0], states[:i+1, 1])
        rover_dot.set_data([states[i, 0]], [states[i, 1]])
        ref_dot.set_data([ref_positions[i, 0]], [ref_positions[i, 1]])

        pred = predicted_trajs[i]
        pred_line.set_data(pred[:, 0], pred[:, 1])

        c, s = states[i, 2], states[i, 3]
        arrow_len = 0.06
        heading_arrow.xy = (states[i, 0] + arrow_len * c, states[i, 1] + arrow_len * s)
        heading_arrow.set_position((states[i, 0], states[i, 1]))

        return actual_line, pred_line, rover_dot, ref_dot

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=len(frames), interval=1000 // fps, blit=False,
    )
    anim.save(str(out_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, nargs="+", default=[2, 5, 10, 20])
    parser.add_argument("--mpc-dt", type=float, default=DT_CTRL,
                        help="MPC prediction timestep (default: same as control dt)")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    for N in args.N:
        print(f"\nRunning N={N}, mpc_dt={args.mpc_dt}s (horizon={N*args.mpc_dt:.2f}s)...")
        result = run_closed_loop(N, mpc_dt=args.mpc_dt)

        states = result["states"]
        ref_positions = result["ref_positions"]

        # Compute tracking error over time
        n = min(len(states) - 1, len(ref_positions))
        errors = np.linalg.norm(states[:n, :2] - ref_positions[:n], axis=1)
        print(f"  Mean tracking error: {np.mean(errors):.4f} m")
        print(f"  Max  tracking error: {np.max(errors):.4f} m")
        print(f"  Final position error: {errors[-1]:.4f} m")

        # Per-waypoint check: find closest state to each waypoint
        for i, wp in enumerate(WAYPOINTS[:-1]):
            dists = np.linalg.norm(states[:, :2] - wp[:2], axis=1)
            min_dist = np.min(dists)
            print(f"  WP{i} closest approach: {min_dist:.4f} m")

        dt_tag = f"_dt{args.mpc_dt}" if args.mpc_dt != DT_CTRL else ""
        out_path = OUT_DIR / f"rover_mpc_N{N}{dt_tag}.gif"
        animate_run(result, out_path, fps=args.fps)


if __name__ == "__main__":
    main()
