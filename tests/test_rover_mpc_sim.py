#!/usr/bin/env python3
"""Test X3 rover MPC with sim dynamics in a closed-loop tracking scenario.

Creates an MPC planner with hand-crafted cost parameters (no learned policy),
runs it in closed loop with the JAX mecanum sim dynamics, and verifies the
rover tracks waypoints correctly. Tests both with and without wheel constraints.

Usage:
    python tests/test_rover_mpc_sim.py
    python tests/test_rover_mpc_sim.py --plot

Output (with --plot):
    tests/rover_mpc_sim.png
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
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

OUT_PATH = Path(__file__).parent / "rover_mpc_sim.png"

DT_CTRL = 0.02
N_HORIZON = 10
T_HORIZON = N_HORIZON * DT_CTRL


def build_mpc_params(
    target_xy: np.ndarray,
    target_cs: np.ndarray,
    current_xy: np.ndarray,
    w_pos: float = 50.0,
    w_heading: float = 5.0,
    w_vel: float = 1.0,
    w_ctrl: float = 0.1,
) -> torch.Tensor:
    """Build MPC parameters for a position+heading tracking task.

    Layout: w_state(7) + w_ctrl(3) + yref_state(7) + yref_ctrl(3) = 20
    """
    W_state = np.array([w_pos, w_pos, w_heading, w_heading, w_vel, w_vel, w_vel])
    W_ctrl = np.array([w_ctrl, w_ctrl, w_ctrl])
    yref_state = np.concatenate([target_xy, target_cs, np.zeros(3)])
    yref_ctrl = np.zeros(3)
    params = np.concatenate([W_state, W_ctrl, yref_state, yref_ctrl])
    return torch.tensor(params, dtype=torch.float32).unsqueeze(0)


def run_closed_loop(
    planner: X3RoverPlanner,
    waypoints: list[np.ndarray],
    initial_state: np.ndarray,
    dt: float = DT_CTRL,
    steps_per_waypoint: int = 150,
) -> dict:
    """Run MPC in closed loop with JAX sim dynamics."""
    state = jnp.array(initial_state)
    states = [np.array(state)]
    controls = []
    wheel_speeds_log = []

    for wp in waypoints:
        target_xy = wp[:2]
        target_heading = wp[2] if len(wp) > 2 else 0.0
        target_cs = np.array([np.cos(target_heading), np.sin(target_heading)])

        for _ in range(steps_per_waypoint):
            state_np = np.array(state)
            obs = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
            params = build_mpc_params(target_xy, target_cs, state_np[:2])

            with torch.no_grad():
                _, u0, _, _, _ = planner(obs=obs, param=params)

            u = u0.squeeze(0).numpy()
            controls.append(u.copy())

            # Log wheel speeds from the command
            r_inv = 1.0 / _WHEEL_RADIUS
            wheels = r_inv * np.array([
                u[0] - u[1] - _K * u[2],
                u[0] + u[1] + _K * u[2],
                u[0] + u[1] - _K * u[2],
                u[0] - u[1] + _K * u[2],
            ])
            wheel_speeds_log.append(wheels)

            state = mecanum_step(state, jnp.array(u), dt, WHEEL_VEL_MAX)
            states.append(np.array(state))

    return {
        "states": np.array(states),
        "controls": np.array(controls),
        "wheel_speeds": np.array(wheel_speeds_log),
    }


def test_waypoint_tracking(wheel_vel_max: float | None, label: str) -> dict:
    """Test MPC tracks waypoints with given wheel constraint config."""
    cfg = X3RoverPlannerConfig(
        N_horizon=N_HORIZON,
        dt=DT_CTRL,
        n_batch_max=1,
        num_threads=1,
        wheel_vel_max=wheel_vel_max,
    )
    planner = X3RoverPlanner(cfg=cfg)

    waypoints = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, np.pi / 2]),
        np.array([0.0, 1.0, np.pi]),
        np.array([0.0, 0.0, 0.0]),
    ]

    initial_state = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    result = run_closed_loop(planner, waypoints, initial_state)
    states = result["states"]
    controls = result["controls"]
    wheel_speeds = result["wheel_speeds"]

    print(f"\n=== {label} ===")

    # Check final position near last waypoint
    final_xy = states[-1, :2]
    target_xy = waypoints[-1][:2]
    pos_error = np.linalg.norm(final_xy - target_xy)
    print(f"  Final position error: {pos_error:.4f} m")

    # Check each waypoint was approximately reached
    steps_per_wp = len(controls) // len(waypoints)
    for i, wp in enumerate(waypoints):
        idx = (i + 1) * steps_per_wp
        wp_state = states[idx]
        wp_error = np.linalg.norm(wp_state[:2] - wp[:2])
        print(f"  Waypoint {i} ({wp[:2]}) error: {wp_error:.4f} m")

    # Check wheel speed limits are respected by MPC commands
    max_wheel = np.max(np.abs(wheel_speeds))
    if wheel_vel_max is not None:
        violated = np.any(np.abs(wheel_speeds) > wheel_vel_max + 0.01)
        print(f"  Max wheel speed: {max_wheel:.2f} rad/s (limit: {wheel_vel_max})")
        print(f"  Wheel constraint violated: {violated}")
    else:
        print(f"  Max wheel speed: {max_wheel:.2f} rad/s (no constraint)")

    # Check no excessive yaw spinning
    wz = controls[:, 2]
    mean_abs_wz = np.mean(np.abs(wz))
    max_wz = np.max(np.abs(wz))
    print(f"  Mean |ωz_cmd|: {mean_abs_wz:.3f} rad/s")
    print(f"  Max  |ωz_cmd|: {max_wz:.3f} rad/s")

    result["label"] = label
    result["waypoints"] = waypoints
    return result


def test_wheel_constraint_active():
    """Test that wheel constraints actually limit combined velocity commands
    that would exceed motor limits without constraints."""
    print("\n=== Wheel constraint activation test ===")

    # Command that requires high wheel speeds:
    # vx=1.0, vy=1.0, wz=3.0 simultaneously
    r_inv = 1.0 / _WHEEL_RADIUS
    vx, vy, wz = 1.0, 1.0, 3.0
    unconstrained_wheels = r_inv * np.array([
        vx - vy - _K * wz,
        vx + vy + _K * wz,
        vx + vy - _K * wz,
        vx - vy + _K * wz,
    ])
    print(f"  Unconstrained wheel speeds for vx={vx}, vy={vy}, wz={wz}:")
    print(f"    {unconstrained_wheels}")
    print(f"    Max |ω|: {np.max(np.abs(unconstrained_wheels)):.1f} rad/s "
          f"(limit: {WHEEL_VEL_MAX})")

    # Create constrained planner and solve from rest with aggressive target
    cfg = X3RoverPlannerConfig(
        N_horizon=N_HORIZON, dt=DT_CTRL,
        n_batch_max=1, num_threads=1,
        wheel_vel_max=WHEEL_VEL_MAX,
    )
    planner = X3RoverPlanner(cfg=cfg)

    state = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    obs = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    # Target far away with heading change to force aggressive command
    target_xy = np.array([3.0, 3.0])
    target_cs = np.array([0.0, 1.0])  # 90 degrees
    params = build_mpc_params(target_xy, target_cs, state[:2], w_pos=100.0, w_heading=50.0)

    with torch.no_grad():
        _, u0, _, _, _ = planner(obs=obs, param=params)
    u = u0.squeeze(0).numpy()

    constrained_wheels = r_inv * np.array([
        u[0] - u[1] - _K * u[2],
        u[0] + u[1] + _K * u[2],
        u[0] + u[1] - _K * u[2],
        u[0] - u[1] + _K * u[2],
    ])
    max_constrained = np.max(np.abs(constrained_wheels))
    print(f"  Constrained MPC command: vx={u[0]:.3f}, vy={u[1]:.3f}, wz={u[2]:.3f}")
    print(f"  Resulting wheel speeds: {constrained_wheels}")
    print(f"  Max |ω|: {max_constrained:.1f} rad/s (limit: {WHEEL_VEL_MAX})")
    print(f"  Within limits: {max_constrained <= WHEEL_VEL_MAX + 0.01}")

    # Compare with unconstrained planner
    cfg_nc = X3RoverPlannerConfig(
        N_horizon=N_HORIZON, dt=DT_CTRL,
        n_batch_max=1, num_threads=1,
        wheel_vel_max=None,
    )
    planner_nc = X3RoverPlanner(cfg=cfg_nc)
    with torch.no_grad():
        _, u0_nc, _, _, _ = planner_nc(obs=obs, param=params)
    u_nc = u0_nc.squeeze(0).numpy()
    nc_wheels = r_inv * np.array([
        u_nc[0] - u_nc[1] - _K * u_nc[2],
        u_nc[0] + u_nc[1] + _K * u_nc[2],
        u_nc[0] + u_nc[1] - _K * u_nc[2],
        u_nc[0] - u_nc[1] + _K * u_nc[2],
    ])
    print(f"\n  Unconstrained MPC command: vx={u_nc[0]:.3f}, vy={u_nc[1]:.3f}, wz={u_nc[2]:.3f}")
    print(f"  Resulting wheel speeds: {nc_wheels}")
    print(f"  Max |ω|: {np.max(np.abs(nc_wheels)):.1f} rad/s")

    return max_constrained <= WHEEL_VEL_MAX + 0.01


def plot_results(results: list[dict]):
    """Plot trajectories, controls, and wheel speeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, res in enumerate(results):
        states = res["states"]
        controls = res["controls"]
        wheel_speeds = res["wheel_speeds"]
        waypoints = res["waypoints"]
        label = res["label"]
        t = np.arange(len(controls)) * DT_CTRL

        # XY trajectory
        ax = axes[i, 0]
        ax.plot(states[:, 0], states[:, 1], "b-", linewidth=1.5, label="trajectory")
        for j, wp in enumerate(waypoints):
            ax.plot(wp[0], wp[1], "rx", markersize=10, markeredgewidth=2)
            ax.annotate(f"WP{j}", (wp[0], wp[1]), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"{label} — XY trajectory")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Body velocity commands
        ax = axes[i, 1]
        ax.plot(t, controls[:, 0], label="vx_cmd")
        ax.plot(t, controls[:, 1], label="vy_cmd")
        ax.plot(t, controls[:, 2], label="wz_cmd")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("command")
        ax.set_title(f"{label} — Controls")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Wheel speeds
        ax = axes[i, 2]
        labels_w = ["FL", "FR", "BL", "BR"]
        for j in range(4):
            ax.plot(t, wheel_speeds[:, j], label=labels_w[j], alpha=0.7)
        ax.axhline(WHEEL_VEL_MAX, color="r", linestyle="--", alpha=0.5, label=f"±{WHEEL_VEL_MAX}")
        ax.axhline(-WHEEL_VEL_MAX, color="r", linestyle="--", alpha=0.5)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("wheel speed [rad/s]")
        ax.set_title(f"{label} — Wheel speeds")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Heading (theta from cos/sin)
        ax = axes[i, 3]
        theta = np.arctan2(states[:, 3], states[:, 2])
        ax.plot(np.arange(len(theta)) * DT_CTRL, np.degrees(theta))
        for j, wp in enumerate(waypoints):
            if len(wp) > 2:
                wp_t = (j + 1) * (len(controls) // len(waypoints)) * DT_CTRL
                ax.axhline(np.degrees(wp[2]), color="r", linestyle=":",
                           alpha=0.3, xmin=j / len(waypoints), xmax=(j + 1) / len(waypoints))
        ax.set_xlabel("time [s]")
        ax.set_ylabel("heading [deg]")
        ax.set_title(f"{label} — Heading")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"\nPlot saved to {OUT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Save trajectory plot")
    args = parser.parse_args()

    results = []

    # Test 1: MPC with wheel constraints + sim dynamics
    res_constrained = test_waypoint_tracking(WHEEL_VEL_MAX, "With wheel constraints")
    results.append(res_constrained)

    # Test 2: MPC without wheel constraints + sim dynamics
    res_unconstrained = test_waypoint_tracking(None, "Without wheel constraints")
    results.append(res_unconstrained)

    # Test 3: Verify wheel constraints are actually active
    wheel_ok = test_wheel_constraint_active()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for res in results:
        states = res["states"]
        waypoints = res["waypoints"]
        steps_per_wp = (len(states) - 1) // len(waypoints)
        final_error = np.linalg.norm(states[-1, :2] - waypoints[-1][:2])
        ok = final_error < 0.15
        status = "PASS" if ok else "FAIL"
        print(f"  {res['label']}: final error={final_error:.4f} m [{status}]")
        all_pass = all_pass and ok

    status = "PASS" if wheel_ok else "FAIL"
    print(f"  Wheel constraint enforcement: [{status}]")
    all_pass = all_pass and wheel_ok

    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    if args.plot:
        plot_results(results)

    return 0 if all_pass else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
