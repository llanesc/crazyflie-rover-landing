#!/usr/bin/env python3
"""Compare rover_dynamics.py (JAX) vs MuJoCo TurtleBot3 Burger simulation.

Applies identical random control sequences from multiple initial conditions
and plots state trajectories for both models side by side.

Usage:
    python tests/test_rover_dynamics_comparison.py

Output:
    tests/rover_dynamics_comparison.png
"""

import numpy as np
import jax.numpy as jnp
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from crazyflie_rover_landing.envs.rover_dynamics import (
    rover_step, WHEEL_RADIUS, WHEEL_VEL_MAX,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BURGER_XML   = (PROJECT_ROOT / "external/robotis_mujoco_menagerie"
                             / "robotis_tb3/turtlebot3_burger.xml")
OUT_PATH     = Path(__file__).parent / "rover_dynamics_comparison.png"

# ── Simulation parameters ─────────────────────────────────────────────────────
DT_CTRL       = 0.02   # control period [s]  (50 Hz)
DT_MJ         = 0.001  # MuJoCo integration timestep [s]
N_MJ_PER_CTRL = int(round(DT_CTRL / DT_MJ))  # 10 sub-steps per control step
N_CTRL_STEPS  = 500    # total control steps per trial (10 s at 50 Hz)
N_TRIALS      = 5
RNG_SEED      = 42


# ── MuJoCo model ──────────────────────────────────────────────────────────────

def build_mujoco_model() -> mujoco.MjModel:
    """Load burger XML, add a ground plane, set timestep."""
    xml = BURGER_XML.read_text()
    # Add stiff ground plane.  Default MuJoCo soft contacts give near-zero normal
    # force at zero penetration, so the robot barely moves (insufficient traction).
    # solref="-3000 -240" gives k=3000 N/m stiffness, b=240 N·s/m (overdamped).
    xml = xml.replace(
        "<worldbody>",
        '<worldbody>\n    '
        '<geom name="floor" type="plane" size="0 0 0.01" rgba=".8 .8 .8 1"'
        ' solref="-3000 -240" solimp="0.999 0.9999 0.001"/>',
    )
    # The burger_base collision mesh bottom is co-planar with the ground at body_z=0,
    # causing it to steal normal force from the wheels.  Disable it for flat-floor
    # sims — the robot is correctly supported by 2 wheels + caster.
    xml = xml.replace(
        '<geom pos="-0.032 0 0.01" mesh="burger_base" class="collision"/>',
        '<geom pos="-0.032 0 0.01" mesh="burger_base" contype="0" conaffinity="0"/>',
    )
    # The tire STL meshes have slightly smaller radius than 0.033 m, so at body_z=0
    # the wheels float ~0.4 mm above the floor and never contact it.  Replace the
    # collision geoms with exact-radius cylinders.  No quat needed: the cylinder
    # Z-axis maps through the wheel body rotation to the world Y axle direction.
    # Stiff contacts + high friction (μ=10) ensure proper support and no-slip.
    for side in ("left", "right"):
        xml = xml.replace(
            f'<geom quat="0.707388 0.706825 0 0" mesh="{side}_tire" class="collision"/>',
            f'<geom type="cylinder" size="0.033 0.009"'
            f' friction="10 0.005 0.0001" condim="4"'
            f' solref="-3000 -240" solimp="0.999 0.9999 0.001" class="collision"/>',
        )
    # condim=4 adds torsional contact friction (prevents spinning on contact patch),
    # which reduces wheel velocity jitter by ~45% vs default condim=3.
    # Stiff contacts + condim=4 introduce ~0.062 N·m of effective rolling resistance.
    # Reduce frictionloss from 0.1 → 0.042 N·m so effective friction ≈ 0.100 N·m.
    # Keep caster ball at original soft contacts (condim=1, no friction).
    # Stiffening the caster worsens jitter — soft contacts let it smoothly absorb
    # body pitch oscillations without creating sharp contact-switching impulses.
    xml = xml.replace('frictionloss="0.1"', 'frictionloss="0.042"')
    # Write next to assets/ so meshdir resolves correctly
    tmp = BURGER_XML.parent / "_tmp_comparison.xml"
    tmp.write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()
    model.opt.timestep = DT_MJ
    return model


def quat_to_yaw(q: np.ndarray) -> float:
    """MuJoCo quaternion [w, x, y, z] → yaw angle [rad]."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z)))


def mj_set_state(model, data, x, y, theta, v_L, v_R):
    """Set MuJoCo state from physical variables."""
    mujoco.mj_resetData(model, data)
    data.qpos[0] = x
    data.qpos[1] = y
    data.qpos[2] = 0.0                    # z: wheels on ground
    data.qpos[3] = np.cos(theta / 2.0)    # quaternion w  (yaw only)
    data.qpos[4] = 0.0                    # quaternion x
    data.qpos[5] = 0.0                    # quaternion y
    data.qpos[6] = np.sin(theta / 2.0)    # quaternion z
    data.qvel[6] = v_L / WHEEL_RADIUS     # wheel_left  ω [rad/s]
    data.qvel[7] = v_R / WHEEL_RADIUS     # wheel_right ω [rad/s]
    mujoco.mj_forward(model, data)


def mj_get_state(data) -> np.ndarray:
    """Extract [x, y, theta, v_L, v_R] from MuJoCo data."""
    return np.array([
        data.qpos[0],
        data.qpos[1],
        quat_to_yaw(data.qpos[3:7]),
        WHEEL_RADIUS * data.qvel[6],   # left  wheel linear velocity [m/s]
        WHEEL_RADIUS * data.qvel[7],   # right wheel linear velocity [m/s]
    ])


# ── JAX helpers ───────────────────────────────────────────────────────────────

def jax_make_state(x, y, theta, v_L, v_R) -> jnp.ndarray:
    return jnp.array([x, y, np.cos(theta), np.sin(theta), v_L, v_R])


def jax_extract(state: jnp.ndarray) -> np.ndarray:
    """Extract [x, y, theta, v_L, v_R] from JAX 6D state."""
    return np.array([
        float(state[0]),
        float(state[1]),
        float(np.arctan2(float(state[3]), float(state[2]))),
        float(state[4]),
        float(state[5]),
    ])


# ── Control sequence generation ───────────────────────────────────────────────

def make_ctrl_sequence(rng: np.random.Generator) -> np.ndarray:
    """Piecewise-constant random commands held for ~0.5s — large excursions."""
    cmds    = np.zeros((N_CTRL_STEPS, 2))
    hold    = max(1, int(0.5 / DT_CTRL))   # steps per segment (~25)
    cmd     = np.zeros(2)
    for k in range(N_CTRL_STEPS):
        if k % hold == 0:
            # Jump to a new random target in full range
            cmd = rng.uniform(-WHEEL_VEL_MAX, WHEEL_VEL_MAX, size=2)
        cmds[k] = cmd
    return cmds


def make_smooth_ctrl_sequence(rng: np.random.Generator) -> np.ndarray:
    """Smooth sinusoidal differential-drive commands — varied forward speed and turning.

    Each call draws random forward speed, differential amplitude, frequency, and phase
    so consecutive calls produce genuinely different profiles.
    """
    t     = np.arange(N_CTRL_STEPS) * DT_CTRL
    fwd   = rng.uniform(0.30, 0.55) * WHEEL_VEL_MAX     # mean forward drive
    amp   = rng.uniform(0.05, 0.20) * WHEEL_VEL_MAX     # differential swing (gentle turns)
    freq  = rng.uniform(0.04, 0.12)                      # turning frequency [Hz]
    phase = rng.uniform(0, 2 * np.pi)
    diff  = amp * np.sin(2 * np.pi * freq * t + phase)
    # No clipping needed: fwd ± amp/2 stays within [-WHEEL_VEL_MAX, WHEEL_VEL_MAX]
    wL    = fwd - diff / 2
    wR    = fwd + diff / 2
    return np.stack([wL, wR], axis=1)   # (T, 2)


# ── Single trial ──────────────────────────────────────────────────────────────

def run_trial(model, data, cmds, x0, y0, theta0, v_L0, v_R0):
    """Run one trial. Returns (mj_traj, jax_traj) each (N_CTRL_STEPS+1, 5)."""

    # ── MuJoCo ──
    mj_set_state(model, data, x0, y0, theta0, v_L0, v_R0)
    mj_traj = np.zeros((N_CTRL_STEPS + 1, 5))
    mj_traj[0] = mj_get_state(data)
    for k in range(N_CTRL_STEPS):
        data.ctrl[0] = cmds[k, 0]
        data.ctrl[1] = cmds[k, 1]
        for _ in range(N_MJ_PER_CTRL):
            mujoco.mj_step(model, data)
        mj_traj[k + 1] = mj_get_state(data)

    # ── JAX ──
    jax_state = jax_make_state(x0, y0, theta0, v_L0, v_R0)
    jax_traj  = np.zeros((N_CTRL_STEPS + 1, 5))
    jax_traj[0] = jax_extract(jax_state)
    for k in range(N_CTRL_STEPS):
        u         = jnp.array(cmds[k])
        jax_state = rover_step(jax_state, u, DT_CTRL)
        jax_traj[k + 1] = jax_extract(jax_state)

    return mj_traj, jax_traj


# ── Main ──────────────────────────────────────────────────────────────────────

def plot_arrow(ax, traj, step, color, scale=0.03):
    """Draw a heading arrow along a trajectory at a given step index."""
    x, y, th = traj[step, 0], traj[step, 1], traj[step, 2]
    ax.annotate("", xy=(x + scale * np.cos(th), y + scale * np.sin(th)),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


def _plot_trial_set(results, t, labels, title_prefix, out_states, out_xy):
    """Generate state time-series and XY trajectory figures for one set of trials."""
    n_trials = len(results)
    cmap     = plt.get_cmap("viridis")

    # Figure 1: state time-series
    fig1, axes1 = plt.subplots(n_trials, 5, figsize=(18, 3.5 * n_trials))
    if n_trials == 1:
        axes1 = axes1[np.newaxis, :]
    for i, (x0, y0, th0, mj_traj, jax_traj) in enumerate(results):
        for j in range(5):
            ax = axes1[i, j]
            ax.plot(t, mj_traj[:, j],  color="tab:blue",   lw=1.5, label="MuJoCo")
            ax.plot(t, jax_traj[:, j], color="tab:orange", lw=1.5, ls="--", label="JAX")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(labels[j], fontsize=11)
            if j == 0:
                ax.set_ylabel(f"IC {i+1}", fontsize=10)
            if i == n_trials - 1:
                ax.set_xlabel("t [s]", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=9)
    fig1.suptitle(f"State time-series ({title_prefix}): JAX vs MuJoCo TurtleBot3 Burger",
                  fontsize=13)
    fig1.tight_layout()
    fig1.savefig(out_states, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_states}")

    # Figure 2: XY trajectories
    fig2, axes2 = plt.subplots(1, n_trials, figsize=(6 * n_trials, 5))
    if n_trials == 1:
        axes2 = [axes2]
    for i, (x0, y0, th0, mj_traj, jax_traj) in enumerate(results):
        ax = axes2[i]
        n  = len(mj_traj)
        for k in range(n - 1):
            frac = k / (n - 1)
            c    = cmap(frac)
            ax.plot(mj_traj[k:k+2,  0], mj_traj[k:k+2,  1],
                    color=c, lw=2.0, solid_capstyle="round")
            ax.plot(jax_traj[k:k+2, 0], jax_traj[k:k+2, 1],
                    color=c, lw=2.0, ls="--", solid_capstyle="round")
        ax.plot(x0, y0, "ko", ms=7, zorder=5, label="start")
        for step in range(0, n, 25):
            plot_arrow(ax, mj_traj,  step, color="tab:blue")
            plot_arrow(ax, jax_traj, step, color="tab:orange")
        ax.plot([], [], color="tab:blue",   lw=2,       label="MuJoCo")
        ax.plot([], [], color="tab:orange", lw=2, ls="--", label="JAX")
        rmse_xy = np.sqrt(np.mean((mj_traj[:, :2] - jax_traj[:, :2]) ** 2))
        ax.set_title(
            f"IC {i+1}:  x₀={x0:.1f}, y₀={y0:.1f}, θ₀={np.degrees(th0):.0f}°\n"
            f"XY RMSE = {rmse_xy*1e3:.1f} mm",
            fontsize=10,
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x [m]", fontsize=10)
        ax.set_ylabel("y [m]", fontsize=10)
        ax.legend(fontsize=9)
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(0, N_CTRL_STEPS * DT_CTRL))
        sm.set_array([])
        fig2.colorbar(sm, ax=ax, label="t [s]", shrink=0.7)
    fig2.suptitle(
        f"XY trajectories ({title_prefix}): JAX vs MuJoCo TurtleBot3 Burger\n"
        "(arrows show heading every 25 steps,  color = time)",
        fontsize=13,
    )
    fig2.tight_layout()
    fig2.savefig(out_xy, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_xy}")
    plt.close(fig1)
    plt.close(fig2)


def main():
    rng   = np.random.default_rng(RNG_SEED)
    model = build_mujoco_model()
    data  = mujoco.MjData(model)

    print(f"MuJoCo timestep : {model.opt.timestep:.4f} s")
    print(f"Control period  : {DT_CTRL:.4f} s  ({N_MJ_PER_CTRL} sub-steps)")
    print(f"Wheel radius    : {WHEEL_RADIUS} m")
    print(f"Max wheel vel   : {WHEEL_VEL_MAX} rad/s\n")

    # Initial conditions: (x0, y0, theta0, v_L0, v_R0)
    ics = [
        (0.0,  0.0,  0.0,          0.0,    0.0),
        (1.0, -1.0,  np.pi / 4,    0.05,   0.05),
        (0.0,  0.5,  np.pi / 2,    0.10,  -0.05),
        (0.5,  0.5,  np.pi,        0.0,    0.10),
        (-1.0, 0.0, -np.pi / 3,    0.08,   0.08),
    ]

    t      = np.arange(N_CTRL_STEPS + 1) * DT_CTRL
    labels = ["x [m]", "y [m]", "θ [rad]", r"$v_L$ [m/s]", r"$v_R$ [m/s]"]

    # ── Piecewise-constant control sequences ──────────────────────────────────
    print("=== Piecewise-constant controls ===")
    step_results = []
    for i, (x0, y0, th0, vL0, vR0) in enumerate(ics):
        cmds = make_ctrl_sequence(rng)
        mj_traj, jax_traj = run_trial(model, data, cmds, x0, y0, th0, vL0, vR0)
        step_results.append((x0, y0, th0, mj_traj, jax_traj))
        rmse = np.sqrt(np.mean((mj_traj - jax_traj) ** 2, axis=0))
        print(f"  Trial {i+1}  RMSE: x={rmse[0]:.4f}m  y={rmse[1]:.4f}m  "
              f"θ={rmse[2]:.4f}rad  v_L={rmse[3]:.4f}m/s  v_R={rmse[4]:.4f}m/s")

    _plot_trial_set(
        step_results, t, labels,
        title_prefix="piecewise-constant controls",
        out_states=OUT_PATH.parent / "rover_dynamics_states.png",
        out_xy=OUT_PATH.parent / "rover_dynamics_comparison.png",
    )

    # ── Smooth sinusoidal control sequences ───────────────────────────────────
    print("\n=== Smooth sinusoidal controls ===")
    smooth_results = []
    for i, (x0, y0, th0, vL0, vR0) in enumerate(ics):
        cmds = make_smooth_ctrl_sequence(rng)
        mj_traj, jax_traj = run_trial(model, data, cmds, x0, y0, th0, vL0, vR0)
        smooth_results.append((x0, y0, th0, mj_traj, jax_traj))
        rmse = np.sqrt(np.mean((mj_traj - jax_traj) ** 2, axis=0))
        print(f"  Trial {i+1}  RMSE: x={rmse[0]:.4f}m  y={rmse[1]:.4f}m  "
              f"θ={rmse[2]:.4f}rad  v_L={rmse[3]:.4f}m/s  v_R={rmse[4]:.4f}m/s")

    _plot_trial_set(
        smooth_results, t, labels,
        title_prefix="smooth sinusoidal controls",
        out_states=OUT_PATH.parent / "rover_dynamics_states_smooth.png",
        out_xy=OUT_PATH.parent / "rover_dynamics_comparison_smooth.png",
    )


if __name__ == "__main__":
    main()
