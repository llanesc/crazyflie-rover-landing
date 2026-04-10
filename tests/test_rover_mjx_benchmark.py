#!/usr/bin/env python3
"""Benchmark JAX rover_dynamics.py vs MJX TurtleBot3 Burger — 128 parallel envs.

Compares:
  1. Trajectory accuracy  (JAX approximation vs MJX ground truth)
  2. Wall-clock step time for 128 parallel environments

Usage:
    python tests/test_rover_mjx_benchmark.py
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
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
OUT_DIR      = Path(__file__).parent

# ── Simulation parameters ─────────────────────────────────────────────────────
N_ENVS        = 128
DT_CTRL       = 0.02    # control period [s]
DT_MJ         = 0.002   # MuJoCo/MJX integration timestep [s]
N_MJ_PER_CTRL = int(round(DT_CTRL / DT_MJ))   # 10 sub-steps
N_CTRL_STEPS  = 500     # steps per trajectory  (10 s)
N_WARMUP      = 5       # JIT warmup iterations
N_TIMING      = 20      # timing iterations
RNG_SEED      = 0


# ── MuJoCo model ──────────────────────────────────────────────────────────────

def build_mj_model(mjx_compat: bool = False) -> mujoco.MjModel:
    """Load burger XML with a ground plane.

    mjx_compat=True replaces the tire mesh collision geoms with cylinders so
    MJX does not explode its collision graph on the complex STL meshes.
    """
    xml = BURGER_XML.read_text()

    # Add ground plane with explicit spring/damper contacts.
    # Default MuJoCo soft contacts give near-zero normal force at zero penetration
    # (tire meshes / cylinder wheels touch at exactly dist=0 → zero traction).
    # solref="-3000 -240": direct spring k=3000 N/m, b=240 N·s/m (overdamped).
    # solimp="0.999 0.9999 0.001": near-rigid impedance so equilibrium depth ≈ 1 mm.
    xml = xml.replace(
        "<worldbody>",
        '<worldbody>\n    <geom name="floor" type="plane" size="0 0 0.01"'
        ' rgba=".8 .8 .8 1" solref="-3000 -240" solimp="0.999 0.9999 0.001"/>',
    )

    if mjx_compat:
        # Tire STL meshes cause MJX OOM — replace collision geoms with cylinders.
        # size="{WHEEL_RADIUS} 0.0094": radius matches JAX WHEEL_RADIUS (0.033 m);
        # half-width 9.44 mm matches the tire mesh AABB (measured 0.00944 m).
        # DO NOT copy the mesh quat — a cylinder's default Z-axis already maps to
        # world Y (wheel axle) through the body frame rotation.  Adding the mesh
        # quat rotates the cylinder axis to world Z (vertical), causing the wheel
        # to float above the ground with no contact.
        # MuJoCo contact friction = product of the two geom frictions.
        # friction=10 → contact μ = 10×1 = 10, exceeds the no-slip threshold
        # (required μ≈7 at max wheel velocity from rest with kv=0.1, r=WHEEL_RADIUS).
        for side in ("left", "right"):
            xml = xml.replace(
                f'<geom quat="0.707388 0.706825 0 0" mesh="{side}_tire" class="collision"/>',
                f'<geom type="cylinder" size="{WHEEL_RADIUS} 0.0094" friction="10 0.005 0.0001" condim="4"'
                f' solref="-3000 -240" solimp="0.999 0.9999 0.001" class="collision"/>',
            )

        # The burger_base mesh bottom is co-planar with the wheel cylinders
        # (both at z=0 when body_z=0).  With co-planar contact the solver gives
        # most of the normal force to the base mesh, leaving the wheels with
        # barely any weight → insufficient traction even at high friction.
        # The base body has an explicit <inertial> tag, so disabling the collision
        # geom does not change the robot's mass/inertia.
        xml = xml.replace(
            '<geom pos="-0.032 0 0.01" mesh="burger_base" class="collision"/>',
            '<geom pos="-0.032 0 0.01" mesh="burger_base" contype="0" conaffinity="0"/>',
        )

    # Stiffen caster ball contact (has its own soft solref by default)
    xml = xml.replace(
        'friction="0.0001 0.0001 0.0001" solref="0.02 1" solimp="0.95 0.99 0.001"',
        'friction="0.0001 0.0001 0.0001" solref="-3000 -240" solimp="0.999 0.9999 0.001"',
    )
    # condim=4 adds torsional contact friction (reduces wheel velocity jitter ~45%).
    # Stiff contacts + condim=4 introduce ~0.062 N·m rolling resistance per wheel.
    # Reducing frictionloss 0.1 → 0.042 N·m makes total effective friction match JAX.
    xml = xml.replace('frictionloss="0.1"', 'frictionloss="0.042"')

    tmp = BURGER_XML.parent / "_tmp_mjx_bench.xml"
    tmp.write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()
    model.opt.timestep = DT_MJ
    return model


def find_equilibrium_z(model: mujoco.MjModel) -> float:
    """Find the equilibrium z height for the cylinder-wheel robot.

    Cylinders at body_z=0 have exactly zero penetration → zero contact force.
    Dropping from height causes the robot to tumble (impact + soft contacts).
    Instead start from the analytical equilibrium depth and run a brief settling
    step to let MuJoCo fine-tune the contact state.

    Analytical: δ_eq = mg / (n_contacts × k) ≈ 0.9×9.81 / (3×3000) ≈ 1 mm.
    """
    data = mujoco.MjData(model)
    data.qpos[3] = 1.0                   # identity quaternion (upright)
    # Start at 1.5× the analytical penetration depth so contacts are engaged
    z_start = -1.5 * 0.9 * 9.81 / (3 * 3000)
    data.qpos[2] = z_start
    # Brief settling (50 ms) — robot is already near equilibrium, just damping out
    for _ in range(25):                  # 50 ms at dt=0.002
        mujoco.mj_step(model, data)
    z_eq = float(data.qpos[2])
    return z_eq


def quat_to_yaw_np(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


# ── Random initial conditions & controls ──────────────────────────────────────

def sample_initial_states(rng: np.random.Generator, N: int) -> np.ndarray:
    """Return (N, 5) array of [x, y, theta, v_L, v_R]."""
    states = np.zeros((N, 5))
    states[:, 0] = rng.uniform(-2.0,  2.0, N)   # x
    states[:, 1] = rng.uniform(-2.0,  2.0, N)   # y
    states[:, 2] = rng.uniform(-np.pi, np.pi, N) # theta
    # start from rest
    return states


def sample_controls(rng: np.random.Generator, N: int, T: int) -> np.ndarray:
    """Smooth sinusoidal commands — moderate speed, gradual turns."""
    t     = np.arange(T) * DT_CTRL
    cmds  = np.zeros((N, T, 2))
    fwd   = rng.uniform(0.3 * WHEEL_VEL_MAX, 0.5 * WHEEL_VEL_MAX, N)
    amp   = rng.uniform(0.1 * WHEEL_VEL_MAX, 0.3 * WHEEL_VEL_MAX, N)
    freq  = rng.uniform(0.05, 0.2, N)
    phase = rng.uniform(0, 2 * np.pi, N)
    for i in range(N):
        diff = amp[i] * np.sin(2 * np.pi * freq[i] * t + phase[i])
        cmds[i, :, 0] = np.clip(fwd[i] - diff / 2, -WHEEL_VEL_MAX, WHEEL_VEL_MAX)
        cmds[i, :, 1] = np.clip(fwd[i] + diff / 2, -WHEEL_VEL_MAX, WHEEL_VEL_MAX)
    return cmds


# ── MJX helpers ───────────────────────────────────────────────────────────────

def build_mjx_batch(model: mujoco.MjModel,
                    states_np: np.ndarray,
                    z_eq: float = 0.0) -> mjx.Data:
    """Create a batched MJX Data from (N, 5) [x, y, theta, v_L, v_R].

    z_eq: equilibrium body z height (from find_equilibrium_z).  Must be set to
    the value where cylinder contacts have enough penetration to support the robot
    weight.  At z=0 the cylinders are tangent to the floor (zero penetration) so
    the soft contact gives near-zero normal force and the robot falls.
    """
    N  = states_np.shape[0]
    nq = model.nq   # 9
    nv = model.nv   # 8

    qpos = np.zeros((N, nq))
    qvel = np.zeros((N, nv))
    for i, (x, y, th, vL, vR) in enumerate(states_np):
        qpos[i, 0] = x
        qpos[i, 1] = y
        qpos[i, 2] = z_eq
        qpos[i, 3] = np.cos(th / 2)   # quaternion w
        qpos[i, 4] = 0.0
        qpos[i, 5] = 0.0
        qpos[i, 6] = np.sin(th / 2)   # quaternion z (yaw only)
        qvel[i, 6] = vL / WHEEL_RADIUS
        qvel[i, 7] = vR / WHEEL_RADIUS

    data   = mujoco.MjData(model)
    dx_one = mjx.put_data(model, data)
    dx_bat = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x[None], (N,) + x.shape).copy(), dx_one
    )
    dx_bat = dx_bat.replace(
        qpos=jnp.array(qpos),
        qvel=jnp.array(qvel),
    )
    return dx_bat


def extract_states_mjx(dx: mjx.Data) -> jnp.ndarray:
    """Return (N, 5) [x, y, theta, v_L, v_R] from batched MJX Data."""
    x  = dx.qpos[:, 0]
    y  = dx.qpos[:, 1]
    w_ = dx.qpos[:, 3]
    qz = dx.qpos[:, 6]
    qy = dx.qpos[:, 5]
    qx = dx.qpos[:, 4]
    theta = jnp.arctan2(2.0 * (w_ * qz + qx * qy),
                        1.0 - 2.0 * (qy ** 2 + qz ** 2))
    v_L = WHEEL_RADIUS * dx.qvel[:, 6]
    v_R = WHEEL_RADIUS * dx.qvel[:, 7]
    return jnp.stack([x, y, theta, v_L, v_R], axis=1)


# ── Batched step functions ────────────────────────────────────────────────────

_batch_rover_step = jax.vmap(rover_step, in_axes=(0, 0, None))
_batch_mjx_step   = jax.vmap(mjx.step,  in_axes=(None, 0))


@jax.jit
def jax_control_step(states: jnp.ndarray, ctrls: jnp.ndarray) -> jnp.ndarray:
    """One control period via RK4 — (N,6) states, (N,2) ctrls → (N,6) states."""
    return _batch_rover_step(states, ctrls, DT_CTRL)


@jax.jit
def mjx_control_step(mx: mjx.Model,
                     dx: mjx.Data,
                     ctrls: jnp.ndarray) -> mjx.Data:
    """One control period via N_MJ_PER_CTRL MJX sub-steps."""
    dx = dx.replace(ctrl=ctrls)

    def substep(dx, _):
        return _batch_mjx_step(mx, dx), None

    dx, _ = jax.lax.scan(substep, dx, None, length=N_MJ_PER_CTRL)
    return dx


# ── JAX state helpers ─────────────────────────────────────────────────────────

def jax_states_from(states_np: np.ndarray) -> jnp.ndarray:
    """(N,5) [x,y,th,vL,vR] → (N,6) [x,y,c,s,vL,vR]."""
    x, y, th, vL, vR = (states_np[:, i] for i in range(5))
    return jnp.stack([x, y, np.cos(th), np.sin(th), vL, vR], axis=1)


def jax_extract(states: jnp.ndarray) -> np.ndarray:
    """(N,6) → (N,5) [x,y,theta,vL,vR]."""
    return np.stack([
        np.array(states[:, 0]),
        np.array(states[:, 1]),
        np.arctan2(np.array(states[:, 3]), np.array(states[:, 2])),
        np.array(states[:, 4]),
        np.array(states[:, 5]),
    ], axis=1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng   = np.random.default_rng(RNG_SEED)
    model = build_mj_model(mjx_compat=True)   # cylinder wheels for MJX
    mx    = mjx.put_model(model)

    print(f"nq={model.nq}  nv={model.nv}  nu={model.nu}")
    print(f"N_ENVS={N_ENVS}  N_CTRL_STEPS={N_CTRL_STEPS}  DT_CTRL={DT_CTRL}s")

    # Find equilibrium z: cylinders are perfect geometry so at z=0 the contact
    # penetration is zero → near-zero normal force → robot falls under gravity.
    # We drop the robot and let it settle so the contact penetration balances
    # gravity, then use that z for all initial states (t=0 starts after settling).
    print("Finding equilibrium z (settling)...")
    z_eq = find_equilibrium_z(model)
    print(f"  z_eq = {z_eq*1e3:.3f} mm\n")

    # ── Sample ICs and controls ───────────────────────────────────────────────
    ic_np   = sample_initial_states(rng, N_ENVS)          # (N, 5)
    ctrl_np = sample_controls(rng, N_ENVS, N_CTRL_STEPS)  # (N, T, 2)

    jax_states0 = jax_states_from(ic_np)                  # (N, 6)
    ctrl_jnp    = jnp.array(ctrl_np)                      # (N, T, 2)

    # t=0 starts at the settled state (robot on floor, ready to drive)
    dx_batch = build_mjx_batch(model, ic_np, z_eq=z_eq)   # batched MJX Data

    # ── Run trajectories ──────────────────────────────────────────────────────
    print("Running trajectories...")
    jax_traj = np.zeros((N_CTRL_STEPS + 1, N_ENVS, 5))
    mjx_traj = np.zeros((N_CTRL_STEPS + 1, N_ENVS, 5))

    jax_states  = jax_states0
    jax_traj[0] = jax_extract(jax_states)
    mjx_traj[0] = np.array(extract_states_mjx(dx_batch))

    for k in range(N_CTRL_STEPS):
        ctrl_k      = ctrl_jnp[:, k, :]
        jax_states  = jax_control_step(jax_states, ctrl_k)
        dx_batch    = mjx_control_step(mx, dx_batch, ctrl_k)
        jax_traj[k + 1] = jax_extract(jax_states)
        mjx_traj[k + 1]  = np.array(extract_states_mjx(dx_batch))
        if (k + 1) % 50 == 0:
            print(f"  step {k+1}/{N_CTRL_STEPS}")

    rmse = np.sqrt(np.mean((jax_traj - mjx_traj) ** 2, axis=(0, 1)))
    print(f"\nRMSE (JAX vs MJX, all envs):")
    print(f"  x={rmse[0]*1e3:.2f}mm  y={rmse[1]*1e3:.2f}mm  "
          f"θ={np.degrees(rmse[2]):.3f}°  "
          f"v_L={rmse[3]*1e3:.2f}mm/s  v_R={rmse[4]*1e3:.2f}mm/s")

    # ── Timing ────────────────────────────────────────────────────────────────
    print(f"\nTiming ({N_WARMUP} warmup + {N_TIMING} timed runs)...")

    ctrl_k = ctrl_jnp[:, 0, :]

    # JAX dynamics
    for _ in range(N_WARMUP):
        jax.block_until_ready(jax_control_step(jax_states, ctrl_k))
    t0 = time.perf_counter()
    for _ in range(N_TIMING):
        jax.block_until_ready(jax_control_step(jax_states, ctrl_k))
    jax_ms = (time.perf_counter() - t0) / N_TIMING * 1e3

    # MJX
    for _ in range(N_WARMUP):
        jax.block_until_ready(mjx_control_step(mx, dx_batch, ctrl_k))
    t0 = time.perf_counter()
    for _ in range(N_TIMING):
        jax.block_until_ready(mjx_control_step(mx, dx_batch, ctrl_k))
    mjx_ms = (time.perf_counter() - t0) / N_TIMING * 1e3

    print(f"  JAX rover_dynamics  : {jax_ms:.3f} ms/step  ({N_ENVS} envs)")
    print(f"  MJX TurtleBot3      : {mjx_ms:.3f} ms/step  ({N_ENVS} envs, {N_MJ_PER_CTRL} sub-steps)")
    print(f"  Speedup (MJX/JAX)   : {mjx_ms/jax_ms:.1f}x slower")

    # ── Figure 1: sampled XY trajectories ─────────────────────────────────────
    N_PLOT   = 8
    n_cols   = 2
    n_rows   = N_PLOT // n_cols

    plotted_envs = [idx * (N_ENVS // N_PLOT) for idx in range(N_PLOT)]

    # Shared ±half_span window: consistent scale, centered per env
    all_disp = np.concatenate([
        (jax_traj[:, plotted_envs, :2]
         - jax_traj[0:1, plotted_envs, :2]).ravel(),
        (mjx_traj[:, plotted_envs, :2]
         - mjx_traj[0:1, plotted_envs, :2]).ravel(),
    ])
    half_span = max(0.3, float(np.abs(all_disp).max()) + 0.2)

    fig1, axes1 = plt.subplots(n_rows, n_cols,
                               figsize=(6 * n_cols, 5 * n_rows))
    for idx, ax in enumerate(axes1.flat):
        env  = plotted_envs[idx]
        x0e  = float(jax_traj[0, env, 0])
        y0e  = float(jax_traj[0, env, 1])

        ax.plot(mjx_traj[:, env, 0], mjx_traj[:, env, 1],
                color="tab:blue", lw=2.0, label="MJX", zorder=2)
        ax.plot(jax_traj[:, env, 0], jax_traj[:, env, 1],
                color="tab:red", lw=2.0, ls="--", label="JAX", zorder=3)

        # Start marker
        ax.plot(x0e, y0e, "ko", ms=7, zorder=5)

        ax.set_xlim(x0e - half_span, x0e + half_span)
        ax.set_ylim(y0e - half_span, y0e + half_span)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_title(f"env {env}", fontsize=10)
        ax.set_xlabel("x [m]", fontsize=9)
        ax.set_ylabel("y [m]", fontsize=9)
        ax.tick_params(labelsize=8)

    fig1.suptitle(
        f"XY trajectories — {N_PLOT} envs, {N_CTRL_STEPS*DT_CTRL:.0f} s\n"
        f"JAX rover_dynamics (red dashed)  vs  MJX TurtleBot3 (blue solid)\n"
        f"XY RMSE = {np.sqrt(rmse[0]**2+rmse[1]**2)*1e3:.1f} mm  |  "
        f"all axes ±{half_span:.2f} m",
        fontsize=12,
    )
    fig1.tight_layout()
    out1 = OUT_DIR / "rover_mjx_trajectories.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out1}")

    # ── Figure 2: timing bar chart ─────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    bars = ax2.bar(["JAX\nrover_dynamics\n(1 RK4 step)",
                    f"MJX\nTurtleBot3 Burger\n({N_MJ_PER_CTRL} sub-steps)"],
                   [jax_ms, mjx_ms],
                   color=["tab:blue", "tab:orange"],
                   width=0.5, edgecolor="black")
    for bar, val in zip(bars, [jax_ms, mjx_ms]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02 * max(jax_ms, mjx_ms),
                 f"{val:.3f} ms", ha="center", va="bottom", fontsize=11)
    ax2.set_ylabel("wall-clock time per control step [ms]", fontsize=10)
    ax2.set_title(
        f"Step time — {N_ENVS} parallel envs  (JAX/MJX, CPU)\n"
        f"MJX is {mjx_ms/jax_ms:.1f}× slower than JAX dynamics",
        fontsize=11,
    )
    ax2.grid(axis="y", alpha=0.4)
    fig2.tight_layout()
    out2 = OUT_DIR / "rover_mjx_timing.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved → {out2}")


if __name__ == "__main__":
    main()
