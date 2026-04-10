#!/usr/bin/env python3
"""Render JAX vs MuJoCo rover trajectories side-by-side using the burger mesh.

Both robots receive identical controls and start from the same initial state.
  Red  robot — JAX rover_dynamics.py (approximate model used in training)
  Blue robot — MuJoCo TurtleBot3 Burger (full physics ground truth)

Output: tests/rover_comparison.mp4

Usage:
    python tests/render_rover_comparison.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import imageio
from pathlib import Path

from crazyflie_rover_landing.envs.rover_dynamics import (
    rover_step, WHEEL_RADIUS, WHEEL_VEL_MAX,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BURGER_XML   = (PROJECT_ROOT / "external/robotis_mujoco_menagerie"
                             / "robotis_tb3/turtlebot3_burger.xml")
OUT_PATH     = Path(__file__).parent / "rover_comparison.mp4"

# ── Parameters ────────────────────────────────────────────────────────────────
DT_CTRL       = 0.02
DT_MJ         = 0.002
N_MJ_PER_CTRL = int(round(DT_CTRL / DT_MJ))
N_CTRL_STEPS  = 300    # 6 seconds
RENDER_FPS    = 30
RENDER_EVERY  = max(1, int(round(1.0 / (RENDER_FPS * DT_CTRL))))
RNG_SEED      = 7


# ── MuJoCo physics model ──────────────────────────────────────────────────────

def build_physics_model() -> mujoco.MjModel:
    """Burger XML with ground plane. Uses full tire meshes (no MJX needed)."""
    xml = BURGER_XML.read_text()
    # Add stiff ground plane (same as comparison test)
    xml = xml.replace(
        "<worldbody>",
        '<worldbody>\n    <geom name="floor" type="plane" size="0 0 0.01"'
        ' rgba=".7 .7 .7 1" solref="-3000 -240" solimp="0.999 0.9999 0.001"/>',
    )
    # Disable burger_base collision mesh
    xml = xml.replace(
        '<geom pos="-0.032 0 0.01" mesh="burger_base" class="collision"/>',
        '<geom pos="-0.032 0 0.01" mesh="burger_base" contype="0" conaffinity="0"/>',
    )
    # Replace tire mesh collision geoms with exact-radius cylinders + stiff contacts
    for side in ("left", "right"):
        xml = xml.replace(
            f'<geom quat="0.707388 0.706825 0 0" mesh="{side}_tire" class="collision"/>',
            f'<geom type="cylinder" size="0.033 0.009"'
            f' friction="10 0.005 0.0001" condim="4"'
            f' solref="-3000 -240" solimp="0.999 0.9999 0.001" class="collision"/>',
        )
    # Stiffen caster ball contact (has its own soft solref by default)
    xml = xml.replace(
        'friction="0.0001 0.0001 0.0001" solref="0.02 1" solimp="0.95 0.99 0.001"',
        'friction="0.0001 0.0001 0.0001" solref="-3000 -240" solimp="0.999 0.9999 0.001"',
    )
    # Reduce frictionloss to compensate for stiff-contact rolling resistance
    xml = xml.replace('frictionloss="0.1"', 'frictionloss="0.042"')
    tmp = BURGER_XML.parent / "_tmp_render_phys.xml"
    tmp.write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()
    model.opt.timestep = DT_MJ
    return model


def quat_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def mj_set_state(model, data, x, y, theta):
    mujoco.mj_resetData(model, data)
    data.qpos[0] = x
    data.qpos[1] = y
    data.qpos[2] = 0.0
    data.qpos[3] = np.cos(theta / 2.0)
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = np.sin(theta / 2.0)
    mujoco.mj_forward(model, data)


# ── Render model (two burger robots, no physics) ──────────────────────────────

def build_render_model() -> tuple[mujoco.MjModel, int, int]:
    """Return (model, jax_qpos_addr, mujoco_qpos_addr)."""
    scene = mujoco.MjSpec.from_string("""
    <mujoco>
      <visual>
        <global offwidth="1280" offheight="720"/>
        <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
      </visual>
      <asset>
        <texture name="checker" type="2d" builtin="checker"
                 rgb1=".18 .18 .18" rgb2=".32 .32 .32"
                 width="512" height="512"/>
        <material name="checker" texture="checker"
                  texrepeat="8 8" texuniform="true" reflectance="0.05"/>
      </asset>
      <worldbody>
        <light pos="0 0 4" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" directional="true"/>
        <geom name="floor" type="plane" size="0 0 0.01" material="checker"/>
      </worldbody>
    </mujoco>
    """)

    scene.copy_during_attach = True
    for prefix in ("jax_", "mj_"):
        b_spec = mujoco.MjSpec.from_file(str(BURGER_XML))
        base   = b_spec.body("base")
        frame  = scene.worldbody.add_frame()
        frame.attach_body(base, prefix, "")

    model = scene.compile()

    # Tint geoms by robot
    for gid in range(model.ngeom):
        bid   = model.geom_bodyid[gid]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if bname.startswith("jax_"):
            model.geom_rgba[gid] = [1.0, 0.3, 0.3, 1.0]   # red
        elif bname.startswith("mj_"):
            model.geom_rgba[gid] = [0.3, 0.55, 1.0, 1.0]  # blue

    jax_jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "jax_base_joint")
    mj_jid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mj_base_joint")
    jax_addr = int(model.jnt_qposadr[jax_jid])
    mj_addr  = int(model.jnt_qposadr[mj_jid])
    return model, jax_addr, mj_addr


def set_pose(data: mujoco.MjData, addr: int, x: float, y: float, theta: float):
    data.qpos[addr    ] = x
    data.qpos[addr + 1] = y
    data.qpos[addr + 2] = 0.0
    data.qpos[addr + 3] = np.cos(theta / 2.0)
    data.qpos[addr + 4] = 0.0
    data.qpos[addr + 5] = 0.0
    data.qpos[addr + 6] = np.sin(theta / 2.0)


# ── Smooth control sequence ───────────────────────────────────────────────────

def make_smooth_controls(rng: np.random.Generator) -> np.ndarray:
    """Sinusoidal differential drive — moderate speed, gentle turns."""
    t     = np.arange(N_CTRL_STEPS) * DT_CTRL
    fwd   = 0.35 * WHEEL_VEL_MAX
    amp   = 0.06 * WHEEL_VEL_MAX
    freq  = rng.uniform(0.05, 0.10)
    phase = rng.uniform(0, 2 * np.pi)
    diff  = amp * np.sin(2 * np.pi * freq * t + phase)
    wL    = np.clip(fwd - diff / 2, 0.0, WHEEL_VEL_MAX)
    wR    = np.clip(fwd + diff / 2, 0.0, WHEEL_VEL_MAX)
    return np.stack([wL, wR], axis=1)   # (T, 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(RNG_SEED)

    # Initial conditions — offset in Y so robots are side by side
    x0, theta0 = 0.0, 0.0
    jax_y0, mj_y0 = +0.3, -0.3

    ctrl_np = make_smooth_controls(rng)   # (T, 2)

    # Physics model for MuJoCo ground truth
    phys_model = build_physics_model()
    phys_data  = mujoco.MjData(phys_model)
    mj_set_state(phys_model, phys_data, x0, mj_y0, theta0)

    # Render model
    render_model, jax_addr, mj_addr = build_render_model()
    render_data = mujoco.MjData(render_model)

    # JAX state: [x, y, cos(θ), sin(θ), v_L, v_R]
    jax_state = jnp.array([[x0, jax_y0, np.cos(theta0), np.sin(theta0), 0.0, 0.0]])

    camera          = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type     = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.5
    camera.elevation = -55.0
    camera.azimuth  = 45.0
    camera.lookat[:] = [0.0, 0.0, 0.0]

    renderer = mujoco.Renderer(render_model, height=720, width=1280)
    frames   = []

    print(f"Rendering {N_CTRL_STEPS} steps ({N_CTRL_STEPS * DT_CTRL:.1f} s) ...")

    for k in range(N_CTRL_STEPS):
        # ── JAX step ──
        u         = jnp.array([[ctrl_np[k, 0], ctrl_np[k, 1]]])
        jax_state = jax.vmap(rover_step, in_axes=(0, 0, None))(jax_state, u, DT_CTRL)
        jax_x     = float(jax_state[0, 0])
        jax_y     = float(jax_state[0, 1])
        jax_theta = float(np.arctan2(float(jax_state[0, 3]), float(jax_state[0, 2])))

        # ── MuJoCo step ──
        phys_data.ctrl[0] = ctrl_np[k, 0]
        phys_data.ctrl[1] = ctrl_np[k, 1]
        for _ in range(N_MJ_PER_CTRL):
            mujoco.mj_step(phys_model, phys_data)
        mj_x     = float(phys_data.qpos[0])
        mj_y     = float(phys_data.qpos[1])
        mj_theta = quat_to_yaw(phys_data.qpos[3:7])

        # Camera tracks midpoint
        camera.lookat[0] = (jax_x + mj_x) / 2.0
        camera.lookat[1] = (jax_y + mj_y) / 2.0

        if k % RENDER_EVERY == 0:
            set_pose(render_data, jax_addr, jax_x, jax_y, jax_theta)
            set_pose(render_data, mj_addr,  mj_x,  mj_y,  mj_theta)
            mujoco.mj_forward(render_model, render_data)
            renderer.update_scene(render_data, camera=camera)
            frames.append(renderer.render().copy())

        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{N_CTRL_STEPS}  JAX=({jax_x:.3f},{jax_y:.3f})  "
                  f"MuJoCo=({mj_x:.3f},{mj_y:.3f})")

    renderer.close()

    print(f"\nSaving {len(frames)} frames → {OUT_PATH}")
    imageio.mimwrite(str(OUT_PATH), frames, fps=RENDER_FPS, quality=8)
    print("Done.")


if __name__ == "__main__":
    main()
