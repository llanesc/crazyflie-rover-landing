#!/usr/bin/env python3
"""Compare rover_dynamics.py (JAX) vs TurtleBot3 Burger in Gazebo Harmonic.

Runs gz sim headlessly, publishes wheel velocity commands via the DiffDrive
cmd_vel topic, records odometry, and compares against JAX model predictions.

The Gazebo model uses cylinders with mu=100000 (no slip) and ODE contact
parameters — a clean reference independent of MuJoCo's contact solver.

Usage:
    python tests/test_gazebo_comparison.py
"""

import sys
import os
import subprocess
import time
import threading
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Import rover_dynamics BEFORE adding system site-packages to sys.path.
# The gz Python bindings live in /usr/lib/python3/dist-packages which also
# contains an older scipy compiled against a different numpy ABI.  Importing
# rover_dynamics first ensures our venv's packages win the import race.
from crazyflie_rover_landing.envs.rover_dynamics import (
    rover_step, WHEEL_RADIUS, WHEEL_VEL_MAX,
)

# Now add the system site-packages so gz modules can be found.
sys.path.insert(0, "/usr/lib/python3/dist-packages")
import gz.transport13 as transport
import gz.msgs10.twist_pb2       as twist_pb2
import gz.msgs10.odometry_pb2    as odom_pb2

# ── Constants ──────────────────────────────────────────────────────────────────
WHEEL_BASE   = 0.160   # m  (burger wheel separation)
DT_CTRL      = 0.02    # s  control period
N_CTRL_STEPS = 500     # 10 s total
RNG_SEED     = 42
OUT_DIR      = Path(__file__).parent

# Gazebo resource path (turtlebot3 models)
GZ_RESOURCE  = "/opt/ros/jazzy/share/turtlebot3_gazebo/models"

# ── World SDF ─────────────────────────────────────────────────────────────────
WORLD_SDF = """\
<?xml version="1.0"?>
<sdf version="1.8">
  <world name="default">
    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>

    <physics type="ode">
      <max_step_size>0.002</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>500</real_time_update_rate>
    </physics>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane><normal>0 0 1</normal><size>100 100</size></plane>
          </geometry>
          <surface>
            <friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction>
          </surface>
        </collision>
      </link>
    </model>

    <!-- TurtleBot3 Burger spawned at origin, facing +X -->
    <include>
      <uri>model://turtlebot3_burger</uri>
      <name>turtlebot3_burger</name>
      <pose>0 0 0 0 0 0</pose>
    </include>

  </world>
</sdf>
"""

# ── JAX helpers ───────────────────────────────────────────────────────────────

def make_smooth_controls(rng: np.random.Generator) -> np.ndarray:
    """Smooth sinusoidal wheel velocity commands — same style as comparison test."""
    t     = np.arange(N_CTRL_STEPS) * DT_CTRL
    fwd   = rng.uniform(0.3, 0.5) * WHEEL_VEL_MAX
    amp   = rng.uniform(0.05, 0.15) * WHEEL_VEL_MAX
    freq  = rng.uniform(0.05, 0.10)
    phase = rng.uniform(0, 2 * np.pi)
    diff  = amp * np.sin(2 * np.pi * freq * t + phase)
    wL = np.clip(fwd - diff / 2, 0.0, WHEEL_VEL_MAX)
    wR = np.clip(fwd + diff / 2, 0.0, WHEEL_VEL_MAX)
    return np.stack([wL, wR], axis=1)   # (T, 2)  rad/s


def run_jax(ctrl: np.ndarray):
    """Run JAX rover model, return (T+1, 5) [x, y, theta, vL, vR]."""
    import jax.numpy as jnp
    state = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])   # x,y,c,s,vL,vR
    traj  = np.zeros((N_CTRL_STEPS + 1, 5))
    traj[0] = [0, 0, 0, 0, 0]
    for k in range(N_CTRL_STEPS):
        state    = rover_step(state, jnp.array(ctrl[k]), DT_CTRL)
        s        = np.array(state)
        traj[k+1] = [s[0], s[1], np.arctan2(s[3], s[2]), s[4], s[5]]
    return traj


# ── cmd_vel conversion ────────────────────────────────────────────────────────

def wheel_to_cmdvel(wL_rads: float, wR_rads: float):
    """Convert wheel angular velocities (rad/s) to Twist (linear, angular)."""
    vL = wL_rads * WHEEL_RADIUS
    vR = wR_rads * WHEEL_RADIUS
    v  = (vL + vR) / 2.0
    w  = (vR - vL) / WHEEL_BASE
    msg = twist_pb2.Twist()
    msg.linear.x  = v
    msg.angular.z = w
    return msg


# ── Gazebo data collection ────────────────────────────────────────────────────

class GazeboRecorder:
    """Subscribes to /odom and records the trajectory."""

    def __init__(self):
        self.node       = transport.Node()
        self.lock       = threading.Lock()
        self.odom_buf   : list = []    # list of (stamp_s, x, y, yaw, vx, vw)
        self._sub_ok    = False

    def _odom_cb(self, msg):
        # msg: gz.msgs.Odometry
        p    = msg.pose.position
        q    = msg.pose.orientation
        # yaw from quaternion
        yaw  = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        vx   = msg.twist.linear.x
        vw   = msg.twist.angular.z
        t_s  = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        with self.lock:
            self.odom_buf.append((t_s, p.x, p.y, yaw, vx, vw))

    def start(self):
        ok = self.node.subscribe(odom_pb2.Odometry, "/odom", self._odom_cb)
        self._sub_ok = ok
        return ok

    def snapshot(self):
        with self.lock:
            return list(self.odom_buf)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng  = np.random.default_rng(RNG_SEED)
    ctrl = make_smooth_controls(rng)   # (T, 2) rad/s

    # ── JAX ──────────────────────────────────────────────────────────────────
    print("Running JAX model...")
    jax_traj = run_jax(ctrl)   # (T+1, 5): x, y, theta, vL, vR
    print(f"  JAX final pos: x={jax_traj[-1,0]:.3f}  y={jax_traj[-1,1]:.3f}  "
          f"θ={np.degrees(jax_traj[-1,2]):.1f}°")

    # ── Gazebo ────────────────────────────────────────────────────────────────
    # Write world SDF to a temp file next to gazebo models
    sdf_path = Path(tempfile.mktemp(suffix=".sdf"))
    sdf_path.write_text(WORLD_SDF)

    env = os.environ.copy()
    env["GZ_SIM_RESOURCE_PATH"] = GZ_RESOURCE

    print("\nLaunching gz sim (headless)...")
    gz_proc = subprocess.Popen(
        ["gz", "sim", "--headless-rendering", "-s", "-r", str(sdf_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for simulator to be ready (poll topic list)
    node_check = transport.Node()
    deadline = time.time() + 15.0
    print("  Waiting for gz sim to start...", end="", flush=True)
    while time.time() < deadline:
        time.sleep(0.5)
        topics = node_check.topic_list()
        if any("/odom" in t for t in topics):
            print(" ready.")
            break
        print(".", end="", flush=True)
    else:
        print("\nERROR: gz sim did not start within 15 s")
        gz_proc.terminate()
        sdf_path.unlink(missing_ok=True)
        return

    # Subscribe to odometry
    rec = GazeboRecorder()
    if not rec.start():
        print("ERROR: could not subscribe to /odom")
        gz_proc.terminate()
        sdf_path.unlink(missing_ok=True)
        return

    # Give the robot a moment to settle on the ground
    print("  Settling (1 s)...")
    time.sleep(1.0)

    # cmd_vel publisher
    pub_node = transport.Node()
    pub      = pub_node.advertise("/cmd_vel", twist_pb2.Twist)
    time.sleep(0.2)   # let publisher register

    # ── Control loop ─────────────────────────────────────────────────────────
    print(f"  Running {N_CTRL_STEPS} control steps × {DT_CTRL}s = "
          f"{N_CTRL_STEPS*DT_CTRL:.0f} s...")
    t_start = time.time()
    for k in range(N_CTRL_STEPS):
        t_next = t_start + (k + 1) * DT_CTRL
        msg    = wheel_to_cmdvel(ctrl[k, 0], ctrl[k, 1])
        pub.publish(msg)
        # busy-wait for the next control period
        now = time.time()
        if now < t_next:
            time.sleep(t_next - now)

    # Stop robot
    pub.publish(twist_pb2.Twist())
    time.sleep(0.2)

    odom_buf = rec.snapshot()
    gz_proc.terminate()
    gz_proc.wait(timeout=5)
    sdf_path.unlink(missing_ok=True)

    if not odom_buf:
        print("ERROR: no odometry data received")
        return

    print(f"  Received {len(odom_buf)} odom samples")
    odom_arr = np.array(odom_buf)   # (M, 6): t, x, y, yaw, vx, vw

    # ── Align odometry to control timesteps ──────────────────────────────────
    # Normalise time relative to first odom message
    t0 = odom_arr[0, 0]
    odom_t = odom_arr[:, 0] - t0

    t_ctrl = np.arange(N_CTRL_STEPS + 1) * DT_CTRL   # (T+1,)
    gz_traj = np.zeros((N_CTRL_STEPS + 1, 3))          # x, y, theta
    for i, tc in enumerate(t_ctrl):
        # nearest odom sample
        idx = int(np.argmin(np.abs(odom_t - tc)))
        gz_traj[i] = odom_arr[idx, 1:4]

    gz_final = gz_traj[-1]
    print(f"\n  Gazebo final pos: x={gz_final[0]:.3f}  y={gz_final[1]:.3f}  "
          f"θ={np.degrees(gz_final[2]):.1f}°")

    # ── RMSE ─────────────────────────────────────────────────────────────────
    rmse_x   = np.sqrt(np.mean((gz_traj[:, 0] - jax_traj[:, 0])**2))
    rmse_y   = np.sqrt(np.mean((gz_traj[:, 1] - jax_traj[:, 1])**2))
    rmse_yaw = np.sqrt(np.mean((gz_traj[:, 2] - jax_traj[:, 2])**2))
    print(f"\nRMSE (JAX vs Gazebo):  x={rmse_x*1e3:.1f}mm  "
          f"y={rmse_y*1e3:.1f}mm  θ={np.degrees(rmse_yaw):.2f}°")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(jax_traj[:, 0], jax_traj[:, 1],
            color="tab:red",  lw=2, ls="--", label="JAX rover_dynamics")
    ax.plot(gz_traj[:, 0],  gz_traj[:, 1],
            color="tab:blue", lw=2,           label="Gazebo (ODE, no-slip)")
    ax.plot(0, 0, "ko", ms=8, zorder=5, label="start")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("XY trajectory")
    ax.legend()

    ax2 = axes[1]
    t_ax = t_ctrl
    ax2.plot(t_ax, jax_traj[:, 0], color="tab:red",  ls="--", label="JAX x")
    ax2.plot(t_ax, gz_traj[:, 0],  color="tab:blue",           label="Gazebo x")
    ax2.plot(t_ax, jax_traj[:, 1], color="orange",   ls="--", label="JAX y")
    ax2.plot(t_ax, gz_traj[:, 1],  color="steelblue",           label="Gazebo y")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("position [m]")
    ax2.set_title("Position vs time")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.suptitle(
        f"JAX rover_dynamics vs Gazebo TurtleBot3 Burger\n"
        f"RMSE: x={rmse_x*1e3:.1f}mm  y={rmse_y*1e3:.1f}mm  "
        f"θ={np.degrees(rmse_yaw):.2f}°\n"
        f"Gazebo: ODE physics, cylinders, mu=100000 (no slip), "
        f"JAX: actuator ODE",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUT_DIR / "rover_gazebo_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
