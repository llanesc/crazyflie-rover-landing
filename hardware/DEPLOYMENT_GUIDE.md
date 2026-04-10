# Crazyflie-Rover Landing: Deployment Guide

## Overview

Three deployment modes for the cooperative drone-rover landing system:
1. **Full Sim** — CrazySim drone + simulated X3 rover (all on PC)
2. **Mixed** — CrazySim drone + real X3 rover
3. **Full Hardware** — Real Crazyflie + real X3 rover + OptiTrack

## Prerequisites

### PC Setup
- Ubuntu 24.04 with ROS2 Jazzy
- NVIDIA GPU (for MuJoCo rendering in CrazySim)
- Python 3.12

### Clone and Setup
```bash
git clone --recurse-submodules git@github.com:llanesc/crazyflie-rover-landing.git
cd crazyflie-rover-landing

# Create training venv
uv venv local_env
VIRTUAL_ENV=local_env uv pip install -e . -e external/crazyflow -e external/leap-c -e external/skrl

# Create hardware venv
uv venv hardware/local_env_hardware
VIRTUAL_ENV=hardware/local_env_hardware uv pip install -e . --no-deps
VIRTUAL_ENV=hardware/local_env_hardware uv pip install torch numpy pyyaml
VIRTUAL_ENV=hardware/local_env_hardware uv pip install -e external/leap-c/external/acados/interfaces/acados_template --use-pep517
VIRTUAL_ENV=hardware/local_env_hardware uv pip install -e external/leap-c --no-deps

# Build ROS2 workspace
source /opt/ros/jazzy/setup.bash
cd hardware/ros2_ws
colcon build
```

### CrazySim Docker
Build the CrazySim Docker image:
```bash
cd external/CrazySim
docker build -t crazysim-landing:latest -f .devcontainer/Dockerfile .
```

---

## Mode 1: Full Sim (CrazySim Drone + Simulated Rover)

Everything runs on PC. CrazySim handles drone physics + firmware SITL. Rover dynamics simulated via mecanum model. Communication between rover and CrazySim via UDP bridge.

### Terminal 1 — CrazySim
```bash
bash hardware/crazysim_sim.sh -d 0.002
```
Options: `--drone-pos X,Y`, `--rover-pos X,Y`, `--no-vis`, `--ground-effect`

### Terminal 2 — Landing System
```bash
source /opt/ros/jazzy/setup.bash
source hardware/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=28
ros2 launch cf_landing_drone launch_full_sim.py
```

### Operation
1. Wait for CrazySim to show the MuJoCo viewer
2. In RViz, click **Takeoff** → wait for HOVER
3. Click **Run** → policy controls both drone and rover
4. Landing detected automatically via dwell timer

---

## Mode 2: Mixed (CrazySim Drone + Real X3 Rover)

Drone runs in CrazySim SITL. Real X3 rover on the network. CrazySim mirrors the real rover position via UDP bridge.

### X3 Rover Setup

#### SSH and Docker
```bash
ssh jetson@192.168.1.11  # X3 on hotspot

# IMPORTANT: Kill the startup script that holds the serial port
pkill -f rosmaster_main.py

# Start Docker container
docker run -d --rm --privileged --network host \
  -v /dev:/dev \
  -v /home/jetson/cf_landing:/workspace \
  -w /workspace \
  --name cf_x3 \
  cf_landing:jazzy \
  bash -c 'sleep 3600'
```

#### Sync Clock (critical — prevents TF_OLD_DATA errors)
```bash
ssh jetson@192.168.1.11 "sudo date -s @$(date +%s)"
```

#### Launch X3 Bringup
```bash
docker exec -d cf_x3 bash -c '\
  source /opt/ros/jazzy/setup.bash && \
  source /workspace/ros2_ws/install/setup.bash && \
  export ROS_DOMAIN_ID=28 && \
  ros2 launch yahboomcar_bringup yahboomcar_bringup_X3_nojoy_launch.py'
```

#### X3 EKF Configuration
The nojoy launch runs:
- **Mcnamu_driver_X3** — reads encoders/IMU from Rosmaster board, publishes `vel_raw`
- **base_node_X3** — integrates `vel_raw` into `odom_raw` with calibration scales
- **imu_filter_madgwick** — fuses IMU data
- **twist_stamper** — wraps `vel_raw` for EKF
- **EKF (robot_localization)** — fuses `odom_raw` (differential mode) + IMU yaw rate

EKF params are passed as Python dict in the launch file (bypasses Jazzy YAML namespace param loading bug). Key settings:
- `odom0: /x3/odom_raw` with `odom0_differential: true` (so `set_pose` resets cleanly)
- `imu0: /x3/imu/data` — **yaw rate only** (no orientation — IMU has ~10° accel bias)
- `two_d_mode: true`
- Watchdog in driver stops motors after 1s of no `cmd_vel`

#### Known Issues
- **Serial crash**: If `rosmaster_main.py` (Yahboom web server) is running, it holds `/dev/myserial` and the driver's receive thread crashes. Always kill it first.
- **IMU orientation**: The board IMU has ~4° roll / ~9° pitch bias. Only yaw rate is fused in EKF.
- **Encoder accuracy**: Good at <0.5 m/s (~3% error). At 1 m/s, encoders overcount by ~44% due to mecanum wheel slip. EKF with IMU constrains this.

### Terminal 1 — CrazySim (mirror mode)
```bash
bash hardware/crazysim_hw.sh -d 0.002
```

### Terminal 2 — Landing System
```bash
source /opt/ros/jazzy/setup.bash
source hardware/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=28
ros2 launch cf_landing_drone launch_mixed.py
```

DDS: Both PC and X3 use Fast DDS (default Jazzy). No Cyclone needed — rover bridge uses UDP.

---

## Mode 3: Full Hardware (Real Crazyflie + Real X3 + OptiTrack)

Same as mixed mode but replace CrazySim with real Crazyflie via Crazyswarm2.

*(Not yet fully tested — needs OptiTrack mocap integration. Add `pose0` to EKF for mocap position input.)*

---

## Architecture

### Control Flow
```
TAKEOFF (high-level firmware) → HOVER (PID position hold, low-level) → RUN (policy) → LANDED
```

- **TAKEOFF**: Crazyswarm2 high-level takeoff command. Firmware Mellinger controller holds altitude.
- **HOVER**: Switches to low-level attitude control. PID position hold (MAPE-style) keeps position.
- **RUN**: Policy takes over seamlessly — no gap between hover PID and policy commands.
- **LANDED**: Dwell timer (0.5s on pad) instead of velocity check (EKF velocity unreliable on contact).

### Firmware Parameters (crazyflies_sim.yaml)
```yaml
stabilizer:
  estimator: 2  # kalman
  controller: 2  # mellinger
```
Mass is NOT set in firmware — the `so_rpy` thrust mapping handles it in the policy.

### Sim-to-Real Notes
- **Pitch rate sign**: `stateEstimateZ.ratePitch` is negated in firmware (`-gyro.y`). The drone agent negates it back.
- **Policy ready gating**: Takeoff blocked until both drone and rover policies report ready (important for ACMPC solver build time).
- **Hover thrust on low-level switch**: Sends hover-thrust setpoint (not zero) after `notify_setpoints_stop` to prevent initial drop.

### UDP Rover Bridge
Replaces rclpy in CrazySim Docker for better RTF (~0.95x vs 0.7x):
- Port 19960: CrazySim → bridge (rover state, 7 doubles)
- Port 19961: Bridge → CrazySim (cmd_vel 3 doubles, OR mirror state 56 bytes)

### Training vs CrazySim Differences Found
1. **so_rpy yaw coupling**: Crazyflow's fitted model has `rpy_coef * euler_angles` term that creates fake yaw torque from initial yaw offset. Not present in CrazySim first-principles physics.
2. **Attitude controller response**: CrazySim firmware Mellinger has stronger yaw damping (kR_z=60000, kw_z=12000) than Crazyflow. Body rates (wx, wy) are 2-3x larger in CrazySim for same commands.
3. **Odom timing**: CrazySim velocity and position come from firmware state estimator at different rates — can be slightly out of sync.

### Disturbance Model
Training supports Ornstein-Uhlenbeck (OU) disturbance for smooth, correlated forces:
```yaml
disturbance_type: "ou"
disturbance_force_std: 0.005
disturbance_torque_std: 5.0e-5
disturbance_ou_theta: 0.5  # lower = smoother
```

### Ground Effect
Per-rotor Cheeseman-Bennett model with 1.5x scale factor. Detects both ground and rover pad surfaces. Matched between training env and CrazySim.
