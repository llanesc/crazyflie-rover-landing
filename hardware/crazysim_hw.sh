#!/bin/bash
# Launch CrazySim with drone + hardware-mirrored X3 rover (mixed/hw mode).
# Rover visual mirrors real X3 position from /rover/odom (no dynamics sim).

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker run --rm -it \
  --network=host --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e ROS_DOMAIN_ID=28 \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "${REPO_DIR}/external/CrazySim:/workspace" \
  -w /workspace \
  -u developer \
  crazysim-landing:latest \
  bash /workspace/crazyflie-firmware/tools/crazyflie-simulation/simulator_files/mujoco/launch/sitl_landing.sh \
  --rover-mirror \
  "$@"
