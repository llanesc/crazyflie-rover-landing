#!/bin/bash
# Launch CrazySim with drone + simulated X3 rover (sim-sim mode).
# Rover uses mecanum dynamics and responds to /cmd_vel.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker run --rm -it \
  --network=host --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e ROS_DOMAIN_ID=28 \
  -e ROS_DISABLE_TYPE_DESCRIPTION_SERVICE=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "${REPO_DIR}/external/CrazySim:/workspace" \
  -w /workspace \
  -u developer \
  crazysim-landing:latest \
  bash /workspace/crazyflie-firmware/tools/crazyflie-simulation/simulator_files/mujoco/launch/sitl_landing.sh \
  "$@"
