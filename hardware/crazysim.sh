#!/bin/bash
# Launch CrazySim with drone + X3 rover for landing experiments.
# Run from anywhere on your PC.
#
# Usage:
#   ./hardware/crazysim.sh
#   ./hardware/crazysim.sh --drone-pos 1.5,0 --rover-pos 0,0
#   ./hardware/crazysim.sh -m cf2x_T350 --wind-speed 2 --turbulence moderate

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker run --rm -it \
  --network=host --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e ROS_DOMAIN_ID=28 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e ROS_DISABLE_TYPE_DESCRIPTION_SERVICE=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "${REPO_DIR}/external/CrazySim:/workspace" \
  -w /workspace \
  -u developer \
  crazysim-landing:latest \
  bash /workspace/crazyflie-firmware/tools/crazyflie-simulation/simulator_files/mujoco/launch/sitl_landing.sh \
  "$@"
