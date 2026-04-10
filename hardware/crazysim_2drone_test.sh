#!/bin/bash
# Launch CrazySim with 2 drones, NO rover — for RTF baseline comparison.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker run --rm -it \
  --network=host --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "${REPO_DIR}/external/CrazySim:/workspace" \
  -w /workspace \
  -u developer \
  crazysim-landing:latest \
  bash -c '
    source /opt/ros/humble/setup.bash
    BUILD=/workspace/crazyflie-firmware/sitl_make/build
    SIM=/workspace/crazyflie-firmware/tools/crazyflie-simulation/simulator_files/mujoco
    pkill -x cf2 2>/dev/null; sleep 1
    mkdir -p $BUILD/0 $BUILD/1
    pushd $BUILD/0 > /dev/null && stdbuf -oL $BUILD/cf2 19950 > out.log 2> error.log & popd > /dev/null
    pushd $BUILD/1 > /dev/null && stdbuf -oL $BUILD/cf2 19951 > out.log 2> error.log & popd > /dev/null
    sleep 1
    python3 $SIM/crazysim.py --vis --dt 0.002 --port 19950 -- 0,0 1,0
  '
