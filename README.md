# Cooperative Drone-Rover Landing

Multi-agent reinforcement learning for autonomous landing of a Crazyflie drone on a moving Yahboom RosMaster X3 mecanum rover.

Both agents (drone + rover) are trained cooperatively using MAPPO with either MLP or Actor-Critic MPC (ACMPC) policies via [LEAP-C](https://github.com/leap-c/leap-c) differentiable MPC.

## Architecture

```
                    Drone (Crazyflie)
                    ┌──────────────┐
                    │  MLP / ACMPC │ ← 29D observation
                    │    Policy    │ → [roll, pitch, yaw, thrust]
                    └──────┬───────┘
                           │ AttitudeSetpoint
                    ┌──────▼───────┐
                    │  Mellinger   │
                    │  Controller  │
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │    Landing Pad (X3)     │
              └────────────▲────────────┘
                           │
                    ┌──────┴───────┐
                    │  MLP / ACMPC │ ← 15D observation
                    │    Policy    │ → [vx, vy, wz]
                    └──────────────┘
                    Rover (RosMaster X3)
```

### Agents
- **Drone**: Crazyflie cf21B_500 brushless, NX=12, NU=4 (roll, pitch, yaw_cmd, thrust)
- **Rover**: Yahboom RosMaster X3 mecanum omnidirectional, NX=7, NU=3 (vx, vy, wz body-frame commands)

### Training
- **Algorithm**: Multi-Agent PPO (MAPPO) with shared critic via [skrl](https://github.com/Toni-SM/skrl)
- **Simulator**: [Crazyflow](https://github.com/utiasDSL/crazyflow) (JAX-based Crazyflie sim)
- **Policy types**: MLP (fast inference) or ACMPC (differentiable MPC via [LEAP-C](https://github.com/leap-c/leap-c) + [acados](https://github.com/acados/acados))
- **Curriculum**: 6 levels from close hover → full map with domain randomization
- **Disturbance**: Ornstein-Uhlenbeck process for smooth, correlated force/torque perturbations

## Setup

```bash
git clone --recurse-submodules git@github.com:llanesc/crazyflie-rover-landing.git
cd crazyflie-rover-landing

# Create virtual environment
uv venv local_env
VIRTUAL_ENV=local_env uv pip install -e . \
  -e external/crazyflow \
  -e external/leap-c \
  -e external/skrl

# Install acados (required for ACMPC)
VIRTUAL_ENV=local_env uv pip install -e external/leap-c/external/acados/interfaces/acados_template --use-pep517
```

## Training

```bash
# MLP policy
python scripts/train_mappo_mlp.py --experiment X3

# ACMPC policy
python scripts/train_mappo_acmpc.py --experiment X3

# Resume from checkpoint
python scripts/train_mappo_mlp.py --experiment X3 --resume-run run_20260409215922 --curriculum-level 4
```

## Evaluation

```bash
# Evaluate with rendering
python scripts/eval_mappo_mlp.py --experiment X3 --run run_20260409215922 \
  --n-episodes 15 --level 6 --deterministic --render --trajectory --cam-distance 8

# Fixed initial conditions
python scripts/eval_mappo_mlp.py --experiment X3 --run run_20260409215922 \
  --drone-pos 2.0,0.5 --rover-pos 0,0 --n-episodes 1 --deterministic --render

# Log observation/action data to CSV
python scripts/eval_mappo_mlp.py --experiment X3 --run run_20260409215922 \
  --drone-pos 2.0,0.5 --rover-pos 0,0 --log-csv /tmp/eval_data.csv --deterministic
```

## Hardware Deployment

Three deployment modes using [CrazySim](https://github.com/gtfactslab/CrazySim) and ROS2 Jazzy:

| Mode | Drone | Rover | Use Case |
|------|-------|-------|----------|
| **Full Sim** | CrazySim SITL | Simulated mecanum | Development & testing |
| **Mixed** | CrazySim SITL | Real X3 rover | Sim-to-real validation |
| **Full Hardware** | Real Crazyflie | Real X3 rover | Deployment |

See [hardware/DEPLOYMENT_GUIDE.md](hardware/DEPLOYMENT_GUIDE.md) for detailed setup instructions.

### Quick Start (Full Sim)

```bash
# Terminal 1: CrazySim
bash hardware/crazysim_sim.sh -d 0.002

# Terminal 2: Landing system
source /opt/ros/jazzy/setup.bash
source hardware/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=28
ros2 launch cf_landing_drone launch_full_sim.py
```

Then use the RViz panel: **Takeoff** → **Run** → watch it land.

## Project Structure

```
crazyflie_rover_landing/
  envs/           Landing environment (Crazyflow-based)
  leap_c/         ACMPC OCP definitions and planners
  policies/       MLP and ACMPC policy networks
  agents/         MAPPO training agent
  utils/          Curriculum, config utilities
  meshes/         X3 rover MuJoCo render model

scripts/
  train_mappo_mlp.py      MLP training
  train_mappo_acmpc.py    ACMPC training
  eval_mappo_mlp.py       MLP evaluation
  eval_mappo_acmpc.py     ACMPC evaluation

hardware/
  ros2_ws/src/
    cf_landing_drone/     Drone/rover agents, mission manager, policy loader
    cf_landing_interfaces/  Custom ROS2 messages/services
    cf_landing_rviz_plugin/ RViz2 control panel
  crazysim_sim.sh         Full sim launch script
  crazysim_hw.sh          Mixed/hardware launch script
  DEPLOYMENT_GUIDE.md     Detailed deployment instructions

results/
  acmpc/X3/config.yaml   ACMPC experiment config
  mlp/X3/config.yaml     MLP experiment config

external/
  CrazySim/       MuJoCo SITL simulator (rover-landing branch)
  crazyflow/      JAX Crazyflie simulator
  leap-c/         Differentiable MPC framework
  skrl/           RL library (resume-support branch)
```

## Configuration

Training configs in `results/{acmpc,mlp}/X3/config.yaml` control:
- Environment: arena size, episode length, rover speed limits
- Curriculum: 6 levels with progressive difficulty
- Rewards: landing bonus, progress, smoothness penalties, descent speed
- Domain randomization: mass/inertia perturbation, OU disturbance forces
- Policy: hidden sizes, activation, action bounds

## Key Findings (Sim-to-Real)

- **so_rpy yaw artifact**: Crazyflow's fitted dynamics model creates fake yaw torque from initial yaw offset (`rpy_coef * euler_angles`). Not present in first-principles physics or CrazySim. Switching to `dynamics: first_principles` in training eliminates this.
- **Action smoothness**: Increasing `action_smoothness_rpy` reduces jittery roll/pitch commands that the firmware Mellinger controller amplifies.
- **OU disturbance**: Ornstein-Uhlenbeck process (theta=0.5) produces smooth disturbances vs Gaussian white noise, preventing the policy from learning high-frequency reactive behavior.
- **Per-rotor ground effect**: Matched between training and CrazySim with 1.5x Cheeseman-Bennett scale factor.

## License

MIT
