"""Basic smoke tests for the drone-rover landing environment."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pytest


def test_import():
    """Test that all major modules can be imported."""
    from crazyflie_rover_landing.envs import LandingEnv, LandingEnvConfig
    from crazyflie_rover_landing.envs.rover_dynamics import rover_step_batched
    from crazyflie_rover_landing.envs.spawn import create_default_spawn_fn
    from crazyflie_rover_landing.envs.wrappers import RescaleActionWrapper


def test_landing_config_defaults():
    """Test LandingEnvConfig creates with defaults without error."""
    from crazyflie_rover_landing.envs.landing_config import LandingEnvConfig
    cfg = LandingEnvConfig()
    assert cfg.n_worlds == 256
    assert cfg.control_freq == 100
    assert cfg.sim_freq == 500
    assert cfg.drone_model == "cf2x_T350"
    assert cfg.dt == pytest.approx(1.0 / 100.0)
    assert cfg.sim_steps_per_control == 5


def test_rover_dynamics():
    """Test JAX unicycle rover dynamics."""
    import jax.numpy as jnp
    from crazyflie_rover_landing.envs.rover_dynamics import (
        rover_step_batched, MAX_SPEED, MIN_SPEED, MAX_OMEGA, MAX_ACCEL
    )

    N = 4
    # State: [x, y, cos(θ), sin(θ), v] — heading=0 means c=1, s=0
    state = jnp.array([[0.0, 0.0, 1.0, 0.0, 1.0]] * N)  # (N, 5)
    control = jnp.array([[0.0, 0.0]] * N)                 # (N, 2)
    dt = 0.1

    next_state = rover_step_batched(state, control, dt, MAX_SPEED, MIN_SPEED, MAX_OMEGA, MAX_ACCEL)
    assert next_state.shape == (N, 5)
    # Moving forward at v=1.0 with no yaw rate, heading=0 (c=1, s=0)
    assert float(next_state[0, 0]) == pytest.approx(0.1, abs=1e-5)  # x += v*c*dt
    assert float(next_state[0, 1]) == pytest.approx(0.0, abs=1e-5)  # y unchanged
    assert float(next_state[0, 2]) == pytest.approx(1.0, abs=1e-5)  # c stays 1
    assert float(next_state[0, 3]) == pytest.approx(0.0, abs=1e-5)  # s stays 0


def test_spawn_function():
    """Test JIT-compiled spawn function."""
    import jax
    from crazyflie_rover_landing.envs.spawn import create_default_spawn_fn

    spawn_fn = create_default_spawn_fn(
        drone_z_min=0.5, drone_z_max=3.0,
        drone_x_half=2.5, drone_y_half=2.5,
        rover_x_half=2.5, rover_y_half=2.5,
        rover_max_speed=1.5,
    )

    key = jax.random.key(42)
    N = 8
    drone_pos, rover_state = spawn_fn(key, N)

    assert drone_pos.shape == (N, 3)
    assert rover_state.shape == (N, 5)
    assert float(drone_pos[:, 2].min()) >= 0.5
    assert float(drone_pos[:, 2].max()) <= 3.0


@pytest.mark.slow
def test_env_reset_step():
    """Test that the environment can reset and step without error."""
    from crazyflie_rover_landing.envs import LandingEnv, LandingEnvConfig

    cfg = LandingEnvConfig(n_worlds=4)
    env = LandingEnv(cfg=cfg)

    obs, info = env.reset(seed=0)

    # Check observation dict
    assert "drone" in obs
    assert "rover" in obs
    assert obs["drone"].shape == (4, env.drone_obs_dim)
    assert obs["rover"].shape == (4, env.rover_obs_dim)

    # Check MPC state in info
    assert "mpc_state" in info
    assert "drone" in info["mpc_state"]
    assert "rover" in info["mpc_state"]
    assert info["mpc_state"]["drone"].shape == (4, 12)
    assert info["mpc_state"]["rover"].shape == (4, 5)

    # Create random actions (in physical bounds — what env expects before rescaling)
    drone_actions = np.zeros((4, 4), dtype=np.float32)  # [roll, pitch, yaw, thrust]
    drone_actions[:, 3] = env.cfg.gravity * env.cfg.mass  # hover thrust
    rover_actions = np.zeros((4, 2), dtype=np.float32)   # [a, ω]

    actions = {"drone": drone_actions, "rover": rover_actions}
    obs2, rewards, terminated, truncated, info2 = env.step(actions)

    assert "drone" in obs2
    assert "rover" in obs2
    assert "drone" in rewards
    assert "rover" in rewards
    assert rewards["drone"].shape == (4,)

    env.close()


@pytest.mark.slow
def test_env_with_wrapper():
    """Test environment with RescaleActionWrapper."""
    import numpy as np
    from crazyflie_rover_landing.envs import LandingEnv, LandingEnvConfig, RescaleActionWrapper

    cfg = LandingEnvConfig(n_worlds=4)
    env = LandingEnv(cfg=cfg)
    wrapped = RescaleActionWrapper(env)

    obs, info = wrapped.reset(seed=0)

    # RescaleActionWrapper expects actions in [-1, 1]
    drone_actions = np.zeros((4, 4), dtype=np.float32)  # neutral in normalized space
    rover_actions = np.zeros((4, 2), dtype=np.float32)
    actions = {"drone": drone_actions, "rover": rover_actions}

    obs2, rewards, terminated, truncated, info2 = wrapped.step(actions)
    assert "drone" in obs2
    assert "rover" in obs2

    wrapped.close()


def test_curriculum_manager():
    """Test curriculum manager advancement logic."""
    from crazyflie_rover_landing.utils.curriculum import (
        CurriculumManager, CurriculumConfig, CurriculumLevel
    )

    levels = [
        CurriculumLevel(name="easy", params={}, spawn={}),
        CurriculumLevel(name="medium", params={}, spawn={}),
        CurriculumLevel(name="hard", params={}, spawn={}),
    ]
    cfg = CurriculumConfig(
        enabled=True,
        advance_threshold=0.7,
        window_size=10,
        levels=levels,
        allow_regression=True,
        regression_threshold=0.2,
    )

    manager = CurriculumManager(cfg)
    assert manager.current_level == 0
    assert manager.landing_rate == 0.0

    # Record 8 successes out of 10 → window full, should advance
    for _ in range(8):
        manager.record_episode(True)
    for _ in range(2):
        manager.record_episode(False)

    assert manager.current_level == 1, f"Expected level 1, got {manager.current_level}"

    # Record poor performance → should regress
    for _ in range(10):
        manager.record_episode(False)

    assert manager.current_level == 0, f"Expected regression to level 0, got {manager.current_level}"


def test_load_curriculum_config():
    """Test loading curriculum from a YAML-like config dict."""
    from crazyflie_rover_landing.utils.curriculum import load_curriculum_config

    config = {
        "curriculum": {
            "enabled": True,
            "advance_threshold": 0.65,
            "window_size": 50,
            "allow_regression": True,
            "regression_threshold": 0.3,
            "levels": [
                {
                    "name": "easy",
                    "drone_spawn": {"z_min": 0.5, "z_max": 1.0},
                    "rover_spawn": {"stationary": True},
                },
                {
                    "name": "hard",
                    "drone_spawn": {"z_min": 0.5, "z_max": 3.0},
                    "rover_spawn": {"max_speed": 1.5},
                },
            ],
        }
    }

    curriculum_cfg = load_curriculum_config(config)
    assert curriculum_cfg is not None
    assert len(curriculum_cfg.levels) == 2
    assert curriculum_cfg.levels[0].name == "easy"
    assert curriculum_cfg.levels[0].spawn.get("drone", {}).get("z_max") == 1.0
    assert curriculum_cfg.advance_threshold == 0.65


def test_partial_running_standard_scaler():
    """Test PartialRunningStandardScaler normalizes and skips dims correctly."""
    import torch
    from crazyflie_rover_landing.preprocessors import PartialRunningStandardScaler

    obs_dim = 10
    skip_dims = [0, 5, 9]  # binary flags at these indices
    scaler = PartialRunningStandardScaler(
        size=obs_dim, skip_dims=skip_dims, device="cpu"
    )

    # Create batch with known values
    x = torch.ones(16, obs_dim)
    # First pass (train=True) should update running stats
    out = scaler(x, train=True)
    assert out.shape == (16, obs_dim)

    # Skipped dims should be unchanged (still 1.0)
    for d in skip_dims:
        assert float(out[0, d]) == pytest.approx(1.0, abs=1e-5), \
            f"Skip dim {d} should be 1.0, got {float(out[0, d])}"


if __name__ == "__main__":
    # Run tests without pytest for quick smoke test
    import sys
    print("Running smoke tests...")

    test_import()
    print("✓ import")

    test_landing_config_defaults()
    print("✓ landing_config_defaults")

    test_rover_dynamics()
    print("✓ rover_dynamics")

    test_spawn_function()
    print("✓ spawn_function")

    test_curriculum_manager()
    print("✓ curriculum_manager")

    test_load_curriculum_config()
    print("✓ load_curriculum_config")

    test_partial_running_standard_scaler()
    print("✓ partial_running_standard_scaler")

    print("\nAll smoke tests passed! (skipping @pytest.mark.slow tests)")
    print("Run with pytest tests/test_env.py to run all tests including slow ones.")
