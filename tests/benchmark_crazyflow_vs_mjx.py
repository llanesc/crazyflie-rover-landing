"""Benchmark: Crazyflow (JAX) vs MJX (MuJoCo JAX) for drone simulation.

Compares step throughput for:
1. Crazyflow first_principles (current approach, no contact)
2. MJX with a drone + landing pad scene (with contact physics)
"""

import time
import numpy as np

N_WORLDS = 128
N_STEPS = 1000
SIM_FREQ = 500
DRONE_MODEL = "cf21B_500"


def benchmark_crazyflow():
    """Benchmark Crazyflow first_principles physics."""
    from crazyflow import Sim, Physics, Control
    import jax
    import jax.numpy as jnp

    sim = Sim(
        n_worlds=N_WORLDS,
        n_drones=1,
        drone_model=DRONE_MODEL,
        physics=Physics.first_principles,
        control=Control.attitude,
        freq=SIM_FREQ,
        attitude_freq=SIM_FREQ,
        device="cpu",
    )

    # Hover command
    hover_cmd = jnp.zeros((N_WORLDS, 1, 4))

    # Warm up JIT
    for _ in range(10):
        sim.attitude_control(hover_cmd)
        sim.step()

    # Benchmark
    start = time.perf_counter()
    for _ in range(N_STEPS):
        sim.attitude_control(hover_cmd)
        sim.step()
    jax.block_until_ready(sim.data.states.pos)
    elapsed = time.perf_counter() - start

    total_steps = N_WORLDS * N_STEPS
    print(f"Crazyflow (first_principles):")
    print(f"  {N_WORLDS} worlds × {N_STEPS} steps = {total_steps:,} total steps")
    print(f"  Wall time: {elapsed:.3f} s")
    print(f"  Throughput: {total_steps / elapsed:,.0f} steps/s")
    print(f"  Per world-step: {elapsed / N_STEPS * 1000:.3f} ms")
    print()
    return elapsed


def benchmark_mjx():
    """Benchmark MJX with drone + landing pad contact scene."""
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    # Minimal drone + pad scene with contact
    xml = """
    <mujoco model="drone_pad_benchmark">
      <option timestep="0.002" gravity="0 0 -9.81"/>
      <worldbody>
        <geom name="ground" type="plane" size="5 5 0.01" rgba="0.5 0.5 0.5 1"/>

        <!-- Landing pad -->
        <body name="pad" pos="0 0 0.12">
          <geom name="pad_geom" type="box" size="0.15 0.15 0.003" rgba="0.8 0.2 0.2 1"/>
        </body>

        <!-- Simplified drone as a sphere with 4 actuated thrusters -->
        <body name="drone" pos="0 0 0.5">
          <freejoint name="drone_joint"/>
          <inertial pos="0 0 0" mass="0.043" diaginertia="1.6e-5 1.6e-5 2.9e-5"/>
          <geom name="drone_body" type="sphere" size="0.05" rgba="0 0.5 1 1"/>

          <!-- Thruster sites -->
          <site name="t1" pos="0.03 0.03 -0.015"/>
          <site name="t2" pos="-0.03 0.03 -0.015"/>
          <site name="t3" pos="-0.03 -0.03 -0.015"/>
          <site name="t4" pos="0.03 -0.03 -0.015"/>
        </body>
      </worldbody>

      <actuator>
        <general name="thrust1" site="t1" gear="0 0 1 0 0 0" ctrlrange="0 0.2"/>
        <general name="thrust2" site="t2" gear="0 0 1 0 0 0" ctrlrange="0 0.2"/>
        <general name="thrust3" site="t3" gear="0 0 1 0 0 0" ctrlrange="0 0.2"/>
        <general name="thrust4" site="t4" gear="0 0 1 0 0 0" ctrlrange="0 0.2"/>
      </actuator>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    mjx_model = mjx.put_model(model)

    # Create batched data
    data = mujoco.MjData(model)
    mjx_data = mjx.put_data(model, data)

    # Batch across worlds using vmap
    batch_data = jax.vmap(lambda _: mjx_data)(jnp.arange(N_WORLDS))

    # Set drone position to 0.5m height for all worlds
    qpos = batch_data.qpos.at[:, 2].set(0.5)  # z position
    qpos = qpos.at[:, 3].set(1.0)  # qw = 1 (identity quaternion, MuJoCo uses wxyz)
    batch_data = batch_data.replace(qpos=qpos)

    # Hover thrust per motor = mass * g / 4
    hover_per_motor = 0.043 * 9.81 / 4
    ctrl = jnp.full((N_WORLDS, 4), hover_per_motor)
    batch_data = batch_data.replace(ctrl=ctrl)

    # JIT compile the step function
    @jax.jit
    @jax.vmap
    def mjx_step(data):
        return mjx.step(mjx_model, data)

    # Warm up JIT
    for _ in range(10):
        batch_data = mjx_step(batch_data)
    jax.block_until_ready(batch_data.qpos)

    # Benchmark
    start = time.perf_counter()
    for _ in range(N_STEPS):
        batch_data = mjx_step(batch_data)
    jax.block_until_ready(batch_data.qpos)
    elapsed = time.perf_counter() - start

    total_steps = N_WORLDS * N_STEPS
    print(f"MJX (drone + pad with contact):")
    print(f"  {N_WORLDS} worlds × {N_STEPS} steps = {total_steps:,} total steps")
    print(f"  Wall time: {elapsed:.3f} s")
    print(f"  Throughput: {total_steps / elapsed:,.0f} steps/s")
    print(f"  Per world-step: {elapsed / N_STEPS * 1000:.3f} ms")
    print()
    return elapsed


if __name__ == "__main__":
    print(f"Benchmark: {N_WORLDS} worlds, {N_STEPS} steps, {SIM_FREQ} Hz\n")

    t_cf = benchmark_crazyflow()
    t_mjx = benchmark_mjx()

    ratio = t_mjx / t_cf
    print(f"MJX / Crazyflow ratio: {ratio:.1f}x slower")
