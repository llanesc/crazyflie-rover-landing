"""Drone agent node for cooperative landing.

Subscribes directly to raw sensor topics, builds observations matching
landing_env.py:_get_observations() (lines 1419-1433), runs drone ACMPC
policy inference, and publishes AttitudeSetpoint commands.
"""

import time
from math import atan2, cos, sin, sqrt

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Duration
from crazyflie_interfaces.msg import AttitudeSetpoint, LogDataGeneric
from crazyflie_interfaces.srv import NotifySetpointsStop, Takeoff
from cf_landing_interfaces.msg import MissionStatus

from cf_landing_drone.policy_loader import (
    get_models_dir, find_checkpoint, load_env_config,
    load_drone_policy, infer_drone_action,
)


def _quat_to_rotmat(qx, qy, qz, qw):
    """Convert quaternion (x,y,z,w) to flattened 3x3 rotation matrix (9,)."""
    r00 = 1 - 2*(qy**2 + qz**2)
    r01 = 2*(qx*qy - qz*qw)
    r02 = 2*(qx*qz + qy*qw)
    r10 = 2*(qx*qy + qz*qw)
    r11 = 1 - 2*(qx**2 + qz**2)
    r12 = 2*(qy*qz - qx*qw)
    r20 = 2*(qx*qz - qy*qw)
    r21 = 2*(qy*qz + qx*qw)
    r22 = 1 - 2*(qx**2 + qy**2)
    return np.array([r00, r01, r02, r10, r11, r12, r20, r21, r22], dtype=np.float32)


def _quat_to_euler(qx, qy, qz, qw):
    """Convert quaternion (x,y,z,w) to euler (roll, pitch, yaw)."""
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx**2 + qy**2)
    roll = atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy**2 + qz**2)
    yaw = atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class DroneAgent(Node):
    def __init__(self, config: dict, training_config: dict):
        super().__init__('drone_agent')

        self.config = config
        self.drone_name = config.get("drone_name", "cf_1")

        # Sensor data QoS (best effort for high-freq topics)
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers — raw topics
        self.create_subscription(
            Odometry, f'/{self.drone_name}/odom',
            self._drone_odom_cb, sensor_qos)
        self.create_subscription(
            LogDataGeneric, f'/{self.drone_name}/body_rates',
            self._drone_body_rates_cb, sensor_qos)
        rover_odom_topic = config.get("rover_odom_topic", "/rover/odom")
        self._use_tf_for_rover = config.get("use_tf_for_rover", False)
        self.create_subscription(
            Odometry, rover_odom_topic,
            self._rover_odom_cb, sensor_qos)
        rover_vel_topic = config.get("rover_vel_topic", "/vel_raw")
        self.create_subscription(
            Twist, rover_vel_topic,
            self._rover_vel_raw_cb, sensor_qos)
        self.create_subscription(
            MissionStatus, '/cf_landing/mission_status',
            self._mission_status_cb, 10)

        # TF listener for AMCL-corrected rover position (hw mode)
        if self._use_tf_for_rover:
            from tf2_ros import Buffer, TransformListener
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)

        # Publisher
        self.cmd_pub = self.create_publisher(
            AttitudeSetpoint, f'/{self.drone_name}/cmd_attitude', 1)

        # Cached sensor data
        self.drone_pos = np.zeros(3, dtype=np.float32)
        self.drone_vel = np.zeros(3, dtype=np.float32)
        self.drone_quat = np.array([0., 0., 0., 1.], dtype=np.float32)  # xyzw
        self.drone_ang_vel = np.zeros(3, dtype=np.float32)
        self.drone_rotmat = np.eye(3, dtype=np.float32).flatten()

        self.rover_xy = np.zeros(2, dtype=np.float32)
        self.rover_cos_theta = 1.0
        self.rover_sin_theta = 0.0
        self.rover_vx_body = 0.0
        self.rover_vy_body = 0.0
        self.rover_wz = 0.0

        self.mission_status = MissionStatus.STATUS_OFF
        self.drone_odom_ready = False
        self.drone_rates_ready = False
        self.rover_odom_ready = False

        # Determine policy type and inference mode
        self.policy_type = config.get("policy_type", "mlp")
        self.deterministic = not config.get("stochastic_policy", False)

        # Load policy
        self.get_logger().info(f"Loading drone {self.policy_type} policy...")
        models_dir = get_models_dir()
        checkpoint_path = find_checkpoint(models_dir, self.policy_type)
        env_config = load_env_config(models_dir, self.policy_type)

        if self.policy_type == "acmpc":
            self.policy, self.preprocessor = load_drone_policy(
                checkpoint_path, env_config, training_config, device="cpu"
            )
        else:
            from cf_landing_drone.policy_loader import load_drone_mlp_policy
            self.policy, self.preprocessor = load_drone_mlp_policy(
                checkpoint_path, env_config, training_config, device="cpu"
            )
        self.get_logger().info(f"Drone policy loaded from {checkpoint_path}")

        # Notify mission manager that drone policy is ready
        from std_msgs.msg import Bool
        ready_pub = self.create_publisher(Bool, '/cf_landing/drone_policy_ready', 10)
        # Publish on a timer so latched subscribers always get it
        self._ready_pub = ready_pub
        self.create_timer(1.0, lambda: self._ready_pub.publish(Bool(data=True)))
        ready_pub.publish(Bool(data=True))

        # Physical params
        env_section = training_config.get("environment", {})
        d_cfg = training_config.get("policy", {}).get("drone", {})
        self.roll_pitch_max = d_cfg.get("roll_pitch_max", env_section.get("roll_pitch_max", 0.1))
        self.yaw_max = d_cfg.get("yaw_max", env_section.get("yaw_max", 0.001))
        self.get_logger().info(f"Action bounds: roll_pitch_max={self.roll_pitch_max}, yaw_max={self.yaw_max}")

        # Thrust params from drone_models directly (avoid importing envs which pulls in jax)
        from drone_models.core import load_params
        drone_model = env_section.get("drone_model", env_config.get("drone_model", "cf21B_500"))
        so_rpy_params = load_params("so_rpy", drone_model)
        fp_params = load_params("first_principles", drone_model)
        self.thrust_min = float(so_rpy_params["thrust_min"]) * 4
        self.thrust_max = float(so_rpy_params["thrust_max"]) * 4
        self.mass = float(fp_params["mass"])
        self.gravity = float(np.abs(fp_params["gravity_vec"][2]))

        # Hardware thrust mapping
        drone_params = so_rpy_params
        self.thrust_max_hw = float(np.abs(drone_params["thrust_max"])) * 4
        self.pwm_max = 65535
        self.pwm_min = 7000

        # Action denormalization
        thrust_mean = (self.thrust_min + self.thrust_max) / 2.0
        thrust_half = (self.thrust_max - self.thrust_min) / 2.0
        self.action_mean = np.array([0., 0., 0., thrust_mean], dtype=np.float32)
        self.action_scale = np.array([self.roll_pitch_max, self.roll_pitch_max, self.yaw_max, thrust_half], dtype=np.float32)

        # Yaw PID state
        self._yaw_integral = 0.0
        self._yaw_error_prev = 0.0
        self._yaw_d_filtered = 0.0

        # Takeoff and low-level control state
        self.takeoff_commanded = False
        self.low_level_enabled = False
        self.takeoff_altitude = config.get("takeoff_altitude", 1.0)
        self.takeoff_duration = config.get("takeoff_duration", 3.0)
        self.altitude_threshold = config.get("altitude_threshold", 0.05)
        self.velocity_threshold = config.get("velocity_threshold", 0.05)

        # Control timer at 50Hz
        self.control_dt = 0.02
        self.create_timer(self.control_dt, self._control_loop)

    # ---- Callbacks ----

    def _drone_odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.drone_pos[:] = [p.x, p.y, p.z]
        v = msg.twist.twist.linear
        self.drone_vel[:] = [v.x, v.y, v.z]
        q = msg.pose.pose.orientation
        self.drone_quat[:] = [q.x, q.y, q.z, q.w]
        self.drone_rotmat = _quat_to_rotmat(q.x, q.y, q.z, q.w)
        self.drone_odom_ready = True

    def _drone_body_rates_cb(self, msg: LogDataGeneric):
        # Crazyswarm2 body_rates from stateEstimateZ: [rateRoll, ratePitch, rateYaw] in millirad/s
        # Firmware negates gyro.y for ratePitch: ratePitch = -gyro.y * deg2millirad
        # Training env (Crazyflow) uses raw body angular velocity without negation,
        # so we negate pitch back to match.
        if len(msg.values) >= 3:
            self.drone_ang_vel[:] = [
                msg.values[0] / 1000.0,    # rateRoll = gyro.x (no sign change)
                -msg.values[1] / 1000.0,   # ratePitch = -gyro.y → negate back
                msg.values[2] / 1000.0,    # rateYaw = gyro.z (no sign change)
            ]
            self.drone_rates_ready = True

    def _rover_odom_cb(self, msg: Odometry):
        if self._use_tf_for_rover:
            # In hw mode, only use odom for velocity. Position comes from TF.
            self.rover_odom_ready = True
            return
        p = msg.pose.pose.position
        self.rover_xy[:] = [p.x, p.y]
        q = msg.pose.pose.orientation
        # Extract yaw from quaternion
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        theta = atan2(siny_cosp, cosy_cosp)
        self.rover_cos_theta = cos(theta)
        self.rover_sin_theta = sin(theta)
        self.rover_odom_ready = True

    def _rover_vel_raw_cb(self, msg: Twist):
        self.rover_vx_body = msg.linear.x
        self.rover_vy_body = msg.linear.y
        self.rover_wz = msg.angular.z

    def _mission_status_cb(self, msg: MissionStatus):
        self.mission_status = msg.status

    # ---- Observation builder (must match landing_env.py lines 1419-1433) ----

    def _build_observation(self) -> np.ndarray:
        """Build 29D drone observation for X3 rover."""
        # World-frame rover velocities
        vel_x_world = self.rover_vx_body * self.rover_cos_theta - self.rover_vy_body * self.rover_sin_theta
        vel_y_world = self.rover_vx_body * self.rover_sin_theta + self.rover_vy_body * self.rover_cos_theta
        rover_speed = sqrt(self.rover_vx_body**2 + self.rover_vy_body**2)

        # Relative position
        rel_pos = self.drone_pos - np.array([self.rover_xy[0], self.rover_xy[1], 0.0])

        # Note: drone obs uses [sin, cos] ordering for rover heading (lines 1426-1427)
        obs = np.concatenate([
            self.drone_pos,                                          # 3
            self.drone_vel,                                          # 3
            self.drone_rotmat,                                       # 9
            self.drone_ang_vel,                                      # 3
            self.rover_xy,                                           # 2
            np.array([vel_x_world, vel_y_world]),                    # 2
            np.array([self.rover_sin_theta]),                        # 1
            np.array([self.rover_cos_theta]),                        # 1
            np.array([rover_speed]),                                 # 1
            np.array([self.rover_vy_body]),                          # 1 (X3 lateral)
            rel_pos,                                                 # 3
        ]).astype(np.float32)                                        # Total: 29
        return obs

    def _build_mpc_state(self) -> np.ndarray:
        """Build 13D drone MPC state: [pos(3), quat_xyzw(4), vel(3), body_rates(3)]."""
        return np.concatenate([
            self.drone_pos,
            self.drone_quat,
            self.drone_vel,
            self.drone_ang_vel,
        ]).astype(np.float32)

    # ---- Control ----

    def _update_rover_from_tf(self):
        """Update rover position from TF map→base_link (AMCL-corrected)."""
        if not self._use_tf_for_rover:
            return
        try:
            import rclpy
            t = self._tf_buffer.lookup_transform('world', 'base_link', rclpy.time.Time())
            p = t.transform.translation
            q = t.transform.rotation
            self.rover_xy[:] = [p.x, p.y]
            theta = atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
            self.rover_cos_theta = cos(theta)
            self.rover_sin_theta = sin(theta)
        except Exception:
            pass

    def _control_loop(self):
        self._update_rover_from_tf()
        if self.mission_status == MissionStatus.STATUS_TAKEOFF:
            if not self.takeoff_commanded:
                self._command_takeoff()
            elif not self.low_level_enabled:
                self._check_ready_for_low_level()
            elif self.drone_odom_ready:
                self._hover_pid()
        elif self.mission_status == MissionStatus.STATUS_HOVER:
            if not self.low_level_enabled:
                self._switch_to_low_level()
            elif self.drone_odom_ready:
                self._hover_pid()
        elif self.mission_status == MissionStatus.STATUS_RUN:
            if not self.low_level_enabled:
                self._switch_to_low_level()
                return
            if not (self.drone_odom_ready and self.drone_rates_ready and self.rover_odom_ready):
                # Fall back to hover PID if policy data not ready yet — no gap
                if self.drone_odom_ready:
                    self._hover_pid()
                return
            self._run_policy()
        elif self.mission_status == MissionStatus.STATUS_LANDED:
            # Kill motors on landing
            self._kill_motors()
            # Reset state so we can takeoff again next time
            self.takeoff_commanded = False
            self.low_level_enabled = False
        elif self.mission_status == MissionStatus.STATUS_ABORT:
            self._kill_motors()
            self.takeoff_commanded = False
            self.low_level_enabled = False
        elif self.mission_status == MissionStatus.STATUS_OFF:
            # Don't send any commands — let firmware idle
            # Reset state for next takeoff
            self.takeoff_commanded = False
            self.low_level_enabled = False

    def _command_takeoff(self):
        """Command high-level takeoff via Crazyswarm2 service."""
        client = self.create_client(Takeoff, f'/{self.drone_name}/takeoff')
        if client.wait_for_service(timeout_sec=1.0):
            request = Takeoff.Request()
            request.group_mask = 0
            request.height = float(self.takeoff_altitude)
            request.duration = Duration(
                sec=int(self.takeoff_duration),
                nanosec=int((self.takeoff_duration % 1) * 1e9))
            client.call_async(request)
            self.get_logger().info(
                f'Commanded takeoff to {self.takeoff_altitude:.2f}m')
        else:
            self.get_logger().warn('Takeoff service not available')
        self.takeoff_commanded = True

    def _check_ready_for_low_level(self):
        """Check if drone reached altitude and switch to low-level control."""
        if not self.drone_odom_ready:
            return
        alt = self.drone_pos[2]
        vel_mag = float(np.linalg.norm(self.drone_vel))
        if (abs(alt - self.takeoff_altitude) < self.altitude_threshold
                and vel_mag < self.velocity_threshold):
            self._switch_to_low_level()

    def _switch_to_low_level(self):
        """Switch from high-level to low-level attitude control.

        Sends notify_setpoints_stop followed by a zero-thrust setpoint
        to register low-level control with the firmware.
        """
        client = self.create_client(
            NotifySetpointsStop, f'/{self.drone_name}/notify_setpoints_stop')
        if client.wait_for_service(timeout_sec=1.0):
            request = NotifySetpointsStop.Request()
            request.remain_valid_millisecs = 0
            client.call_async(request)
            self.get_logger().info('Switched to low-level attitude control')

            # Send zero setpoint immediately — required to register low-level control
            zero_setpoint = AttitudeSetpoint()
            zero_setpoint.roll = 0.0
            zero_setpoint.pitch = 0.0
            zero_setpoint.yaw_rate = 0.0
            zero_setpoint.thrust = 0
            self.cmd_pub.publish(zero_setpoint)
            self.get_logger().info('Sent zero setpoint')
        else:
            self.get_logger().warn('notify_setpoints_stop service not available')

        self.low_level_enabled = True

    def _run_policy(self):
        obs = self._build_observation()
        mpc_state = self._build_mpc_state() if self.policy_type == "acmpc" else None

        normalized_action = infer_drone_action(
            self.policy, self.preprocessor, obs, mpc_state, device="cpu",
            deterministic=self.deterministic,
        )

        # Denormalize: physical = mean + normalized * scale
        physical = self.action_mean + normalized_action * self.action_scale
        physical[3] = np.clip(physical[3], self.thrust_min, self.thrust_max)

        # CSV logging
        if not hasattr(self, '_csv_file'):
            import csv
            self._csv_file = open('/tmp/crazysim_drone_log.csv', 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            obs_cols = [f'obs_{i}' for i in range(len(obs))]
            self._csv_writer.writerow(['step', 'time'] + obs_cols + ['act_roll', 'act_pitch', 'act_yaw', 'act_thrust'])
            self._csv_step = 0
            self._csv_t0 = time.time()
            self.get_logger().info('Logging drone data to /tmp/crazysim_drone_log.csv')
        self._csv_step += 1
        t = time.time() - self._csv_t0
        self._csv_writer.writerow([self._csv_step, f'{t:.4f}'] + obs.tolist() + physical.tolist())
        if self._csv_step % 250 == 0:
            self._csv_file.flush()

        # Get current RPY for yaw PID
        roll, pitch, yaw = _quat_to_euler(*self.drone_quat)
        self._publish_attitude_setpoint(physical, current_rpy=np.array([roll, pitch, yaw]))

    def _kill_motors(self):
        """Send zero thrust to cut motors."""
        setpoint = AttitudeSetpoint()
        setpoint.roll = 0.0
        setpoint.pitch = 0.0
        setpoint.yaw_rate = 0.0
        setpoint.thrust = 0
        self.cmd_pub.publish(setpoint)

    def _hover_pid(self):
        """PID position hold at hover position. Matches MAPE _pid_control exactly.

        Uses pure pursuit PD for XY, PID for Z (with integral to eliminate
        steady-state altitude error). Converts acceleration to attitude+thrust
        using the same rotation-matrix method as MAPE _accel_to_attitude.
        """
        if not hasattr(self, '_hover_target'):
            self._hover_target = self.drone_pos.copy()
            self._hover_z_integral = 0.0
            self._hover_last_time = time.time()
            self.get_logger().info(
                f'Hover PID target: [{self._hover_target[0]:.2f}, '
                f'{self._hover_target[1]:.2f}, {self._hover_target[2]:.2f}]')

        # MAPE gains
        k_pxy = 6.1624
        k_vxy = 3.39
        k_pz = 20.0
        k_vz = 10.0
        ki_z = 6.0
        integral_cap = 1.0

        # dt from wall clock (like MAPE)
        now = time.time()
        dt = np.clip(now - self._hover_last_time, 0.001, 0.1)
        self._hover_last_time = now

        # Position and velocity errors (target_vel = 0 for hover)
        pos_err = self._hover_target - self.drone_pos
        vel_err = -self.drone_vel

        # Pure pursuit acceleration (matches MAPE _pure_pursuit)
        acc_xy = k_pxy * pos_err[:2] + k_vxy * vel_err[:2]
        acc_z = k_pz * pos_err[2] + k_vz * vel_err[2] + self.gravity

        # Z integral (matches MAPE _pid_control)
        self._hover_z_integral += pos_err[2] * dt
        self._hover_z_integral = float(np.clip(self._hover_z_integral, -integral_cap, integral_cap))
        acc_z += ki_z * self._hover_z_integral

        accel = np.array([acc_xy[0], acc_xy[1], acc_z])

        # Convert to attitude + thrust (matches MAPE _accel_to_attitude)
        roll, pitch, yaw = _quat_to_euler(*self.drone_quat)
        target_thrust_vec = accel * self.mass

        # Current body z-axis from rotation matrix
        rotmat = _quat_to_rotmat(*self.drone_quat).reshape(3, 3)
        z_axis = rotmat[:, 2]

        # Thrust = projection of desired force onto body z
        current_thrust = float(np.dot(target_thrust_vec, z_axis))
        current_thrust = float(np.clip(current_thrust, self.thrust_min, self.thrust_max))

        # Desired z-axis from desired force direction
        force_norm = max(np.linalg.norm(target_thrust_vec), 1e-6)
        z_des = target_thrust_vec / force_norm

        # Desired rotation matrix (yaw = 0)
        x_c = np.array([cos(0.0), sin(0.0), 0.0])
        y_des = np.cross(z_des, x_c)
        y_norm = max(np.linalg.norm(y_des), 1e-6)
        y_des = y_des / y_norm
        x_des = np.cross(y_des, z_des)

        # Extract RPY from desired rotation matrix
        R_des = np.column_stack([x_des, y_des, z_des])
        pitch_des = -np.arcsin(np.clip(R_des[2, 0], -1, 1))
        roll_des = np.arctan2(R_des[2, 1], R_des[2, 2])

        control = np.array([float(roll_des), float(pitch_des), 0.0, current_thrust])
        self._publish_attitude_setpoint(control, current_rpy=np.array([roll, pitch, yaw]))

    def _publish_attitude_setpoint(self, control: np.ndarray, current_rpy: np.ndarray):
        """Publish AttitudeSetpoint with yaw PID.

        Adapted from MAPE pursuer_evader.py:cmd_attitude_setpoint() (lines 320-391).
        """
        roll_rad, pitch_rad, yaw_des, thrust_N = control

        current_roll, current_pitch, current_yaw = current_rpy

        # Yaw PID
        yaw_error = yaw_des - current_yaw
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi

        kP_yaw = 8.0
        kI_yaw = 0.3
        kD_yaw = 0.3
        max_yaw_rate = 3.0
        dt = self.control_dt

        d_error_raw = (yaw_error - self._yaw_error_prev) / dt
        self._yaw_error_prev = yaw_error
        alpha_d = 0.2
        self._yaw_d_filtered = alpha_d * d_error_raw + (1.0 - alpha_d) * self._yaw_d_filtered
        d_error = self._yaw_d_filtered

        yaw_rate_unclamped = -(kP_yaw * yaw_error + kI_yaw * self._yaw_integral - kD_yaw * d_error)
        yaw_rate_body = float(np.clip(yaw_rate_unclamped, -max_yaw_rate, max_yaw_rate))

        if abs(yaw_rate_unclamped) < max_yaw_rate:
            self._yaw_integral = float(np.clip(self._yaw_integral + yaw_error * dt, -0.5, 0.5))

        # Euler coupling feedforward
        cp = cos(current_pitch)
        sp = sin(current_pitch)
        cr = cos(current_roll)
        sr = sin(current_roll)

        denom = cr * cp
        yaw_rate_euler = yaw_rate_body / denom if abs(denom) > 0.1 else yaw_rate_body
        roll_ff = -yaw_rate_euler * sp * dt
        pitch_ff = yaw_rate_euler * sr * cp * dt

        setpoint = AttitudeSetpoint()
        setpoint.roll = float(roll_rad + roll_ff)
        setpoint.pitch = float(pitch_rad + pitch_ff)
        setpoint.yaw_rate = yaw_rate_body
        setpoint.thrust = self._thrust_to_pwm(thrust_N)

        self.cmd_pub.publish(setpoint)

    def _thrust_to_pwm(self, collective_thrust: float) -> int:
        collective_thrust = float(np.clip(collective_thrust, self.thrust_min, self.thrust_max))
        pwm = (collective_thrust / self.thrust_max_hw) * self.pwm_max
        return int(np.clip(pwm, self.pwm_min, self.pwm_max))


def main(args=None):
    rclpy.init(args=args)

    # TODO: Load training config from landing_config.yaml or args
    import yaml
    from ament_index_python.packages import get_package_share_directory
    config_dir = get_package_share_directory('cf_landing_drone')
    with open(f'{config_dir}/config/landing_config.yaml') as f:
        landing_config = yaml.safe_load(f)

    # Load training config (the original results/acmpc/X3/config.yaml)
    training_config_path = landing_config.get("training_config_path")
    if training_config_path:
        with open(training_config_path) as f:
            training_config = yaml.safe_load(f)
    else:
        # Fallback: look for config in models dir
        models_dir = get_models_dir()
        config_path = models_dir / "acmpc" / "config.yaml"
        with open(config_path) as f:
            training_config = yaml.safe_load(f)

    node = DroneAgent(landing_config, training_config)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
