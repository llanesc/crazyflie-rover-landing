"""Rover agent node for cooperative landing.

Subscribes directly to raw sensor topics, builds observations matching
landing_env.py:_get_observations() (lines 1439-1461), runs rover ACMPC
policy inference, and publishes cmd_vel commands.

Works identically on PC (sim mode) or X3 Docker (hw mode).
"""

from math import atan2, cos, sin, sqrt

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cf_landing_interfaces.msg import MissionStatus

from cf_landing_drone.policy_loader import (
    get_models_dir, find_checkpoint, load_env_config,
    load_rover_policy, infer_rover_action,
)


class RoverAgent(Node):
    def __init__(self, config: dict, training_config: dict):
        super().__init__('rover_agent')

        self.config = config
        self.drone_name = config.get("drone_name", "cf_1")

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers — raw topics
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
            Odometry, f'/{self.drone_name}/odom',
            self._drone_odom_cb, sensor_qos)
        self.create_subscription(
            MissionStatus, '/cf_landing/mission_status',
            self._mission_status_cb, 10)

        # TF listener for AMCL-corrected rover position (hw mode)
        if self._use_tf_for_rover:
            from tf2_ros import Buffer, TransformListener
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)

        # Publisher
        rover_cmd_topic = config.get("rover_cmd_topic", "/cmd_vel")
        self.cmd_pub = self.create_publisher(Twist, rover_cmd_topic, 1)

        # Cached sensor data
        self.rover_xy = np.zeros(2, dtype=np.float32)
        self.rover_cos_theta = 1.0
        self.rover_sin_theta = 0.0
        self.rover_vx_body = 0.0
        self.rover_vy_body = 0.0
        self.rover_wz = 0.0

        self.drone_pos = np.zeros(3, dtype=np.float32)
        self.drone_vel = np.zeros(3, dtype=np.float32)

        self.mission_status = MissionStatus.STATUS_OFF
        self.rover_odom_ready = False
        self.drone_odom_ready = False

        # Action limits from training config
        env_section = training_config.get("environment", {})
        self.vx_max = env_section.get("rover_vx_max", 1.0)
        self.vy_max = env_section.get("rover_vy_max", 1.0)
        self.wz_max = env_section.get("rover_wz_max", 5.0)

        # Determine policy type and inference mode
        self.policy_type = config.get("policy_type", "mlp")
        self.deterministic = not config.get("stochastic_policy", False)

        # Load policy
        self.get_logger().info(f"Loading rover {self.policy_type} policy...")
        models_dir = get_models_dir()
        checkpoint_path = find_checkpoint(models_dir, self.policy_type)
        env_config = load_env_config(models_dir, self.policy_type)

        if self.policy_type == "acmpc":
            self.policy, self.preprocessor = load_rover_policy(
                checkpoint_path, env_config, training_config, device="cpu"
            )
        else:
            from cf_landing_drone.policy_loader import load_rover_mlp_policy
            self.policy, self.preprocessor = load_rover_mlp_policy(
                checkpoint_path, env_config, training_config, device="cpu"
            )
        self.get_logger().info(f"Rover policy loaded from {checkpoint_path}")

        # Notify mission manager that rover policy is ready
        from std_msgs.msg import Bool
        ready_pub = self.create_publisher(Bool, '/cf_landing/rover_policy_ready', 10)
        self._ready_pub = ready_pub
        self.create_timer(1.0, lambda: self._ready_pub.publish(Bool(data=True)))
        ready_pub.publish(Bool(data=True))

        # Control timer at 50Hz
        self.control_dt = 0.02
        self.create_timer(self.control_dt, self._control_loop)

    # ---- Callbacks ----

    def _rover_odom_cb(self, msg: Odometry):
        if self._use_tf_for_rover:
            # In hw mode, only mark as ready. Position comes from TF.
            self.rover_odom_ready = True
            return
        p = msg.pose.pose.position
        self.rover_xy[:] = [p.x, p.y]
        q = msg.pose.pose.orientation
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

    def _drone_odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.drone_pos[:] = [p.x, p.y, p.z]
        v = msg.twist.twist.linear
        self.drone_vel[:] = [v.x, v.y, v.z]
        self.drone_odom_ready = True

    def _mission_status_cb(self, msg: MissionStatus):
        self.mission_status = msg.status

    # ---- Observation builder (must match landing_env.py lines 1439-1461) ----

    def _build_observation(self) -> np.ndarray:
        """Build 15D rover observation for X3."""
        drone_speed = float(np.linalg.norm(self.drone_vel))
        rel_pos = self.drone_pos - np.array([self.rover_xy[0], self.rover_xy[1], 0.0])
        dist = float(np.linalg.norm(rel_pos))

        # Note: rover obs uses [cos, sin] ordering for heading (lines 1441-1442)
        obs = np.concatenate([
            self.rover_xy,                                     # 2
            np.array([self.rover_cos_theta]),                  # 1
            np.array([self.rover_sin_theta]),                  # 1
            np.array([self.rover_vx_body]),                    # 1
            np.array([self.rover_vy_body]),                    # 1
            np.array([self.rover_wz]),                         # 1
            rel_pos,                                           # 3
            self.drone_vel,                                    # 3
            np.array([drone_speed]),                           # 1
            np.array([dist]),                                  # 1
        ]).astype(np.float32)                                  # Total: 15
        return obs

    def _build_mpc_state(self) -> np.ndarray:
        """Build 7D rover MPC state: [x, y, cos(θ), sin(θ), vx, vy, wz]."""
        return np.array([
            self.rover_xy[0], self.rover_xy[1],
            self.rover_cos_theta, self.rover_sin_theta,
            self.rover_vx_body, self.rover_vy_body,
            self.rover_wz,
        ], dtype=np.float32)

    # ---- Control ----

    def _update_rover_from_tf(self):
        """Update rover position from TF world→base_link (AMCL-corrected)."""
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
        if self.mission_status == MissionStatus.STATUS_RUN:
            if not (self.rover_odom_ready and self.drone_odom_ready):
                return
            self._run_policy()
        else:
            # TAKEOFF, HOVER, OFF, LANDED, ABORT: stop the rover
            self._stop()

    def _run_policy(self):
        obs = self._build_observation()
        mpc_state = self._build_mpc_state() if self.policy_type == "acmpc" else None

        normalized_action = infer_rover_action(
            self.policy, self.preprocessor, obs, mpc_state, device="cpu",
            deterministic=self.deterministic,
        )

        # Denormalize: X3 uses [vx_cmd, vy_cmd, wz_cmd]
        vx_cmd = float(np.clip(normalized_action[0] * self.vx_max, -self.vx_max, self.vx_max))
        vy_cmd = float(np.clip(normalized_action[1] * self.vy_max, -self.vy_max, self.vy_max))
        wz_cmd = float(np.clip(normalized_action[2] * self.wz_max, -self.wz_max, self.wz_max))

        twist = Twist()
        twist.linear.x = vx_cmd
        twist.linear.y = vy_cmd
        twist.angular.z = wz_cmd
        self.cmd_pub.publish(twist)

    def _stop(self):
        """Publish zero velocity."""
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)

    import yaml
    from ament_index_python.packages import get_package_share_directory
    config_dir = get_package_share_directory('cf_landing_drone')
    with open(f'{config_dir}/config/landing_config.yaml') as f:
        landing_config = yaml.safe_load(f)

    training_config_path = landing_config.get("training_config_path")
    if training_config_path:
        with open(training_config_path) as f:
            training_config = yaml.safe_load(f)
    else:
        models_dir = get_models_dir()
        config_path = models_dir / "acmpc" / "config.yaml"
        with open(config_path) as f:
            training_config = yaml.safe_load(f)

    node = RoverAgent(landing_config, training_config)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
