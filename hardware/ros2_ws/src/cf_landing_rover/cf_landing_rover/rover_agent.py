"""Standalone rover agent node for X3 Docker deployment.

Subscribes directly to raw sensor topics, builds observations matching
landing_env.py:_get_observations() (lines 1439-1461), runs rover ACMPC
policy inference, and publishes cmd_vel commands.
"""

from math import atan2, cos, sin, sqrt

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cf_landing_interfaces.msg import MissionStatus

from cf_landing_rover.policy_loader import (
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

        # Subscribers
        self.create_subscription(Odometry, '/rover/odom', self._rover_odom_cb, sensor_qos)
        self.create_subscription(Twist, '/vel_raw', self._rover_vel_raw_cb, sensor_qos)
        self.create_subscription(Odometry, f'/{self.drone_name}/odom', self._drone_odom_cb, sensor_qos)
        self.create_subscription(MissionStatus, '/cf_landing/mission_status', self._mission_status_cb, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)

        # Cached data
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

        env_section = training_config["environment"]
        self.vx_max = env_section.get("rover_vx_max", 1.0)
        self.vy_max = env_section.get("rover_vy_max", 1.0)
        self.wz_max = env_section.get("rover_wz_max", 5.0)

        # Load policy
        self.get_logger().info("Loading rover ACMPC policy...")
        models_dir = get_models_dir()
        checkpoint_path = find_checkpoint(models_dir)
        env_config = load_env_config(models_dir)
        self.policy, self.preprocessor = load_rover_policy(
            checkpoint_path, env_config, training_config, device="cpu"
        )
        self.get_logger().info(f"Rover policy loaded from {checkpoint_path}")

        self.control_dt = 0.02
        self.create_timer(self.control_dt, self._control_loop)

    def _rover_odom_cb(self, msg: Odometry):
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

    def _build_observation(self) -> np.ndarray:
        """Build 15D rover observation for X3 (cos, sin ordering)."""
        drone_speed = float(np.linalg.norm(self.drone_vel))
        rel_pos = self.drone_pos - np.array([self.rover_xy[0], self.rover_xy[1], 0.0])
        dist = float(np.linalg.norm(rel_pos))

        return np.concatenate([
            self.rover_xy,
            np.array([self.rover_cos_theta]),
            np.array([self.rover_sin_theta]),
            np.array([self.rover_vx_body]),
            np.array([self.rover_vy_body]),
            np.array([self.rover_wz]),
            rel_pos,
            self.drone_vel,
            np.array([drone_speed]),
            np.array([dist]),
        ]).astype(np.float32)

    def _build_mpc_state(self) -> np.ndarray:
        """Build 7D rover MPC state."""
        return np.array([
            self.rover_xy[0], self.rover_xy[1],
            self.rover_cos_theta, self.rover_sin_theta,
            self.rover_vx_body, self.rover_vy_body,
            self.rover_wz,
        ], dtype=np.float32)

    def _control_loop(self):
        if self.mission_status == MissionStatus.STATUS_RUN:
            if not (self.rover_odom_ready and self.drone_odom_ready):
                return
            self._run_policy()
        elif self.mission_status in (MissionStatus.STATUS_TAKEOFF, MissionStatus.STATUS_HOVER):
            self.cmd_pub.publish(Twist())

    def _run_policy(self):
        obs = self._build_observation()
        mpc_state = self._build_mpc_state()
        normalized_action = infer_rover_action(self.policy, self.preprocessor, obs, mpc_state)

        twist = Twist()
        twist.linear.x = float(np.clip(normalized_action[0] * self.vx_max, -self.vx_max, self.vx_max))
        twist.linear.y = float(np.clip(normalized_action[1] * self.vy_max, -self.vy_max, self.vy_max))
        twist.angular.z = float(np.clip(normalized_action[2] * self.wz_max, -self.wz_max, self.wz_max))
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)

    import yaml
    from ament_index_python.packages import get_package_share_directory
    config_dir = get_package_share_directory('cf_landing_rover')
    with open(f'{config_dir}/config/rover_config.yaml') as f:
        config = yaml.safe_load(f)

    training_config_path = config.get("training_config_path")
    if training_config_path:
        with open(training_config_path) as f:
            training_config = yaml.safe_load(f)
    else:
        models_dir = get_models_dir()
        config_path = models_dir / "acmpc" / "config.yaml"
        with open(config_path) as f:
            training_config = yaml.safe_load(f)

    node = RoverAgent(config, training_config)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
