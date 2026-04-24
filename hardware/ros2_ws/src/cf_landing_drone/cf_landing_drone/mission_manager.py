"""Lightweight mission manager for cooperative drone-rover landing.

Only handles state machine transitions and landing detection.
Does NOT relay sensor data — agents subscribe to raw topics directly.
"""

from math import atan2, cos, sin, sqrt

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from cf_landing_interfaces.msg import MissionStatus
from cf_landing_interfaces.srv import Command


class MissionManager(Node):
    def __init__(self, config: dict):
        super().__init__('mission_manager')

        self.config = config
        self.drone_name = config.get("drone_name", "cf_1")

        # Thresholds from config
        self.takeoff_altitude = config.get("takeoff_altitude", 1.0)
        self.altitude_threshold = config.get("altitude_threshold", 0.05)
        self.velocity_threshold = config.get("velocity_threshold", 0.05)
        self.rover_height = config.get("rover_height", 0.213)
        self.rover_platform_radius = config.get("rover_platform_radius", 0.127)
        self.landing_z_tol = config.get("landing_z_tol", 0.05)
        self.landing_vel_xy_tol = config.get("landing_vel_xy_tol", 0.1)
        self.landing_vel_z_tol = config.get("landing_vel_z_tol", 0.1)
        self.landing_attitude_tol = config.get("landing_attitude_tol", 0.05)
        self.landing_zone_radius = config.get("landing_zone_radius", 0.05)
        self.map_size_x = config.get("map_size_x", 5.0)
        self.map_size_y = config.get("map_size_y", 5.0)
        self.drone_z_max = config.get("drone_z_max", 3.0)

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscriptions — only for state checking
        self.create_subscription(
            Odometry, f'/{self.drone_name}/odom',
            self._drone_odom_cb, sensor_qos)
        rover_odom_topic = config.get("rover_odom_topic", "/rover/odom")
        self.create_subscription(
            Odometry, rover_odom_topic,
            self._rover_odom_cb, sensor_qos)

        # Publishers
        self.status_pub = self.create_publisher(MissionStatus, '/cf_landing/mission_status', 10)
        self.drone_marker_pub = self.create_publisher(Marker, '/cf_landing/drone_marker', 1)
        self.rover_marker_pub = self.create_publisher(Marker, '/cf_landing/rover_marker', 1)
        self.tf_broadcaster = TransformBroadcaster(self)

        self._rover_odom_repub = None

        # Publish rover TF (world → base_footprint) in sim mode
        # In hw mode with AMCL, the X3 handles its own TF
        self._publish_rover_tf = not config.get("use_tf_for_rover", False)

        # Service for state transitions
        self.create_service(Command, '/cf_landing/command', self._command_cb)

        # State
        self.status = MissionStatus.STATUS_OFF
        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.drone_quat = np.array([0., 0., 0., 1.])
        self.rover_xy = np.zeros(2)
        self.rover_vel = np.zeros(3)
        self.rover_quat = np.array([0., 0., 0., 1.])
        self.drone_odom_ready = False
        self.rover_odom_ready = False

        # Landing dwell timer — once on pad, must stay for this duration
        self._on_pad_since = None
        self._landing_dwell_s = 0.5  # seconds on pad before declaring landed

        # Policy ready tracking — block takeoff until agents report ready
        self.drone_policy_ready = False
        self.rover_policy_ready = False
        self.create_subscription(
            Bool, '/cf_landing/drone_policy_ready',
            lambda msg: setattr(self, 'drone_policy_ready', msg.data), 10)
        self.create_subscription(
            Bool, '/cf_landing/rover_policy_ready',
            lambda msg: setattr(self, 'rover_policy_ready', msg.data), 10)

        # Mesh paths for rviz markers — use absolute paths
        import os
        # Find repo root by walking up from this file looking for 'external/'
        d = os.path.abspath(__file__)
        repo_root = None
        for _ in range(15):
            d = os.path.dirname(d)
            if os.path.isdir(os.path.join(d, 'external', 'CrazySim')):
                repo_root = d
                break
        if repo_root is None:
            # Fallback: hardcoded
            repo_root = '/home/llanesc/crazyflie-rover-landing'
        self.get_logger().info(f'Repo root: {repo_root}')

        rover_meshes = f'{repo_root}/external/CrazySim/crazyflie-firmware/tools/crazyflie-simulation/simulator_files/mujoco/rover-models/meshes'
        # Use Crazyswarm2's .dae mesh (has embedded colors per part including green props)
        try:
            from ament_index_python.packages import get_package_share_directory
            cf_share = get_package_share_directory('crazyflie')
            self._drone_mesh = f'file://{cf_share}/urdf/cf2_assembly_with_props.dae'
        except Exception:
            self._drone_mesh = f'file://{repo_root}/external/CrazySim/crazyflie-firmware/tools/crazyflie-simulation/simulator_files/mujoco/drone-models/drone_models/data/assets/cf21B/cf21B_full.stl'
        self._rover_mesh = f'file://{rover_meshes}/base_link_X3_bracket_only.STL'
        self._pad_mesh = f'file://{rover_meshes}/landing_pad.STL'

        # Verify paths exist
        for name, path in [('drone', self._drone_mesh), ('rover', self._rover_mesh), ('pad', self._pad_mesh)]:
            fpath = path.replace('file://', '')
            if os.path.isfile(fpath):
                self.get_logger().info(f'{name} mesh: OK')
            else:
                self.get_logger().warn(f'{name} mesh NOT FOUND: {fpath}')

        # Publish at 10Hz
        self.create_timer(0.1, self._publish_status)
        self.create_timer(0.02, self._publish_markers)  # 50Hz for smooth rviz visualization
        # Check state transitions at 50Hz
        self.create_timer(0.02, self._check_transitions)

        self.get_logger().info("Mission manager started. Waiting for commands...")

    # ---- Callbacks ----

    def _drone_odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.drone_pos[:] = [p.x, p.y, p.z]
        v = msg.twist.twist.linear
        self.drone_vel[:] = [v.x, v.y, v.z]
        q = msg.pose.pose.orientation
        self.drone_quat[:] = [q.x, q.y, q.z, q.w]
        self.drone_odom_ready = True

    def _rover_odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.rover_xy[:] = [p.x, p.y]
        v = msg.twist.twist.linear
        self.rover_vel[:] = [v.x, v.y, v.z]
        q = msg.pose.pose.orientation
        self.rover_quat[:] = [q.x, q.y, q.z, q.w]
        self.rover_odom_ready = True

    def _command_cb(self, request, response):
        cmd = request.command
        prev = self.status

        if cmd == Command.Request.CMD_TAKEOFF:
            if self.status != MissionStatus.STATUS_OFF:
                self.get_logger().warn(f"Cannot TAKEOFF from state {self.status}")
                response.success = False
            elif not (self.drone_policy_ready and self.rover_policy_ready):
                waiting = []
                if not self.drone_policy_ready:
                    waiting.append("drone")
                if not self.rover_policy_ready:
                    waiting.append("rover")
                self.get_logger().warn(
                    f"Cannot TAKEOFF — waiting for policy build: {', '.join(waiting)}")
                response.success = False
            else:
                self.status = MissionStatus.STATUS_TAKEOFF
                self.get_logger().info("TAKEOFF commanded")
                response.success = True

        elif cmd == Command.Request.CMD_RUN:
            if self.status == MissionStatus.STATUS_HOVER:
                self.status = MissionStatus.STATUS_RUN
                self.get_logger().info("RUN commanded — policies active")
                response.success = True
            else:
                self.get_logger().warn(f"Cannot RUN from state {self.status}")
                response.success = False

        elif cmd == Command.Request.CMD_ABORT:
            self.status = MissionStatus.STATUS_ABORT
            self.get_logger().warn("ABORT commanded")
            response.success = True

        elif cmd == Command.Request.CMD_OFF:
            self.status = MissionStatus.STATUS_OFF
            self.get_logger().info("OFF commanded")
            response.success = True

        else:
            self.get_logger().error(f"Unknown command: {cmd}")
            response.success = False

        return response

    # ---- State transitions ----

    def _check_transitions(self):
        if not self.drone_odom_ready:
            return

        if self.status == MissionStatus.STATUS_TAKEOFF:
            self._check_takeoff_complete()
        elif self.status == MissionStatus.STATUS_RUN:
            self._check_landing()
            self._check_boundaries()

    def _check_takeoff_complete(self):
        alt = self.drone_pos[2]
        vel_mag = float(np.linalg.norm(self.drone_vel))

        if (abs(alt - self.takeoff_altitude) < self.altitude_threshold
                and vel_mag < self.velocity_threshold):
            self.status = MissionStatus.STATUS_HOVER
            self.get_logger().info(
                f"Takeoff complete (alt={alt:.2f}m, vel={vel_mag:.3f}m/s). "
                f"Hovering — waiting for RUN command."
            )

    def _check_landing(self):
        """Replicate _jit_check_landing logic from landing_env.py."""
        if not self.rover_odom_ready:
            return

        # Relative position to pad center (pad is offset from rover body origin)
        # Pad offset in rover body frame: (-0.0384, 0.0004)
        q = self.rover_quat
        rover_yaw = atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1]**2 + q[2]**2))
        pad_offset_body = np.array([-0.0384, 0.0004])
        c_r, s_r = cos(rover_yaw), sin(rover_yaw)
        pad_offset_world = np.array([
            pad_offset_body[0] * c_r - pad_offset_body[1] * s_r,
            pad_offset_body[0] * s_r + pad_offset_body[1] * c_r,
        ])
        pad_xy = self.rover_xy + pad_offset_world
        rel_xy = self.drone_pos[:2] - pad_xy
        horiz_dist = float(np.linalg.norm(rel_xy))

        # Altitude above rover pad
        alt_above_pad = self.drone_pos[2] - self.rover_height

        # Velocity
        vel_xy = float(np.linalg.norm(self.drone_vel[:2]))
        vel_z = self.drone_vel[2]

        # Attitude (roll, pitch from quaternion)
        qx, qy, qz, qw = self.drone_quat
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx**2 + qy**2)
        roll = abs(np.arctan2(sinr_cosp, cosr_cosp))
        sinp = np.clip(2 * (qw * qy - qz * qx), -1, 1)
        pitch = abs(np.arcsin(sinp))

        # Landing detection: drone is on pad — use dwell timer instead of velocity
        # (firmware EKF velocity estimate is unreliable on contact)
        near_pad_height = alt_above_pad < 0.08 and alt_above_pad > -0.05
        within_pad = horiz_dist < self.rover_platform_radius
        on_pad = near_pad_height and within_pad

        import time as _time
        now = _time.time()
        if on_pad:
            if self._on_pad_since is None:
                self._on_pad_since = now
            dwell = now - self._on_pad_since
        else:
            self._on_pad_since = None
            dwell = 0.0

        landed = on_pad and dwell >= self._landing_dwell_s

        # Debug: log landing check when close to pad
        if not hasattr(self, '_land_dbg_cnt'):
            self._land_dbg_cnt = 0
        self._land_dbg_cnt += 1
        if self._land_dbg_cnt % 50 == 0 and horiz_dist < 0.5:
            self.get_logger().info(
                f'LAND CHECK: dist={horiz_dist:.3f} within={within_pad} '
                f'alt_above={alt_above_pad:.3f} near_h={near_pad_height} '
                f'dwell={dwell:.2f}s rover_h={self.rover_height:.3f}')

        if landed:
            self.status = MissionStatus.STATUS_LANDED
            self.get_logger().info(
                f"LANDED! dist={horiz_dist:.3f}m, alt={alt_above_pad:.3f}m, "
                f"vel_xy={vel_xy:.3f}m/s, vel_z={vel_z:.3f}m/s"
            )

    def _check_boundaries(self):
        x, y, z = self.drone_pos
        if (abs(x) > self.map_size_x or abs(y) > self.map_size_y
                or z > self.drone_z_max):
            self.status = MissionStatus.STATUS_ABORT
            self.get_logger().warn(
                f"Boundary violation! pos=({x:.2f}, {y:.2f}, {z:.2f})"
            )
        elif z < 0.1:
            self.status = MissionStatus.STATUS_ABORT
            self.get_logger().warn(
                f"CRASH! Drone below 0.1m: z={z:.3f}m"
            )

    # ---- Publishers ----

    def _publish_status(self):
        msg = MissionStatus()
        msg.status = self.status
        msg.drone_policy_ready = self.drone_policy_ready
        msg.rover_policy_ready = self.rover_policy_ready
        self.status_pub.publish(msg)

    def _publish_markers(self):
        now = self.get_clock().now().to_msg()

        # Publish rover TF: world → base_footprint (only in sim mode)
        # In hw mode, the X3's EKF publishes odom → base_footprint and AMCL publishes map → odom
        if self.rover_odom_ready and self._publish_rover_tf:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = 'world'
            t.child_frame_id = 'base_footprint'
            t.transform.translation.x = float(self.rover_xy[0])
            t.transform.translation.y = float(self.rover_xy[1])
            t.transform.translation.z = 0.0
            t.transform.rotation.x = float(self.rover_quat[0])
            t.transform.rotation.y = float(self.rover_quat[1])
            t.transform.rotation.z = float(self.rover_quat[2])
            t.transform.rotation.w = float(self.rover_quat[3])
            self.tf_broadcaster.sendTransform(t)

        # Drone marker
        if self.drone_odom_ready:
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = 'world'
            m.ns = 'drone'
            m.id = 0
            m.type = Marker.MESH_RESOURCE
            m.action = Marker.ADD
            m.pose.position.x = float(self.drone_pos[0])
            m.pose.position.y = float(self.drone_pos[1])
            m.pose.position.z = float(self.drone_pos[2])
            m.pose.orientation.x = float(self.drone_quat[0])
            m.pose.orientation.y = float(self.drone_quat[1])
            m.pose.orientation.z = float(self.drone_quat[2])
            m.pose.orientation.w = float(self.drone_quat[3])
            m.scale.x = 1.0
            m.scale.y = 1.0
            m.scale.z = 1.0
            m.color.r = 0.522
            m.color.g = 0.902
            m.color.b = 0.145
            m.color.a = 1.0
            m.mesh_resource = self._drone_mesh
            self.drone_marker_pub.publish(m)

        # Rover marker (chassis)
        if self.rover_odom_ready:
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = 'world'
            m.ns = 'rover'
            m.id = 0
            m.type = Marker.MESH_RESOURCE
            m.action = Marker.ADD
            m.pose.position.x = float(self.rover_xy[0])
            m.pose.position.y = float(self.rover_xy[1])
            m.pose.position.z = 0.0325  # base_link at wheel_radius above ground
            m.pose.orientation.x = float(self.rover_quat[0])
            m.pose.orientation.y = float(self.rover_quat[1])
            m.pose.orientation.z = float(self.rover_quat[2])
            m.pose.orientation.w = float(self.rover_quat[3])
            m.scale.x = 1.0
            m.scale.y = 1.0
            m.scale.z = 1.0
            m.color.r = 0.15
            m.color.g = 0.15
            m.color.b = 0.15
            m.color.a = 1.0
            m.mesh_resource = self._rover_mesh
            self.rover_marker_pub.publish(m)

            # Landing pad marker
            m2 = Marker()
            m2.header.stamp = now
            m2.header.frame_id = 'base_link'
            m2.ns = 'rover'
            m2.id = 1
            m2.type = Marker.MESH_RESOURCE
            m2.action = Marker.ADD
            # Pad position relative to base_link (fixed offset, same as CrazySim)
            # CrazySim body is at z=0.065, URDF base_link is at z=0.0815
            # Pad offset in CrazySim: (-0.0384, 0.0004, 0.138) from body
            # Adjust z: 0.138 - (0.0815 - 0.065) = 0.138 - 0.0165 = 0.1215...
            # Actually: pad is at z=0.203 above ground, base_link at z=0.0815
            # So pad z relative to base_link = 0.203 - 0.0815 = 0.1215
            m2.pose.position.x = -0.0384
            m2.pose.position.y = 0.0004
            m2.pose.position.z = 0.138  # match CrazySim offset from body origin
            # 180 degree yaw rotation
            m2.pose.orientation.z = 1.0
            m2.pose.orientation.w = 0.0
            m2.scale.x = 1.0
            m2.scale.y = 1.0
            m2.scale.z = 1.0
            m2.color.r = 0.8
            m2.color.g = 0.2
            m2.color.b = 0.2
            m2.color.a = 1.0
            m2.mesh_resource = self._pad_mesh
            self.rover_marker_pub.publish(m2)


def main(args=None):
    rclpy.init(args=args)

    import yaml
    from ament_index_python.packages import get_package_share_directory
    config_dir = get_package_share_directory('cf_landing_drone')
    with open(f'{config_dir}/config/landing_config.yaml') as f:
        config = yaml.safe_load(f)

    node = MissionManager(config)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
