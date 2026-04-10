#!/usr/bin/env python3
"""UDP ↔ ROS2 bridge for CrazySim rover.

Runs outside Docker (Jazzy) — receives rover state via UDP from CrazySim,
publishes as /x3/odom and /x3/vel_raw. Subscribes to /x3/cmd_vel and
forwards via UDP to CrazySim.

UDP protocol (all little-endian doubles):
  State (sim→bridge, port 19960): 7 doubles [x, y, cosθ, sinθ, vx_body, vy_body, wz]
  Cmd   (bridge→sim, port 19961): 3 doubles [vx_cmd, vy_cmd, wz_cmd]
"""

import math
import socket
import struct
import threading

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion, Point, Vector3


STATE_PORT = 19960   # receive state from CrazySim
CMD_PORT = 19961     # send cmd_vel to CrazySim
STATE_SIZE = 7 * 8   # 7 doubles = 56 bytes
CMD_SIZE = 3 * 8     # 3 doubles = 24 bytes


class RoverUDPBridge(Node):
    def __init__(self):
        super().__init__('rover_udp_bridge')

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/x3/odom', 10)
        self.vel_pub = self.create_publisher(Twist, '/x3/vel_raw', 10)

        # Subscribers
        self.create_subscription(Twist, '/x3/cmd_vel', self._cmd_vel_cb, 10)
        # Mirror mode: forward real X3 odom to CrazySim so it can update the visual
        self.create_subscription(Odometry, '/x3/odom', self._odom_mirror_cb, 10)

        # UDP sockets
        self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_sock.bind(('0.0.0.0', STATE_PORT))
        self._state_sock.settimeout(1.0)

        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sim_addr = ('127.0.0.1', CMD_PORT)

        # Receive thread
        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

        self.get_logger().info(
            f'Rover UDP bridge: state←:{STATE_PORT}, cmd→:{CMD_PORT}')

    def _recv_loop(self):
        """Receive rover state from CrazySim and publish as ROS2 topics."""
        while self._running:
            try:
                data = self._state_sock.recv(STATE_SIZE)
            except socket.timeout:
                continue
            except Exception:
                break
            if len(data) != STATE_SIZE:
                continue

            x, y, c, s, vx_body, vy_body, wz = struct.unpack('<7d', data)
            theta = math.atan2(s, c)

            # Publish Odometry
            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = 'odom'
            odom.child_frame_id = 'base_footprint'
            odom.pose.pose.position = Point(x=x, y=y, z=0.0)
            qw = math.cos(theta / 2)
            qz = math.sin(theta / 2)
            odom.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
            vx_world = vx_body * c - vy_body * s
            vy_world = vx_body * s + vy_body * c
            odom.twist.twist.linear = Vector3(x=vx_world, y=vy_world, z=0.0)
            odom.twist.twist.angular = Vector3(x=0.0, y=0.0, z=wz)
            self.odom_pub.publish(odom)

            # Publish vel_raw (body-frame)
            vel = Twist()
            vel.linear.x = vx_body
            vel.linear.y = vy_body
            vel.angular.z = wz
            self.vel_pub.publish(vel)

    def _cmd_vel_cb(self, msg: Twist):
        """Forward cmd_vel to CrazySim via UDP."""
        pkt = struct.pack('<3d', msg.linear.x, msg.linear.y, msg.angular.z)
        self._cmd_sock.sendto(pkt, self._sim_addr)

    def _odom_mirror_cb(self, msg: Odometry):
        """Forward real X3 odom to CrazySim for mirror mode visual update.

        Sends 7-double state packet on CMD_PORT. CrazySim distinguishes from
        cmd_vel by packet size (56 bytes vs 24 bytes).
        """
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular
        theta = 2.0 * math.atan2(q.z, q.w)
        c, s = math.cos(theta), math.sin(theta)
        # Convert world velocity to body velocity
        vx_body = v.x * c + v.y * s
        vy_body = -v.x * s + v.y * c
        pkt = struct.pack('<7d', p.x, p.y, c, s, vx_body, vy_body, w.z)
        self._cmd_sock.sendto(pkt, self._sim_addr)

    def destroy_node(self):
        self._running = False
        self._state_sock.close()
        self._cmd_sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RoverUDPBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
