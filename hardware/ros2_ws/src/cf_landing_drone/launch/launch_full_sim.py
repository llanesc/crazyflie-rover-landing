"""Full simulation launch: CrazySim drone + CrazySim rover + all agents on PC."""

import os
import site
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

package_name = 'cf_landing_drone'

# Find the hardware venv site-packages for policy dependencies
# Walk up from the install share dir to find the workspace root, then to repo root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WS_ROOT = _THIS_DIR
for _ in range(10):
    if os.path.basename(_WS_ROOT) == 'ros2_ws' or os.path.exists(os.path.join(_WS_ROOT, 'src')):
        break
    _WS_ROOT = os.path.dirname(_WS_ROOT)
_HARDWARE_ROOT = os.path.dirname(_WS_ROOT)
_REPO_ROOT = os.path.dirname(_HARDWARE_ROOT)
_VENV_SITE = os.path.join(_HARDWARE_ROOT, 'local_env_hardware',
                           'lib', 'python3.12', 'site-packages')
_ACADOS_ROOT = os.path.join(_REPO_ROOT, 'external', 'leap-c', 'external', 'acados')


def generate_launch_description():
    pkg_dir = get_package_share_directory(package_name)

    crazyflies_yaml = os.path.join(pkg_dir, 'config', 'crazyflies_sim.yaml')

    rviz_config = os.path.join(pkg_dir, 'config', 'landing.rviz')

    # Rover URDF for robot_state_publisher
    # Use the urdf/ directory version (correct mesh paths), not the yahboomcar_description/ subdir version
    rover_urdf_path = os.path.join(
        get_package_share_directory('yahboomcar_description'), 'urdf', 'yahboomcar_X3.urdf')
    # If the installed URDF has wrong mesh paths, fall back to src
    with open(rover_urdf_path, 'r') as f:
        rover_urdf = f.read()
    # Verify mesh paths - if it references meshes/mecanum/, fix to meshes/
    if 'meshes/mecanum/' in rover_urdf:
        rover_urdf = rover_urdf.replace('meshes/mecanum/base_link.STL', 'meshes/base_link_X3.STL')
        rover_urdf = rover_urdf.replace('meshes/mecanum/front_right_wheel.STL', 'meshes/front_right_wheel_X3.STL')
        rover_urdf = rover_urdf.replace('meshes/mecanum/front_left_wheel.STL', 'meshes/front_left_wheel_X3.STL')
        rover_urdf = rover_urdf.replace('meshes/mecanum/back_right_wheel.STL', 'meshes/back_right_wheel_X3.STL')
        rover_urdf = rover_urdf.replace('meshes/mecanum/back_left_wheel.STL', 'meshes/back_left_wheel_X3.STL')
        rover_urdf = rover_urdf.replace('meshes/sensor/camera_link.STL', 'meshes/camera_link.STL')
        rover_urdf = rover_urdf.replace('meshes/sensor/laser_link.STL', 'meshes/laser_link.STL')

    # Build PYTHONPATH: venv site-packages + existing PYTHONPATH
    existing_pp = os.environ.get('PYTHONPATH', '')
    new_pp = _VENV_SITE + (':' + existing_pp if existing_pp else '')

    return LaunchDescription([
        # Ensure venv packages are available to all nodes
        SetEnvironmentVariable('PYTHONPATH', new_pp),
        SetEnvironmentVariable('SCIPY_ARRAY_API', '1'),
        SetEnvironmentVariable('ACADOS_SOURCE_DIR', _ACADOS_ROOT),
        SetEnvironmentVariable('LD_LIBRARY_PATH',
                               os.path.join(_ACADOS_ROOT, 'lib') + ':' +
                               os.environ.get('LD_LIBRARY_PATH', '')),
        # Crazyswarm2 (CrazySim backend for drone)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('crazyflie'), 'launch'),
                '/launch.py',
            ]),
            launch_arguments={
                'crazyflies_yaml_file': crazyflies_yaml,
                'gui': 'False',
                'rviz': 'False',
                'mocap': 'False',
                'teleop': 'False',
                'backend': 'cpp',
            }.items(),
        ),
        # Rover robot_state_publisher (publishes URDF + TF for joints)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='rover_state_publisher',
            namespace='rover',
            parameters=[{'robot_description': rover_urdf}],
        ),
        # RViz2 with landing panel
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
        ),
        # UDP ↔ ROS2 bridge for CrazySim rover (replaces in-Docker rclpy)
        Node(
            package=package_name,
            executable='rover_udp_bridge',
            name='rover_udp_bridge',
            output='screen',
        ),
        # Main executor: mission_manager + drone_agent + rover_agent
        Node(
            package=package_name,
            executable='main_executor',
            name='cf_landing_executor',
            output='screen',
            arguments=['--drone-mode', 'sim', '--rover-mode', 'sim'],
            additional_env={'PYTHONPATH': new_pp},
        ),
    ])
