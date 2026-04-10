"""Full hardware launch: real Crazyflie + real X3 rover + OptiTrack mocap.

Drone agent runs on PC. Rover agent runs on X3 Docker (not launched here).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

package_name = 'cf_landing_drone'


def generate_launch_description():
    pkg_dir = get_package_share_directory(package_name)

    crazyflies_yaml = os.path.join(pkg_dir, 'config', 'crazyflies_hw.yaml')
    motion_capture_yaml = os.path.join(pkg_dir, 'config', 'motion_capture.yaml')

    return LaunchDescription([
        # Crazyswarm2 (real hardware + motion capture)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('crazyflie'), 'launch'),
                '/launch.py',
            ]),
            launch_arguments={
                'crazyflies_yaml_file': crazyflies_yaml,
                'gui': 'False',
                'rviz': 'False',
                'mocap': 'True',
                'mocap_config': motion_capture_yaml,
                'teleop': 'False',
                'backend': 'cpp',
            }.items(),
        ),
        # Main executor: mission_manager + drone_agent only (rover on X3 Docker)
        Node(
            package=package_name,
            executable='main_executor',
            name='cf_landing_executor',
            output='screen',
            arguments=['--drone-mode', 'hw', '--rover-mode', 'hw'],
        ),
    ])
