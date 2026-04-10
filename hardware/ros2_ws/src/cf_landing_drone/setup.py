import os
from glob import glob
from setuptools import setup

package_name = 'cf_landing_drone'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.rviz')),

        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'models', 'acmpc'), glob('models/acmpc/*')),
        (os.path.join('share', package_name, 'models', 'mlp'), glob('models/mlp/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Christian Llanes',
    maintainer_email='christian@todo.todo',
    description='Drone-side package for cooperative drone-rover landing',
    license='MIT',
    entry_points={
        'console_scripts': [
            'main_executor = cf_landing_drone.main:main',
            'drone_agent = cf_landing_drone.drone_agent:main',
            'rover_agent = cf_landing_drone.rover_agent:main',
            'mission_manager = cf_landing_drone.mission_manager:main',
            'rover_udp_bridge = cf_landing_drone.rover_udp_bridge:main',
        ],
    },
)
