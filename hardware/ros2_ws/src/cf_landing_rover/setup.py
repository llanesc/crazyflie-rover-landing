import os
from glob import glob
from setuptools import setup

package_name = 'cf_landing_rover'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'models', 'acmpc'), glob('models/acmpc/*')),
        (os.path.join('share', package_name, 'models', 'mlp'), glob('models/mlp/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Christian Llanes',
    maintainer_email='christian@todo.todo',
    description='Rover-side package for cooperative drone-rover landing',
    license='MIT',
    entry_points={
        'console_scripts': [
            'rover_executor = cf_landing_rover.rover_agent:main',
        ],
    },
)
