from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'balloon_hunter'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Balloon Hunter Team',
    maintainer_email='team@balloonhunter.com',
    description='Real-flight balloon tracking with IBVS+PNG',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'balloon_detector = balloon_hunter.balloon_detector:main',
            'filter_node = balloon_hunter.filter_node:main',
            'ibvs_controller = balloon_hunter.ibvs_controller:main',
            'png_guidance = balloon_hunter.png_guidance:main',
            'logger = balloon_hunter.logger:main',
            'drone_manager_real = balloon_hunter.drone_manager_real:main',
            'drone_visualizer_real = balloon_hunter.drone_visualizer_real:main',
        ],
    },
)
