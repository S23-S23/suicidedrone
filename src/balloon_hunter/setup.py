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
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        # 빨간 풍선 모델
        (os.path.join('share', package_name, 'models', 'red_balloon'),
            glob('models/red_balloon/model.*')),
        # 드론 모델 (typhoon_h480) 추가
        (os.path.join('share', package_name, 'models', 'typhoon_h480'),
            glob('models/typhoon_h480/*.sdf*') + glob('models/typhoon_h480/*.config')),
        # 드론 모델 메쉬 파일 (meshes) 추가
        (os.path.join('share', package_name, 'models', 'typhoon_h480', 'meshes'),
            glob('models/typhoon_h480/meshes/*.stl')),
        # iris 모델 추가
        (os.path.join('share', package_name, 'models', 'iris'),
            glob('models/iris/*.sdf*') + glob('models/iris/*.config')),
        (os.path.join('share', package_name, 'models', 'iris', 'meshes'),
            glob('models/iris/meshes/*')),
        # iris_depth_camera 모델 추가
        (os.path.join('share', package_name, 'models', 'iris_depth_camera'),
            glob('models/iris_depth_camera/*.sdf*') + glob('models/iris_depth_camera/*.config')),
        # iris_stereo_camera 모델 추가 (백업)
        (os.path.join('share', package_name, 'models', 'iris_stereo_camera'),
            glob('models/iris_stereo_camera/*.sdf*') + glob('models/iris_stereo_camera/*.config')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') if os.path.exists('config') else []),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Balloon Hunter Team',
    maintainer_email='team@balloonhunter.com',
    description='Drone balloon hunting simulation package using YOLO and PX4',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'balloon_detector = balloon_hunter.balloon_detector:main',
            'position_estimator = balloon_hunter.position_estimator:main',
            'drone_manager = balloon_hunter.drone_manager:main',
            'collision_handler = balloon_hunter.collision_handler:main',
            'gcs_station = balloon_hunter.gcs_station:main',
        ],
    },
)