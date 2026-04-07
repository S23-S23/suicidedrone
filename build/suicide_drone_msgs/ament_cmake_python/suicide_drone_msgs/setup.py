from setuptools import find_packages
from setuptools import setup

setup(
    name='suicide_drone_msgs',
    version='0.0.0',
    packages=find_packages(
        include=('suicide_drone_msgs', 'suicide_drone_msgs.*')),
)
