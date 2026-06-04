import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'soa_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files: top-level and include/
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'launch', 'include'),
            glob('launch/include/*.py')),
        # Config files
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        # RViz configs
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
        # Rosetta contracts
        (os.path.join('share', package_name, 'rosetta_contracts'),
            glob('rosetta_contracts/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='42076119+htchr@users.noreply.github.com',
    description='Launch files and configurations for SOA robot arm bringup',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
