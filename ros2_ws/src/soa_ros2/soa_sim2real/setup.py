import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'soa_sim2real'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='42076119+htchr@users.noreply.github.com',
    description='Sim-to-real PyTorch JIT policy runner for the SOA lift task.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lift_object_server = soa_sim2real.lift_object_server:main',
        ],
    },
)
