from setuptools import find_packages, setup

package_name = 'soa_functions'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='42076119+htchr@users.noreply.github.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'move_to_pose_server = soa_functions.move_to_pose_server:main',
            'move_to_joint_states_server = soa_functions.move_to_joint_states_server:main',
            'gripper_server = soa_functions.gripper_server:main',
            'controller_switcher = soa_functions.controller_switcher:main',
            'save_joint_states = soa_functions.save_joint_states:main',
            'planning_scene = soa_functions.planning_scene:main',
        ],
    },
)
