"""Parameterized camera include launch file for SOA robot arms.

Accepts a 'config_file' launch argument (filename within soa_bringup/config/)
that lists cameras to spawn. Reads the YAML and spawns one ROS2 node per entry.

Usage from a parent launch file:
    IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([bringup_share, 'launch', 'include', 'cameras.launch.py'])
        ]),
        launch_arguments={'config_file': 'soa_cameras.yaml'},
        condition=IfCondition(...),
    )

Camera config YAML format (see soa_cameras.yaml / bi_soa_cameras.yaml):
    cameras:
      - name: wrist_cam
        camera_type: usb_camera          # 'usb_camera' or 'realsense2_camera'
        param_path: soa_usb_cam.yaml     # filename in soa_bringup/config/
        attached_to: follower            # ROS2 node namespace
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch_ros.actions import Node

PKG = 'soa_bringup'

TYPE_REGISTRY = {
    'usb_camera':        ('usb_cam',          'usb_cam_node_exe'),
    'realsense2_camera': ('realsense2_camera', 'realsense2_camera_node'),
}


def _build_camera_nodes(context, *args, **kwargs):
    share_dir = get_package_share_directory(PKG)
    config_file = context.launch_configurations['config_file']
    config_path = os.path.join(share_dir, 'config', config_file)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if not cfg or 'cameras' not in cfg:
        return []

    nodes = []
    for cam in cfg['cameras']:
        name = cam['name']
        cam_type = cam['camera_type']
        param_path = os.path.join(share_dir, 'config', cam['param_path'])
        namespace = cam['attached_to']

        if cam_type not in TYPE_REGISTRY:
            raise ValueError(
                f"Unknown camera_type '{cam_type}' for camera '{name}'. "
                f"Valid types: {list(TYPE_REGISTRY.keys())}"
            )

        pkg, exe = TYPE_REGISTRY[cam_type]

        # Per-node parameter overrides: hardware identity from registry takes precedence
        # over any shared defaults in the type-specific param_path YAML.
        overrides = {'camera_name': name}
        if cam_type == 'usb_camera':
            if 'frame_id' in cam:
                overrides['frame_id'] = cam['frame_id']
            if 'video_device' in cam:
                overrides['video_device'] = cam['video_device']
        elif cam_type == 'realsense2_camera':
            overrides['camera_namespace'] = namespace
            if 'serial_no' in cam:
                overrides['serial_no'] = cam['serial_no']
            if 'usb_port_id' in cam:
                overrides['usb_port_id'] = cam['usb_port_id']

        nodes.append(Node(
            package=pkg,
            executable=exe,
            name=name,
            namespace=namespace,
            output='screen',
            parameters=[param_path, overrides],
        ))

    return [GroupAction(nodes)]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value='soa_cameras.yaml',
            description='Camera config YAML filename (relative to soa_bringup/config/). '
                        'Use soa_cameras.yaml for single arm, bi_soa_cameras.yaml for bimanual.',
        ),
        OpaqueFunction(function=_build_camera_nodes),
    ])
