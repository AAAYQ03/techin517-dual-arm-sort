"""Single SOA arm bringup launch file.

Supports four modes depending on launch arguments:
  1. Follower only (default):
       ros2 launch soa_bringup soa_bringup.launch.py
  2. Leader + follower (overlapping in RViz for teleop monitoring):
       ros2 launch soa_bringup soa_bringup.launch.py leader:=true
  3. Follower in Gazebo (pending soa_sim development):
       ros2 launch soa_bringup soa_bringup.launch.py use_sim:=true
  4. Follower in Gazebo + leader in real life:
       ros2 launch soa_bringup soa_bringup.launch.py leader:=true use_sim:=true

Configuration:
  Edit soa_bringup/config/soa_params.yaml to set USB ports, robot IDs,
  and calibration directory paths.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from soa_bringup.calibration_loader import load_arm_calibration

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']


def _calib_xacro_args(prefix: str, calib: dict) -> list:
    """Return flat list of xacro ' key:=value' strings for one arm.

    Args:
        prefix: Arg name prefix ('' for single arm, 'left_'/'right_' for bimanual).
        calib:  Dict from load_arm_calibration: {joint: {id, offset}}.
    """
    args = []
    for j in JOINTS:
        args += [
            f' {prefix}{j}_id:=',     str(calib[j]['id']),
            f' {prefix}{j}_offset:=',  str(calib[j]['offset']),
        ]
    return args


def _make_xacro_cmd(xacro_file: str, usb_port: str, leader_mode: str,
                    use_sim: str, calib: dict, prefix: str = '') -> Command:
    """Build a xacro Command substitution for soa.urdf.xacro."""
    return Command([
        FindExecutable(name='xacro'), ' ',
        xacro_file,
        ' usb_port:=', usb_port,
        ' leader_mode:=', leader_mode,
        ' use_sim:=', use_sim,
        *_calib_xacro_args(prefix, calib),
    ])


def _make_spawner_chain(controllers: list, cm_namespace: str, inactive: set = None) -> list:
    """Build a sequenced list of controller spawner nodes.

    Controllers are spawned one after another using OnProcessExit handlers
    to avoid race conditions on startup.

    Args:
        controllers: List of controller names to spawn in order.
        cm_namespace: Namespace of the controller_manager to target.
        inactive: Set of controller names to spawn in inactive state.
    """
    inactive = inactive or set()
    spawners = []
    for name in controllers:
        args = [name, '--controller-manager', f'/{cm_namespace}/controller_manager']
        if name in inactive:
            args.append('--inactive')
        spawners.append(Node(
            package='controller_manager',
            executable='spawner',
            arguments=args,
            output='screen',
        ))

    # Chain spawners: each waits for the previous to exit before starting
    actions = [spawners[0]]
    for i in range(1, len(spawners)):
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=spawners[i - 1],
                    on_exit=[spawners[i]],
                )
            )
        )
    return actions


def launch_setup(context, *args, **kwargs):
    use_sim = context.launch_configurations['use_sim']
    leader = context.launch_configurations['leader']
    display = context.launch_configurations['display']
    cameras = context.launch_configurations['cameras']
    controller_mode = context.launch_configurations['controller']

    bringup_share = get_package_share_directory('soa_bringup')
    description_share = get_package_share_directory('soa_description')

    # Load hardware parameters
    params_file = os.path.join(bringup_share, 'config', 'soa_params.yaml')
    with open(params_file) as f:
        hw = yaml.safe_load(f)['/**']['ros__parameters']

    follower_params = hw['follower']
    leader_params = hw['leader']

    # Load follower calibration
    try:
        follower_calib = load_arm_calibration(
            follower_params['calibration_dir'],
            follower_params['id'],
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f'[soa_bringup] Follower calibration not found.\n{e}\n'
            'Run LeRobot calibration and update soa_params.yaml with the correct path.'
        )

    xacro_file = os.path.join(description_share, 'urdf', 'soa.urdf.xacro')
    follower_controllers_yaml = os.path.join(
        description_share, 'config', 'follower_controllers.yaml'
    )
    leader_controllers_yaml = os.path.join(
        description_share, 'config', 'leader_controllers.yaml'
    )

    # --- Follower nodes ---
    follower_urdf_cmd = _make_xacro_cmd(
        xacro_file=xacro_file,
        usb_port=follower_params['usb_port'],
        leader_mode='false',
        use_sim=use_sim,
        calib=follower_calib,
    )
    follower_robot_description = {'robot_description': ParameterValue(follower_urdf_cmd, value_type=str)}

    follower_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='follower',
        output='screen',
        parameters=[follower_robot_description, {'frame_prefix': 'follower/'}],
    )

    follower_cm = Node(
        package='controller_manager',
        executable='ros2_control_node',
        namespace='follower',
        output='screen',
        parameters=[follower_robot_description, follower_controllers_yaml],
    )

    if controller_mode == 'forward':
        active = ['arm_fwd_controller', 'gripper_fwd_controller']
        inactive = {'arm_controller', 'gripper_controller'}
    else:
        active = ['arm_controller', 'gripper_controller']
        inactive = {'arm_fwd_controller', 'gripper_fwd_controller'}

    follower_spawners = _make_spawner_chain(
        ['joint_state_broadcaster'] + active + list(inactive),
        cm_namespace='follower',
        inactive=inactive,
    )

    all_actions = [follower_rsp, follower_cm] + follower_spawners

    # --- Leader nodes (only when leader:=true) ---
    if leader == 'true':
        try:
            leader_calib = load_arm_calibration(
                leader_params['calibration_dir'],
                leader_params['id'],
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f'[soa_bringup] Leader calibration not found.\n{e}\n'
                'Run LeRobot calibration and update soa_params.yaml with the correct path.'
            )

        leader_urdf_cmd = _make_xacro_cmd(
            xacro_file=xacro_file,
            usb_port=leader_params['usb_port'],
            leader_mode='true',
            use_sim='false',  # leaders are always real hardware
            calib=leader_calib,
        )
        leader_robot_description = {'robot_description': ParameterValue(leader_urdf_cmd, value_type=str)}

        leader_rsp = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace='leader',
            output='screen',
            parameters=[leader_robot_description, {'frame_prefix': 'leader/'}],
        )

        leader_cm = Node(
            package='controller_manager',
            executable='ros2_control_node',
            namespace='leader',
            output='screen',
            parameters=[leader_robot_description, leader_controllers_yaml],
        )

        leader_spawners = _make_spawner_chain(
            ['joint_state_broadcaster'],
            cm_namespace='leader',
        )

        # Static transforms: both arms anchored to 'world' at same origin
        # This makes them overlap in RViz so the operator can see leader vs follower movement
        follower_static_tf = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='follower_world_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'follower/base_link'],
            output='screen',
        )
        leader_static_tf = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='leader_world_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'leader/base_link'],
            output='screen',
        )

        teleop_node = Node(
            package='soa_teleop',
            executable='teleop_node',
            name='teleop_node',
            output='screen',
        )

        all_actions += [leader_rsp, leader_cm] + leader_spawners
        all_actions += [follower_static_tf, leader_static_tf, teleop_node]

    # --- Display (RViz) ---
    if display == 'true':
        rviz_config = os.path.join(
            bringup_share, 'rviz',
            'soa_leader_follower.rviz' if leader == 'true' else 'soa.rviz'
        )
        rviz = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
        )
        all_actions.append(rviz)

    # --- Cameras ---
    if cameras == 'true' and use_sim == 'false':
        cameras_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_share, 'launch', 'include', 'cameras.launch.py')
            ),
            launch_arguments={'config_file': 'soa_cameras.yaml'}.items(),
        )
        all_actions.append(cameras_launch)

    # --- Simulation note ---
    if use_sim == 'true':
        all_actions.append(LogInfo(msg=(
            '[soa_bringup] use_sim:=true — ensure Gazebo is running via the '
            'soa_sim package before controllers are spawned.'
        )))

    return all_actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'leader',
            default_value='false',
            description='Launch leader arm in addition to follower. '
                        'Both arms overlap in RViz for teleoperation monitoring.',
        ),
        DeclareLaunchArgument(
            'use_sim',
            default_value='false',
            description='Use Gazebo simulation for the follower arm. '
                        'When leader:=true, the leader still uses real hardware.',
        ),
        DeclareLaunchArgument(
            'display',
            default_value='false',
            description='Launch RViz for visualization.',
        ),
        DeclareLaunchArgument(
            'cameras',
            default_value='true',
            description='Launch camera nodes (usb_cam / realsense2_camera). '
                        'Cameras are always on the follower arm only.',
        ),
        DeclareLaunchArgument(
            'controller',
            default_value='forward',
            description='Controller type: "jtc" (JointTrajectory) or '
                        '"forward" (ForwardCommand).',
        ),
        OpaqueFunction(function=launch_setup),
    ])
