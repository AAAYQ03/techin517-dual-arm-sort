from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('soa_description')
    xacro_cmd = PathJoinSubstitution([FindExecutable(name='xacro')])

    robot_arg = DeclareLaunchArgument(
        'robot',
        default_value='single',
        description='Robot configuration to display: "single" or "bi"',
        choices=['single', 'bi'],
    )

    robot = LaunchConfiguration('robot')
    is_single = IfCondition(PythonExpression(["'", robot, "' == 'single'"]))
    is_bi = IfCondition(PythonExpression(["'", robot, "' == 'bi'"]))

    single_description = ParameterValue(
        Command([
            xacro_cmd, ' ',
            PathJoinSubstitution([pkg_share, 'urdf', 'soa.urdf.xacro']),
        ]),
        value_type=str,
    )

    bi_description = ParameterValue(
        Command([
            xacro_cmd, ' ',
            PathJoinSubstitution([pkg_share, 'urdf', 'bi_soa.urdf.xacro']),
        ]),
        value_type=str,
    )

    rsp_single = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': single_description}],
        condition=is_single,
    )

    rsp_bi = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': bi_description}],
        condition=is_bi,
    )

    jspg = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution([pkg_share, 'rviz', 'soa.rviz'])],
    )

    return LaunchDescription([
        robot_arg,
        rsp_single,
        rsp_bi,
        jspg,
        rviz,
    ])
