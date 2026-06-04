from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('soa_sim2real')
    default_model = PathJoinSubstitution([pkg, 'models', 'policy.pt'])
    default_config = PathJoinSubstitution([pkg, 'config', 'lift_object.yaml'])

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=default_model,
        description='Absolute path to the TorchScript policy file.',
    )
    config_arg = DeclareLaunchArgument(
        'config',
        default_value=default_config,
        description='YAML parameter file for lift_object_server.',
    )

    node = Node(
        package='soa_sim2real',
        executable='lift_object_server',
        name='lift_object_server',
        output='screen',
        parameters=[
            LaunchConfiguration('config'),
            {'model_path': LaunchConfiguration('model_path')},
        ],
    )

    return LaunchDescription([model_path_arg, config_arg, node])
