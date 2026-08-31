from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    share = FindPackageShare('lesta_ros2')
    params = PathJoinSubstitution([
        share, 'config', 'temporal_trav_prediction_node.yaml'
    ])
    rviz_config = PathJoinSubstitution([
        share, 'launch', 'include', 'traversability_prediction_ros2.rviz'
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_rviz', default_value='false',
            description='Start RViz2 with the traversability prediction displays.'
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='rampway2_lidar_extrinsics',
            arguments=[
                '--x', '0.04', '--y', '0.0', '--z', '0.572',
                '--qx', '-0.0079423844', '--qy', '0.1197083765',
                '--qz', '0.0009576874', '--qw', '0.9927768661',
                '--frame-id', 'base_link',
                '--child-frame-id', 'velodyne_tilted',
            ],
            output='screen',
        ),
        Node(
            package='lesta_ros2',
            executable='trav_prediction_node',
            name='lesta_trav_prediction',
            output='screen',
            parameters=[params],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            condition=IfCondition(LaunchConfiguration('use_rviz')),
            output='screen',
        ),
    ])
