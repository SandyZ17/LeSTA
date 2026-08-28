from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    share = FindPackageShare('lesta_ros2')
    params = PathJoinSubstitution([share, 'config', 'trav_mapping_node.yaml'])
    return LaunchDescription([
        Node(package='lesta_ros2', executable='trav_mapping_node',
             name='lesta_trav_mapping', output='screen', parameters=[params]),
    ])
