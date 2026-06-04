"""Save pose service node.

Provides the /follower/save_pose service (soa_interfaces/srv/SavePose)
to capture the current pose of the gripper_link in the base_link frame
and optionally append it to a CSV file for later analysis or replay.

The pose is sampled from tf2 (base_link -> gripper_link) at request time.

Can be run standalone:
    ros2 run soa_functions save_pose

Services:
    /follower/save_pose (soa_interfaces/srv/SavePose)
        request:  csv_path - path to CSV file; if empty, pose is not saved
        response: success, pose
"""

# References:
# - tf2 listener tutorial:
#   https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Listener-Py.html
# - geometry_msgs/Pose:
#   https://docs.ros.org/en/humble/p/geometry_msgs/msg/Pose.html

import os
import csv

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import Pose
from soa_interfaces.srv import SavePose

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


# Frames we look up the gripper pose in.
# Edit these via parameters if your URDF uses different link names.
DEFAULT_TARGET_FRAME = 'gripper_link'
DEFAULT_SOURCE_FRAME = 'base_link'


class SavePoseNode(Node):

    def __init__(self):
        super().__init__('save_pose')

        # Allow overriding the frame names from the launch / CLI.
        self.declare_parameter('target_frame', DEFAULT_TARGET_FRAME)
        self.declare_parameter('source_frame', DEFAULT_SOURCE_FRAME)

        self._target_frame = (
            self.get_parameter('target_frame').get_parameter_value().string_value
        )
        self._source_frame = (
            self.get_parameter('source_frame').get_parameter_value().string_value
        )

        self._cb_group = ReentrantCallbackGroup()

        # tf2 buffer + listener; the listener subscribes to /tf and /tf_static
        # and feeds the buffer with transforms.
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Create the save_pose service under the /follower namespace,
        # matching the convention used by save_joint_states.
        self.create_service(
            SavePose,
            '/follower/save_pose',
            self._handle_save_pose,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f'SavePose service ready. Looking up '
            f'{self._source_frame} -> {self._target_frame}.'
        )

    def _lookup_gripper_pose(self) -> Pose | None:
        """Return the current gripper pose in base_link, or None on failure."""
        try:
            # Time() with default = latest available transform
            t = self._tf_buffer.lookup_transform(
                self._source_frame,
                self._target_frame,
                Time(),
                timeout=Duration(seconds=1.0),
            )
        except TransformException as ex:
            self.get_logger().warn(
                f'Could not transform {self._source_frame} -> '
                f'{self._target_frame}: {ex}'
            )
            return None

        pose = Pose()
        pose.position.x = t.transform.translation.x
        pose.position.y = t.transform.translation.y
        pose.position.z = t.transform.translation.z
        pose.orientation.x = t.transform.rotation.x
        pose.orientation.y = t.transform.rotation.y
        pose.orientation.z = t.transform.rotation.z
        pose.orientation.w = t.transform.rotation.w
        return pose

    def _handle_save_pose(self, req, res):
        """Handle the /follower/save_pose service request."""
        pose = self._lookup_gripper_pose()
        if pose is None:
            self.get_logger().warn(
                'No transform available yet; cannot save pose. '
                'Is the robot publishing tf?'
            )
            res.success = False
            return res

        res.pose = pose
        res.success = True

        if req.csv_path:
            try:
                self._append_to_csv(req.csv_path, pose)
            except OSError as e:
                self.get_logger().error(
                    f'Failed to write pose to {req.csv_path}: {e}'
                )
                res.success = False

        return res

    def _append_to_csv(self, path: str, pose: Pose) -> None:
        """Append a single row (x, y, z, qx, qy, qz, qw) to a CSV file.

        Writes a header row the first time the file is created.
        """
        file_exists = os.path.isfile(path)

        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
            writer.writerow([
                f'{pose.position.x:.6f}',
                f'{pose.position.y:.6f}',
                f'{pose.position.z:.6f}',
                f'{pose.orientation.x:.6f}',
                f'{pose.orientation.y:.6f}',
                f'{pose.orientation.z:.6f}',
                f'{pose.orientation.w:.6f}',
            ])

        self.get_logger().info(
            f'Appended pose to {path}: '
            f'pos=({pose.position.x:.4f}, {pose.position.y:.4f}, '
            f'{pose.position.z:.4f}), '
            f'quat=({pose.orientation.x:.4f}, {pose.orientation.y:.4f}, '
            f'{pose.orientation.z:.4f}, {pose.orientation.w:.4f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = SavePoseNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
