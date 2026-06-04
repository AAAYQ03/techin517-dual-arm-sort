#!/usr/bin/env python3
"""
Pick an object at a given (x, y, z) position in the robot's base frame.

The script follows the Lab 5 pick sequence:
  1. Move above the object (z + APPROACH_HEIGHT, gripper pointing down)
  2. Open the gripper
  3. Move down to pick the object
  4. Close the gripper (not all the way -- preserves cube)
  5. Lift the object back to the above pose

Usage:
    # 1. Launch the MoveIt stack:
    ros2 launch soa_moveit_config soa_moveit_bringup.launch.py
    # 2. Start the action servers:
    ros2 run soa_functions move_to_pose_server
    ros2 run soa_functions gripper_server
    # 3. Run this app with x/y/z parameters:
    ros2 run soa_apps pick_by_position --ros-args -p x:=0.28 -p y:=0.025 -p z:=0.05
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import Pose
from soa_interfaces.action import Gripper, MoveToPose


# Gripper joint positions (rad). Same convention as go_to_poses.py.
GRIPPER_OPEN = 1.6
GRIPPER_PICK = 0.7   # closed enough to grasp 3 cm cube, but NOT all the way

# How far above the object to approach from. 10 cm is a comfortable lift.
APPROACH_HEIGHT = 0.09

# Fixed gripper orientation: pointing straight down.
# gripper_frame_link's +z points away from the wrist along the gripper.
# Rotating 180 deg about the world Y-axis flips +z to -z (i.e. straight down).
# Quaternion (xyzw) = [0, 1, 0, 0]
GRIPPER_DOWN_QUAT = (0.0, -0.3827, 0.0, 0.9239)


def make_pose(x: float, y: float, z: float,
              quat_xyzw=GRIPPER_DOWN_QUAT) -> Pose:
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.x = float(quat_xyzw[0])
    pose.orientation.y = float(quat_xyzw[1])
    pose.orientation.z = float(quat_xyzw[2])
    pose.orientation.w = float(quat_xyzw[3])
    return pose


class PickByPosition(Node):

    def __init__(self):
        super().__init__('pick_by_position')

        # Target object position (in follower/base_link frame).
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('z', 0.0)

        # Action clients (server-side names match move_to_pose_server / gripper_server).
        self._pose_client = ActionClient(self, MoveToPose, 'move_to_pose')
        self._gripper_client = ActionClient(self, Gripper, 'gripper_command')

    # ---------- helpers ----------

    def send_pose_goal(self, pose: Pose, label: str = '') -> bool:
        goal = MoveToPose.Goal()
        goal.target_pose = pose

        self.get_logger().info(
            f'Sending pose goal ({label}): '
            f'pos=({pose.position.x:.3f}, {pose.position.y:.3f}, '
            f'{pose.position.z:.3f}), '
            f'quat=({pose.orientation.x:.3f}, {pose.orientation.y:.3f}, '
            f'{pose.orientation.z:.3f}, {pose.orientation.w:.3f})'
        )

        self._pose_client.wait_for_server()
        future = self._pose_client.send_goal_async(
            goal, feedback_callback=self._pose_feedback_callback
        )
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Pose goal rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.success:
            self.get_logger().info(f'Pose goal succeeded: {result.message}')
        else:
            self.get_logger().error(f'Pose goal failed: {result.message}')
        return result.success

    def send_gripper_goal(self, target_position: float, label: str = '') -> bool:
        goal = Gripper.Goal()
        goal.target_position = float(target_position)

        self.get_logger().info(
            f'Sending gripper goal ({label}): target={target_position:.4f}'
        )

        self._gripper_client.wait_for_server()
        future = self._gripper_client.send_goal_async(
            goal, feedback_callback=self._gripper_feedback_callback
        )
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Gripper goal rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.success:
            self.get_logger().info(f'Gripper goal succeeded: {result.message}')
        else:
            self.get_logger().error(f'Gripper goal failed: {result.message}')
        return result.success

    def _pose_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'  [feedback] distance_to_goal={feedback.distance_to_goal:.4f} m'
        )

    def _gripper_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'  [feedback] gripper current_position={feedback.current_position:.4f}'
        )

    # ---------- pick sequence ----------

    def run(self):
        x = self.get_parameter('x').get_parameter_value().double_value
        y = self.get_parameter('y').get_parameter_value().double_value
        z = self.get_parameter('z').get_parameter_value().double_value

        self.get_logger().info(
            f'=== pick_by_position: target=({x:.3f}, {y:.3f}, {z:.3f}) ==='
        )

        above_pose = make_pose(x, y, z + APPROACH_HEIGHT)
        PICK_Z_OFFSET = 0.02  # raise pick height to clear gripper mechanism
        pick_pose = make_pose(x, y, z + PICK_Z_OFFSET)

        # Step 1: Move above the object
        self.get_logger().info('--- Step 1/5: Move above object ---')
        if not self.send_pose_goal(above_pose, label='above'):
            return

        # Step 2: Open the gripper
        self.get_logger().info('--- Step 2/5: Open gripper ---')
        if not self.send_gripper_goal(GRIPPER_OPEN, label='OPEN'):
            return

        # Step 3: Move down to pick
        self.get_logger().info('--- Step 3/5: Descend to pick ---')
        if not self.send_pose_goal(pick_pose, label='pick'):
            return

        # Step 4: Close gripper (not all the way)
        self.get_logger().info('--- Step 4/5: Close gripper (partial) ---')
        if not self.send_gripper_goal(GRIPPER_PICK, label='PICK'):
            return

        # Step 5: Lift the object
        self.get_logger().info('--- Step 5/5: Lift object ---')
        if not self.send_pose_goal(above_pose, label='lift'):
            return

        self.get_logger().info('=== pick_by_position complete ===')


def main(args=None):
    rclpy.init(args=args)

    node = PickByPosition()
    try:
        node.run()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()