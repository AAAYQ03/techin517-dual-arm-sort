#!/usr/bin/env python3
"""
Pick-and-place using hardcoded poses recorded from the real follower arm.
Bypasses CSV loading; poses are baked in directly.

Prerequisites:
    ros2 launch soa_moveit_config soa_moveit_bringup.launch.py
    ros2 run soa_functions move_to_pose_server
    ros2 run soa_functions gripper_server
    ros2 run soa_apps go_to_poses
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import Pose
from soa_interfaces.action import Gripper, MoveToPose


GRIPPER_OPEN = 1.7453
GRIPPER_CLOSED = 0.1


def make_pose(x, y, z, qx, qy, qz, qw):
    p = Pose()
    p.position.x = x
    p.position.y = y
    p.position.z = z
    p.orientation.x = qx
    p.orientation.y = qy
    p.orientation.z = qz
    p.orientation.w = qw
    return p


# Hardcoded poses recorded from real follower hardware
POSES = [
    make_pose(0.299264, 0.020239, 0.034028, -0.002369, 0.041334, 0.020896, 0.998924),  # 0: overhead
    make_pose(0.268866, -0.026402, 0.111932, 0.045620, 0.208934, 0.098933, 0.971842),  # 1: pick
    make_pose(0.287774, -0.028558, 0.099952, 0.028421, 0.157831, 0.063753, 0.984996),  # 2: retract
    make_pose(0.289319, -0.037727, 0.177729, -0.010403, -0.048739, 0.061451, 0.996865),  # 3: place
]

SEQUENCE = [
    ('gripper', GRIPPER_OPEN),
    ('pose', 0),
    ('pose', 1),
    ('gripper', GRIPPER_CLOSED),
    ('pose', 2),
    ('pose', 3),
    ('gripper', GRIPPER_OPEN),
]


class GoToPoses(Node):

    def __init__(self):
        super().__init__('go_to_poses')
        self._pose_client = ActionClient(self, MoveToPose, 'move_to_pose')
        self._gripper_client = ActionClient(self, Gripper, 'gripper_command')

    def send_pose_goal(self, pose, label=''):
        goal = MoveToPose.Goal()
        goal.target_pose = pose
        self.get_logger().info(
            f'Sending pose goal ({label}): '
            f'pos=({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})'
        )
        self._pose_client.wait_for_server()
        future = self._pose_client.send_goal_async(goal, feedback_callback=self._pose_fb)
        rclpy.spin_until_future_complete(self, future)
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('Pose goal rejected')
            return False
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        result = rf.result().result
        if result.success:
            self.get_logger().info(f'Pose succeeded: {result.message}')
        else:
            self.get_logger().error(f'Pose failed: {result.message}')
        return result.success

    def send_gripper_goal(self, target, label=''):
        goal = Gripper.Goal()
        goal.target_position = float(target)
        self.get_logger().info(f'Sending gripper goal ({label}): target={target:.4f}')
        self._gripper_client.wait_for_server()
        future = self._gripper_client.send_goal_async(goal, feedback_callback=self._gripper_fb)
        rclpy.spin_until_future_complete(self, future)
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('Gripper goal rejected')
            return False
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        result = rf.result().result
        if result.success:
            self.get_logger().info(f'Gripper succeeded: {result.message}')
        else:
            self.get_logger().error(f'Gripper failed: {result.message}')
        return result.success

    def _pose_fb(self, msg):
        self.get_logger().info(f'  [fb] dist={msg.feedback.distance_to_goal:.4f}')

    def _gripper_fb(self, msg):
        self.get_logger().info(f'  [fb] gripper={msg.feedback.current_position:.4f}')

    def run(self):
        self.get_logger().info(f'Loaded {len(POSES)} hardcoded poses')
        for i, step in enumerate(SEQUENCE):
            kind = step[0]
            self.get_logger().info(f'--- Step {i+1}/{len(SEQUENCE)}: {kind} ---')
            if kind == 'pose':
                ok = self.send_pose_goal(POSES[step[1]], label=f'pose#{step[1]}')
            elif kind == 'gripper':
                lbl = 'OPEN' if step[1] > 0.9 else 'CLOSED'
                ok = self.send_gripper_goal(step[1], label=lbl)
            else:
                ok = False
            if not ok:
                self.get_logger().error(f'Step {i+1} failed; aborting')
                return
        self.get_logger().info('=== Sequence complete ===')


def main(args=None):
    rclpy.init(args=args)
    node = GoToPoses()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
