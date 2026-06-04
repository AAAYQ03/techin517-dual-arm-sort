#!/usr/bin/env python3
"""
Iterate through a CSV of joint states and drive the SOA arm + gripper to each row in order.

Each row in the CSV specifies all 5 arm joint positions plus a gripper position.
For every row: move arm to joint configuration, then set the gripper position.

CSV format: shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper (with header)

Prerequisites:
    # 1. Launch the MoveIt stack:
    ros2 launch soa_moveit_config soa_moveit_bringup.launch.py

    # 2. Start the action servers:
    ros2 run soa_actions move_to_joint_states_server
    ros2 run soa_actions gripper_server

    # 3. Run this node (default CSV path):
    ros2 run soa_apps go_to_joint_states

    # Or with a custom CSV path:
    ros2 run soa_apps go_to_joint_states --ros-args -p csv_path:=/path/to/joints.csv
"""

import csv

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

# TODO: import the Gripper and MoveToJointStates actions from soa_interfaces


DEFAULT_CSV_PATH = '/home/ubuntu/techin517/soa_ws/joints.csv'

ARM_JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']


def load_rows(path: str) -> list:
    """
    Parse CSV file and return a list of dicts (one per row), e.g.
    [
        {'joint1': '0.0', 'joint2': '0.0', etc.},
        {'joint1': '0.0', 'joint2': '0.0', etc.},
    ]
    """
    # TODO: open the csv file of joint states and load them into a list of dictionaries 
    #       return the list


class GoToJointStates(Node):

    def __init__(self):
        super().__init__('go_to_joint_states')

        self.declare_parameter('csv_path', DEFAULT_CSV_PATH)

        self._joint_client = # TODO: create the action client for the MoveToJointStates action
        self._gripper_client = # TODO: creat the action client for the Gripper action

    def send_joint_goal(self, joint_positions: list, joint_names: list, label: str = '') -> bool:
        """Send a MoveToJointStates goal and block until the result is received."""
        goal = # TODO: create the goal for the move to joint states action

        self.get_logger().info(
            f'Sending joint goal ({label}): '
            + ', '.join(f'{n}={p:.4f}' for n, p in zip(joint_names, joint_positions))
        )

        self._joint_client.wait_for_server()

        future = # TODO: send the goal asynchronously
        #                refer to the documentation to understand what "future" is here
        #                https://docs.ros.org/en/humble/Tutorials/Intermediate/Writing-an-Action-Server-Client/Py.html#writing-an-action-client   
        rclpy.spin_until_future_complete(self, future)

        goal_handle = # TODO: get the result from future to see if the goal was accepted
        if not goal_handle.accepted:
            self.get_logger().error('Joint goal rejected')
            return False

        # wait until the action is complete to get the action result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if result.success:
            self.get_logger().info(f'Joint goal succeeded: {result.message}')
        else:
            self.get_logger().error(f'Joint goal failed: {result.message}')
        return result.success

    def send_gripper_goal(self, target_position: float, label: str = '') -> bool:
        """Send a Gripper goal and block until the result is received."""
        # TODO: copy the pattern above for the gripper action goal
        #       1. create the goal and set the target position
        #       2. send the goal asynchronously
        #       3. create the goal handle 
        #       4. check if the goal was accepted
        #       5. get the action result from the goal handle
        #       6. use the ros logger to send informative information
        #       7. return the result success boolean

    def _joint_feedback_callback(self, feedback_msg):
        # TODO: log the feedback from the joint action

    def _gripper_feedback_callback(self, feedback_msg):
        # TODO: log the feedback from the gripper action

    def run(self):
        """Load joint states from CSV and execute each row in order."""
        csv_path = self.get_parameter('csv_path').get_parameter_value().string_value
        self.get_logger().info(f'Loading joint states from: {csv_path}')

        rows = load_rows(csv_path)
        self.get_logger().info(f'Loaded {len(rows)} row(s)')

        self.get_logger().info('=== Starting go_to_joint_states sequence ===')

        for i, row in enumerate(rows):
            self.get_logger().info(f'--- Row {i + 1}/{len(rows)} ---')

            # Move arm to joint configuration
            positions = [float(row[j]) for j in ARM_JOINTS]
            success = self.send_joint_goal(positions, ARM_JOINTS, label=f'row {i}')
            if not success:
                self.get_logger().error(f'Arm goal failed at row {i}. Aborting.')
                return

            # Set gripper position
            success = self.send_gripper_goal(float(row['gripper']), label=f'row {i} gripper')
            if not success:
                self.get_logger().error(f'Gripper goal failed at row {i}. Aborting.')
                return

        self.get_logger().info('=== go_to_joint_states sequence complete ===')


def main(args=None):
    rclpy.init(args=args)

    node = GoToJointStates()
    try:
        node.run()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
