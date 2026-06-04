#!/usr/bin/env python3
"""
MoveToPose action server for the SOA 5-DOF arm.

Uses pymoveit2 to plan and execute IK-based motion to a target pose.
Implements a fallback strategy for the 5-DOF arm:
  1. Attempt full pose (position + orientation)
  2. Fall back to position-only IK if full pose planning fails

Usage:
    ros2 run soa_functions move_to_pose_server
"""

import math
import time
from threading import Thread

# TODO: import rclpy
# TODO: from rclpy, import:
#       ActionServer
#       ReentrantCallbackGroup
#       MultiThreadedExecutor
#       Node

# TODO: import MoveIt2 and MoveIt2State from pymoveit2

# TODO: import the MoveToPose action from soa_interfaces 

from soa_functions import soa_robot


class MoveToPoseServer(Node):

    def __init__(self):
        super().__init__('move_to_pose_server')

        # Declare configurable parameters
        self.declare_parameter('max_velocity', 0.5)
        self.declare_parameter('max_acceleration', 0.5)
        self.declare_parameter('tolerance_position', 0.01)
        self.declare_parameter('tolerance_orientation', 0.1)
        self.declare_parameter('tolerance_orientation_relaxed', 0.5)
        self.declare_parameter('num_planning_attempts', 5)
        self.declare_parameter('allowed_planning_time', 3.0)
        self.declare_parameter('max_reach', 0.4)

        # Callback group for pymoveit2 (must be reentrant)
        self._cb_group = #TODO: create the callback group

        # Initialize MoveIt2 interface
        self._moveit2 = MoveIt2(
            # TODO: initialize moveit2 using pymoveit2
            #       checkout the example from last week's lab
            #       use the information provided in the soa_robot object provided above
        )

        # Apply velocity/acceleration scaling
        self._moveit2.max_velocity = (
            # TODO: set the maximum velocity to move the arm safely
        )
        self._moveit2.max_acceleration = (
            # TODO: set the maximum acceleration to move the arm safely
        )

        # Increase planning budget (pymoveit2 defaults: 0.5s / 5 attempts)
        self._moveit2.num_planning_attempts = (
            # TODO: set the number of planning attempts
        )
        self._moveit2.allowed_planning_time = (
            # TODO: set the allowed planning time
        )

        # Create action server
        self._action_server = ActionServer(
            # TODO: create the action server
        )

        self.get_logger().info('MoveToPose action server ready')

    def _wait_and_publish_feedback(self, goal_handle, target_position):
        """Wait for MoveIt2 execution, publishing feedback each iteration."""
        # TODO: publish feedback while waiting for MoveIt to finish

    def _plan_and_execute(self, goal_handle, position, quat_xyzw=None,
                          tol_pos=0.01, tol_orient=0.1,
                          planning_time=None) -> bool:
        """Plan and execute a single motion attempt. Returns True on success."""
        base_time = (
            self.get_parameter('allowed_planning_time')
            .get_parameter_value().double_value
        )
        self._moveit2.allowed_planning_time = (
            planning_time if planning_time is not None else base_time
        )
        self._moveit2.clear_goal_constraints()

        kwargs = dict(
            position=position,
            tolerance_position=tol_pos,
            start_joint_state=self._moveit2.joint_state,
        )
        if quat_xyzw is not None:
            kwargs['quat_xyzw'] = quat_xyzw
            kwargs['tolerance_orientation'] = tol_orient

        future = self._moveit2.plan_async(**kwargs)
        if future is None:
            return False

        while not future.done():
            time.sleep(0.1)

        trajectory = self._moveit2.get_trajectory(future)
        if trajectory is None:
            return False

        self._moveit2.execute(trajectory)
        return self._wait_and_publish_feedback(goal_handle, position)

    def _execute_callback(self, goal_handle):
        self.get_logger().info('Received MoveToPose goal')

        tol_pos = (
            self.get_parameter('tolerance_position')
            .get_parameter_value().double_value
        )
        tol_orient = (
            self.get_parameter('tolerance_orientation')
            .get_parameter_value().double_value
        )
        tol_orient_relaxed = (
            self.get_parameter('tolerance_orientation_relaxed')
            .get_parameter_value().double_value
        )
        planning_time = (
            self.get_parameter('allowed_planning_time')
            .get_parameter_value().double_value
        )

        max_reach = (
            self.get_parameter('max_reach')
            .get_parameter_value().double_value
        )

        # TODO: write the action callback function
        #       1. retireve the target pose from the goal_handle
        #           reference the action definition in the interface package
        #       2. deconstruction the position into position and orientation
        #       3. initialize the action result from the MoveToPose object that you imported above

        # --- Pre-flight validation ---
        dist = math.sqrt(sum(p ** 2 for p in position))
        if dist > max_reach:
            # TODO: abort the goal if the position distance is further than the maximum distance
            #       set result success field to false
            #       set the result message to something informative
            #       use the ros logger to log an informative warning
            #       return the result

        quat_norm = math.sqrt(sum(q ** 2 for q in quat_xyzw))
        if abs(quat_norm - 1.0) > 0.01:
            # TODO: abort the goal if the user provides an invalid orientation
            #       set the result success field to false
            #       set the result message to something informative
            #       use the ros logger to log an informative warning
            #       return the result

        attempts = [
            # (label, quat_xyzw, tol_pos, tol_orient, planning_time)
            ('Attempt 1: full pose (tight)',
             quat_xyzw, tol_pos, tol_orient, planning_time),
            ('Attempt 2: full pose (relaxed orientation)',
             quat_xyzw, tol_pos, tol_orient_relaxed, planning_time),
            ('Attempt 3: position-only',
             None, tol_pos * 2, None, planning_time + 2.0),
        ]

        success_messages = [
            'Reached target: full pose (tight)',
            'Reached target: full pose (relaxed orientation)',
            'Reached target: position-only IK (orientation ignored)',
        ]

        self.get_logger().info(
            f'Target position: {position}, orientation: {quat_xyzw}'
        )

        for i, (label, 
                quaternion, 
                tolerance_position, 
                tolerance_orientation, 
                plan_time) in enumerate(attempts):
            self.get_logger().info(label)
            success = self._plan_and_execute(
                # TODO: use the information provided to call the plan and execute function defined above
            )
            if success:
                # TODO: return positive results
                #       call the goal_handle's succeed function
                #       set the result's success field to true
                #       use the appropriate success message defined above based on the current iteration
                #       use the ros logger to log something informative
                #       return the result

            self.get_logger().warn(f'{label} — failed')

        # TODO: abort the goal
        #       if the code gets this far, that means all attmpets have failed
        #       abort the goal
        #       set the result object's fields
        #       log the failure
        #       return the result

    def _publish_feedback(self, goal_handle, target_position):
        """Publish distance feedback from current EE pose to target."""
        feedback = MoveToPose.Feedback()
        try:
            # Use compute_fk_async() to avoid rclpy.spin_once() in compute_fk()
            fk_future = self._moveit2.compute_fk_async()
            if fk_future is not None:
                while not fk_future.done():
                    time.sleep(0.1)
                fk_result = self._moveit2.get_compute_fk_result(fk_future)
            else:
                fk_result = None

            if fk_result is not None:
                current = fk_result.pose.position
                dx = current.x - target_position[0]
                dy = current.y - target_position[1]
                dz = current.z - target_position[2]
                feedback.distance_to_goal = math.sqrt(dx**2 + dy**2 + dz**2)
            else:
                feedback.distance_to_goal = -1.0
        except Exception:
            feedback.distance_to_goal = -1.0
        goal_handle.publish_feedback(feedback)


def main(args=None):
    # TODO: define the main function


if __name__ == '__main__':
    main()
