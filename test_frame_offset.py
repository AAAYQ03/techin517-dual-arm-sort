#!/usr/bin/env python3
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import tf2_ros
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as MsgDuration

from lerobot_kinematics import lerobot_FK, get_robot

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

TEST_POSES = [
    ("safe_home",       [-0.020, -1.313, 1.433, 1.014, 0.147]),
    ("pan_left_0.3",    [+0.300, -1.313, 1.433, 1.014, 0.147]),
    ("more_extended",   [-0.020, -0.900, 1.000, 1.014, 0.147]),
]


class Tester(Node):
    def __init__(self):
        super().__init__("frame_offset_tester")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.joints = None
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.client = ActionClient(self, FollowJointTrajectory,
                                   "/arm_controller/follow_joint_trajectory")
        self.client.wait_for_server(timeout_sec=10.0)
        self.robot = get_robot('so101')
        # 等 joint_states 第一次到
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.joints is not None:
                break

    def _on_js(self, msg):
        m = dict(zip(msg.name, msg.position))
        if all(n in m for n in JOINT_NAMES):
            self.joints = [m[n] for n in JOINT_NAMES]

    def go(self, target):
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [float(j) for j in target]
        pt.time_from_start = MsgDuration(sec=3)
        traj.points.append(pt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        gh = future.result()
        if not gh or not gh.accepted:
            print("  GOAL REJECTED")
            return False
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf, timeout_sec=15.0)
        # 等臂稳定
        time.sleep(1.0)
        return True

    def sample(self):
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)
        print(f"  joints: {[f'{j:+.3f}' for j in self.joints]}")
        seed = np.array(self.joints[1:5])
        fk = lerobot_FK(seed, robot=self.robot)
        print(f"  lerobot_FK: x={fk[0]:+.4f} y={fk[1]:+.4f} z={fk[2]:+.4f} "
              f"roll={fk[3]:+.4f} pitch={fk[4]:+.4f}")
        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link", "gripper_link", rclpy.time.Time(),
                timeout=Duration(seconds=2.0))
            t = tf.transform.translation
            print(f"  tf2: x={t.x:+.4f} y={t.y:+.4f} z={t.z:+.4f}")
            print(f"  OFFSET (tf2-FK): dx={t.x-fk[0]:+.4f} "
                  f"dy={t.y-fk[1]:+.4f} dz={t.z-fk[2]:+.4f}")
        except Exception as ex:
            print(f"  TF: {ex}")


def main():
    rclpy.init()
    node = Tester()
    print("\n=== Frame Offset Test ===\n")
    for name, joints in TEST_POSES:
        print(f"--- {name} ---")
        print(f"  target: {joints}")
        if not node.go(joints):
            print("  skip\n")
            continue
        node.sample()
        print()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
