#!/usr/bin/env python3
"""
pipeline_to_above_only.py - v0.7 派生的简化版
用途：方案 A 重录数据的"起点定位器"
  CV → IK → MoveIt 送到 above → 打印 follower 当前关节角 → 退出
退出后：ROS launch 还在跑，follower servo 保持在 above 位置；
        然后人眼摆 leader 对齐，再杀 ROS 跑 lerobot-record
"""

import sys
import os
import time
import math
from types import SimpleNamespace

sys.path.insert(0, os.path.expanduser("~/techin517/cv_module"))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, JointState
from cv_bridge import CvBridge

import tf2_ros
from tf2_ros import TransformException
import geometry_msgs.msg
from tf2_geometry_msgs import do_transform_point

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint
from control_msgs.action import GripperCommand

from cv_module import load_detector, detect_objects
from lerobot_kinematics import lerobot_IK, get_robot


TOPIC_COLOR   = "/static_camera/overhead_cam/color/image_raw"
TOPIC_DEPTH   = "/static_camera/overhead_cam/aligned_depth_to_color/image_raw"
TOPIC_CAMINFO = "/static_camera/overhead_cam/color/camera_info"
TARGET_FRAME = "base_link"
SOURCE_FRAME = "overhead_camoverhead_cam_color_optical_frame"

# 物体相关——录别的物体时改这里
TEXT_PROMPT    = "a rounded white case."
TARGET_CLASSES = {"case", "rounded white case"}
BOX_THRESHOLD  = 0.20
DEPTH_SCALE    = 0.001
DEVICE         = "cuda"

SAFE_HOME_JOINTS = {
    "shoulder_pan":  -0.020,
    "shoulder_lift": -1.313,
    "elbow_flex":    +1.433,
    "wrist_flex":    +1.014,
    "wrist_roll":    +0.147,
}
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def is_valid_position(base_xyz):
    """工作空间过滤: 滤掉机械臂误识别 (跟 dispatch_pick_f7.py 一致)"""
    x, y, z = base_xyz
    return (0.18 < x < 0.30) and (abs(y) < 0.12) and (-0.02 < z < 0.06)

URDF_TO_LEROBOT_DX = 0.030
URDF_TO_LEROBOT_DZ = 0.056
LEROBOT_ROLL  = 0.147
LEROBOT_PITCH = 1.30   # 注意：与 5.13 总结里的 1.134 不一致，跟现版 v0.7 保持一致

ARM_GROUP       = "arm"
GRIPPER_OPEN    = 1.7
APPROACH_HEIGHT = 0.165
PLANNING_TIME   = 5.0


def make_joint_goal(joint_dict, tolerance=0.01):
    constraints = Constraints()
    for name, val in joint_dict.items():
        c = JointConstraint()
        c.joint_name = name
        c.position = float(val)
        c.tolerance_above = tolerance
        c.tolerance_below = tolerance
        c.weight = 1.0
        constraints.joint_constraints.append(c)
    return constraints


def base_xyz_to_arm_joints(x_base, y_base, z_base, seed_qpos):
    shoulder_pan = math.atan2(y_base, x_base)
    horizontal_dist = math.sqrt(x_base**2 + y_base**2)
    target_x_lerobot = horizontal_dist - URDF_TO_LEROBOT_DX
    target_z_lerobot = z_base - URDF_TO_LEROBOT_DZ
    target_gpos = np.array([target_x_lerobot, 0.0, target_z_lerobot,
                            LEROBOT_ROLL, LEROBOT_PITCH, 0.0])
    qpos_inv, ok = lerobot_IK(seed_qpos, target_gpos, robot=get_robot('so101'))
    if not ok:
        return None, False
    return {
        "shoulder_pan":  float(shoulder_pan),
        "shoulder_lift": float(qpos_inv[0]),
        "elbow_flex":    float(qpos_inv[1]),
        "wrist_flex":    float(qpos_inv[2]),
        "wrist_roll":    float(qpos_inv[3]),
    }, True


class PipelineToAboveOnly(Node):
    def __init__(self):
        super().__init__("pipeline_to_above_only")
        self.bridge = CvBridge()
        self.top_color_msg = None
        self.top_depth_msg = None
        self.top_caminfo = None
        self.joint_state_msg = None

        cam_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(Image, TOPIC_COLOR, self._on_top_color, cam_qos)
        self.create_subscription(Image, TOPIC_DEPTH, self._on_top_depth, cam_qos)
        self.create_subscription(CameraInfo, TOPIC_CAMINFO, self._on_top_caminfo, cam_qos)
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.move_client = ActionClient(self, MoveGroup, "/move_action")
        self.move_client.wait_for_server(timeout_sec=10.0)
        self.get_logger().info("/move_action ready.")
        self.gripper_client = ActionClient(self, GripperCommand, "/gripper_controller/gripper_cmd")
        self.gripper_client.wait_for_server(timeout_sec=10.0)
        self.get_logger().info("/gripper_controller/gripper_cmd ready.")

        self.get_logger().info("Loading Grounding DINO ...")
        self.processor_cv, self.model_cv = load_detector(device=DEVICE)
        self.get_logger().info("Detector ready.")

    def _on_top_color(self, msg):   self.top_color_msg = msg
    def _on_top_depth(self, msg):   self.top_depth_msg = msg
    def _on_top_caminfo(self, msg): self.top_caminfo = msg
    def _on_joint_state(self, msg): self.joint_state_msg = msg

    def wait_for_top_camera(self, timeout_s=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.top_color_msg and self.top_depth_msg and self.top_caminfo:
                return True
        return False

    def warmup(self, seconds=3.0):
        t0 = time.time()
        while time.time() - t0 < seconds:
            rclpy.spin_once(self, timeout_sec=0.1)

    def send_joint_goal_moveit(self, joint_dict, label="goal", tolerance=0.01):
        c = make_joint_goal(joint_dict, tolerance=tolerance)
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = ARM_GROUP
        req.num_planning_attempts = 5
        req.allowed_planning_time = PLANNING_TIME
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3
        req.goal_constraints.append(c)
        goal.request = req
        goal.planning_options.plan_only = False
        future = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        gh = future.result()
        if gh is None or not gh.accepted:
            return False
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf, timeout_sec=30.0)
        r = rf.result()
        if r is None or r.result.error_code.val != 1:
            return False
        return True

    def set_gripper(self, position, label="gripper", wait=True):
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = 10.0
        future = self.gripper_client.send_goal_async(goal)
        if wait:
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            gh = future.result()
            if gh and gh.accepted:
                rf = gh.get_result_async()
                rclpy.spin_until_future_complete(self, rf, timeout_sec=5.0)
        return True

    def go_safe_home(self):
        return self.send_joint_goal_moveit(SAFE_HOME_JOINTS, "safe_home", 0.05)

    def detect_target(self):
        if not self.wait_for_top_camera():
            return None
        color = self.bridge.imgmsg_to_cv2(self.top_color_msg, desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(self.top_depth_msg, desired_encoding="passthrough")
        K = np.array(self.top_caminfo.k).reshape(3, 3)
        intrin = SimpleNamespace(fx=K[0,0], fy=K[1,1], ppx=K[0,2], ppy=K[1,2])
        dets = detect_objects(
            color_image=color, depth_image=depth, intrin=intrin,
            depth_scale=DEPTH_SCALE, processor=self.processor_cv,
            model=self.model_cv, text_prompt=TEXT_PROMPT,
            target_classes=TARGET_CLASSES, box_threshold=BOX_THRESHOLD,
            device=DEVICE)

        targets = [d for d in dets if d.get("is_target") and d.get("depth_valid")]
        if not targets:
            return None
        targets.sort(key=lambda d: d["score"], reverse=True)

        # 工作空间过滤: 转 base_link 看是否合法, 不合法跳到下一个 (滤掉机械臂误识别)
        stamp = self.top_color_msg.header.stamp
        for t in targets:
            base_xyz = self.cam_to_base(t["cam_xyz_m"], stamp)
            if base_xyz is None:
                self.get_logger().warn(f"  [SKIP] TF lookup failed: score={t['score']:.2f}")
                continue
            if is_valid_position(base_xyz):
                self.get_logger().info(f"Target: score={t['score']:.2f} cam_xyz={t['cam_xyz_m']}")
                self.get_logger().info(f"  ✓ base_link valid: {base_xyz}")
                return t
            else:
                self.get_logger().warn(f"  [SKIP] out of workspace: score={t['score']:.2f} base_xyz={base_xyz}")
        self.get_logger().warn("All targets out of workspace")
        return None

    def cam_to_base(self, cam_xyz, stamp):
        pt = geometry_msgs.msg.PointStamped()
        pt.header.frame_id = SOURCE_FRAME
        pt.header.stamp = stamp
        pt.point.x, pt.point.y, pt.point.z = cam_xyz
        try:
            tf = self.tf_buffer.lookup_transform(
                TARGET_FRAME, SOURCE_FRAME, rclpy.time.Time(),
                timeout=Duration(seconds=2.0))
        except TransformException:
            return None
        p = do_transform_point(pt, tf)
        return (p.point.x - 0.030, -p.point.y + 0.020, p.point.z)

    def go_above(self, base_xyz):
        x, y, z = base_xyz
        z_above = z + APPROACH_HEIGHT
        self.get_logger().info(f"Above target: ({x:+.3f}, {y:+.3f}, {z_above:+.3f})")
        seed = np.array([SAFE_HOME_JOINTS["shoulder_lift"],
                         SAFE_HOME_JOINTS["elbow_flex"],
                         SAFE_HOME_JOINTS["wrist_flex"],
                         SAFE_HOME_JOINTS["wrist_roll"]])
        joint_dict, ok = base_xyz_to_arm_joints(x, y, z_above, seed)
        if not ok:
            return False
        self.get_logger().info(f"IK: {joint_dict}")
        return self.send_joint_goal_moveit(joint_dict, "above", 0.05)

    def report_current_joints(self):
        """打印 follower 当前关节角，方便人眼摆 leader"""
        # 多吃几次 spin 确保 joint_state 是最新的
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.joint_state_msg is None:
            self.get_logger().warn("No joint_state received.")
            return
        m = dict(zip(self.joint_state_msg.name, self.joint_state_msg.position))
        print("\n" + "="*60)
        print("FOLLOWER 当前关节角 (摆 leader 时参照):")
        print("="*60)
        for n in JOINT_NAMES:
            if n in m:
                rad = m[n]
                deg = math.degrees(rad)
                print(f"  {n:15s}: {rad:+.4f} rad  ({deg:+7.2f} deg)")
        if "gripper" in m:
            rad = m["gripper"]
            print(f"  {'gripper':15s}: {rad:+.4f} rad  ({math.degrees(rad):+7.2f} deg)")
        print("="*60)

    def run(self):
        self.get_logger().info("=== Step 0: safe_home ===")
        if not self.go_safe_home():
            return False
        self.get_logger().info("=== Step 1: detect ===")
        self.top_color_msg = self.top_depth_msg = None
        self.warmup(2.0)
        target = self.detect_target()
        if not target:
            self.get_logger().error("No target detected.")
            return False
        base_xyz = self.cam_to_base(target["cam_xyz_m"], self.top_color_msg.header.stamp)
        if not base_xyz:
            self.get_logger().error("TF transform failed.")
            return False
        self.get_logger().info(f"Target base_link: ({base_xyz[0]:+.3f}, {base_xyz[1]:+.3f}, {base_xyz[2]:+.3f})")
        self.get_logger().info("=== Step 1.5: open gripper ===")
        self.set_gripper(GRIPPER_OPEN, "open")
        self.get_logger().info("=== Step 2: go above ===")
        if not self.go_above(base_xyz):
            self.get_logger().error("go_above failed.")
            return False
        # 等机械臂到位 + servo 锁定
        self.warmup(1.5)
        self.report_current_joints()
        self.get_logger().info("=== DONE (follower 停在 above；ACT 阶段跳过) ===")
        return True


def main():
    rclpy.init()
    node = PipelineToAboveOnly()
    try:
        ok = node.run()
        print("\n=== PIPELINE", "OK" if ok else "FAILED", "===")
        if ok:
            print("\n下一步操作:")
            print("  1. 本脚本已退出。ROS launch 仍在运行，follower servo 锁定在 above 位置。")
            print("  2. 人眼把 leader 摆到上面打印的关节角 (粗略对照即可)。")
            print("  3. 杀 ROS launch (在 ROS 终端 Ctrl+C，或新开终端跑:")
            print("     pkill -9 -f ros2; pkill -9 -f move_group; pkill -9 -f controller_manager; sleep 3)")
            print("  4. 启动录制: bash ~/techin517/record_one_demo.sh battery")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
