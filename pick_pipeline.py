#!/usr/bin/env python3
"""
pick_pipeline.py v0.7 - CV + IK + ACT 完整混合架构
"""

import sys
import os
import time
import math
from types import SimpleNamespace

sys.path.insert(0, os.path.expanduser("~/techin517/cv_module"))

import numpy as np
import torch
import cv2
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
from control_msgs.action import GripperCommand, FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as MsgDuration

from cv_module import load_detector, detect_objects
from lerobot_kinematics import lerobot_IK, get_robot

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors, get_policy_class
from lerobot.utils.control_utils import predict_action


TOPIC_COLOR   = "/static_camera/overhead_cam/color/image_raw"
TOPIC_DEPTH   = "/static_camera/overhead_cam/aligned_depth_to_color/image_raw"
TOPIC_CAMINFO = "/static_camera/overhead_cam/color/camera_info"
TARGET_FRAME = "base_link"
SOURCE_FRAME = "overhead_camoverhead_cam_color_optical_frame"
WRIST_DEV = "/dev/v4l/by-id/usb-BC-231220-A_XWF-1080P-video-index0"

TEXT_PROMPT    = "a battery."
TARGET_CLASSES = {"battery"}
BOX_THRESHOLD  = 0.30
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

URDF_TO_LEROBOT_DX = 0.030
URDF_TO_LEROBOT_DZ = 0.056
LEROBOT_ROLL  = 0.147
LEROBOT_PITCH = 1.30

ARM_GROUP       = "arm"
GRIPPER_OPEN    = 1.7
GRIPPER_CLOSED  = 0.7
APPROACH_HEIGHT = 0.165
PLANNING_TIME   = 5.0

ACT_CHECKPOINT = "/home/ubuntu/techin517/outputs/train/act_battery_v2/checkpoints/last/pretrained_model"
ACT_DURATION_S = 15.0
ACT_FPS        = 30
ACT_TASK_NAME  = "Pick up the battery"


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


class PickPipeline(Node):
    def __init__(self):
        super().__init__("pick_pipeline_v07")
        self.bridge = CvBridge()
        self.top_color_msg = None
        self.top_depth_msg = None
        self.top_caminfo = None
        self.joint_state_msg = None

        self.wrist_cap = cv2.VideoCapture(WRIST_DEV, cv2.CAP_V4L2)
        self.wrist_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.wrist_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.wrist_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.wrist_cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.wrist_cap.isOpened():
            raise RuntimeError(f"wrist cam open failed: {WRIST_DEV}")
        self.get_logger().info(f"Wrist cam opened: {WRIST_DEV}")

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
        self.traj_client = ActionClient(self, FollowJointTrajectory,
                                        "/arm_controller/follow_joint_trajectory")
        self.traj_client.wait_for_server(timeout_sec=10.0)
        self.get_logger().info("/arm_controller/follow_joint_trajectory ready.")

        self.get_logger().info("Loading Grounding DINO ...")
        self.processor_cv, self.model_cv = load_detector(device=DEVICE)
        self.get_logger().info("Detector ready.")

        self.get_logger().info("Loading ACT policy ...")
        self.act_policy_cfg = PreTrainedConfig.from_pretrained(ACT_CHECKPOINT)
        policy_class = get_policy_class(self.act_policy_cfg.type)
        self.act_policy = policy_class.from_pretrained(ACT_CHECKPOINT)
        self.act_policy.to(self.act_policy_cfg.device)
        self.act_policy.eval()
        self.act_preprocessor, self.act_postprocessor = make_pre_post_processors(
            policy_cfg=self.act_policy_cfg,
            pretrained_path=ACT_CHECKPOINT,
        )
        self.act_device = torch.device(self.act_policy_cfg.device)
        self.get_logger().info("ACT loaded.")

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

    def send_trajectory(self, joints_rad, time_s=0.1):
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [float(j) for j in joints_rad]
        pt.time_from_start = MsgDuration(
            sec=int(time_s),
            nanosec=int((time_s - int(time_s)) * 1e9))
        traj.points.append(pt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        self.traj_client.send_goal_async(goal)

    def set_gripper(self, position, label="gripper", wait=True):
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = 800.0
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
        t = targets[0]
        self.get_logger().info(f"Target: score={t['score']:.2f} cam_xyz={t['cam_xyz_m']}")
        return t

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

    def get_state_for_act(self):
        if self.joint_state_msg is None:
            return None
        m = dict(zip(self.joint_state_msg.name, self.joint_state_msg.position))
        try:
            arm_rad = [m[n] for n in JOINT_NAMES]
            gripper_rad = m["gripper"]
        except KeyError:
            return None
        return np.array([math.degrees(r) for r in arm_rad] + [math.degrees(gripper_rad)],
                        dtype=np.float32)

    def run_act(self):
        self.get_logger().info(f"=== Step 3: ACT ({ACT_DURATION_S}s) ===")
        self.act_policy.reset()
        self.act_preprocessor.reset()
        self.act_postprocessor.reset()
        period = 1.0 / ACT_FPS
        t_end = time.time() + ACT_DURATION_S
        n = 0
        while time.time() < t_end:
            t_step = time.time()
            rclpy.spin_once(self, timeout_sec=0.001)
            state = self.get_state_for_act()
            if state is None or self.top_color_msg is None:
                time.sleep(0.01)
                continue
            top_img = self.bridge.imgmsg_to_cv2(self.top_color_msg, desired_encoding="rgb8")
            ret, wrist_bgr = self.wrist_cap.read()
            if not ret:
                continue
            wrist_img = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)
            obs = {
                "observation.images.wrist": wrist_img,
                "observation.images.top": top_img,
                "observation.state": state,
            }
            action_t = predict_action(
                observation=obs, policy=self.act_policy,
                device=self.act_device,
                preprocessor=self.act_preprocessor,
                postprocessor=self.act_postprocessor,
                use_amp=False, task=ACT_TASK_NAME)
            action = action_t.cpu().numpy().flatten()
            arm_rad = [math.radians(action[i]) for i in range(5)]
            gripper_rad = math.radians(action[5])
            self.send_trajectory(arm_rad, time_s=period * 2)
            self.set_gripper(gripper_rad, wait=False)
            n += 1
            el = time.time() - t_step
            if el < period:
                time.sleep(period - el)
        self.get_logger().info(f"ACT done. {n} steps.")
        return True

    def run(self):
        self.get_logger().info("=== Step 0: safe_home ===")
        if not self.go_safe_home():
            return False
        self.get_logger().info("=== Step 1: detect ===")
        self.top_color_msg = self.top_depth_msg = None
        self.warmup(2.0)
        target = self.detect_target()
        if not target:
            return False
        base_xyz = self.cam_to_base(target["cam_xyz_m"], self.top_color_msg.header.stamp)
        if not base_xyz:
            return False
        self.get_logger().info(f"Target base_link: ({base_xyz[0]:+.3f}, {base_xyz[1]:+.3f}, {base_xyz[2]:+.3f})")
        self.get_logger().info("=== Step 1.5: open gripper ===")
        self.set_gripper(GRIPPER_OPEN, "open")
        self.get_logger().info("=== Step 2: go above ===")
        if not self.go_above(base_xyz):
            return False
        self.warmup(2.0)
        self.run_act()
        self.get_logger().info("=== DONE ===")
        return True


def main():
    rclpy.init()
    node = PickPipeline()
    try:
        ok = node.run()
        print("\n=== PIPELINE", "OK" if ok else "FAILED", "===")
    finally:
        if hasattr(node, "wrist_cap"):
            node.wrist_cap.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
