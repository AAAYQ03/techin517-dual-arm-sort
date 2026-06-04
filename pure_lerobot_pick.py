#!/usr/bin/env python3
"""pure_lerobot_pick.py v2 - 加 send_to_box 阶段"""

import sys, os, time, math
from types import SimpleNamespace

sys.path.insert(0, os.path.expanduser("~/techin517/cv_module"))

import numpy as np
import cv2
import torch
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

from lerobot.robots.so101_follower.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors, get_policy_class
from lerobot.utils.control_utils import predict_action

from cv_module import load_detector, detect_objects
from lerobot_kinematics import lerobot_IK, get_robot

# ============ 配置 ============
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0181426-if00"
FOLLOWER_ID   = "follower8"
WRIST_DEV     = "/dev/v4l/by-id/usb-BC-231220-A_XWF-1080P-video-index0"
TOP_SERIAL    = "243222072732"
CAM_W, CAM_H, CAM_FPS = 1280, 720, 30

TEXT_PROMPT    = "a battery."
TARGET_CLASSES = {"battery"}
BOX_THRESHOLD  = 0.30
DEPTH_SCALE    = 0.001
DEVICE         = "cuda"

URDF_TO_LEROBOT_DX = 0.030
URDF_TO_LEROBOT_DZ = 0.056
LEROBOT_ROLL  = 0.147
LEROBOT_PITCH = 1.30
APPROACH_HEIGHT = 0.165

SAFE_HOME_RAD = {
    "shoulder_pan":  -0.020,
    "shoulder_lift": -1.313,
    "elbow_flex":    +1.433,
    "wrist_flex":    +1.014,
    "wrist_roll":    +0.147,
}
GRIPPER_OPEN   = 1.7
GRIPPER_CLOSED = 0.7

# === 盒子位置 (base_link 系, 单位米) ===
BOX_XYZ = (0.20, -0.10, 0.10)   # 左盒中心 (X前, Y左, Z高)

CAM_T = np.array([0.0198, -0.1215, 0.3372])
CAM_Q = np.array([0.0419, 0.3597, -0.0051, 0.9321])
OPT_T = np.array([0.0, 0.015, 0.0])
OPT_Q = np.array([-0.497, 0.504, -0.497, 0.502])

ACT_CHECKPOINT = "/home/ubuntu/techin517/outputs/train/act_battery_v2/checkpoints/last/pretrained_model"
ACT_DURATION_S = 15.0
ACT_FPS        = 30
ACT_TASK_NAME  = "Pick up the battery"


def make_T(t, q):
    T = np.eye(4); T[:3,:3] = R.from_quat(q).as_matrix(); T[:3,3] = t
    return T

T_base_camlink    = make_T(CAM_T, CAM_Q)
T_camlink_optical = make_T(OPT_T, OPT_Q)
T_base_optical    = T_base_camlink @ T_camlink_optical


def cam_xyz_to_base(cam_xyz):
    p = np.array([cam_xyz[0], cam_xyz[1], cam_xyz[2], 1.0])
    pb = T_base_optical @ p
    return (float(pb[0] - 0.030), float(-pb[1] + 0.020), float(pb[2]))


def base_xyz_to_arm_joints(x, y, z, seed_qpos):
    shoulder_pan = math.atan2(y, x)
    horizontal = math.sqrt(x*x + y*y)
    target_gpos = np.array([horizontal - URDF_TO_LEROBOT_DX, 0.0, z - URDF_TO_LEROBOT_DZ,
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


def ramp_to(follower, target_deg, duration_s=3.0, hz=30, label=""):
    obs = follower.get_observation()
    start_deg = {k: v for k, v in obs.items() if k.endswith('.pos')}
    common = [k for k in target_deg if k in start_deg]
    n_steps = int(duration_s * hz)
    dt = 1.0 / hz
    print(f"  [ramp:{label}] {n_steps} steps over {duration_s}s")
    for i in range(1, n_steps + 1):
        t = i / n_steps
        interp = {k: start_deg[k] * (1 - t) + target_deg[k] * t for k in common}
        follower.send_action(interp)
        time.sleep(dt)
    time.sleep(0.3)


class RealSenseTop:
    def __init__(self, serial, w, h, fps):
        self.serial = serial; self.w = w; self.h = h; self.fps = fps
        self.pipeline = None; self.align = None; self.intrin = None
    def start(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.w, self.h, rs.format.z16, self.fps)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrin = SimpleNamespace(fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy)
        for _ in range(15):
            self.pipeline.wait_for_frames()
        print(f"  [RS] ready: fx={intr.fx:.1f} fy={intr.fy:.1f}")
    def read(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        bgr = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        return bgr, depth, self.intrin
    def stop(self):
        if self.pipeline:
            self.pipeline.stop()


class WristCam:
    def __init__(self, path, w, h, fps):
        self.cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"wrist cam open failed: {path}")
        print(f"  [Wrist] opened: {path}")
    def read_rgb(self):
        ok, bgr = self.cap.read()
        if not ok:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    def stop(self):
        self.cap.release()


def main():
    print("=" * 60)
    print("  pure_lerobot_pick.py v2 - 含 send_to_box")
    print(f"  BOX_XYZ = {BOX_XYZ}")
    print("=" * 60)

    print("\n[1/4] Connecting follower...")
    cfg = SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID,
                              disable_torque_on_disconnect=False)
    follower = SO101Follower(cfg)
    follower.connect()
    print(f"  ✓ Follower connected: {FOLLOWER_ID}")

    print("\n[2/4] Starting cameras...")
    top = RealSenseTop(TOP_SERIAL, CAM_W, CAM_H, CAM_FPS); top.start()
    wrist = WristCam(WRIST_DEV, CAM_W, CAM_H, CAM_FPS)
    for _ in range(5): wrist.read_rgb()

    print("\n[3/4] Loading models...")
    processor_cv, model_cv = load_detector(device=DEVICE)
    print("  ✓ CV ready")
    _ = get_robot('so101')
    print("  ✓ IK ready")
    print("  Loading ACT...")
    policy_cfg = PreTrainedConfig.from_pretrained(ACT_CHECKPOINT)
    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(ACT_CHECKPOINT)
    policy.to(policy_cfg.device); policy.eval()
    pre_proc, post_proc = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=ACT_CHECKPOINT)
    act_device = torch.device(policy_cfg.device)
    print("  ✓ ACT ready")

    try:
        # === Step 0: safe_home ===
        print("\n=== Step 0: safe_home ===")
        target = {f"{k}.pos": math.degrees(v) for k, v in SAFE_HOME_RAD.items()}
        target["gripper.pos"] = math.degrees(GRIPPER_OPEN)
        ramp_to(follower, target, duration_s=3.0, label="safe_home")

        # === Step 1: CV detect ===
        print("\n=== Step 1: CV detect ===")
        bgr, depth_raw, intrin = top.read()
        dets = detect_objects(
            color_image=bgr, depth_image=depth_raw, intrin=intrin,
            depth_scale=DEPTH_SCALE, processor=processor_cv,
            model=model_cv, text_prompt=TEXT_PROMPT,
            target_classes=TARGET_CLASSES, box_threshold=BOX_THRESHOLD,
            device=DEVICE)
        targets = [d for d in dets if d.get("is_target") and d.get("depth_valid")]
        if not targets:
            print("  ✗ No detection"); return
        targets.sort(key=lambda d: d["score"], reverse=True)
        det = targets[0]
        base_xyz = cam_xyz_to_base(det['cam_xyz_m'])
        print(f"  ✓ score={det['score']:.2f}, base_link={base_xyz}")

        # === Step 2: IK to above ===
        print("\n=== Step 2: IK to above ===")
        z_above = base_xyz[2] + APPROACH_HEIGHT
        seed = np.array([SAFE_HOME_RAD["shoulder_lift"], SAFE_HOME_RAD["elbow_flex"],
                         SAFE_HOME_RAD["wrist_flex"], SAFE_HOME_RAD["wrist_roll"]])
        joints_rad, ok = base_xyz_to_arm_joints(base_xyz[0], base_xyz[1], z_above, seed)
        if not ok:
            print("  ✗ IK failed"); return
        target_above = {f"{k}.pos": math.degrees(v) for k, v in joints_rad.items()}
        target_above["gripper.pos"] = math.degrees(GRIPPER_OPEN)
        ramp_to(follower, target_above, duration_s=2.5, label="above")

        # === Step 3: ACT ===
        print(f"\n=== Step 3: ACT ({ACT_DURATION_S}s) ===")
        policy.reset(); pre_proc.reset(); post_proc.reset()
        period = 1.0 / ACT_FPS
        t_start = time.time(); n = 0
        while time.time() - t_start < ACT_DURATION_S:
            t_loop = time.time()
            obs = follower.get_observation()
            state = np.array([
                obs.get("shoulder_pan.pos", 0.0), obs.get("shoulder_lift.pos", 0.0),
                obs.get("elbow_flex.pos", 0.0), obs.get("wrist_flex.pos", 0.0),
                obs.get("wrist_roll.pos", 0.0), obs.get("gripper.pos", 0.0),
            ], dtype=np.float32)
            wrist_rgb = wrist.read_rgb()
            if wrist_rgb is None: continue
            top_bgr, _, _ = top.read()
            top_rgb = cv2.cvtColor(top_bgr, cv2.COLOR_BGR2RGB)
            act_obs = {
                "observation.images.wrist": wrist_rgb,
                "observation.images.top":   top_rgb,
                "observation.state":        state,
            }
            action_t = predict_action(
                observation=act_obs, policy=policy, device=act_device,
                preprocessor=pre_proc, postprocessor=post_proc,
                use_amp=False, task=ACT_TASK_NAME)
            action = action_t.cpu().numpy().flatten()
            follower.send_action({
                "shoulder_pan.pos":  float(action[0]),
                "shoulder_lift.pos": float(action[1]),
                "elbow_flex.pos":    float(action[2]),
                "wrist_flex.pos":    float(action[3]),
                "wrist_roll.pos":    float(action[4]),
                "gripper.pos":       float(action[5]),
            })
            n += 1
            el = time.time() - t_loop
            if el < period: time.sleep(period - el)
        print(f"  ✓ ACT done. {n} steps")

        # === Step 4: Send to box ===
        print(f"\n=== Step 4: Send to box {BOX_XYZ} ===")
        obs = follower.get_observation()
        # 保持当前 gripper 值 (ACT 学到的闭合状态), 不要变!
        current_gripper = obs.get("gripper.pos", math.degrees(GRIPPER_CLOSED))
        print(f"  Current gripper = {current_gripper:.2f} deg (keep closed)")
        seed = np.array([
            math.radians(obs.get("shoulder_lift.pos", 0)),
            math.radians(obs.get("elbow_flex.pos", 0)),
            math.radians(obs.get("wrist_flex.pos", 0)),
            math.radians(obs.get("wrist_roll.pos", 0)),
        ])
        box_joints, ok = base_xyz_to_arm_joints(BOX_XYZ[0], BOX_XYZ[1], BOX_XYZ[2], seed)
        if not ok:
            print("  ✗ Box IK failed (不应该发生, 之前扫描可达)")
        else:
            target_box = {f"{k}.pos": math.degrees(v) for k, v in box_joints.items()}
            target_box["gripper.pos"] = current_gripper  # 保持当前 (闭合)
            ramp_to(follower, target_box, duration_s=3.0, label="to_box")
            time.sleep(0.5)
            
            # 张开 gripper 释放
            print("  Releasing...")
            target_box["gripper.pos"] = math.degrees(GRIPPER_OPEN)
            follower.send_action(target_box)
            time.sleep(1.5)
            print("  ✓ Released into box")
        
        # === Step 5: 回 safe_home ===
        print("\n=== Step 5: Back to safe_home ===")
        target = {f"{k}.pos": math.degrees(v) for k, v in SAFE_HOME_RAD.items()}
        target["gripper.pos"] = math.degrees(GRIPPER_OPEN)
        ramp_to(follower, target, duration_s=3.0, label="safe_home")
        print("\n=== ALL DONE ===")

    finally:
        print("\n[4/4] Cleanup...")
        try: top.stop()
        except: pass
        try: wrist.stop()
        except: pass
        try: follower.disconnect()
        except: pass


if __name__ == "__main__":
    main()
