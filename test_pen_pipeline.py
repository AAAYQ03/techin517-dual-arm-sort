#!/usr/bin/env python3
"""
test_pen_pipeline.py - 单物体 pen pipeline 测试 (follower7)
基于 dispatch_pick.py v3 简化, 用于验证 follower7 + act_pen_f7 端到端

不依赖 ROS launch (LeRobot 直连串口)

流程:
  1. CV (DINO) 检测 pen → cam_xyz
  2. cam_xyz → base_link 坐标 (用 follower7 hand-eye 标定)
  3. IK → follower7 移动到 pen 上方 ~5cm
  4. ACT (act_pen_f7) 接管 → 自动下降、闭合、抬起
  5. 回 safe_home

用法:
  # 不要启动 ROS launch! 直接跑:
  python3 ~/techin517/test_pen_pipeline.py
"""
import sys, os, time, math
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

# ============ follower7 配置 ============
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6057204-if00"
FOLLOWER_ID   = "gix-follower7"
WRIST_DEV     = "/dev/v4l/by-path/pci-0000:0e:00.0-usb-0:1.1:1.0-video-index0"
TOP_SERIAL    = "243222072732"

CAM_W, CAM_H, CAM_FPS = 1280, 720, 30
BOX_THRESHOLD  = 0.20
DEPTH_SCALE    = 0.001
DEVICE         = "cuda"

# ============ IK 参数 (与 dispatch_pick 一致) ============
URDF_TO_LEROBOT_DX = 0.030
URDF_TO_LEROBOT_DZ = 0.056
LEROBOT_ROLL  = 0.147
LEROBOT_PITCH = 1.30
APPROACH_HEIGHT = 0.165   # follower8 调好的值, follower7 估计差不多

# ============ Safe home (与 dispatch_pick 一致) ============
SAFE_HOME_RAD = {
    "shoulder_pan":  -0.020,
    "shoulder_lift": -1.313,
    "elbow_flex":    +1.433,
    "wrist_flex":    +1.014,
    "wrist_roll":    +0.147,
}
GRIPPER_OPEN   = 1.7
GRIPPER_CLOSED = 0.7

# ============ follower7 hand-eye calibration ============
CAM_T = np.array([0.0305, 0.1557, 0.3337])
CAM_Q = np.array([-0.0179, 0.3854, 0.0403, 0.9217])
OPT_T = np.array([0.0, 0.015, 0.0])
OPT_Q = np.array([-0.497, 0.504, -0.497, 0.502])

# ============ Pen ACT model ============
ACT_CHECKPOINT = "/home/ubuntu/techin517/outputs/train/act_pen_f7/checkpoints/last/pretrained_model"
ACT_DURATION_S = 15.0
ACT_FPS        = 30
DINO_PROMPT    = "a black pen."
DINO_TARGETS   = {"black pen", "pen", "marker"}
TASK_NAME      = "Pick up the pen"

# ============ 工作空间过滤 (follower7 在右, 允许 y >= 0) ============
def is_valid_position(base_xyz):
    x, y, z = base_xyz
    return (0.18 < x < 0.32) and (abs(y) < 0.12) and (-0.05 < z < 0.06)


# ============ 矩阵换算 ============
def make_T(t, q):
    T = np.eye(4); T[:3,:3] = R.from_quat(q).as_matrix(); T[:3,3] = t
    return T

T_base_camlink    = make_T(CAM_T, CAM_Q)
T_camlink_optical = make_T(OPT_T, OPT_Q)
T_base_optical    = T_base_camlink @ T_camlink_optical

def cam_xyz_to_base(cam_xyz):
    """relesense 光学坐标 → follower/base_link 坐标。补偿暂设 0, 必要再调"""
    p = np.array([cam_xyz[0], cam_xyz[1], cam_xyz[2], 1.0])
    pb = T_base_optical @ p
    # follower7 系统性补偿: 先设 0, 看物理表现再调
    return (float(pb[0] - 0.0), float(pb[1] + 0.0), float(pb[2]))


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
    for i in range(1, n_steps + 1):
        t = i / n_steps
        interp = {k: start_deg[k] * (1 - t) + target_deg[k] * t for k in common}
        follower.send_action(interp)
        time.sleep(dt)
    print(f"  ✓ ramp_to {label} done")


# ============ Camera classes ============
class RealSenseTop:
    def __init__(self, serial, w, h, fps):
        self.serial = serial; self.w = w; self.h = h; self.fps = fps
        self.pipeline = None
    def start(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.w, self.h, rs.format.z16, self.fps)
        profile = self.pipeline.start(cfg)
        intrin_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrin = intrin_profile.get_intrinsics()
        self.align = rs.align(rs.stream.color)
        for _ in range(15):
            self.pipeline.wait_for_frames()
        print(f"  ✓ D435i top started")
    def read(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        return color, depth, self.intrin
    def stop(self):
        if self.pipeline: self.pipeline.stop()


class WristCam:
    def __init__(self, dev, w, h, fps):
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
    def read_rgb(self):
        ok, bgr = self.cap.read()
        if not ok: return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    def stop(self):
        self.cap.release()


# ============ ACT inference ============
def load_act_policy(ckpt):
    print(f"  Loading ACT: {ckpt.split('/')[-3]}")
    policy_cfg = PreTrainedConfig.from_pretrained(ckpt)
    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(ckpt)
    policy.to(policy_cfg.device); policy.eval()
    pre_proc, post_proc = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=ckpt)
    return {
        "policy": policy, "pre_proc": pre_proc, "post_proc": post_proc,
        "device": torch.device(policy_cfg.device),
    }


def run_act(follower, top, wrist, bundle, task_name, duration_s):
    policy = bundle["policy"]; pre = bundle["pre_proc"]; post = bundle["post_proc"]
    act_device = bundle["device"]
    policy.reset(); pre.reset(); post.reset()
    period = 1.0 / ACT_FPS
    t_start = time.time(); n = 0
    while time.time() - t_start < duration_s:
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
            preprocessor=pre, postprocessor=post,
            use_amp=False, task=task_name)
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


def go_to_safe_home(follower):
    target = {f"{k}.pos": math.degrees(v) for k, v in SAFE_HOME_RAD.items()}
    target["gripper.pos"] = math.degrees(GRIPPER_OPEN)
    ramp_to(follower, target, duration_s=3.0, label="safe_home")


# ============ Main ============
def main():
    print("=" * 60)
    print("  test_pen_pipeline.py - follower7 + act_pen_f7")
    print("=" * 60)

    print("\n[1/4] Connecting follower7...")
    cfg = SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID,
                              disable_torque_on_disconnect=False)
    follower = SO101Follower(cfg); follower.connect()
    print(f"  ✓ Follower connected: {FOLLOWER_ID}")

    print("\n[2/4] Starting cameras...")
    top = RealSenseTop(TOP_SERIAL, CAM_W, CAM_H, CAM_FPS); top.start()
    wrist = WristCam(WRIST_DEV, CAM_W, CAM_H, CAM_FPS)
    for _ in range(5): wrist.read_rgb()
    print(f"  ✓ Wrist cam started")

    print("\n[3/4] Loading models...")
    processor_cv, model_cv = load_detector(device=DEVICE)
    print("  ✓ CV ready")
    _ = get_robot('so101')
    print("  ✓ IK ready")
    bundle = load_act_policy(ACT_CHECKPOINT)
    print("  ✓ ACT loaded")

    try:
        print("\n=== Step 1: safe_home ===")
        go_to_safe_home(follower)

        print("\n=== Step 2: CV detect pen ===")
        bgr, depth_raw, intrin = top.read()
        dets = detect_objects(
            color_image=bgr, depth_image=depth_raw, intrin=intrin,
            depth_scale=DEPTH_SCALE, processor=processor_cv, model=model_cv,
            text_prompt=DINO_PROMPT, target_classes=DINO_TARGETS,
            box_threshold=BOX_THRESHOLD, device=DEVICE)
        valid = [d for d in dets if d.get("is_target") and d.get("depth_valid")]
        print(f"  DINO 找到 {len(valid)} 个 candidate")

        target = None
        for d in valid:
            base_xyz = cam_xyz_to_base(d['cam_xyz_m'])
            print(f"  cand: cam_xyz={d['cam_xyz_m']} base_xyz={base_xyz} score={d['score']:.2f}")
            if is_valid_position(base_xyz):
                target = d
                target['base_xyz'] = base_xyz
                break
            else:
                print(f"    [SKIP] out of workspace")
        if target is None:
            print("  ✗ No valid pen detection")
            return

        print(f"\n  ✓ Target pen: base_xyz={target['base_xyz']}")

        print("\n=== Step 3: IK → above ===")
        bx, by, bz = target['base_xyz']
        z_above = bz + APPROACH_HEIGHT
        seed = np.array([SAFE_HOME_RAD["shoulder_lift"], SAFE_HOME_RAD["elbow_flex"],
                         SAFE_HOME_RAD["wrist_flex"], SAFE_HOME_RAD["wrist_roll"]])
        joints_rad, ok = base_xyz_to_arm_joints(bx, by, z_above, seed)
        if not ok:
            print("  ✗ IK fail")
            return
        target_above = {f"{k}.pos": math.degrees(v) for k, v in joints_rad.items()}
        target_above["gripper.pos"] = math.degrees(GRIPPER_OPEN)
        print(f"  IK above: {joints_rad}")
        ramp_to(follower, target_above, duration_s=2.5, label="above")

        print(f"\n=== Step 4: ACT ({ACT_DURATION_S}s) ===")
        run_act(follower, top, wrist, bundle, TASK_NAME, ACT_DURATION_S)

        print("\n=== Step 5: back to safe_home ===")
        go_to_safe_home(follower)

        print("\n" + "=" * 60)
        print("  === TEST DONE ===")
        print("=" * 60)

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
