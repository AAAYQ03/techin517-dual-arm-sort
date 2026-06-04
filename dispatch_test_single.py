#!/usr/bin/env python3
"""
单物品 dispatch 测试 v2 (基于 dispatch_pick.py 5/26 跑通版改最小化):
  python3 dispatch_test_single.py pen
  python3 dispatch_test_single.py earbuds
  python3 dispatch_test_single.py battery
  python3 dispatch_test_single.py usb
  python3 dispatch_test_single.py pen --skip-box      # 不送盒子, 抓完结束
"""
import sys, os, time, math, argparse
from types import SimpleNamespace

sys.path.insert(0, os.path.expanduser("~/techin517/cv_module"))
sys.path.insert(0, os.path.expanduser("~/techin517"))

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


# ============ 配置 (从 dispatch_pick.py 直接拷贝) ============
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0181426-if00"
FOLLOWER_ID   = "follower8"
WRIST_DEV     = "/dev/v4l/by-path/pci-0000:0b:00.0-usb-0:3:1.0-video-index0"
TOP_SERIAL    = "243222072732"
CAM_W, CAM_H, CAM_FPS = 1280, 720, 30

BOX_THRESHOLD  = 0.20
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

# === 单物品配置 ===
SINGLE_ITEMS = {
    "pen": {
        "name": "pen",
        "dino_prompt": "a black pen.",
        "dino_targets": {"pen", "marker"},
        "act_checkpoint": "/home/ubuntu/techin517/outputs/train/act_pen_f8/checkpoints/last/pretrained_model",
        "task_name": "Pick up the pen",
        "box_xyz": (0.210, +0.160, 0.12),  # f8 跨侧右盒
    },
    "earbuds": {
        "name": "earbuds",
        "dino_prompt": "a rounded white case.",
        "dino_targets": {"case", "rounded white case"},
        "act_checkpoint": "/home/ubuntu/techin517/outputs/train/act_earbuds_f8/checkpoints/last/pretrained_model",
        "task_name": "Pick up the earbuds",
        "box_xyz": (0.155, +0.058, 0.10),  # f8 自己侧左盒
    },
    "battery": {
        "name": "battery",
        "dino_prompt": "a battery.",
        "dino_targets": {"battery"},
        "act_checkpoint": "/home/ubuntu/techin517/outputs/train/act_battery_v2/checkpoints/last/pretrained_model",
        "task_name": "Pick up the battery",
        "box_xyz": (0.155, +0.058, 0.10),
    },
    "usb": {
        "name": "usb stick",
        "dino_prompt": "a usb stick.",
        "dino_targets": {"usb stick", "usb"},
        "act_checkpoint": "/home/ubuntu/techin517/outputs/train/act_usb_v2/checkpoints/last/pretrained_model",
        "task_name": "Pick up the usb",
        "box_xyz": (0.155, +0.058, 0.10),
    },
}

ACT_DURATION_S = 15.0
ACT_FPS        = 30

CAM_T = np.array([0.0198, -0.1215, 0.3372])
CAM_Q = np.array([0.0419, 0.3597, -0.0051, 0.9321])
OPT_T = np.array([0.0, 0.015, 0.0])
OPT_Q = np.array([-0.497, 0.504, -0.497, 0.502])


def is_valid_position(base_xyz):
    x, y, z = base_xyz
    return (0.18 < x < 0.30) and (abs(y) < 0.12) and (-0.02 < z < 0.06)


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
    target_x_lerobot = horizontal - URDF_TO_LEROBOT_DX
    target_z_lerobot = z - URDF_TO_LEROBOT_DZ
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


def load_act_policy(ckpt):
    name = ckpt.split('/')[-3]
    print(f"  Loading ACT: {name}")
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


def run_act(follower, top, wrist, act_bundle, task_name, duration_s):
    policy = act_bundle["policy"]
    pre_proc = act_bundle["pre_proc"]
    post_proc = act_bundle["post_proc"]
    act_device = act_bundle["device"]
    policy.reset(); pre_proc.reset(); post_proc.reset()
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
            preprocessor=pre_proc, postprocessor=post_proc,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("item", choices=list(SINGLE_ITEMS.keys()))
    ap.add_argument("--skip-box", action="store_true",
                    help="不送盒子, 抓完结束")
    args = ap.parse_args()

    cfg = SINGLE_ITEMS[args.item]
    print(f"\n{'='*60}")
    print(f"  单物品 dispatch 测试: {cfg['name']}")
    print(f"  DINO prompt: {cfg['dino_prompt']}")
    print(f"  ACT: {cfg['act_checkpoint']}")
    if not args.skip_box:
        print(f"  Box: {cfg['box_xyz']}")
    print(f"{'='*60}\n")

    follower = None
    top = None
    wrist = None
    try:
        print("[1/4] Connecting follower...")
        # 关键: disable_torque_on_disconnect=False, 避免出错时 follower 掉下去
        f_cfg = SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID,
                                     disable_torque_on_disconnect=False)
        follower = SO101Follower(f_cfg)
        follower.connect()
        print(f"  ✓ Follower connected: {FOLLOWER_ID}")

        print("\n[2/4] Starting cameras...")
        top = RealSenseTop(TOP_SERIAL, CAM_W, CAM_H, CAM_FPS); top.start()
        wrist = WristCam(WRIST_DEV, CAM_W, CAM_H, CAM_FPS)

        print("\n[3/4] Loading DINO + ACT...")
        processor_cv, model_cv = load_detector(device=DEVICE,
                                                model_name="IDEA-Research/grounding-dino-base")
        act_bundle = load_act_policy(cfg["act_checkpoint"])

        print("\n[4/4] === Starting pipeline ===\n")

        # Step 0: safe home
        print("=== Step 0: safe_home ===")
        go_to_safe_home(follower)
        time.sleep(1.0)

        # Step 1: DINO 检测
        print(f"\n=== Step 1: DINO detect ({cfg['dino_prompt']}) ===")
        bgr, depth_raw, intrin = top.read()
        dets = detect_objects(
            color_image=bgr, depth_image=depth_raw, intrin=intrin,
            depth_scale=DEPTH_SCALE, processor=processor_cv, model=model_cv,
            text_prompt=cfg["dino_prompt"], target_classes=cfg["dino_targets"],
            box_threshold=BOX_THRESHOLD, device=DEVICE)
        valid = [d for d in dets if d.get("depth_valid")]
        print(f"  DINO 找到 {len(valid)} 个 candidate")
        if not valid:
            print("  ✗ No detection, abort")
            return
        valid.sort(key=lambda d: d["score"], reverse=True)
        target = valid[0]
        base_xyz = cam_xyz_to_base(target['cam_xyz_m'])
        print(f"  Target: score={target['score']:.2f} bbox={target['bbox_xyxy']}")
        print(f"  base_xyz={base_xyz}")
        if not is_valid_position(base_xyz):
            print(f"  ✗ Out of workspace, abort")
            return

        # Step 2: IK 送 above
        print(f"\n=== Step 2: IK → above (z+{APPROACH_HEIGHT}m) ===")
        z_above = base_xyz[2] + APPROACH_HEIGHT
        seed = np.array([SAFE_HOME_RAD["shoulder_lift"], SAFE_HOME_RAD["elbow_flex"],
                         SAFE_HOME_RAD["wrist_flex"], SAFE_HOME_RAD["wrist_roll"]])
        joints_rad, ok = base_xyz_to_arm_joints(base_xyz[0], base_xyz[1], z_above, seed)
        if not ok:
            print(f"  ✗ IK fail, abort")
            return
        target_above = {f"{k}.pos": math.degrees(v) for k, v in joints_rad.items()}
        target_above["gripper.pos"] = math.degrees(GRIPPER_OPEN)
        ramp_to(follower, target_above, duration_s=2.5, label="above")
        time.sleep(0.5)

        # Step 3: ACT 接管
        print(f"\n=== Step 3: ACT ({ACT_DURATION_S}s) ===")
        run_act(follower, top, wrist, act_bundle, cfg["task_name"], ACT_DURATION_S)

        if args.skip_box:
            print("\n=== --skip-box: 不送盒子, 测试结束 ===")
            return

        # Step 4: IK 送盒子
        print(f"\n=== Step 4: IK → box {cfg['box_xyz']} ===")
        obs = follower.get_observation()
        current_gripper = obs.get("gripper.pos", math.degrees(GRIPPER_CLOSED))
        seed = np.array([
            math.radians(obs.get("shoulder_lift.pos", 0)),
            math.radians(obs.get("elbow_flex.pos", 0)),
            math.radians(obs.get("wrist_flex.pos", 0)),
            math.radians(obs.get("wrist_roll.pos", 0)),
        ])
        box_joints, ok = base_xyz_to_arm_joints(*cfg['box_xyz'], seed)
        if not ok:
            print(f"  ✗ Box IK fail")
            return
        target_box = {f"{k}.pos": math.degrees(v) for k, v in box_joints.items()}
        target_box["gripper.pos"] = current_gripper
        ramp_to(follower, target_box, duration_s=3.0, label="to_box")
        time.sleep(0.5)
        target_box["gripper.pos"] = math.degrees(GRIPPER_OPEN)
        follower.send_action(target_box)
        time.sleep(1.5)
        print(f"  ✓ Released")

        # Step 5: 回 safe_home
        print(f"\n=== Step 5: safe_home ===")
        go_to_safe_home(follower)

        print(f"\n{'='*60}")
        print(f"  ✓ DONE: {cfg['name']}")
        print(f"{'='*60}\n")

    finally:
        try:
            if top is not None: top.stop()
        except: pass
        try:
            if wrist is not None: wrist.stop()
        except: pass
        try:
            if follower is not None: follower.disconnect()
        except: pass


if __name__ == "__main__":
    main()
