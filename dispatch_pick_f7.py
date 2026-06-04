#!/usr/bin/env python3
"""
dispatch_pick.py v2 - 分开 detect + 位置过滤

修复 v1 的两个问题:
  1. 混合 prompt "a battery. a usb stick." 下 DINO mis-classify
     → 改成每个 prompt 单独 detect, 强制用 prompt 对应 label
  2. DINO 把机械臂误识别为物体
     → 加 base_link 位置过滤, 工作空间外的都丢
"""

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
from clip_classifier import ClipClassifier
from lerobot_kinematics import lerobot_IK, get_robot


# ============ 配置 ============
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6057204-if00"
FOLLOWER_ID   = "gix-follower7"
WRIST_DEV     = "/dev/v4l/by-path/pci-0000:0e:00.0-usb-0:1.1:1.0-video-index0"
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

# === f7 盒子坐标 ===
BOX_SAME_F7  = (0.155, -0.058, 0.10)   # f7 自侧右盒 (stationery: pen, glue)
BOX_CROSS_F7 = (0.195, -0.180, 0.12)   # f7 跨侧左盒 (electronics: battery, earbuds)

# === 全部物品配置 (3 个; f7 不抓 earbuds 同侧, 但保留 cross 用) ===
ALL_ITEMS = {
    "pen": {
        "name": "pen",
        "dino_prompt": "a black pen.",
        "dino_targets": {"black pen", "pen", "marker"},
        "act_checkpoint": "/home/ubuntu/techin517/outputs/train/act_pen_f7/checkpoints/last/pretrained_model",
        "task_name": "Pick up the pen",
        "category": "stationery",
        "score_threshold": 0.22,
    },
    "glue": {
        "name": "glue",
        "dino_prompt": "a glue stick.",
        "dino_targets": {"glue stick", "stick", "glue"},
        "act_checkpoint": "/home/ubuntu/techin517/outputs/train/act_glue_f7/checkpoints/last/pretrained_model",
        "task_name": "Pick up the glue",
        "category": "stationery",
        "score_threshold": 0.45,
    },
    "battery": {
        "name": "battery",
        "dino_prompt": "a battery.",
        "dino_targets": {"battery"},
        "act_checkpoint": "/home/ubuntu/techin517/outputs/train/act_battery_f7/checkpoints/last/pretrained_model",
        "task_name": "Pick up the battery",
        "category": "electronics",
        "score_threshold": 0.30,
    },
}

def get_box_for_item(item_cfg):
    """f7: stationery → 同侧右盒, electronics → 跨侧左盒"""
    if item_cfg["category"] == "stationery":
        return BOX_SAME_F7
    else:
        return BOX_CROSS_F7

# 兼容旧 ITEMS 引用
ITEMS = list(ALL_ITEMS.values())

# === 工作空间过滤 (滤掉机械臂误识别) ===
def is_valid_position(base_xyz):
    """物体必须在合理工作空间内 (桌面物体), 否则可能是机械臂误识别"""
    x, y, z = base_xyz
    return (0.18 < x < 0.30) and (abs(y) < 0.12) and (-0.025 < z < 0.06)

CAM_T = np.array([0.0305, 0.1557, 0.3337])
CAM_Q = np.array([-0.0179, 0.3854, 0.0403, 0.9217])
OPT_T = np.array([0.0, 0.015, 0.0])
OPT_Q = np.array([-0.497, 0.504, -0.497, 0.502])

ACT_DURATION_S = 10.0
ACT_FPS        = 30


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


def load_act_policies(items_to_detect=None):
    policies = {}
    if items_to_detect is not None:
        unique_ckpts = set(ALL_ITEMS[n]["act_checkpoint"] for n in items_to_detect)
    else:
        unique_ckpts = set(item["act_checkpoint"] for item in ITEMS)
    for ckpt in unique_ckpts:
        name = ckpt.split('/')[-3]
        print(f"  Loading: {name}...")
        policy_cfg = PreTrainedConfig.from_pretrained(ckpt)
        policy_class = get_policy_class(policy_cfg.type)
        policy = policy_class.from_pretrained(ckpt)
        policy.to(policy_cfg.device); policy.eval()
        pre_proc, post_proc = make_pre_post_processors(
            policy_cfg=policy_cfg, pretrained_path=ckpt)
        policies[ckpt] = {
            "policy": policy, "pre_proc": pre_proc, "post_proc": post_proc,
            "device": torch.device(policy_cfg.device),
        }
    return policies


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


def redetect_single_item(top, processor_cv, model_cv, item_cfg):
    """单 item 重新 detect, 用于抓下一个物品前 (避免多物品干扰)"""
    print(f"  [Re-detect] {item_cfg['name']}...")
    bgr, depth_raw, intrin = top.read()
    # 保存 top image debug 图
    import cv2 as _cv2
    import time as _t
    debug_path = f"/home/ubuntu/techin517/debug_redetect_{int(_t.time())}.png"
    _cv2.imwrite(debug_path, bgr)
    print(f"    [DEBUG] redetect top 图: {debug_path}")

    dets = detect_objects(
        color_image=bgr, depth_image=depth_raw, intrin=intrin,
        depth_scale=DEPTH_SCALE, processor=processor_cv, model=model_cv,
        text_prompt=item_cfg['dino_prompt'],
        target_classes=item_cfg['dino_targets'],
        box_threshold=item_cfg.get('score_threshold', BOX_THRESHOLD), device=DEVICE)
    valid = [d for d in dets if d.get("is_target") and d.get("depth_valid")]

    if len(dets) > 0:
        print(f"    [DEBUG] DINO raw {len(dets)} detection(s):")
        for i, d in enumerate(dets):
            print(f"      [{i}] score={d.get('score',0):.3f} label={d.get('label','?')}")
    else:
        print(f"    [DEBUG] DINO raw 0")

    for d in valid:
        base_xyz = cam_xyz_to_base(d['cam_xyz_m'])
        d['base_xyz'] = base_xyz
        if not is_valid_position(base_xyz):
            continue
        d['item_config'] = item_cfg
        d['box_xyz'] = get_box_for_item(item_cfg)
        print(f"    [Re-detect ACCEPT] score={d.get('score',0):.2f} base_xyz={base_xyz}")
        return d  # 返回第一个有效

    print(f"    [Re-detect FAIL] no valid {item_cfg['name']} found")
    return None



def color_prior_pass(bgr, bbox, item_name):
    """颜色硬约束: battery 不能太绿 (拒绝绿色饮料盒等干扰).
       earbuds 必须有足够白色比例 (避免黄色机械臂误识别).
       其他物品不做颜色检查."""
    import cv2 as _cv2
    import numpy as _np
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(W,x2), min(H,y2)
    if x2<=x1 or y2<=y1:
        return True, "empty_bbox"
    crop = bgr[y1:y2, x1:x2]
    hsv = _cv2.cvtColor(crop, _cv2.COLOR_BGR2HSV)
    h, s, v = hsv[...,0], hsv[...,1], hsv[...,2]
    if item_name == 'battery':
        green = ((h > 35) & (h < 85) & (s > 40)).mean()
        if green > 0.20:
            return False, f"too_green:{green:.2f}"
    return True, "ok"


def detect_all_objects(top, processor_cv, model_cv, clip_clf, items_to_detect=None):
    """每个 item 单独跑 DINO prompt (不用 CLIP), 工作区过滤 + 去重 + 排序"""
    print("\n  Snapshot from D435i...")
    bgr, depth_raw, intrin = top.read()
    debug_bgr = bgr.copy()  # 保存 raw 图

    if items_to_detect is None:
        items_to_detect = list(ALL_ITEMS.keys())

    all_detections = []
    for item_name in items_to_detect:
        item_cfg = ALL_ITEMS[item_name]
        print(f"  DINO detect: {item_cfg['dino_prompt']!r}")
        dets = detect_objects(
            color_image=bgr, depth_image=depth_raw, intrin=intrin,
            depth_scale=DEPTH_SCALE, processor=processor_cv, model=model_cv,
            text_prompt=item_cfg['dino_prompt'],
            target_classes=item_cfg['dino_targets'],
            box_threshold=item_cfg.get('score_threshold', BOX_THRESHOLD), device=DEVICE)
        # DEBUG: 打印 raw DINO 检测结果 + 在图上画框
        import cv2 as _cv2
        if len(dets) > 0:
            print(f"    [DEBUG] DINO raw {len(dets)} detection(s):")
            for i, d in enumerate(dets):
                print(f"      [{i}] score={d.get('score',0):.3f} is_target={d.get('is_target')} depth_valid={d.get('depth_valid')} label={d.get('label','?')}")
                bbox = d.get('bbox_xyxy', None)
                if bbox is not None:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    # 不同 item 用不同颜色
                    color_map = {"pen": (0,255,255), "glue": (255,200,0), "battery": (0,0,255), "earbuds": (255,0,255)}
                    color = color_map.get(item_name, (200,200,200))
                    _cv2.rectangle(debug_bgr, (x1,y1), (x2,y2), color, 2)
                    label_text = f"{item_name}:{d.get('score',0):.2f}"
                    _cv2.putText(debug_bgr, label_text, (x1, y1-5), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            print(f"    [DEBUG] DINO raw 0 detection")
        valid = [d for d in dets if d.get("is_target") and d.get("depth_valid")]
        print(f"    found {len(valid)} candidate(s) after filters")

        for d in valid:
            # 颜色 prior 过滤 (battery 拒绝绿色; earbuds 需要足够白色)
            cp_pass, cp_reason = color_prior_pass(bgr, d['bbox_xyxy'], item_name)
            if not cp_pass:
                print(f"    [SKIP] color_prior fail ({cp_reason}): score={d['score']:.2f}")
                continue
            base_xyz = cam_xyz_to_base(d['cam_xyz_m'])
            d['base_xyz'] = base_xyz
            if not is_valid_position(base_xyz):
                print(f"    [SKIP] out of workspace: base_xyz={base_xyz}")
                continue
            d['item_config'] = item_cfg
            d['box_xyz'] = get_box_for_item(item_cfg)
            bb = tuple(int(x) for x in d['bbox_xyxy'])
            print(f"    [ACCEPT] {item_cfg['name']} bbox={bb} score={d['score']:.2f} base_xyz={base_xyz}")
            all_detections.append(d)

    # 保存 debug 图
    import time as _time
    import cv2 as _cv2
    debug_path = f"/home/ubuntu/techin517/debug_detect_{int(_time.time())}.png"
    _cv2.imwrite(debug_path, debug_bgr)
    print(f"\n  [DEBUG] 检测可视化图: {debug_path}")

    if not all_detections:
        return []

    # bbox 去重
    def bbox_center(d):
        x1, y1, x2, y2 = d['bbox_xyxy']
        return ((x1+x2)/2, (y1+y2)/2)
    deduped = []
    sorted_dets = sorted(all_detections, key=lambda d: -d['score'])
    for d in sorted_dets:
        cx, cy = bbox_center(d)
        item_name = d['item_config']['name']
        # 只跟同类别物品去重 (不同类别 bbox 接近不该合并)
        same_class_kept = [k for k in deduped if k['item_config']['name'] == item_name]
        if not any((cx-bbox_center(k)[0])**2 + (cy-bbox_center(k)[1])**2 < 200**2 for k in same_class_kept):
            deduped.append(d)
    return deduped



def main():
    import argparse
    ap = argparse.ArgumentParser(description="Universal f7 dispatch")
    ap.add_argument("items", nargs="+", choices=list(ALL_ITEMS.keys()),
                    help="物品列表 (例: pen glue, 或 earbuds glue)")
    args = ap.parse_args()
    items_to_detect = args.items

    print("=" * 60)
    print("  dispatch_pick_f7.py - Universal f7 multi-item dispatch")
    print(f"  Items: {items_to_detect}")
    for n in items_to_detect:
        c = ALL_ITEMS[n]
        print(f"    - {c['name']:10s} -> {c['category']:11s} box={get_box_for_item(c)}")
    print("=" * 60)

    print("\n[1/4] Connecting follower...")
    cfg = SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID,
                              disable_torque_on_disconnect=False)
    follower = SO101Follower(cfg); follower.connect()
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
    print("  Loading ACT policies...")
    policies = load_act_policies(items_to_detect)
    print(f"  ✓ {len(policies)} ACT models loaded")

    try:
        clip_clf = None  # 不用 CLIP, 每个 item 单独 prompt

        print("\n=== safe_home ===")
        go_to_safe_home(follower)

        print("\n=== CV detect all objects ===")
        targets = detect_all_objects(top, processor_cv, model_cv, clip_clf, items_to_detect)
        # 按 base_xyz[0] 降序: 远的物体先抓 (避免前方 gripper 抓近物时撞远物)
        # pen 优先, battery 次优先 (避免胶棒干扰), 其他按 x 降序
        def _sort_key(d):
            name = d['item_config']['name']
            if name == 'pen':
                return (0, -d['base_xyz'][0])
            elif name == 'battery':
                return (1, -d['base_xyz'][0])
            else:
                return (2, -d['base_xyz'][0])
        targets.sort(key=_sort_key)
        if not targets:
            print("  ✗ No valid detection")
            return
        print(f"\n  ✓ {len(targets)} object(s) to pick:")
        for d in targets:
            print(f"    - {d['item_config']['name']}: score={d['score']:.2f}")

        for idx, det in enumerate(targets):
            item = det['item_config']
            act_bundle = policies[item['act_checkpoint']]

            # 抓之前重新 detect 该 item (避免多物品干扰)
            # 第 1 个用初始 detect 结果, 后续重新 detect
            if idx > 0:
                new_det = redetect_single_item(top, processor_cv, model_cv, item)
                if new_det is None:
                    print(f"  [SKIP] re-detect 失败, 跳过 {item['name']}")
                    continue
                base_xyz = new_det['base_xyz']
                _redet_box_xyz = new_det['box_xyz']  # 保存供后面用
            else:
                base_xyz = det['base_xyz']
                _redet_box_xyz = det['box_xyz']

            print(f"\n{'='*60}")
            print(f"  [{idx+1}/{len(targets)}] Picking '{item['name']}'")
            print(f"  base_xyz = {base_xyz}")
            print('='*60)

            z_above = base_xyz[2] + APPROACH_HEIGHT
            seed = np.array([SAFE_HOME_RAD["shoulder_lift"], SAFE_HOME_RAD["elbow_flex"],
                             SAFE_HOME_RAD["wrist_flex"], SAFE_HOME_RAD["wrist_roll"]])
            joints_rad, ok = base_xyz_to_arm_joints(base_xyz[0], base_xyz[1], z_above, seed)
            if not ok:
                print(f"  ✗ IK fail, skipping")
                continue
            target_above = {f"{k}.pos": math.degrees(v) for k, v in joints_rad.items()}
            target_above["gripper.pos"] = math.degrees(GRIPPER_OPEN)
            ramp_to(follower, target_above, duration_s=2.5, label="above")

            print(f"\n  ACT ({ACT_DURATION_S}s, task='{item['task_name']}')")
            run_act(follower, top, wrist, act_bundle, item['task_name'], ACT_DURATION_S)

            box_xyz = _redet_box_xyz  # 用 redetect 结果 (或第 1 个的初始结果)
            print(f"\n  Send to box {box_xyz}")
            obs = follower.get_observation()
            current_gripper = obs.get("gripper.pos", math.degrees(GRIPPER_CLOSED))
            seed = np.array([
                math.radians(obs.get("shoulder_lift.pos", 0)),
                math.radians(obs.get("elbow_flex.pos", 0)),
                math.radians(obs.get("wrist_flex.pos", 0)),
                math.radians(obs.get("wrist_roll.pos", 0)),
            ])
            box_joints, ok = base_xyz_to_arm_joints(*box_xyz, seed)
            if not ok:
                print(f"  ✗ Box IK fail")
                continue
            target_box = {f"{k}.pos": math.degrees(v) for k, v in box_joints.items()}
            target_box["gripper.pos"] = current_gripper
            ramp_to(follower, target_box, duration_s=3.0, label="to_box")
            time.sleep(0.5)
            target_box["gripper.pos"] = math.degrees(GRIPPER_OPEN)
            follower.send_action(target_box)
            time.sleep(1.5)
            print(f"  ✓ Released")

            print(f"\n  Back to safe_home")
            go_to_safe_home(follower)

        print("\n\n" + "=" * 60)
        print(f"  === ALL DONE: picked {len(targets)} objects ===")
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
