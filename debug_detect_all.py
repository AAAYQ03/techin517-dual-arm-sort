"""跑 DINO 检测 4 个物品 (pen, glue, battery, earbuds), 保存可视化图"""
import sys, os, time
sys.path.insert(0, os.path.expanduser('~/techin517'))

import numpy as np
import cv2
import pyrealsense2 as rs
from cv_module.cv_module import load_detector, detect_objects, draw_detections

ITEMS = [
    {"prompt": "a black pen.",         "targets": {"black pen", "pen", "marker"}, "thr": 0.22},
    {"prompt": "a glue stick.",        "targets": {"glue stick", "glue"},          "thr": 0.25},
    {"prompt": "a battery.",           "targets": {"battery"},                     "thr": 0.28},
    {"prompt": "a rounded white case.","targets": {"case", "rounded white case"},  "thr": 0.27},
]

# 1. 拍 D435i
pipeline = rs.pipeline()
config = rs.config()
config.enable_device('243222072732')
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

time.sleep(2)
frames = pipeline.wait_for_frames()
aligned = align.process(frames)
color = aligned.get_color_frame()
depth = aligned.get_depth_frame()
intr = depth.profile.as_video_stream_profile().get_intrinsics()
color_bgr = np.asanyarray(color.get_data())
depth_raw = np.asanyarray(depth.get_data())
pipeline.stop()

# 2. DINO 跑每个 item
processor, model = load_detector(device="cuda")
from types import SimpleNamespace
intrin = SimpleNamespace(fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy)

all_dets = []
for item in ITEMS:
    print(f"\n=== DINO '{item['prompt']}' threshold={item['thr']} ===")
    dets = detect_objects(
        color_image=color_bgr, depth_image=depth_raw, intrin=intrin,
        depth_scale=depth_scale, processor=processor, model=model,
        text_prompt=item['prompt'], target_classes=item['targets'],
        box_threshold=item['thr'],
    )
    print(f"  Found {len(dets)} detection(s):")
    for d in dets:
        tag = "[TARGET]" if d.get('is_target') else "[noise ]"
        print(f"    {tag} {d['label']} score={d['score']:.3f} bbox={d['bbox_xyxy']}")
    all_dets.extend(dets)

# 3. 画框存图
out = draw_detections(color_bgr, all_dets)
cv2.imwrite('/home/ubuntu/techin517/debug_detect_hard.png', out)
print("\n✓ ~/techin517/debug_detect_hard.png")
print("  绿框 = is_target=True (会被 dispatch 接受)")
print("  橙框 = DINO 检测到但 label 不匹配 (会被过滤)")
