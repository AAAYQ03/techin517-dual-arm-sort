"""单独跑 DINO 检测 earbuds, 保存可视化图"""
import sys, os, time
sys.path.insert(0, os.path.expanduser('~/techin517'))

import numpy as np
import cv2
import pyrealsense2 as rs
from cv_module.cv_module import load_detector, detect_objects, draw_detections

PROMPT = "a rounded white case."
TARGETS = {"case", "rounded white case"}
BOX_THRESHOLD = 0.30

# 1. D435i 拍照 + depth
pipeline = rs.pipeline()
config = rs.config()
config.enable_device('243222072732')
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"depth_scale = {depth_scale}")

time.sleep(2)  # warmup

frames = pipeline.wait_for_frames()
aligned = align.process(frames)
color = aligned.get_color_frame()
depth = aligned.get_depth_frame()
intr = depth.profile.as_video_stream_profile().get_intrinsics()
color_bgr = np.asanyarray(color.get_data())
depth_raw = np.asanyarray(depth.get_data())
pipeline.stop()

cv2.imwrite('/home/ubuntu/techin517/debug_color_only.png', color_bgr)
print("✓ 原图保存: ~/techin517/debug_color_only.png")

# 2. DINO 跑检测
print(f"\n=== DINO 检测 (prompt: '{PROMPT}', threshold {BOX_THRESHOLD}) ===")
processor, model = load_detector(device="cuda")

from types import SimpleNamespace
intrin = SimpleNamespace(fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy)

dets = detect_objects(
    color_image=color_bgr,
    depth_image=depth_raw,
    intrin=intrin,
    depth_scale=depth_scale,
    processor=processor,
    model=model,
    text_prompt=PROMPT,
    target_classes=TARGETS,
    box_threshold=BOX_THRESHOLD,
)

print(f"\n=== 检测到 {len(dets)} 个候选 ===")
for i, d in enumerate(dets):
    tag = "[TARGET]" if d.get('is_target') else "[noise ]"
    print(f"  {tag} {d['label']} score={d['score']:.3f} bbox={d['bbox_xyxy']}")
    print(f"           cam_xyz={d.get('cam_xyz_m')}")

# 3. 画框存图
out = draw_detections(color_bgr, dets)
cv2.imwrite('/home/ubuntu/techin517/debug_earbuds_detected.png', out)
print("\n✓ 检测可视化: ~/techin517/debug_earbuds_detected.png")
print("   绿框=匹配 target_classes, 橙框=DINO 检测到但不在 target_classes")
