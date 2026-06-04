#!/usr/bin/env python3
"""拍 D435i 照片 + 跑 detect_objects + 在图上画所有 bbox (含低 score)"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/techin517/cv_module"))

from types import SimpleNamespace
import numpy as np, cv2, pyrealsense2 as rs
from cv_module import load_detector, detect_objects

TEXT_PROMPT = "a battery. a usb stick."
TARGET_CLASSES = {"battery", "usb stick", "usb"}
BOX_THRESHOLD = 0.15  # 调低看更多 detection

print("启动 D435i...")
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_device('243222072732')
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(cfg)
align = rs.align(rs.stream.color)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
intrin = SimpleNamespace(fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy)
for _ in range(15): pipeline.wait_for_frames()
frames = pipeline.wait_for_frames()
frames = align.process(frames)
bgr = np.asanyarray(frames.get_color_frame().get_data())
depth = np.asanyarray(frames.get_depth_frame().get_data())
pipeline.stop()
print("拍照完成.")

print("\n加载 detector...")
processor, model = load_detector(device='cuda')

print(f"\nDetect with prompt: {TEXT_PROMPT!r}")
print(f"Threshold: {BOX_THRESHOLD}")
dets = detect_objects(
    color_image=bgr, depth_image=depth, intrin=intrin,
    depth_scale=0.001, processor=processor, model=model,
    text_prompt=TEXT_PROMPT, target_classes=TARGET_CLASSES,
    box_threshold=BOX_THRESHOLD, device='cuda')

print(f"\n=== Detected {len(dets)} objects ===")
for i, d in enumerate(dets):
    print(f"  [{i}] label={d['label']!r}  score={d['score']:.2f}  "
          f"is_target={d['is_target']}  depth_valid={d['depth_valid']}  "
          f"bbox={tuple(int(x) for x in d['bbox_xyxy'])}")

out = bgr.copy()
for d in dets:
    x1, y1, x2, y2 = [int(x) for x in d['bbox_xyxy']]
    color = (0, 255, 0) if d['is_target'] else (0, 0, 255)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    text = f"{d['label']} {d['score']:.2f}"
    cv2.putText(out, text, (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)

cv2.imwrite('/tmp/detect_debug.jpg', out)
cv2.imwrite('/tmp/detect_raw.jpg', bgr)
print("\nSaved:")
print("  /tmp/detect_raw.jpg (原图)")
print("  /tmp/detect_debug.jpg (带 bbox + label + score)")
