"""DINO + CLIP 二阶段, 跟 dispatch_pick.py 一致"""
import sys, os, time
sys.path.insert(0, os.path.expanduser('~/techin517'))

import numpy as np
import cv2
import pyrealsense2 as rs
from cv_module.cv_module import load_detector, detect_objects, draw_detections
from clip_classifier import ClipClassifier

ITEMS = [
    {"prompt": "a black pen.",         "targets": {"black pen", "pen", "marker"}, "thr": 0.22, "clip_class": "pen"},
    {"prompt": "a glue stick.",        "targets": {"glue stick", "glue"},          "thr": 0.25, "clip_class": "glue stick"},
    {"prompt": "a battery.",           "targets": {"battery"},                     "thr": 0.30, "clip_class": "battery"},
    {"prompt": "a rounded white case.","targets": {"case", "rounded white case"},  "thr": 0.27, "clip_class": "earbuds case"},
]

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
color_bgr = np.asanyarray(aligned.get_color_frame().get_data())
depth_raw = np.asanyarray(aligned.get_depth_frame().get_data())
intr = aligned.get_depth_frame().profile.as_video_stream_profile().get_intrinsics()
pipeline.stop()

processor, model = load_detector(device="cuda")
clip_clf = ClipClassifier(device="cuda")
from types import SimpleNamespace
intrin = SimpleNamespace(fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy)

all_dets = []
for item in ITEMS:
    print(f"\n=== DINO '{item['prompt']}' thr={item['thr']} ===")
    dets = detect_objects(
        color_image=color_bgr, depth_image=depth_raw, intrin=intrin,
        depth_scale=depth_scale, processor=processor, model=model,
        text_prompt=item['prompt'], target_classes=item['targets'],
        box_threshold=item['thr'],
    )
    for d in dets:
        if not d.get('is_target'):
            continue
        # CLIP 二阶段
        cls, conf, _ = clip_clf.classify(color_bgr, d['bbox_xyxy'])
        match = (cls == item['clip_class'])
        tag = "✓" if match else "✗"
        d['clip_match'] = match
        d['clip_class'] = cls
        d['clip_conf'] = conf
        print(f"  {tag} DINO={d['label']} score={d['score']:.3f} | CLIP={cls} conf={conf:.3f}  ({'KEEP' if match else 'REJECT'})")
        if match:
            all_dets.append(d)

out = draw_detections(color_bgr, all_dets)
cv2.imwrite('/home/ubuntu/techin517/debug_detect_hard_clip.png', out)
print("\n✓ ~/techin517/debug_detect_hard_clip.png")
print("  只显示通过 CLIP 验证的物品")
