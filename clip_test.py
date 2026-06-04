#!/usr/bin/env python3
"""CLIP zero-shot 二次分类 - 验证能否区分 U 盘 vs 电池"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/techin517/cv_module"))
from types import SimpleNamespace
import numpy as np, cv2, torch
import pyrealsense2 as rs
from cv_module import load_detector, detect_objects
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

TEXT_PROMPT = "a battery. a usb stick."
TARGET_CLASSES = {"battery", "usb stick", "usb"}
BOX_THRESHOLD = 0.20

# 1. D435i
print("拍照...")
pl = rs.pipeline(); cfg = rs.config()
cfg.enable_device('243222072732')
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pl.start(cfg)
align = rs.align(rs.stream.color)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
intrin = SimpleNamespace(fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy)
for _ in range(15): pl.wait_for_frames()
frames = align.process(pl.wait_for_frames())
bgr = np.asanyarray(frames.get_color_frame().get_data())
depth = np.asanyarray(frames.get_depth_frame().get_data())
pl.stop()
print("OK\n")

# 2. DINO detect
print("DINO detect...")
processor, model = load_detector(device='cuda')
dets = detect_objects(
    color_image=bgr, depth_image=depth, intrin=intrin,
    depth_scale=0.001, processor=processor, model=model,
    text_prompt=TEXT_PROMPT, target_classes=TARGET_CLASSES,
    box_threshold=BOX_THRESHOLD, device='cuda')
print(f"DINO 找到 {len(dets)} 个 candidate\n")

# 3. CLIP load + encode text
print("加载 CLIP...")
device = 'cuda'
clip_m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
classes = ['battery', 'usb stick', 'sharpie marker', 'robotic arm', 'empty background']
templates = ["a photo of a {}", "a close-up of a {}", "a {} on a desk"]
text_in = [t.format(c) for c in classes for t in templates]
ti = clip_p(text=text_in, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    tf = clip_m.get_text_features(**ti)
    tf = tf / tf.norm(dim=-1, keepdim=True)
    tf = tf.view(len(classes), len(templates), -1).mean(dim=1)
    tf = tf / tf.norm(dim=-1, keepdim=True)

# 4. 每个 DINO bbox 跑 CLIP
print("=== CLIP 分类 ===\n")
vis = bgr.copy()
H, W = bgr.shape[:2]
for i, d in enumerate(dets):
    x1, y1, x2, y2 = [int(v) for v in d['bbox_xyxy']]
    pad = 15
    cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
    cx2, cy2 = min(W, x2+pad), min(H, y2+pad)
    crop = bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0: continue
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    ii = clip_p(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        f = clip_m.get_image_features(**ii)
        f = f / f.norm(dim=-1, keepdim=True)
        s = (clip_m.logit_scale.exp() * (f @ tf.T)).softmax(-1)[0]
    idx = s.argmax().item()
    print(f"[{i}] DINO: '{d['label']}' score={d['score']:.2f} bbox=({x1},{y1},{x2},{y2})")
    print(f"    CLIP: {classes[idx]} ({s[idx].item():.2f})")
    for j, c in enumerate(classes):
        bar = '#' * int(s[j].item()*30)
        print(f"      {c:18s} {s[j].item():.3f} {bar}")
    print()
    cv2.imwrite(f'/tmp/clip_crop_{i}.jpg', crop)
    col = (0,255,0) if classes[idx] in ('battery','usb stick') else (0,0,255)
    cv2.rectangle(vis, (x1,y1), (x2,y2), col, 2)
    txt = f"D:{d['label'][:8]}({d['score']:.2f}) C:{classes[idx][:8]}({s[idx].item():.2f})"
    cv2.putText(vis, txt, (x1, max(20,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

cv2.imwrite('/tmp/clip_debug.jpg', vis)
print("保存: /tmp/clip_debug.jpg + /tmp/clip_crop_*.jpg")
