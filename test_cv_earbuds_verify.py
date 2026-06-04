#!/usr/bin/env python3
"""
验证 top prompt 的 false positive:
对每个候选 prompt, 输出所有 score > 0.15 的检测,
看会不会把其他物品也当成耳机盒
"""
import sys, os, cv2, numpy as np
from types import SimpleNamespace
sys.path.insert(0, os.path.expanduser("~/techin517/cv_module"))
from cv_module import load_detector, detect_objects

# Top 5 候选 (语义中性的)
CANDIDATES = [
    "a rounded white case.",
    "a small white case.",
    "a white plastic case.",
    "a smooth white case.",
    "a white soap bar.",        # 备选: score 最高但语义偏
]

img_path = sys.argv[1]
img = cv2.imread(img_path)
H, W = img.shape[:2]
print(f"图像: {img_path}, {W}x{H}\n")

processor, model = load_detector(device="cuda", model_name="IDEA-Research/grounding-dino-base")

depth_fake = np.zeros((H, W), dtype=np.uint16)
intrin_fake = SimpleNamespace(fx=600, fy=600, ppx=W/2, ppy=H/2)

# 各物品的大致中心位置 (从你的图判断)
LANDMARKS = {
    "earbuds_case": (318, 248),   # 真耳机盒
    "battery":      (193, 316),   # 9V 电池
    "glue_stick":   (513, 244),   # 胶棒
    "marker":       (542, 328),   # 马克笔
    "robot_arm_L":  (55, 380),    # 左下黑色机械臂
    "yellow_box":   (250, 460),   # 黄色盒子
}

def closest_landmark(bbox):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)/2, (y1+y2)/2
    best = ("?", 99999)
    for name, (lx, ly) in LANDMARKS.items():
        d = ((cx-lx)**2 + (cy-ly)**2) ** 0.5
        if d < best[1]:
            best = (name, d)
    return best[0] if best[1] < 80 else f"unknown({int(cx)},{int(cy)})"

print("=" * 80)
print("各 prompt 的所有 score > 0.15 检测 (看 FP):")
print("=" * 80)

for prompt in CANDIDATES:
    dets = detect_objects(
        color_image=img, depth_image=depth_fake, intrin=intrin_fake,
        depth_scale=0.001, processor=processor, model=model,
        text_prompt=prompt, target_classes=set(),
        box_threshold=0.15, text_threshold=0.10, device="cuda",
    )
    dets.sort(key=lambda d: d["score"], reverse=True)
    
    print(f"\nPrompt: '{prompt}'")
    print(f"  检测数: {len(dets)}")
    for i, d in enumerate(dets):
        landmark = closest_landmark(d["bbox_xyxy"])
        is_target = "✓" if landmark == "earbuds_case" else "✗ FP"
        print(f"  [{i+1}] {is_target}  score={d['score']:.3f}  bbox={d['bbox_xyxy']}  near={landmark}")

print("\n" + "=" * 80)
print("决策建议:")
print("  ✓ 只命中 earbuds_case → 好 prompt")
print("  ✗ 命中多个物品 → 有 FP, dispatch 会混淆")
print("=" * 80)
