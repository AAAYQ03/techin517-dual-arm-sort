#!/usr/bin/env python3
"""快速测试多个耳机盒 prompt, 看 DINO 哪个认"""
import sys, os, cv2, numpy as np
from types import SimpleNamespace
sys.path.insert(0, os.path.expanduser("~/techin517/cv_module"))
from cv_module import load_detector, detect_objects

CANDIDATES = [
    "a wireless earbuds case.",
    "a white earbuds case.",
    "an airpods case.",
    "a white plastic case.",
    "a small white case.",
    "a white soap bar.",
    "a white pill.",
    "a white capsule.",
    "a small white box.",
    "a white object.",
    "a rounded white case.",
    "a smooth white case.",
    "a white pebble.",
]

img_path = sys.argv[1] if len(sys.argv) > 1 else None
if not img_path or not os.path.exists(img_path):
    print(f"Usage: {sys.argv[0]} <image.png>")
    sys.exit(1)

img = cv2.imread(img_path)
H, W = img.shape[:2]
print(f"图像: {img_path}, {W}x{H}\n")

processor, model = load_detector(device="cuda", model_name="IDEA-Research/grounding-dino-base")

depth_fake = np.zeros((H, W), dtype=np.uint16)
intrin_fake = SimpleNamespace(fx=600, fy=600, ppx=W/2, ppy=H/2)

EXPECT_X_RANGE = (240, 360)
EXPECT_Y_RANGE = (180, 290)

def in_expected_area(bbox):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)/2, (y1+y2)/2
    return EXPECT_X_RANGE[0] <= cx <= EXPECT_X_RANGE[1] and EXPECT_Y_RANGE[0] <= cy <= EXPECT_Y_RANGE[1]

results = []
for prompt in CANDIDATES:
    dets = detect_objects(
        color_image=img, depth_image=depth_fake, intrin=intrin_fake,
        depth_scale=0.001, processor=processor, model=model,
        text_prompt=prompt, target_classes=set(),
        box_threshold=0.15, text_threshold=0.10, device="cuda",
    )
    dets.sort(key=lambda d: d["score"], reverse=True)

    hit = None
    for d in dets[:5]:
        if in_expected_area(d["bbox_xyxy"]):
            hit = d
            break

    if hit:
        results.append((prompt, hit["score"], hit["bbox_xyxy"]))
        print(f"✓ '{prompt}' → score={hit['score']:.3f} bbox={hit['bbox_xyxy']}")
    else:
        if dets:
            top = dets[0]
            print(f"✗ '{prompt}' → top bbox 在其他区域: {top['bbox_xyxy']} (score={top['score']:.3f})")
        else:
            print(f"✗ '{prompt}' → no detection")

print("\n=== 命中耳机盒区域的 prompt 排名 ===")
for prompt, score, bbox in sorted(results, key=lambda x: -x[1]):
    print(f"  score={score:.3f}  bbox={bbox}  prompt='{prompt}'")
