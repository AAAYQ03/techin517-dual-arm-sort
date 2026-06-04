#!/usr/bin/env python3
"""
4 物品识别测试 (不依赖 ROS):
  - 读磁盘 PNG (而不是 D435i topic)
  - 每个物品单独跑 DINO prompt (跟 dispatch_pick.py v3 一致)
  - CLIP 二次分类
  - shape prior (aspect ratio)
  - 可视化结果存到 PNG, 宿主机查看

用法:
  python3 test_cv_4items.py <输入图.png> [--threshold 0.20]
"""
import sys, os, argparse, cv2, numpy as np, torch
from types import SimpleNamespace

sys.path.insert(0, os.path.expanduser("~/techin517/cv_module"))
sys.path.insert(0, os.path.expanduser("~/techin517"))

from cv_module import load_detector, detect_objects, draw_detections
from clip_classifier import ClipClassifier

# ============ 4 物品配置 ============
# DINO prompt 候选 (单独跑每个 prompt, 强制 label)
PROMPTS = [
    {"name": "battery",     "prompt": "a battery.",                "category": "electronics"},
    {"name": "earbuds_case","prompt": "a rounded white case.",  "category": "electronics"},
    {"name": "glue_stick",  "prompt": "a glue stick.",             "category": "stationery"},
    {"name": "marker",      "prompt": "a black marker pen.",       "category": "stationery"},
]

# shape prior (aspect ratio = long_side / short_side)
# 用于区分形状差异大的物品 (尤其 marker 是细长 vs 其他)
ASPECT_RANGES = {
    "battery":      (1.2, 2.5),   # 矩形, 偏短
    "earbuds_case": (1.2, 2.2),   # 椭圆, 接近正方形
    "glue_stick":   (2.3, 4.5),   # 圆柱, 中等细长
    "marker":       (4.5, 12.0),  # 细长
}

BOX_THRESHOLD = 0.20
TEXT_THRESHOLD = 0.15
DEVICE = "cuda"


def bbox_aspect_ratio(bbox):
    x1, y1, x2, y2 = bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    long_side  = max(w, h)
    short_side = min(w, h)
    return long_side / short_side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_image")
    ap.add_argument("--threshold", type=float, default=BOX_THRESHOLD)
    ap.add_argument("--use-clip", action="store_true",
                    help="也跑 CLIP (但 CLIP 4 类需要重训, 这里仅展示当前 CLIP 输出)")
    args = ap.parse_args()

    if not os.path.exists(args.input_image):
        print(f"[ERROR] 文件不存在: {args.input_image}")
        sys.exit(1)

    img = cv2.imread(args.input_image)
    if img is None:
        print(f"[ERROR] 读图失败: {args.input_image}")
        sys.exit(1)
    H, W = img.shape[:2]
    print(f"[INFO] 输入图: {args.input_image}, 尺寸: {W}x{H}\n")

    # === 加载模型 ===
    print("[INFO] Loading DINO base...")
    processor, model = load_detector(device=DEVICE, model_name="IDEA-Research/grounding-dino-base")

    clip_clf = None
    if args.use_clip:
        print("[INFO] Loading CLIP (注: CLIP CLASSES 当前只有 battery/usb stick, 4 物品分类需要扩展)...")
        clip_clf = ClipClassifier(device=DEVICE)

    # === 假 depth + 假 intrin (我们只看 bbox, 不算 3D xyz) ===
    depth_fake = np.zeros((H, W), dtype=np.uint16)
    intrin_fake = SimpleNamespace(fx=600, fy=600, ppx=W/2, ppy=H/2)

    # === 逐个 prompt 检测 ===
    all_results = []
    for item in PROMPTS:
        print(f"\n=== Detecting '{item['name']}' (prompt='{item['prompt']}') ===")
        dets = detect_objects(
            color_image=img,
            depth_image=depth_fake,
            intrin=intrin_fake,
            depth_scale=0.001,
            processor=processor,
            model=model,
            text_prompt=item["prompt"],
            target_classes={item["name"]},  # 不依赖 normalize, 我们直接用 prompt name
            box_threshold=args.threshold,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE,
        )

        if not dets:
            print(f"  [MISS] no detection")
            continue

        # 按 score 排序, 显示前 3 个
        dets.sort(key=lambda d: d["score"], reverse=True)
        for i, d in enumerate(dets[:3]):
            bbox = d["bbox_xyxy"]
            score = d["score"]
            raw = d["raw_label"]
            aspect = bbox_aspect_ratio(bbox)

            # shape prior 检查
            lo, hi = ASPECT_RANGES.get(item["name"], (0, 999))
            in_range = lo <= aspect <= hi

            tag = "✓" if in_range else "✗"
            print(f"  [{i+1}] {tag} score={score:.3f} bbox={bbox} aspect={aspect:.2f} "
                  f"(expected {lo:.1f}-{hi:.1f}) raw='{raw}'")

            # CLIP 验证 (如果加载了)
            if clip_clf is not None:
                clip_label, clip_conf, clip_scores = clip_clf.classify(img, bbox)
                top3 = sorted(clip_scores.items(), key=lambda x: -x[1])[:3]
                print(f"      CLIP: {clip_label} ({clip_conf:.2f}) - "
                      f"top3: {[(l, f'{s:.2f}') for l,s in top3]}")

            # 保存最高分的结果用于可视化
            if i == 0 and in_range:
                d["display_label"] = f"{item['name']} ({score:.2f}, AR={aspect:.1f})"
                d["pass_shape"] = True
                all_results.append(d)
            elif i == 0:
                d["display_label"] = f"{item['name']}? AR={aspect:.1f} OUT"
                d["pass_shape"] = False
                all_results.append(d)

    # === 可视化 ===
    print("\n=== Visualizing ===")
    vis = img.copy()
    colors = {
        "battery":      (0, 0, 255),    # red
        "earbuds_case": (255, 0, 0),    # blue
        "glue_stick":   (0, 165, 255),  # orange
        "marker":       (0, 255, 0),    # green
    }
    for d in all_results:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        name = d["display_label"].split(" ")[0].rstrip("?")
        color = colors.get(name, (255, 255, 255))
        thick = 3 if d.get("pass_shape", False) else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
        label = d["display_label"]
        font_scale = 0.6
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(vis, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(vis, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 2)

    out_path = args.input_image.replace(".png", "_detected.png")
    cv2.imwrite(out_path, vis)
    print(f"[INFO] 可视化保存到: {out_path}")
    print(f"\n[INFO] 检测总结:")
    print(f"  通过 shape prior: {sum(1 for d in all_results if d.get('pass_shape'))}")
    print(f"  未通过 shape prior: {sum(1 for d in all_results if not d.get('pass_shape'))}")


if __name__ == "__main__":
    main()
