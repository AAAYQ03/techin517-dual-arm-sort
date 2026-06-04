# cv_module.py
"""
CV 检测模块 - 基于 Grounding DINO + RealSense 深度

主要导出:
    load_detector(device='cuda', model_name='IDEA-Research/grounding-dino-base')
        加载模型,返回 (processor, model)
    
    detect_objects(color_image, depth_image, intrin, depth_scale,
                   processor, model, text_prompt, target_classes,
                   box_threshold=0.25, text_threshold=0.20, nms_iou=0.5,
                   device='cuda')
        对一帧图像做物体检测,返回检测结果列表
        
    Detection 数据结构:
        {
            "label": str,                  # 规范化后的类别名
            "raw_label": str,              # 模型原始输出
            "score": float,                # 置信度
            "is_target": bool,             # 是否在 target_classes 里
            "bbox_xyxy": (x1,y1,x2,y2),    # 检测框,像素
            "pixel_center": (cx, cy),      # 中心像素
            "cam_xyz_m": (X, Y, Z),        # 相机坐标系下 3D 点,米
            "depth_valid": bool,           # 深度查询是否有效
        }
"""

import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def load_detector(device="cuda", model_name="IDEA-Research/grounding-dino-base"):
    """加载 Grounding DINO 模型"""
    print(f"[cv_module] 加载 {model_name} 到 {device}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
    model.eval()
    print(f"[cv_module] ✓ 模型已就绪")
    return processor, model


def _query_depth(depth_image, cx, cy, depth_scale, patch=2, bbox=None):
    """在 (cx,cy) 周围方窗取深度中位数; 若提供 bbox 则用 bbox 内全部有效深度.
    单位米. 对细长/黑色物体 (pen) 更稳定."""
    h, w = depth_image.shape
    if bbox is not None:
        # 用 bbox 内全部有效深度的中位数 (对 pen 这种细长黑色物体更稳定)
        x1b, y1b, x2b, y2b = bbox
        x1b, y1b = max(0, int(x1b)), max(0, int(y1b))
        x2b, y2b = min(w, int(x2b)), min(h, int(y2b))
        if x2b <= x1b or y2b <= y1b:
            # bbox 无效, 退回到中心 patch
            pass
        else:
            region = depth_image[y1b:y2b, x1b:x2b]
            valid = region[region > 0]
            if len(valid) > 5:  # 至少 5 个有效点
                return float(np.median(valid)) * depth_scale, True
    # 默认: 中心 patch
    y1, y2 = max(0, cy - patch), min(h, cy + patch + 1)
    x1, x2 = max(0, cx - patch), min(w, cx + patch + 1)
    region = depth_image[y1:y2, x1:x2]
    valid = region[region > 0]
    if len(valid) == 0:
        return 0.0, False
    return float(np.median(valid)) * depth_scale, True


def _pixel_to_cam(cx, cy, z_m, intrin):
    """像素 + 深度 -> 相机坐标系 3D 点(米)"""
    X = (cx - intrin.ppx) * z_m / intrin.fx
    Y = (cy - intrin.ppy) * z_m / intrin.fy
    Z = z_m
    return float(X), float(Y), float(Z)


def _normalize_label(raw_label, target_classes):
    """
    规范化标签:
    - 如果 raw_label 包含任一 target 类别词作为子串,返回最长的那个匹配
    - 否则返回 raw_label 本身
    返回: (normalized_label, is_target)
    """
    matched = [t for t in target_classes if t in raw_label]
    if matched:
        return max(matched, key=len), True
    return raw_label, False


def _nms(boxes, scores, labels, score_threshold, iou_threshold):
    """对检测结果做 NMS 去重"""
    if len(boxes) == 0:
        return [], [], []
    boxes_xywh = [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                  for x1, y1, x2, y2 in boxes]
    keep = cv2.dnn.NMSBoxes(
        boxes_xywh, scores.tolist() if hasattr(scores, "tolist") else list(scores),
        score_threshold=score_threshold,
        nms_threshold=iou_threshold,
    )
    if len(keep) == 0:
        return [], [], []
    keep = np.array(keep).flatten()
    return boxes[keep], scores[keep], [labels[i] for i in keep]


def detect_objects(
    color_image,
    depth_image,
    intrin,
    depth_scale,
    processor,
    model,
    text_prompt,
    target_classes,
    box_threshold=0.25,
    text_threshold=0.20,
    nms_iou=0.5,
    device="cuda",
):
    """
    主检测函数。输入一帧 RGB+Depth,返回所有检测结果。
    
    color_image: np.ndarray, BGR, 形状 (H, W, 3)
    depth_image: np.ndarray, uint16, 形状 (H, W),单位由 depth_scale 决定
    intrin:      pyrealsense2 内参对象,有 fx/fy/ppx/ppy 字段
    depth_scale: float,depth_image * depth_scale = 米
    processor, model: load_detector() 返回的两个对象
    text_prompt: str,Grounding DINO 风格,每类用 ". " 分隔
    target_classes: set[str],目标类别词(将作为子串匹配 raw_label)
    
    返回: List[Detection dict],参见模块文档
    """
    # 1. 模型推理
    rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[pil.size[::-1]],
    )[0]
    
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    raw_labels = results.get("text_labels", results.get("labels"))
    
    # 2. NMS
    boxes, scores, raw_labels = _nms(
        boxes, scores, raw_labels,
        score_threshold=box_threshold, iou_threshold=nms_iou
    )
    
    # 3. 组装输出
    detections = []
    for box, score, raw_label in zip(boxes, scores, raw_labels):
        x1, y1, x2, y2 = box.astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        z_m, depth_valid = _query_depth(depth_image, cx, cy, depth_scale, bbox=(x1, y1, x2, y2))
        X, Y, Z = _pixel_to_cam(cx, cy, z_m, intrin)
        norm_label, is_target = _normalize_label(raw_label, target_classes)
        
        detections.append({
            "label": norm_label,
            "raw_label": raw_label,
            "score": float(score),
            "is_target": is_target,
            "bbox_xyxy": (int(x1), int(y1), int(x2), int(y2)),
            "pixel_center": (int(cx), int(cy)),
            "cam_xyz_m": (X, Y, Z),
            "depth_valid": depth_valid,
        })
    
    return detections


def draw_detections(image, detections):
    """在图像上画出检测结果(返回新图像,不修改原图)"""
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        cx, cy = det["pixel_center"]
        X, Y, Z = det["cam_xyz_m"]
        color = (0, 255, 0) if det["is_target"] else (0, 165, 255)
        
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(out, f"{det['label']} {det['score']:.2f}",
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        coord_text = f"({X*100:+.1f},{Y*100:+.1f},{Z*100:+.1f})cm"
        if not det["depth_valid"]:
            coord_text += " [no depth]"
        cv2.putText(out, coord_text, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out