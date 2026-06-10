# cv_module.py
"""
CV detection module - based on Grounding DINO + RealSense depth.

Main exports:
    load_detector(device='cuda', model_name='IDEA-Research/grounding-dino-base')
        Load the model and return (processor, model).
    
    detect_objects(color_image, depth_image, intrin, depth_scale,
                   processor, model, text_prompt, target_classes,
                   box_threshold=0.25, text_threshold=0.20, nms_iou=0.5,
                   device='cuda')
        Run object detection on a single frame; returns a list of detection results.
        
    Detection data structure:
        {
            "label": str,                  # Normalized class name
            "raw_label": str,              # Raw model output
            "score": float,                # Confidence
            "is_target": bool,             # Whether it is in target_classes
            "bbox_xyxy": (x1,y1,x2,y2),    # Detection box, pixels
            "pixel_center": (cx, cy),      # Center pixel
            "cam_xyz_m": (X, Y, Z),        # 3D point in camera frame, meters
            "depth_valid": bool,           # Whether depth lookup succeeded
        }
"""

import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def load_detector(device="cuda", model_name="IDEA-Research/grounding-dino-base"):
    """Load the Grounding DINO model."""
    print(f"[cv_module] Loading {model_name} on {device}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
    model.eval()
    print(f"[cv_module] ✓ Model ready")
    return processor, model


def _query_depth(depth_image, cx, cy, depth_scale, patch=2, bbox=None):
    """Take the median depth from a square window around (cx, cy);
    if bbox is provided, use all valid depth values inside the bbox.
    Units in meters. More stable for thin/dark objects (pen)."""
    h, w = depth_image.shape
    if bbox is not None:
        # Use the median of all valid depth values inside the bbox
        # (more stable for thin dark objects like pens).
        x1b, y1b, x2b, y2b = bbox
        x1b, y1b = max(0, int(x1b)), max(0, int(y1b))
        x2b, y2b = min(w, int(x2b)), min(h, int(y2b))
        if x2b <= x1b or y2b <= y1b:
            # bbox invalid, fall back to center patch
            pass
        else:
            region = depth_image[y1b:y2b, x1b:x2b]
            valid = region[region > 0]
            if len(valid) > 5:  # at least 5 valid points
                return float(np.median(valid)) * depth_scale, True
    # Default: center patch
    y1, y2 = max(0, cy - patch), min(h, cy + patch + 1)
    x1, x2 = max(0, cx - patch), min(w, cx + patch + 1)
    region = depth_image[y1:y2, x1:x2]
    valid = region[region > 0]
    if len(valid) == 0:
        return 0.0, False
    return float(np.median(valid)) * depth_scale, True


def _pixel_to_cam(cx, cy, z_m, intrin):
    """Pixel + depth -> 3D point in camera frame (meters)."""
    X = (cx - intrin.ppx) * z_m / intrin.fx
    Y = (cy - intrin.ppy) * z_m / intrin.fy
    Z = z_m
    return float(X), float(Y), float(Z)


def _normalize_label(raw_label, target_classes):
    """
    Normalize the label:
    - If raw_label contains any target class word as a substring,
      return the longest matching one.
    - Otherwise return raw_label itself.
    Returns: (normalized_label, is_target)
    """
    matched = [t for t in target_classes if t in raw_label]
    if matched:
        return max(matched, key=len), True
    return raw_label, False


def _nms(boxes, scores, labels, score_threshold, iou_threshold):
    """Apply NMS to deduplicate detection results."""
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
    Main detection function. Input a single RGB+Depth frame; returns all detections.
    
    color_image: np.ndarray, BGR, shape (H, W, 3)
    depth_image: np.ndarray, uint16, shape (H, W); units determined by depth_scale
    intrin:      pyrealsense2 intrinsics object, with fx/fy/ppx/ppy fields
    depth_scale: float, depth_image * depth_scale = meters
    processor, model: the two objects returned by load_detector()
    text_prompt: str, Grounding DINO style, classes separated by ". "
    target_classes: set[str], target class words (matched as substrings against raw_label)
    
    Returns: List[Detection dict], see module docstring.
    """
    # 1. Model inference
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
    
    # 3. Assemble outputs
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
    """Draw detection results on the image (returns a new image; does not modify the original)."""
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
