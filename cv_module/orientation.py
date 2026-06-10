"""
orientation.py - Object orientation extraction from RGB crops.

Used by the CV pipeline to provide 6-DOF pose (position + orientation) to IK,
not just position. This addresses a limitation in v1 where the dispatcher only
passed (x, y, z) to IK, causing grasp issues on elongated objects (pen, glue
stick) because the gripper had no information about the object's long-axis
direction.

Method: For each detected bbox, we segment the object via adaptive threshold,
find the largest contour, fit a minimum-area rotated rectangle, and extract
the principal-axis angle. Confidence is reported as a function of aspect
ratio (low for symmetric objects, high for elongated ones).
"""

import cv2
import numpy as np
import math


def extract_orientation_pca(bgr, bbox, debug=False):
    """
    Extract object orientation (image-plane angle) via min-area rotated rect.

    Args:
        bgr: full BGR image (H, W, 3)
        bbox: (x1, y1, x2, y2) pixel bbox

    Returns:
        (angle_deg, confidence)
            angle_deg : float in [-90, 90], rotation of the long axis in image
                        coords; None if extraction failed.
            confidence : float in [0, 1], based on aspect ratio.
                         <0.3 means symmetric (orientation not reliable).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h_img, w_img = bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    if x2 - x1 < 5 or y2 - y1 < 5:
        return None, 0.0

    crop = bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold (lighting-robust)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 5,
    )
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 20:
        return None, 0.0

    rect = cv2.minAreaRect(largest)
    (cx, cy), (w, h), angle = rect

    # cv2.minAreaRect angle convention: (-90, 0], width is "first" side.
    # Normalize so `angle` is the LONG axis in [-90, 90].
    if w < h:
        angle = angle + 90.0
    if angle > 90.0:
        angle -= 180.0

    long_side = max(w, h)
    short_side = max(min(w, h), 1.0)
    aspect = long_side / short_side
    confidence = float(np.clip((aspect - 1.0) / 4.0, 0.0, 1.0))

    return float(angle), confidence


def image_angle_to_base_yaw(image_angle_deg):
    """
    Convert image-plane angle (deg) to base_link yaw (rad).

    First-order approximation for an overhead D435i mount where image x-axis
    is roughly aligned with base_link -Y. For perfect accuracy this should
    consult the actual hand-eye TF, but this approximation is enough for
    wrist_roll alignment within ±5°.

    Returns yaw in [-pi/2, pi/2].
    """
    if image_angle_deg is None:
        return None
    yaw_rad = math.radians(90.0 - image_angle_deg)
    while yaw_rad > math.pi / 2:
        yaw_rad -= math.pi
    while yaw_rad < -math.pi / 2:
        yaw_rad += math.pi
    return yaw_rad


# Per-item switch: only elongated objects benefit from orientation.
# Symmetric objects keep wrist_roll at the IK default.
USE_ORIENTATION = {
    "pen":      True,
    "glue":     True,
    "battery":  False,
    "earbuds":  False,
}


def get_object_yaw_for_dispatch(bgr, bbox, item_name,
                                 min_confidence=0.3):
    """
    Top-level helper called by dispatch_pick.py.

    Returns:
        yaw_rad : object yaw in base_link [-pi/2, pi/2], or
        None if orientation should not be used (symmetric object,
              low-confidence detection, or extraction failed).
    """
    if not USE_ORIENTATION.get(item_name, False):
        return None

    angle_deg, conf = extract_orientation_pca(bgr, bbox)
    if angle_deg is None or conf < min_confidence:
        return None

    return image_angle_to_base_yaw(angle_deg)
