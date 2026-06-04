# cv_pipeline.py - 交互式测试脚本
import pyrealsense2 as rs
import numpy as np
import cv2

from cv_module import load_detector, detect_objects, draw_detections

# ============ 任务配置 ============
TEXT_PROMPT = "a battery. a usb stick. a black pen."
TARGET_CLASSES = {"battery", "usb stick", "pen"}

# ============ 加载模型 ============
processor, model = load_detector(device="cuda")

# ============ RealSense ============
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

color_stream = profile.get_stream(rs.stream.color)
intrin = color_stream.as_video_stream_profile().get_intrinsics()
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
print(f"内参: fx={intrin.fx:.2f} fy={intrin.fy:.2f} cx={intrin.ppx:.2f} cy={intrin.ppy:.2f}")
print(f"深度单位: {depth_scale} 米/unit")
print("\n窗口已打开,按 'd' 检测一帧, 's' 保存当前检测图, 'q' 退出")

shot_idx = 0
last_display = None

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        cv2.imshow("RGB", color_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        if key == ord('s') and last_display is not None:
            shot_idx += 1
            fname = f"detection_shot{shot_idx}.png"
            cv2.imwrite(fname, last_display)
            print(f"  [已保存 {fname}]")
            continue
        if key != ord('d'):
            continue
        
        # ===== 触发检测 =====
        print("\n--- 检测中 ---")
        detections = detect_objects(
            color_image, depth_image, intrin, depth_scale,
            processor, model,
            text_prompt=TEXT_PROMPT,
            target_classes=TARGET_CLASSES,
            box_threshold=0.30,
        )
        
        if len(detections) == 0:
            print("  (没检测到任何物体)")
        for det in detections:
            cx, cy = det["pixel_center"]
            X, Y, Z = det["cam_xyz_m"]
            tag = "[TARGET]" if det["is_target"] else "[DISTRACTOR]"
            print(f"  {det['label']:15s} score={det['score']:.2f}  "
                  f"pixel=({cx},{cy})  "
                  f"cam_3d=({X*100:+6.1f}, {Y*100:+6.1f}, {Z*100:+6.1f}) cm  {tag}")
        
        display = draw_detections(color_image, detections)
        cv2.imshow("Detection", display)
        last_display = display
        cv2.waitKey(1)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("\n已退出")