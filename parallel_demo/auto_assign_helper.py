"""auto_assign_helper.py - dual_arm_smart 的 --auto 模式辅助函数
关键: 每个物品按归属臂用各自标定转 base_xyz (f7/f8 hand-eye 不同)"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/techin517/cv_module"))
from cv_module import detect_objects


def build_unified_items(df8, df7):
    items = {}
    for name in ["pen", "glue", "battery", "earbuds"]:
        f8_item = df8.ALL_ITEMS.get(name)
        f7_item = df7.ALL_ITEMS.get(name)
        ref = f8_item or f7_item
        if ref is None:
            continue
        items[name] = {
            "name": name,
            "dino_prompt": ref["dino_prompt"],
            "dino_targets": ref["dino_targets"],
            "category": ref["category"],
            # battery 特殊: 取 f7/f8 中较低的 threshold (f7=0.30 < f8=0.45)
            "score_threshold": (
                min(f8_item.get("score_threshold", 0.45) if f8_item else 1.0,
                    f7_item.get("score_threshold", 0.45) if f7_item else 1.0)
                if name == "battery" else
                ref.get("score_threshold", 0.25)
            ),
            "task_name": ref["task_name"],
            "f8_act": f8_item["act_checkpoint"] if f8_item else None,
            "f7_act": f7_item["act_checkpoint"] if f7_item else None,
        }
    return items


def detect_unified(top, processor_cv, model_cv, df8, unified_items):
    """只 detect + 颜色 prior, 不转 base_xyz (留给 assign 时按臂转)"""
    import cv2, time as _time
    print("\n  Snapshot from D435i (unified detect)...")
    bgr, depth_raw, intrin = top.read()
    _bgr_for_debug = bgr.copy()
    all_dets = []
    for item_name, cfg in unified_items.items():
        print(f"  DINO detect: {cfg['dino_prompt']!r}")
        dets = detect_objects(
            color_image=bgr, depth_image=depth_raw, intrin=intrin,
            depth_scale=df8.DEPTH_SCALE, processor=processor_cv, model=model_cv,
            text_prompt=cfg['dino_prompt'], target_classes=cfg['dino_targets'],
            box_threshold=cfg['score_threshold'], device=df8.DEVICE,
        )
        valid = [d for d in dets if d.get("is_target") and d.get("depth_valid")]
        for d in valid:
            cp_pass, cp_reason = df8.color_prior_pass(bgr, d['bbox_xyxy'], item_name)
            if not cp_pass:
                print(f"    [SKIP] color_prior fail ({cp_reason}): score={d['score']:.2f}")
                continue
            d['unified_name'] = item_name
            d['unified_cfg'] = cfg
            print(f"    [DET] {item_name} score={d['score']:.2f} bbox={d['bbox_xyxy']} cam_xyz={d['cam_xyz_m']}")
            all_dets.append(d)
    # 保存 debug 可视化图
    for d in all_dets:
        x1, y1, x2, y2 = [int(v) for v in d['bbox_xyxy']]
        cv2.rectangle(_bgr_for_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d['unified_name']}:{d['score']:.2f}"
        cv2.putText(_bgr_for_debug, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    ts = int(_time.time())
    out_path = f"/home/ubuntu/techin517/debug_detect_auto_{ts}.png"
    cv2.imwrite(out_path, _bgr_for_debug)
    print(f"  [DEBUG] 检测可视化图: {out_path}")
    return all_dets


def assign_to_arms(deduped, df8, df7):
    """按 bbox u 判断臂, 用对应臂的标定转 base_xyz."""
    targets_f8, targets_f7 = [], []
    BOX_F8_SAME = (0.155, +0.058, 0.10)
    BOX_F8_CROSS = (0.210, +0.160, 0.12)
    BOX_F7_SAME = (0.155, -0.058, 0.10)
    BOX_F7_CROSS = (0.195, -0.180, 0.12)

    for d in deduped:
        cfg = d['unified_cfg']
        name = d['unified_name']
        x1, _, x2, _ = d['bbox_xyxy']
        u_center = (x1 + x2) / 2.0
        cat = cfg['category']

        # 1. 按 u 判断 preferred 臂
        if u_center < 640:
            preferred = 'f8'; other = 'f7'
            df_pref = df8; df_other = df7
        else:
            preferred = 'f7'; other = 'f8'
            df_pref = df7; df_other = df8

        prefer_act = cfg[preferred + '_act']
        other_act = cfg[other + '_act']

        # 2. 决定 arm (preferred 优先, 没 ACT 就跨臂)
        if prefer_act:
            arm = preferred
            df_use = df_pref
            act_ckpt = prefer_act
        elif other_act:
            arm = other
            df_use = df_other
            act_ckpt = other_act
            print(f"  [cross-arm] {preferred} no ACT for {name}, fallback to {arm}")
        else:
            print(f"  [SKIP] {name} - no ACT for either arm")
            continue

        # 3. 用对应臂的标定转 base_xyz
        base_xyz = df_use.cam_xyz_to_base(d['cam_xyz_m'])
        if not df_use.is_valid_position(base_xyz):
            print(f"  [SKIP] {name} out of {arm} workspace: base_xyz={base_xyz}")
            continue

        # 4. 确定 box (用 arm 视角的 same/cross)
        if arm == 'f8':
            box = BOX_F8_SAME if cat == 'electronics' else BOX_F8_CROSS
        else:
            box = BOX_F7_SAME if cat == 'stationery' else BOX_F7_CROSS

        d['base_xyz'] = base_xyz
        d['item_config'] = {
            'name': name,
            'category': cat,
            'task_name': cfg['task_name'],
            'act_checkpoint': act_ckpt,
        }
        d['box_xyz'] = box
        d['arm'] = arm

        if arm == 'f8':
            targets_f8.append(d)
        else:
            targets_f7.append(d)
        print(f"  [assign] {name} u={u_center:.0f} -> {arm} base_xyz={base_xyz} box={box}")

    return targets_f8, targets_f7
