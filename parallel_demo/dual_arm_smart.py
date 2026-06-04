"""
dual_arm_smart.py - 智能双臂调度
检测后自动选:
  - 两臂所有 picks 都自侧→自侧 box  → 并行
  - 任一臂有跨侧 box                → 顺序 (f8 先 f7 后)
"""

import sys, os, time, math, threading
import argparse
import numpy as np

sys.path.insert(0, os.path.expanduser("~/techin517"))
sys.path.insert(0, os.path.expanduser("~/techin517/cv_module"))

import importlib.util
sys.path.insert(0, os.path.expanduser('~/techin517/parallel_demo'))
from auto_assign_helper import build_unified_items, detect_unified, assign_to_arms


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


df8 = load_module("df8", "/home/ubuntu/techin517/dispatch_pick.py")
df7 = load_module("df7", "/home/ubuntu/techin517/dispatch_pick_f7.py")

from lerobot.robots.so101_follower.so101_follower import SO101Follower, SO101FollowerConfig


# ============ SharedTop ============
class SharedTop:
    def __init__(self, real_top):
        self.real = real_top
        self.latest = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            try:
                bgr, depth, intrin = self.real.read()
                with self.lock:
                    self.latest = (bgr.copy(), depth.copy(), intrin)
            except Exception:
                time.sleep(0.05)

    def read(self):
        while self.latest is None:
            time.sleep(0.01)
        with self.lock:
            bgr, depth, intrin = self.latest
            return bgr.copy(), depth.copy(), intrin

    def stop(self):
        self.running = False
        time.sleep(0.1)


# ============ Pick one ============
def _pick(follower, top, wrist, det, policies, df, label):
    item = det['item_config']
    base_xyz = det['base_xyz']
    box_xyz = det['box_xyz']
    act_bundle = policies[item['act_checkpoint']]

    print(f"  [{label}] Picking '{item['name']}' at base_xyz={base_xyz}")

    z_above = base_xyz[2] + df.APPROACH_HEIGHT
    seed = np.array([df.SAFE_HOME_RAD["shoulder_lift"], df.SAFE_HOME_RAD["elbow_flex"],
                     df.SAFE_HOME_RAD["wrist_flex"], df.SAFE_HOME_RAD["wrist_roll"]])
    joints_rad, ok = df.base_xyz_to_arm_joints(base_xyz[0], base_xyz[1], z_above, seed)
    if not ok:
        print(f"  [{label}] ✗ IK fail (above), skipping")
        return
    target_above = {f"{k}.pos": math.degrees(v) for k, v in joints_rad.items()}
    target_above["gripper.pos"] = math.degrees(df.GRIPPER_OPEN)
    df.ramp_to(follower, target_above, duration_s=2.5, label=f"{label}:above")

    print(f"  [{label}] ACT ({df.ACT_DURATION_S}s, task='{item['task_name']}')")
    df.run_act(follower, top, wrist, act_bundle, item['task_name'], df.ACT_DURATION_S)

    obs = follower.get_observation()
    current_gripper = obs.get("gripper.pos", math.degrees(df.GRIPPER_CLOSED))
    seed = np.array([
        math.radians(obs.get("shoulder_lift.pos", 0)),
        math.radians(obs.get("elbow_flex.pos", 0)),
        math.radians(obs.get("wrist_flex.pos", 0)),
        math.radians(obs.get("wrist_roll.pos", 0)),
    ])
    box_joints, ok = df.base_xyz_to_arm_joints(*box_xyz, seed)
    if not ok:
        print(f"  [{label}] ✗ Box IK fail")
        return
    target_box = {f"{k}.pos": math.degrees(v) for k, v in box_joints.items()}
    target_box["gripper.pos"] = current_gripper
    df.ramp_to(follower, target_box, duration_s=3.0, label=f"{label}:to_box")
    time.sleep(0.5)
    target_box["gripper.pos"] = math.degrees(df.GRIPPER_OPEN)
    follower.send_action(target_box)
    time.sleep(1.5)
    print(f"  [{label}] ✓ Released")

    df.go_to_safe_home(follower)


def worker(follower, top, wrist, targets, policies, df, label):
    for det in targets:
        _pick(follower, top, wrist, det, policies, df, label)
    print(f"  [{label}] ALL DONE")


# ============ IoU 跨类别去重 ============
def bbox_iou(b1, b2):
    """计算 IoMin (containment) = inter / min(area1, area2).
    比 IoU 更鲁棒: 小框被大框包含时 IoMin=1.0 (IoU 会很低)"""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / min(a1, a2)


def dedupe_iou(targets, threshold=0.7):
    """跨类别 IoU 去重, 同一物体被识别为多类时保留 score 最高的."""
    if len(targets) <= 1:
        return targets
    sorted_t = sorted(targets, key=lambda d: -d['score'])
    deduped = []
    for d in sorted_t:
        keep = True
        for k in deduped:
            iou = bbox_iou(d['bbox_xyxy'], k['bbox_xyxy'])
            if iou > threshold:
                d_name = d.get('unified_name') or d.get('item_config', {}).get('name', '?')
                k_name = k.get('unified_name') or k.get('item_config', {}).get('name', '?')
                print(f"    [IoU dedup] dropping {d_name} score={d['score']:.2f} "
                      f"(IoU={iou:.2f} with kept {k_name} score={k['score']:.2f})")
                keep = False
                break
        if keep:
            deduped.append(d)
    return deduped


# ============ Side analysis ============
def is_same_side(det, arm):
    """f8 的自侧是 electronics; f7 的自侧是 stationery"""
    cat = det['item_config']['category']
    if arm == 'f8':
        return cat == 'electronics'
    else:
        return cat == 'stationery'


def all_same_side(targets_f8, targets_f7):
    return (all(is_same_side(d, 'f8') for d in targets_f8) and
            all(is_same_side(d, 'f7') for d in targets_f7))


# ============ Main ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f8items", nargs="+", default=["battery", "earbuds"])
    parser.add_argument("--f7items", nargs="+", default=["pen", "glue"])
    parser.add_argument("--force_sequential", action="store_true",
                        help="强制顺序执行 (跳过并行判断)")
    parser.add_argument("--auto", action="store_true",
                        help="自动检测+分配 (不需 --f8items/--f7items)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Dual-Arm SMART Dispatch")
    print(f"  f8 items: {args.f8items}")
    print(f"  f7 items: {args.f7items}")
    print("=" * 60)

    # Connect followers
    print("\n[1/6] Connecting followers...")
    f8 = SO101Follower(SO101FollowerConfig(port=df8.FOLLOWER_PORT, id=df8.FOLLOWER_ID))
    f8.connect()
    f7 = SO101Follower(SO101FollowerConfig(port=df7.FOLLOWER_PORT, id=df7.FOLLOWER_ID))
    f7.connect()
    print(f"  ✓ {df8.FOLLOWER_ID} + {df7.FOLLOWER_ID}")

    print("\n[2/6] Opening D435i (SharedTop)...")
    real_top = df8.RealSenseTop(df8.TOP_SERIAL, df8.CAM_W, df8.CAM_H, df8.CAM_FPS)
    real_top.start()
    shared_top = SharedTop(real_top)
    time.sleep(1.5)
    print("  ✓ ready")

    print("\n[3/6] Opening wrist cams...")
    wrist8 = df8.WristCam(df8.WRIST_DEV, df8.CAM_W, df8.CAM_H, df8.CAM_FPS)
    wrist7 = df7.WristCam(df7.WRIST_DEV, df7.CAM_W, df7.CAM_H, df7.CAM_FPS)

    print("\n[4/6] Loading CV + ACT models...")
    from cv_module import load_detector
    processor_cv, model_cv = load_detector(device=df8.DEVICE)
    clip_clf = df8.ClipClassifier(device=df8.DEVICE)
    if args.auto:
        # auto 模式: 加载两臂所有可用 ACT
        policies_f8 = df8.load_act_policies(list(df8.ALL_ITEMS.keys()))
        policies_f7 = df7.load_act_policies(list(df7.ALL_ITEMS.keys()))
    else:
        policies_f8 = df8.load_act_policies(args.f8items)
        policies_f7 = df7.load_act_policies(args.f7items)
    print("  ✓ All loaded")

    try:
        print("\n[5/6] Initial poses (parallel)...")
        t1 = threading.Thread(target=df8.go_to_detect_home, args=(f8,))
        t2 = threading.Thread(target=df7.go_to_safe_home, args=(f7,))
        t1.start(); t2.start()
        t1.join(); t2.join()

        print("\n=== CV detect ===")
        if args.auto:
            print("  [AUTO MODE] detect all 4 items, auto-assign by bbox u-coordinate")
            unified = build_unified_items(df8, df7)
            all_dets = detect_unified(shared_top, processor_cv, model_cv, df8, unified)
            print("\n=== IoU dedup (cross-class) ===")
            all_dets = dedupe_iou(all_dets)
            print("\n=== Assign to arms ===")
            targets_f8, targets_f7 = assign_to_arms(all_dets, df8, df7)
        else:
            targets_f8 = df8.detect_all_objects(shared_top, processor_cv, model_cv, clip_clf, args.f8items)
            targets_f7 = df7.detect_all_objects(shared_top, processor_cv, model_cv, clip_clf, args.f7items)

        # IoU dedup (non-auto path)
        if not args.auto:
            print("\n=== IoU dedup ===")
            targets_f8 = dedupe_iou(targets_f8)
            targets_f7 = dedupe_iou(targets_f7)

        df8.go_to_safe_home(f8)

        def _sort_f8(d):
            name = d['item_config']['name']
            return (0 if name == 'pen' else 1 if name == 'earbuds' else 2, -d['base_xyz'][0])
        def _sort_f7(d):
            name = d['item_config']['name']
            return (0 if name == 'pen' else 1 if name == 'battery' else 2, -d['base_xyz'][0])
        targets_f8.sort(key=_sort_f8)
        targets_f7.sort(key=_sort_f7)

        print(f"\nf8 targets: {[d['item_config']['name'] for d in targets_f8]}")
        print(f"f7 targets: {[d['item_config']['name'] for d in targets_f7]}")

        # ============ 分组: same-side vs cross-side ============
        if args.force_sequential:
            # 强制顺序: 全部当 cross-side 处理
            f8_same, f8_cross = [], list(targets_f8)
            f7_same, f7_cross = [], list(targets_f7)
        else:
            f8_same = [d for d in targets_f8 if is_same_side(d, 'f8')]
            f8_cross = [d for d in targets_f8 if not is_same_side(d, 'f8')]
            f7_same = [d for d in targets_f7 if is_same_side(d, 'f7')]
            f7_cross = [d for d in targets_f7 if not is_same_side(d, 'f7')]

        print(f"\n  Phase 1 (SEQUENTIAL cross-side):")
        print(f"    f8: {[d['item_config']['name'] for d in f8_cross]}")
        print(f"    f7: {[d['item_config']['name'] for d in f7_cross]}")
        print(f"  Phase 2 (PARALLEL same-side):")
        print(f"    f8: {[d['item_config']['name'] for d in f8_same]}")
        print(f"    f7: {[d['item_config']['name'] for d in f7_same]}")

        T0 = time.time()

        # ============ Phase 1: 顺序抓 cross-side (f8 先 f7 后) ============
        if f8_cross or f7_cross:
            print(f"\n[6/6] → Phase 1: SEQUENTIAL (cross-side picks, f8 first then f7)")
            if f8_cross:
                worker(f8, shared_top, wrist8, f8_cross, policies_f8, df8, "f8")
                time.sleep(1.0)
            if f7_cross:
                worker(f7, shared_top, wrist7, f7_cross, policies_f7, df7, "f7")
            print(f"  Phase 1 done ({time.time()-T0:.1f}s)")

        # ============ Phase 2: 并行抓 same-side ============
        if f8_same or f7_same:
            print(f"\n[6/6] ★ Phase 2: PARALLEL (same-side picks) ★")
            T2 = time.time()
            threads = []
            if f8_same:
                tw_f8 = threading.Thread(target=worker, args=(f8, shared_top, wrist8, f8_same, policies_f8, df8, "f8"))
                threads.append(tw_f8)
            if f7_same:
                tw_f7 = threading.Thread(target=worker, args=(f7, shared_top, wrist7, f7_same, policies_f7, df7, "f7"))
                threads.append(tw_f7)
            for i, t in enumerate(threads):
                t.start()
                if i == 0 and len(threads) > 1:
                    time.sleep(0.3)  # stagger
            for t in threads:
                t.join()
            print(f"  Phase 2 done ({time.time()-T2:.1f}s)")

        print(f"\n  Total exec time: {time.time()-T0:.1f}s")

        print("\n" + "=" * 60)
        print("  ✓ Done")
        print("=" * 60)

    finally:
        print("\n[Cleanup]")
        try: shared_top.stop()
        except: pass
        try: real_top.stop()
        except: pass
        try: wrist8.stop()
        except: pass
        try: wrist7.stop()
        except: pass
        try: f8.disconnect()
        except: pass
        try: f7.disconnect()
        except: pass


if __name__ == "__main__":
    main()
