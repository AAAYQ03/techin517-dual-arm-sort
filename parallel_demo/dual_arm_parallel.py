"""
dual_arm_parallel.py - Single-process dual-arm parallel ACT dispatch
For EASY MODE ONLY (both arms pick from their own side, no collision risk).

Reuses functions from dispatch_pick.py / dispatch_pick_f7.py via importlib,
runs two ACT pick threads in parallel sharing one D435i top cam.
"""

import sys, os, time, math, threading
import argparse
import numpy as np

sys.path.insert(0, os.path.expanduser("~/techin517"))
sys.path.insert(0, os.path.expanduser("~/techin517/cv_module"))

import importlib.util


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
    """Background thread reads D435i continuously; threads share latest frame."""
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


# ============ Pick one object (extracted from dispatch_pick.py main) ============
def _pick(follower, top, wrist, det, policies, df, label):
    item = det['item_config']
    base_xyz = det['base_xyz']
    box_xyz = det['box_xyz']
    act_bundle = policies[item['act_checkpoint']]

    print(f"  [{label}] Picking '{item['name']}' at base_xyz={base_xyz}")

    # IK to above
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

    # ACT
    print(f"  [{label}] ACT ({df.ACT_DURATION_S}s, task='{item['task_name']}')")
    df.run_act(follower, top, wrist, act_bundle, item['task_name'], df.ACT_DURATION_S)

    # IK to box
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


# ============ Main ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f8items", nargs="+", default=["battery", "earbuds"])
    parser.add_argument("--f7items", nargs="+", default=["pen", "glue"])
    args = parser.parse_args()

    print("=" * 60)
    print("  Dual-Arm PARALLEL Dispatch (Easy mode)")
    print(f"  f8 items: {args.f8items}")
    print(f"  f7 items: {args.f7items}")
    print("=" * 60)

    # 1. Connect followers
    print("\n[1/6] Connecting followers...")
    f8 = SO101Follower(SO101FollowerConfig(port=df8.FOLLOWER_PORT, id=df8.FOLLOWER_ID))
    f8.connect()
    f7 = SO101Follower(SO101FollowerConfig(port=df7.FOLLOWER_PORT, id=df7.FOLLOWER_ID))
    f7.connect()
    print(f"  ✓ {df8.FOLLOWER_ID} + {df7.FOLLOWER_ID}")

    # 2. D435i shared
    print("\n[2/6] Opening D435i (SharedTop wrapper)...")
    real_top = df8.RealSenseTop(df8.TOP_SERIAL, df8.CAM_W, df8.CAM_H, df8.CAM_FPS)
    real_top.start()
    shared_top = SharedTop(real_top)
    time.sleep(1.5)  # wait for first frame
    print("  ✓ SharedTop ready")

    # 3. Wrist cams
    print("\n[3/6] Opening wrist cams...")
    wrist8 = df8.WristCam(df8.WRIST_DEV, df8.CAM_W, df8.CAM_H, df8.CAM_FPS)
    wrist7 = df7.WristCam(df7.WRIST_DEV, df7.CAM_W, df7.CAM_H, df7.CAM_FPS)

    # 4. CV + ACT loading
    print("\n[4/6] Loading CV + ACT models...")
    from cv_module import load_detector
    processor_cv, model_cv = load_detector(device=df8.DEVICE)
    clip_clf = df8.ClipClassifier(device=df8.DEVICE)
    policies_f8 = df8.load_act_policies(args.f8items)
    policies_f7 = df7.load_act_policies(args.f7items)
    print("  ✓ All models loaded")

    try:
        # 5. Initial pose (parallel)
        print("\n[5/6] Initial poses (parallel)...")
        t1 = threading.Thread(target=df8.go_to_detect_home, args=(f8,))
        t2 = threading.Thread(target=df7.go_to_safe_home, args=(f7,))
        t1.start(); t2.start()
        t1.join(); t2.join()

        # CV detect
        print("\n=== CV detect (both arms, same top frame snapshot) ===")
        targets_f8 = df8.detect_all_objects(shared_top, processor_cv, model_cv, clip_clf, args.f8items)
        targets_f7 = df7.detect_all_objects(shared_top, processor_cv, model_cv, clip_clf, args.f7items)

        # f8 → safe_home (f7 already at safe_home)
        df8.go_to_safe_home(f8)

        # Sort
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

        if not targets_f8 and not targets_f7:
            print("\n✗ Nothing to pick")
            return

        # 6. PARALLEL pick threads
        print("\n[6/6] Starting PARALLEL pick threads...")
        T0 = time.time()
        tw_f8 = threading.Thread(target=worker,
                                 args=(f8, shared_top, wrist8, targets_f8, policies_f8, df8, "f8"))
        tw_f7 = threading.Thread(target=worker,
                                 args=(f7, shared_top, wrist7, targets_f7, policies_f7, df7, "f7"))
        tw_f8.start()
        time.sleep(0.3)  # tiny stagger
        tw_f7.start()
        tw_f8.join()
        tw_f7.join()
        print(f"\n  Parallel exec time: {time.time()-T0:.1f}s")

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
