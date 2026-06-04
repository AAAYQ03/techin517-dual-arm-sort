"""
测试：单进程能否同时:
  1. 打开 D435i (top cam)
  2. 连接 follower7 + follower8
  3. 打开 wrist7 + wrist8

如果这步 OK, 后续才能写 ACT 并行版.
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~/techin517"))
sys.path.insert(0, os.path.expanduser("~/techin517/cv_module"))

import importlib.util


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    print("=" * 60)
    print("  HARDWARE TEST: dual-arm + dual-wrist + shared D435i")
    print("=" * 60)

    # 加载两个 dispatch 模块 (复用 RealSenseTop / WristCam 类)
    print("\n[1/5] Loading dispatch modules...")
    df8 = load_module("df8", "/home/ubuntu/techin517/dispatch_pick.py")
    df7 = load_module("df7", "/home/ubuntu/techin517/dispatch_pick_f7.py")
    print("  ✓ df8 + df7 loaded")

    # 连接 follower8
    print("\n[2/5] Connecting follower8...")
    from lerobot.robots.so101_follower.so101_follower import SO101Follower, SO101FollowerConfig
    f8 = SO101Follower(SO101FollowerConfig(port=df8.FOLLOWER_PORT, id=df8.FOLLOWER_ID))
    f8.connect()
    print(f"  ✓ {df8.FOLLOWER_ID} connected")

    # 连接 follower7
    print("\n[3/5] Connecting follower7...")
    f7 = SO101Follower(SO101FollowerConfig(port=df7.FOLLOWER_PORT, id=df7.FOLLOWER_ID))
    f7.connect()
    print(f"  ✓ {df7.FOLLOWER_ID} connected")

    # 打开 D435i top cam
    print("\n[4/5] Opening D435i top cam...")
    top = df8.RealSenseTop(df8.TOP_SERIAL, df8.CAM_W, df8.CAM_H, df8.CAM_FPS)
    top.start()
    print("  ✓ D435i ready")

    # 打开两个 wrist cams
    print("\n[5/5] Opening wrist cams...")
    wrist8 = df8.WristCam(df8.WRIST_DEV, df8.CAM_W, df8.CAM_H, df8.CAM_FPS)
    wrist7 = df7.WristCam(df7.WRIST_DEV, df7.CAM_W, df7.CAM_H, df7.CAM_FPS)
    print("  ✓ wrist8 + wrist7 ready")

    # 读几帧验证不冲突
    print("\n=== Reading frames for 3 seconds (verify no conflict) ===")
    t_start = time.time()
    n_top = n_w8 = n_w7 = 0
    while time.time() - t_start < 3.0:
        bgr, _, _ = top.read(); n_top += 1
        if wrist8.read_rgb() is not None: n_w8 += 1
        if wrist7.read_rgb() is not None: n_w7 += 1
    print(f"  top frames:    {n_top} ({n_top/3.0:.1f} fps)")
    print(f"  wrist8 frames: {n_w8} ({n_w8/3.0:.1f} fps)")
    print(f"  wrist7 frames: {n_w7} ({n_w7/3.0:.1f} fps)")

    # 读 follower state 验证不冲突
    print("\n=== Reading follower states (verify no conflict) ===")
    s8 = f8.get_observation()
    s7 = f7.get_observation()
    print(f"  f8 shoulder_pan = {s8.get('shoulder_pan.pos', '?'):.2f} deg")
    print(f"  f7 shoulder_pan = {s7.get('shoulder_pan.pos', '?'):.2f} deg")

    print("\n[Cleanup]")
    top.stop()
    wrist8.stop()
    wrist7.stop()
    f8.disconnect()
    f7.disconnect()
    print("\n✓✓✓ ALL OK — proceed to ACT integration")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"\n✗ FAILED: {e}")
        sys.exit(1)
