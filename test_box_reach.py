#!/usr/bin/env python3
"""
test_box_reach.py - 测试某臂能否到达某个 base_link 坐标
用法:
  python3 ~/techin517/test_box_reach.py <follower7|follower8> <x> <y> <z>
例:
  python3 ~/techin517/test_box_reach.py follower7 0.22 0.10 0.10

坐标系: 该臂自己的 base_link 系
  x = 前方 (远离基座)
  y = 左 (+) / 右 (-)
  z = 上 (桌面上方)

流程:
  1. IK 计算是否可达
  2. 如果可达, 询问是否物理 ramp 过去
  3. ramp: safe_home → target → safe_home
"""
import sys, math, time
import numpy as np
from lerobot.robots.so101_follower.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot_kinematics import lerobot_IK, get_robot

ARMS = {
    "follower7": {
        "port": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6057204-if00",
        "id":   "gix-follower7",
    },
    "follower8": {
        "port": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0181426-if00",
        "id":   "follower8",
    },
}

URDF_TO_LEROBOT_DX = 0.030
URDF_TO_LEROBOT_DZ = 0.056
LEROBOT_ROLL  = 0.147
LEROBOT_PITCH = 1.30

SAFE_HOME_RAD = {
    "shoulder_pan":  -0.020,
    "shoulder_lift": -1.313,
    "elbow_flex":    +1.433,
    "wrist_flex":    +1.014,
    "wrist_roll":    +0.147,
}
GRIPPER_OPEN = 1.7


def base_to_arm(x, y, z, seed):
    shoulder_pan = math.atan2(y, x)
    horizontal = math.sqrt(x*x + y*y)
    target = np.array([horizontal - URDF_TO_LEROBOT_DX, 0.0, z - URDF_TO_LEROBOT_DZ,
                       LEROBOT_ROLL, LEROBOT_PITCH, 0.0])
    qpos, ok = lerobot_IK(seed, target, robot=get_robot('so101'))
    if not ok:
        return None, False
    return {
        "shoulder_pan":  float(shoulder_pan),
        "shoulder_lift": float(qpos[0]),
        "elbow_flex":    float(qpos[1]),
        "wrist_flex":    float(qpos[2]),
        "wrist_roll":    float(qpos[3]),
    }, True


def ramp_to(follower, target_deg, duration_s=3.0, hz=30):
    obs = follower.get_observation()
    start = {k: v for k, v in obs.items() if k.endswith('.pos')}
    common = [k for k in target_deg if k in start]
    n = int(duration_s * hz); dt = 1.0 / hz
    for i in range(1, n + 1):
        t = i / n
        interp = {k: start[k] * (1 - t) + target_deg[k] * t for k in common}
        follower.send_action(interp)
        time.sleep(dt)


def main():
    if len(sys.argv) != 5:
        print("用法: python3 test_box_reach.py <follower7|follower8> <x> <y> <z>")
        print("例:   python3 test_box_reach.py follower7 0.22 0.10 0.10")
        sys.exit(1)

    arm = sys.argv[1]
    x = float(sys.argv[2]); y = float(sys.argv[3]); z = float(sys.argv[4])

    if arm not in ARMS:
        print(f"未知 arm: {arm}")
        print(f"可用: {list(ARMS.keys())}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  测试 {arm} 到 base_link 坐标 ({x:+.3f}, {y:+.3f}, {z:+.3f})")
    print(f"{'='*60}\n")

    # Step 1: IK 检查
    print("[1/3] IK feasibility check...")
    seed = np.array([SAFE_HOME_RAD["shoulder_lift"], SAFE_HOME_RAD["elbow_flex"],
                     SAFE_HOME_RAD["wrist_flex"], SAFE_HOME_RAD["wrist_roll"]])
    joints, ok = base_to_arm(x, y, z, seed)
    if not ok:
        print(f"  ✗ IK FAIL — {arm} 够不到 ({x:+.3f}, {y:+.3f}, {z:+.3f})")
        sys.exit(0)

    print(f"  ✓ IK OK:")
    for k, v in joints.items():
        print(f"      {k:15s} = {math.degrees(v):+7.2f}°")

    # Step 2: 询问物理验证
    print()
    confirm = input(f"  要物理 ramp {arm} 过去验证吗? (y/n): ")
    if confirm.lower() != 'y':
        print("\n  跳过物理验证. 退出.")
        sys.exit(0)

    # Step 3: 物理 ramp
    cfg = ARMS[arm]
    print(f"\n[2/3] Connecting {arm}...")
    follower = SO101Follower(SO101FollowerConfig(
        port=cfg["port"], id=cfg["id"], disable_torque_on_disconnect=False))
    follower.connect()
    print(f"  ✓ Connected.")

    try:
        target_home = {f"{k}.pos": math.degrees(v) for k, v in SAFE_HOME_RAD.items()}
        target_home["gripper.pos"] = math.degrees(GRIPPER_OPEN)
        target_box = {f"{k}.pos": math.degrees(v) for k, v in joints.items()}
        target_box["gripper.pos"] = math.degrees(GRIPPER_OPEN)

        print(f"\n[3/3] ramp: current → safe_home → target → safe_home")

        print(f"  → safe_home (3s)...")
        ramp_to(follower, target_home, duration_s=3.0)
        time.sleep(1)

        print(f"  → target ({x:+.3f}, {y:+.3f}, {z:+.3f}) (3s)...")
        ramp_to(follower, target_box, duration_s=3.0)

        print(f"\n  ✓ 已到达. 检查物理位置是否对应你想放盒子的地方.")
        input(f"  按 ENTER 回 safe_home...")

        print(f"  → safe_home (3s)...")
        ramp_to(follower, target_home, duration_s=3.0)
        print(f"\n  ✓ 完成.")

    finally:
        follower.disconnect()
        print(f"  Disconnected.")


if __name__ == "__main__":
    main()
