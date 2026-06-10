#!/usr/bin/env python3
"""
leader_to_follower_f7.py - 让 leader 主动 servo 到 follower 当前位置, 然后 torque-off
解决 lerobot-record 启动时 follower 被 snap 到 leader 的痛点

用法 (在桀 ROS 之后, 启动 lerobot-record 之前):
    python3 ~/techin517/data_collection/leader_to_follower_f7.py

前置:
    - ROS launch 已杀干净 (释放 dynamixel 串口)
    - follower 通过 dynamixel 硬件 torque 保持在 above 位置 (杀 ROS 不丢)
"""
import time
import sys
from lerobot.robots.so101_follower.so101_follower import SO101Follower, SO101FollowerConfig

FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6057204-if00"
LEADER_PORT   = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0181425-if00"
FOLLOWER_ID = "gix-follower7"
LEADER_ID   = "gix-leader7"

RAMP_DURATION_S = 3.0
RAMP_HZ = 30


def main():
    print(">>> [1/4] Connecting follower (read-only)...")
    follower = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID, disable_torque_on_disconnect=False))
    follower.connect()
    obs_f = follower.get_observation()
    target = {k: v for k, v in obs_f.items() if k.endswith('.pos')}
    print(f">>> Follower current joints:")
    for k, v in target.items():
        print(f"      {k:25s} = {v:+.4f}")
    if not target:
        print("ERROR: 没有 .pos 键. obs keys:", list(obs_f.keys()))
        follower.disconnect()
        sys.exit(1)

    print(">>> [2/4] Connecting leader as a follower (复用 SO101Follower 类发 action)...")
    leader = SO101Follower(SO101FollowerConfig(port=LEADER_PORT, id=LEADER_ID))
    leader.connect()
    obs_l = leader.get_observation()
    start = {k: v for k, v in obs_l.items() if k.endswith('.pos')}
    print(f">>> Leader start joints:")
    for k, v in start.items():
        print(f"      {k:25s} = {v:+.4f}  (delta={target.get(k,0)-v:+.4f})")

    # 检查 delta 不要太离谱
    max_delta = max(abs(target[k] - start.get(k, 0)) for k in target)
    print(f">>> Max joint delta = {max_delta:+.4f} rad")
    if max_delta > 1.5:
        print(f"WARNING: delta 很大 (>1.5 rad), leader 移动幅度大, 注意避免碰撞")
        input(">>> 按 Enter 继续 (Ctrl+C 取消)...")

    print(f">>> [3/4] Ramping leader to follower over {RAMP_DURATION_S}s ({int(RAMP_DURATION_S*RAMP_HZ)} steps)...")
    n_steps = int(RAMP_DURATION_S * RAMP_HZ)
    dt = 1.0 / RAMP_HZ
    common_keys = [k for k in target if k in start]
    for i in range(1, n_steps + 1):
        t = i / n_steps
        interp = {k: start[k] * (1 - t) + target[k] * t for k in common_keys}
        leader.send_action(interp)
        time.sleep(dt)

    obs_l2 = leader.get_observation()
    final = {k: v for k, v in obs_l2.items() if k.endswith('.pos')}
    print(f">>> Leader final joints:")
    for k, v in final.items():
        delta = v - target.get(k, 0)
        print(f"      {k:25s} = {v:+.4f}  (vs target delta={delta:+.4f})")

    print(">>> [4/4] Disconnecting (torque off both)...")
    leader.disconnect()
    follower.disconnect()

    print()
    print("="*60)
    print("DONE. Leader 跟 follower 同位置, 都 torque off.")
    print("立刻启动录制 (snap 距离 ≈ 0):")
    print("  bash ~/techin517/record_one_demo.sh battery")
    print("="*60)


if __name__ == "__main__":
    main()
