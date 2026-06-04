#!/usr/bin/env python3
"""Reset follower8 to safe_home. 这版用度数."""
import math, time
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

# 5.10 PDF SAFE_HOME 是弧度. 转成度.
SAFE_HOME = {
    "shoulder_pan.pos":  math.degrees(-0.020),   # -1.15°
    "shoulder_lift.pos": math.degrees(-1.313),   # -75.2°
    "elbow_flex.pos":    math.degrees(+1.433),   # +82.1°
    "wrist_flex.pos":    math.degrees(+1.014),   # +58.1°
    "wrist_roll.pos":    math.degrees(+0.147),   # +8.4°
    "gripper.pos":       math.degrees(+1.700),   # +97.4° (OPEN)
}

cfg = SO101FollowerConfig(
    port='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0181426-if00',
    id='follower8',
    disable_torque_on_disconnect=False,
)
robot = SO101Follower(cfg)
robot.connect()

obs = robot.get_observation()
print("=== Current obs (臂'伸直了'状态) ===")
for k, v in obs.items():
    if k.endswith('.pos'):
        print(f"  {k:25s} = {v:+.3f}")

print("\n=== Target SAFE_HOME (度数) ===")
for k, v in SAFE_HOME.items():
    print(f"  {k:25s} = {v:+.2f}")

current = {k: obs[k] for k in SAFE_HOME if k in obs}
N, duration = 90, 3.0
print(f"\nRamping {N} steps over {duration}s...")
for step in range(N + 1):
    alpha = step / N
    action = {k: current[k] + alpha * (SAFE_HOME[k] - current[k]) for k in SAFE_HOME if k in current}
    robot.send_action(action)
    time.sleep(duration / N)

print("\nDone.")
robot.disconnect()
