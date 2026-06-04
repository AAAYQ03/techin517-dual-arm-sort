#!/bin/bash
# dual_arm_dispatch.sh - 双臂顺序 dispatch
# 流程: f7 抓 pen 送右盒 → f8 抓 battery 送左盒
#
# 用法: bash ~/techin517/dual_arm_dispatch.sh
#
# 前提:
#   - 桌面同时放: battery (f8 半区) + pen (f7 半区)
#   - 两个 motor + 两个 wrist cam + D435i 都在线
#   - 没有 ROS launch 在跑

# set -e removed  # 任何步骤失败立刻停

echo "============================================================"
echo "  Dual-arm dispatch start"
echo "============================================================"

# 清场, 防止之前 LeRobot 进程残留
pkill -9 -f 'pipeline_to' 2>/dev/null
pkill -9 -f 'test_pen' 2>/dev/null
pkill -9 -f 'test_box' 2>/dev/null
sleep 2

# 修权限
sudo chmod 666 /dev/serial/by-id/* /dev/v4l/by-path/* 2>/dev/null

echo ""
echo "============================================================"
echo "  Phase 1/2: follower7 抓 pen 送右盒"
echo "============================================================"
python3 ~/techin517/dispatch_pick_f7.py
PHASE1_STATUS=$?

if [ $PHASE1_STATUS -ne 0 ]; then
    echo "✗ Phase 1 failed (exit code $PHASE1_STATUS), abort"
    exit 1
fi

echo ""
echo "  Phase 1 done. Waiting 3s for hardware to settle..."
sleep 3

echo ""
echo "============================================================"
echo "  Phase 2/2: follower8 抓 battery 送左盒"
echo "============================================================"
python3 ~/techin517/dispatch_pick.py
PHASE2_STATUS=$?

if [ $PHASE2_STATUS -ne 0 ]; then
    echo "✗ Phase 2 failed (exit code $PHASE2_STATUS)"
    exit 1
fi

echo ""
echo "============================================================"
echo "  ✓ Dual-arm dispatch complete!"
echo "============================================================"
