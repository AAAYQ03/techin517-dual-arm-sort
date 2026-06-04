#!/bin/bash
# dual_arm_cross_dispatch.sh - 双臂顺序跨侧 dispatch 测试
# 流程:
#   Phase 1: f7 抓 pen 送跨侧左盒 (0.195, -0.180, 0.12)
#   Phase 2: f8 抓 earbuds 送跨侧右盒 (0.210, +0.160, 0.12)

echo "============================================================"
echo "  Dual-arm CROSS-SIDE dispatch test"
echo "============================================================"

# 清场
pkill -9 -f 'pipeline_to' 2>/dev/null
pkill -9 -f 'test_pen' 2>/dev/null
pkill -9 -f 'test_box' 2>/dev/null
sleep 2

# 修权限
sudo chmod 666 /dev/serial/by-id/* /dev/v4l/by-path/* 2>/dev/null

echo ""
echo "============================================================"
echo "  Phase 1/2: follower7 抓 pen → 跨侧左盒"
echo "============================================================"
python3 ~/techin517/dispatch_pick_f7_cross.py
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
echo "  Phase 2/2: follower8 抓 earbuds → 跨侧右盒"
echo "============================================================"
python3 ~/techin517/dispatch_pick_f8_cross.py
PHASE2_STATUS=$?

if [ $PHASE2_STATUS -ne 0 ]; then
    echo "✗ Phase 2 failed (exit code $PHASE2_STATUS)"
    exit 1
fi

echo ""
echo "============================================================"
echo "  ✓ Dual-arm CROSS-SIDE dispatch complete!"
echo "============================================================"
