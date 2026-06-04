#!/bin/bash
# dual_arm_full_dispatch.sh - 双臂顺序集成 dispatch (参数版)
# 用法:
#   ./dual_arm_full_dispatch.sh f7items=<items> f8items=<items>
# 例:
#   ./dual_arm_full_dispatch.sh f7items="battery" f8items="pen earbuds"
#   ./dual_arm_full_dispatch.sh f7items="pen" f8items="battery earbuds"
# 顺序: f8 先 f7 后
#
# f7 可选 items: pen, glue, battery   (f7 ALL_ITEMS, 没 earbuds 因为没训 act_earbuds_f7)
# f8 可选 items: battery, earbuds, pen (f8 ALL_ITEMS, 没 glue 因为没训 act_glue_f8)

F7_ITEMS=""
F8_ITEMS=""

for arg in "$@"; do
    case "$arg" in
        f7items=*)
            F7_ITEMS="${arg#f7items=}"
            ;;
        f8items=*)
            F8_ITEMS="${arg#f8items=}"
            ;;
        *)
            echo "未知参数: $arg"
            echo "用法: $0 f7items=\"<items>\" f8items=\"<items>\""
            echo "例:   $0 f7items=\"battery\" f8items=\"pen earbuds\""
            exit 1
            ;;
    esac
done

if [ -z "$F7_ITEMS" ] && [ -z "$F8_ITEMS" ]; then
    echo "至少需要给 f7items 或 f8items 之一"
    echo "用法: $0 f7items=\"<items>\" f8items=\"<items>\""
    exit 1
fi

echo "============================================================"
echo "  Dual-arm Full Dispatch"
echo "  F7 抓 (后): $F7_ITEMS"
echo "  F8 抓 (先): $F8_ITEMS"
echo "============================================================"

# 清场
pkill -9 -f 'ros2' 2>/dev/null
pkill -9 -f 'realsense' 2>/dev/null
pkill -9 -f 'controller_manager' 2>/dev/null
pkill -9 -f 'move_group' 2>/dev/null
pkill -9 -f 'static_transform' 2>/dev/null
pkill -9 -f 'robot_state' 2>/dev/null
pkill -9 -f 'rviz' 2>/dev/null
sleep 5
rm -rf /tmp/launch_params_* 2>/dev/null
sudo chmod 666 /dev/serial/by-id/* /dev/v4l/by-path/* /dev/video* 2>/dev/null

# Phase 1: f8 (先)
if [ -n "$F8_ITEMS" ]; then
    echo ""
    echo "============================================================"
    echo "  Phase 1/2: follower8 抓 [$F8_ITEMS]"
    echo "============================================================"
    python3 ~/techin517/dispatch_pick.py $F8_ITEMS
    F8_STATUS=$?
    if [ $F8_STATUS -ne 0 ]; then
        echo ""
        echo "✗ Phase 1 (f8) failed (exit code $F8_STATUS), abort"
        exit 1
    fi
    echo ""
    echo "  Phase 1 done. Waiting 5s..."
    sleep 5
fi

# Phase 2: f7 (后)
if [ -n "$F7_ITEMS" ]; then
    echo ""
    echo "============================================================"
    echo "  Phase 2/2: follower7 抓 [$F7_ITEMS]"
    echo "============================================================"
    python3 ~/techin517/dispatch_pick_f7.py $F7_ITEMS
    F7_STATUS=$?
    if [ $F7_STATUS -ne 0 ]; then
        echo ""
        echo "✗ Phase 2 (f7) failed (exit code $F7_STATUS)"
        exit 1
    fi
fi

echo ""
echo "============================================================"
echo "  ✓ All done!"
echo "============================================================"
