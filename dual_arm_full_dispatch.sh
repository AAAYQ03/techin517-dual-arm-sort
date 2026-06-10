#!/bin/bash
# dual_arm_full_dispatch.sh - Sequential dual-arm integrated dispatch (parameterized)
# Usage:
#   ./dual_arm_full_dispatch.sh f7items=<items> f8items=<items>
# Examples:
#   ./dual_arm_full_dispatch.sh f7items="battery" f8items="pen earbuds"
#   ./dual_arm_full_dispatch.sh f7items="pen" f8items="battery earbuds"
# Order: f8 first, f7 second
#
# f7 available items: pen, glue, battery   (f7 ALL_ITEMS; earbuds excluded because act_earbuds_f7 was not trained)
# f8 available items: battery, earbuds, pen (f8 ALL_ITEMS; glue excluded because act_glue_f8 was not trained)
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
            echo "Unknown argument: $arg"
            echo "Usage: $0 f7items=\"<items>\" f8items=\"<items>\""
            echo "Example: $0 f7items=\"battery\" f8items=\"pen earbuds\""
            exit 1
            ;;
    esac
done
if [ -z "$F7_ITEMS" ] && [ -z "$F8_ITEMS" ]; then
    echo "At least one of f7items or f8items is required"
    echo "Usage: $0 f7items=\"<items>\" f8items=\"<items>\""
    exit 1
fi
echo "============================================================"
echo "  Dual-arm Full Dispatch"
echo "  F7 picks (second): $F7_ITEMS"
echo "  F8 picks (first):  $F8_ITEMS"
echo "============================================================"
# Cleanup
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
# Phase 1: f8 (first)
if [ -n "$F8_ITEMS" ]; then
    echo ""
    echo "============================================================"
    echo "  Phase 1/2: follower8 picks [$F8_ITEMS]"
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
# Phase 2: f7 (second)
if [ -n "$F7_ITEMS" ]; then
    echo ""
    echo "============================================================"
    echo "  Phase 2/2: follower7 picks [$F7_ITEMS]"
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
