#!/bin/bash
# Usage: bash deploy_one_position.sh <pos_id>
# Example: bash deploy_one_position.sh pos1
#
# 一键流程: ROS launch → pipeline_to_above_only → kill ROS → leader_to_follower → 纯 ACT deploy

POS_ID=${1:-pos1}
DATASET_ID="ycui77/eval_battery_v2_${POS_ID}"
ACT_CHECKPOINT="outputs/train/act_battery_v2/checkpoints/last/pretrained_model"

echo "=========================================="
echo "  Deploy: position '${POS_ID}'"
echo "  Dataset: ${DATASET_ID}"
echo "=========================================="

# Setup
export DISPLAY=:3
export XAUTHORITY=$HOME/techin517/.Xauthority_host
sudo chmod 666 /dev/serial/by-id/* /dev/v4l/by-id/* 2>/dev/null

# [1/5] Clean residual
echo ""
echo "[1/5] Cleaning residual ROS processes..."
pkill -9 -f 'ros2' 2>/dev/null
pkill -9 -f 'move_group' 2>/dev/null
pkill -9 -f 'controller_manager' 2>/dev/null
pkill -9 -f 'realsense' 2>/dev/null
rm -rf /dev/shm/fastrtps_* 2>/dev/null
sleep 3

# [2/5] Start ROS launch
echo ""
echo "[2/5] Starting ROS launch (background)..."
cd ~/techin517/ros2_ws
source install/setup.bash
nohup ros2 launch soa_moveit_config soa_moveit_bringup.launch.py > /tmp/ros_launch_${POS_ID}.log 2>&1 &
LAUNCH_PID=$!
echo "  ROS launch PID: $LAUNCH_PID"
echo "  Waiting for ROS to be ready (max 90s)..."

ROS_READY=false
for i in {1..90}; do
    if grep -q "You can start planning now" /tmp/ros_launch_${POS_ID}.log 2>/dev/null; then
        echo "  ✓ ROS ready in ${i}s"
        ROS_READY=true
        break
    fi
    sleep 1
done

if [ "$ROS_READY" = false ]; then
    echo "  ✗ ROS launch failed in 90s. Check /tmp/ros_launch_${POS_ID}.log"
    exit 1
fi

echo "  Extra 5s for RealSense node..."
sleep 5

# [3/5] Pipeline to above
echo ""
echo "[3/5] Running pipeline_to_above_only..."
cd ~/techin517
python3 ~/techin517/pipeline_to_above_only.py
RESULT=$?
if [ $RESULT -ne 0 ]; then
    echo "  ✗ pipeline_to_above_only failed (exit code $RESULT)"
    echo "  Check /tmp/ros_launch_${POS_ID}.log for details"
    pkill -9 -f 'ros2' 2>/dev/null
    exit 1
fi
echo "  ✓ Follower at above position"

# [4/5] Kill ROS to release serial port
echo ""
echo "[4/5] Killing ROS to release serial port..."
pkill -9 -f 'ros2' 2>/dev/null
pkill -9 -f 'move_group' 2>/dev/null
pkill -9 -f 'controller_manager' 2>/dev/null
pkill -9 -f 'realsense' 2>/dev/null
rm -rf /dev/shm/fastrtps_* 2>/dev/null
sleep 3
echo "  ✓ ROS killed"

# [5/5] leader_to_follower + pure ACT
echo ""
echo "[5/5] Running leader_to_follower..."
python3 ~/techin517/leader_to_follower.py
sleep 1
echo "  ✓ Leader synced"

# Clean old dataset if exists (one-shot test, overwrite each time)
rm -rf ~/techin517/huggingface/lerobot/${DATASET_ID} 2>/dev/null

echo ""
echo "[5/5] Starting pure ACT deploy (20s)..."
echo "  Watch the arm — press Ctrl+C if it crashes"
echo ""

HF_HOME=/home/ubuntu/techin517/huggingface lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0181426-if00 \
  --robot.id=follower8 \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: /dev/v4l/by-id/usb-BC-231220-A_XWF-1080P-video-index0, width: 1280, height: 720, fps: 30, fourcc: MJPG}, top: {type: intelrealsense, serial_number_or_name: '243222072732', width: 1280, height: 720, fps: 30}}" \
  --display_data=false \
  --dataset.repo_id=${DATASET_ID} \
  --dataset.num_episodes=1 \
  --dataset.single_task="Pick up the battery" \
  --dataset.push_to_hub=False \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=5 \
  --policy.path=${ACT_CHECKPOINT}

echo ""
echo "=========================================="
echo "  Deploy ${POS_ID} done"
echo "  Reset: take battery off, place at next position"
echo "=========================================="
