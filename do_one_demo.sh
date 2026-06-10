#!/bin/bash
# do_one_demo.sh - Single demo (leader_to_follower + 10s + lerobot-record)
# Automatically detects resume / Automatically cleans up leftovers

ITEM=${1:-battery}
REPO_ID="ycui77/so101_${ITEM}_f8"
DS_DIR="$HOME/techin517/huggingface/lerobot/$REPO_ID"

export DISPLAY=:2
export XAUTHORITY=$HOME/techin517/.Xauthority_host

echo "============================================================"
echo ">>> [Step 1/3] leader_to_follower"
echo "============================================================"
python3 ~/techin517/leader_to_follower.py
if [ $? -ne 0 ]; then
    echo ">>> leader_to_follower FAILED, abort"
    exit 1
fi

echo ""
echo "============================================================"
echo ">>> [Step 2/3] 4-second countdown (support the leader to prevent falling due to gravity)"
echo "============================================================"
for i in $(seq 4 -1 1); do
    printf "\r>>> Recording will start in %2d seconds.... " $i
    sleep 1
done
printf "\n>>> Launching lerobot-record!\n\n"

# Determine resume mode
if [ -d "$DS_DIR/data/chunk-000" ] && [ -n "$(ls -A $DS_DIR/data/chunk-000/ 2>/dev/null)" ]; then
    RESUME_FLAG="--resume=true"
    echo ">>> Dataset 已存在, append 模式"
else
    RESUME_FLAG=""
    # Clean up debris in the top-level directory (previous failures may have left behind empty skeletons)
    if [ -d "$DS_DIR" ]; then
        echo ">>> Clean up debris $DS_DIR"
        rm -rf "$DS_DIR"
    fi
    echo ">>> Dataset does not exist; creating new mode."
fi

echo "============================================================"
echo ">>> [Step 3/3] lerobot-record"
echo "============================================================"
echo ""

HF_HOME=/home/ubuntu/techin517/huggingface lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0181426-if00 \
  --robot.id=follower8 \
  --robot.cameras='{ wrist: {type: opencv, index_or_path: /dev/v4l/by-path/pci-0000:0b:00.0-usb-0:3:1.0-video-index0, width: 1280, height: 720, fps: 30, fourcc: MJPG}, top: {type: intelrealsense, serial_number_or_name: "243222072732", width: 1280, height: 720, fps: 30}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0183407-if00 \
  --teleop.id=leader8 \
  --display_data=false \
  --dataset.repo_id=${REPO_ID} \
  --dataset.num_episodes=1 \
  --dataset.single_task="Pick up the ${ITEM}" \
  --dataset.push_to_hub=False \
  --dataset.episode_time_s=9999 \
  --dataset.reset_time_s=9999 \
  $RESUME_FLAG

echo ""
echo ">>> Finish"
