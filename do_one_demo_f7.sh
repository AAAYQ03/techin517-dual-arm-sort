#!/bin/bash
# do_one_demo_f7.sh - 单条 demo for follower7 (right arm)
# 用法: ./do_one_demo_f7.sh <item>   例: ./do_one_demo_f7.sh pen

ITEM=${1:-pen}
REPO_ID="ycui77/so101_${ITEM}_f7"
DS_DIR="$HOME/techin517/huggingface/lerobot/$REPO_ID"

export DISPLAY=:2
export XAUTHORITY=$HOME/techin517/.Xauthority_host

echo "============================================================"
echo ">>> [Step 1/3] leader_to_follower (摆 leader 到 follower 当前姿态)"
echo "============================================================"
python3 ~/techin517/leader_to_follower_f7.py
if [ $? -ne 0 ]; then
    echo ">>> leader_to_follower FAILED, abort"
    exit 1
fi

echo ""
echo "============================================================"
echo ">>> [Step 2/3] 4 秒倒计时 (扶 leader 防重力掉)"
echo "============================================================"
for i in $(seq 4 -1 1); do
    printf "\r>>> 录制将在 %2d 秒后启动... " $i
    sleep 1
done
printf "\n>>> 启动 lerobot-record!\n\n"

# 判断 resume 模式
if [ -d "$DS_DIR/data/chunk-000" ] && [ -n "$(ls -A $DS_DIR/data/chunk-000/ 2>/dev/null)" ]; then
    RESUME_FLAG="--resume=true"
    echo ">>> Dataset 已存在, append 模式"
else
    RESUME_FLAG=""
    if [ -d "$DS_DIR" ]; then
        echo ">>> 清理残骸目录 $DS_DIR"
        rm -rf "$DS_DIR"
    fi
    echo ">>> Dataset 不存在, 新建模式"
fi

echo "============================================================"
echo ">>> [Step 3/3] lerobot-record (录完按 → 结束)"
echo "============================================================"
echo ""

HF_HOME=/home/ubuntu/techin517/huggingface lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6057204-if00 \
  --robot.id=gix-follower7 \
  --robot.cameras='{ wrist: {type: opencv, index_or_path: /dev/v4l/by-path/pci-0000:0e:00.0-usb-0:1.1:1.0-video-index0, width: 1280, height: 720, fps: 30, fourcc: MJPG}, top: {type: intelrealsense, serial_number_or_name: "243222072732", width: 1280, height: 720, fps: 30}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0181425-if00 \
  --teleop.id=gix-leader7 \
  --display_data=false \
  --dataset.repo_id=${REPO_ID} \
  --dataset.num_episodes=1 \
  --dataset.single_task="Pick up the ${ITEM}" \
  --dataset.push_to_hub=False \
  --dataset.episode_time_s=9999 \
  --dataset.reset_time_s=9999 \
  $RESUME_FLAG

echo ""
echo ">>> 完成"
