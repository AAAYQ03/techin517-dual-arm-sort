#!/bin/bash
# do_one_demo.sh - 单条 demo (leader_to_follower + 10s + lerobot-record)
# 自动判断 resume / 自动清残骸

ITEM=${1:-battery}
REPO_ID="ycui77/so101_${ITEM}_f8"
DS_DIR="$HOME/techin517/huggingface/lerobot/$REPO_ID"

export DISPLAY=:2
export XAUTHORITY=$HOME/techin517/.Xauthority_host

echo "============================================================"
echo ">>> [Step 1/3] leader_to_follower"
echo "============================================================"
python3 ~/techin517/data_collection/leader_to_follower.py
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
    # 清残骸顶层目录 (上次失败可能留下空骨架)
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
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0181426-if00 \
  --robot.id=follower8 \
  --robot.cameras='{ wrist: {type: opencv, index_or_path: /dev/v4l/by-path/pci-0000:0b:00.0-usb-0:3:1.0-video-index0, width: 1280, height: 720, fps: 30, fourcc: MJPG}, top: {type: intelrealsense, [...]' \
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
echo ">>> 完成"
