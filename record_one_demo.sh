#!/bin/bash
# record_one_demo.sh - 录 1 条 v2 格式 demo（起点 = IK 给的 above 位置）
# 用法：bash record_one_demo.sh <物体名>
# 例：  bash record_one_demo.sh battery
# 数据集名自动 = ycui77/so101_<物体>_v2

set -e

ITEM=${1:-battery}
REPO_ID="ycui77/so101_${ITEM}_v2"

echo "========================================"
echo "录 1 条 demo"
echo "  物体:    ${ITEM}"
echo "  数据集:  ${REPO_ID}"
echo "========================================"
echo ""
echo "确认清单:"
echo "  [1] follower 在 IK 给的 above 位置 (跑过 pipeline_to_above_only)"
echo "  [2] ROS 已关闭 (pkill 干净了)"
echo "  [3] leader 已摆到接近 follower 当前姿态"
echo "  [4] 物体在桌面预定位置"
echo ""
read -p "按 Enter 启动 lerobot-record (启动瞬间会 snap follower) ..."

# 修权限
sudo chmod 666 /dev/serial/by-id/* /dev/v4l/by-id/* 2>/dev/null || true

HF_HOME=/home/ubuntu/techin517/huggingface lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0181426-if00 \
  --robot.id=follower8 \
  --robot.cameras='{ wrist: {type: opencv, index_or_path: /dev/v4l/by-id/usb-BC-231220-A_XWF-1080P-video-index0, width: 1280, height: 720, fps: 30, fourcc: MJPG}, top: {type: intelrealsense, serial_number_or_name: "243222072732", width: 1280, height: 720, fps: 30}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0183407-if00 \
  --teleop.id=leader8 \
  --display_data=false \
  --dataset.repo_id=${REPO_ID} \
  --dataset.num_episodes=1 \
  --dataset.single_task="Pick up the ${ITEM}" \
  --dataset.push_to_hub=False \
  --dataset.episode_time_s=9999 \
  --dataset.reset_time_s=9999

echo ""
echo "录制完成。${REPO_ID} 已 +1 条 episode。"
