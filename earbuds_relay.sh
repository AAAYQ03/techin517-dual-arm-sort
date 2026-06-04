#!/bin/bash
LOG=~/techin517/train_earbuds_f8_wait.log
PEN_LOG=~/techin517/train_pen_f8.log

log_msg() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG"
}

# 清空旧 log
> "$LOG"

log_msg "=== 接力守护启动 ==="
log_msg "PID: $$"
log_msg "等待 act_pen_f8 完成..."
log_msg ""

# 阶段 1: 等 pen_f8 进程消失, 连续 3 次确认
CONSECUTIVE_GONE=0
while [ $CONSECUTIVE_GONE -lt 3 ]; do
    if pgrep -f "act_pen_f8" > /dev/null; then
        if [ $CONSECUTIVE_GONE -gt 0 ]; then
            log_msg "  ⚠️  进程又出现了 (假阴性), 重置计数"
        fi
        CONSECUTIVE_GONE=0
    else
        CONSECUTIVE_GONE=$((CONSECUTIVE_GONE + 1))
        log_msg "  pen_f8 进程消失 ${CONSECUTIVE_GONE}/3 次"
    fi
    sleep 30
done
log_msg "✓ 阶段 1: pen_f8 进程确认结束"
log_msg ""

# 阶段 2: 等 60 秒让 checkpoint 写盘
log_msg "阶段 2: 等待 60 秒让 checkpoint 写盘完成..."
sleep 60
log_msg "✓ 阶段 2: 等待结束"
log_msg ""

# 阶段 3: 检查 log 修改时间
LAST_MOD=$(stat -c %Y "$PEN_LOG" 2>/dev/null || echo 0)
NOW=$(date +%s)
DIFF=$((NOW - LAST_MOD))
log_msg "阶段 3: pen_f8 log 上次修改 ${DIFF} 秒前"
if [ $DIFF -lt 30 ]; then
    log_msg "  ⚠️  log 还在更新, 再等 60 秒..."
    sleep 60
fi
log_msg "✓ 阶段 3: log 确认稳定"
log_msg ""

# 阶段 4: 检查 GPU 显存
log_msg "阶段 4: 检查 GPU 显存..."
for i in 1 2 3; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d " ")
    log_msg "  GPU 显存 ${MEM} MiB (尝试 ${i}/3)"
    if [ "$MEM" -lt 5000 ]; then
        log_msg "  ✓ GPU 已空闲"
        break
    fi
    if [ $i -lt 3 ]; then
        log_msg "  ⚠️  GPU 还占用 > 5GB, 等 60 秒"
        sleep 60
    fi
done
log_msg ""

# 阶段 5: 验证 pen_f8 checkpoint 存在
PEN_CKPT=~/techin517/outputs/train/act_pen_f8/checkpoints/last/pretrained_model
if [ -f "$PEN_CKPT/model.safetensors" ] || [ -f "$PEN_CKPT/policy.safetensors" ]; then
    log_msg "✓ 阶段 5: pen_f8 checkpoint 存在"
else
    log_msg "⚠️  阶段 5: pen_f8 checkpoint 文件不在预期位置"
    log_msg "    查 $PEN_CKPT/:"
    ls -la "$PEN_CKPT" 2>&1 | tee -a "$LOG"
fi
log_msg ""

# 阶段 6: 启动 earbuds_f8 训练
log_msg "=== 启动 act_earbuds_f8 训练 ==="
log_msg ""

cd ~/techin517

HF_HOME=/home/ubuntu/techin517/huggingface \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
lerobot-train \
  --dataset.repo_id=ycui77/so101_earbuds_f8 \
  --policy.type=act \
  --policy.repo_id=ycui77/act_earbuds_f8 \
  --policy.push_to_hub=False \
  --policy.device=cuda \
  --output_dir=outputs/train/act_earbuds_f8 \
  --job_name=act_earbuds_f8 \
  --steps=80000 \
  --batch_size=8 \
  --save_freq=10000 \
  --log_freq=200 \
  --num_workers=2 2>&1 | tee -a "$LOG"

EXIT_CODE=$?
log_msg ""
log_msg "=== act_earbuds_f8 训练结束 (exit code: $EXIT_CODE) ==="
