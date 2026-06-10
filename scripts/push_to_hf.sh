#!/bin/bash
# push_to_hf.sh - 把当前 ACT 模型 + 数据集 push 到 HuggingFace
set -u
HF_USER="ycui77"

push_model() {
    local NAME=$1
    local LOCAL_DIR=~/techin517/outputs/train/${NAME}/checkpoints/last/pretrained_model
    if [ ! -d "$LOCAL_DIR" ]; then
        echo "⚠️  $NAME: 本地目录不存在, 跳过"
        return
    fi
    echo ""
    echo "============================================================"
    echo "📤 Push model: ${HF_USER}/${NAME}"
    echo "   from $LOCAL_DIR"
    echo "============================================================"
    hf upload "${HF_USER}/${NAME}" "$LOCAL_DIR" --repo-type model
}

push_dataset() {
    local NAME=$1
    local LOCAL_DIR=~/techin517/huggingface/lerobot/${HF_USER}/${NAME}
    if [ ! -d "$LOCAL_DIR" ]; then
        echo "⚠️  $NAME: 本地目录不存在, 跳过"
        return
    fi
    echo ""
    echo "============================================================"
    echo "📤 Push dataset: ${HF_USER}/${NAME}"
    echo "   from $LOCAL_DIR"
    echo "============================================================"
    hf upload "${HF_USER}/${NAME}" "$LOCAL_DIR" --repo-type dataset
}

echo "============================================================"
echo "  Push 当前 ACT 模型 + 数据集到 HuggingFace"
echo "  HF User: $HF_USER"
echo "============================================================"

# 模型 (训练完成的, 不含 pen_f8_v2 因为还在训练)
push_model act_pen_f7
push_model act_battery_f7
push_model act_glue_f7
push_model act_battery_f8
push_model act_earbuds_f8

# 数据集 (当前活跃版本)
push_dataset so101_pen_f7
push_dataset so101_battery_f7
push_dataset so101_glue_f7
push_dataset so101_battery_f8
push_dataset so101_earbuds_f8
push_dataset so101_pen_f8   # 这个是新录的 v2 数据集

echo ""
echo "============================================================"
echo "✅ 全部 push 完成!"
echo "   查看: https://huggingface.co/${HF_USER}"
echo "============================================================"
