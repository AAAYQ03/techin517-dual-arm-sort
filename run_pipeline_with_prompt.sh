#!/bin/bash
# 用法: ./run_pipeline_with_prompt.sh <item>
#   item: pen | earbuds | glue | battery
ITEM=${1:-pen}

case "$ITEM" in
    pen)
        PROMPT="a black pen."
        TARGETS='{"pen", "marker"}'
        ;;
    earbuds)
        PROMPT="a rounded white case."
        TARGETS='{"case", "rounded white case"}'
        ;;
    glue)
        PROMPT="a glue stick."
        TARGETS='{"glue stick", "stick"}'
        ;;
    battery)
        PROMPT="a battery."
        TARGETS='{"battery"}'
        ;;
    *)
        echo "Unknown item: $ITEM"
        echo "Usage: $0 [pen|earbuds|glue|battery]"
        exit 1
        ;;
esac

echo "=== 切换 pipeline prompt 到 $ITEM ==="
echo "  TEXT_PROMPT = \"$PROMPT\""
echo "  TARGET_CLASSES = $TARGETS"

# 用 python 替换 prompt 行
PROMPT_VAL="$PROMPT" TARGETS_VAL="$TARGETS" python3 << 'PY_EOF'
import os, re
path = "/home/ubuntu/techin517/pipeline_to_above_only.py"
prompt = os.environ['PROMPT_VAL']
targets = os.environ['TARGETS_VAL']
with open(path) as f:
    src = f.read()
src = re.sub(r'TEXT_PROMPT\s*=\s*".*?"', f'TEXT_PROMPT    = "{prompt}"', src)
src = re.sub(r'TARGET_CLASSES\s*=\s*\{[^}]*\}', f'TARGET_CLASSES = {targets}', src)
with open(path, "w") as f:
    f.write(src)
print("✓ prompt 已更新")
PY_EOF

echo ""
echo "=== 验证 ==="
grep "TEXT_PROMPT\|TARGET_CLASSES" /home/ubuntu/techin517/pipeline_to_above_only.py

echo ""
echo "=== 启动 pipeline ==="
export DISPLAY=:2
python3 ~/techin517/pipeline_to_above_only.py
