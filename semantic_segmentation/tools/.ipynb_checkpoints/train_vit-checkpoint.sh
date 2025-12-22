# #!/usr/bin/env bash

# # PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# # python $(dirname "$0")/train.py --config configs/mae/mae_vit_linear_probe.py --work-dir logs/vit_b_mae_pretrain_linear_probe --launcher none --gpu_ids 0

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python $(dirname "$0")/train.py --config configs/mae/mae_vit_linear_probe.py --work-dir logs/vit_l_mae_pretrain_linear_probe --launcher none --gpu_ids 0

#!/bin/bash

# === CONFIG ===
# Keyword to match the currently running job(s)
WATCH_KEYWORD="mae_vit_linear_probe.py"

# Command to run after the above jobs finish
NEXT_JOB="PYTHONPATH=\"\$(dirname \$0)/..\":\$PYTHONPATH \
python \$(dirname \"\$0\")/train.py \
  --config configs/mae/mae_vit_linear_probe.py \
  --work-dir logs/vit_l_mae_pretrain_linear_probe \
  --launcher none --gpu_ids 0"

# === MAIN LOOP ===
echo "Watching for processes with keyword: $WATCH_KEYWORD"
while pgrep -f "$WATCH_KEYWORD" > /dev/null; do
    echo "$(date): Training still running... waiting 5 minutes."
    sleep 300  # check every 5 minutes
done

echo "All matching processes finished. Starting next job..."
eval $NEXT_JOB