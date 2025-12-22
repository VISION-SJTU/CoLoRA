#!/usr/bin/env bash

CONFIG_PATH="configs/mae/convnext_peft.py"
BASE_WORKDIR="logs/convnext_s_colora_aircraft"
TRAIN_SCRIPT="$(dirname "$0")/train.py" 
RUN=1

echo "========================"
echo "Training on task: AIR"
echo "========================"
echo "----- Run AIR -----"
WORKDIR="${BASE_WORKDIR}/${TASK}/run${RUN}"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python "$TRAIN_SCRIPT" \
    --config "$CONFIG_PATH" \
    --work-dir "$WORKDIR" \
    --sub_name "$TASK" \
    --launcher none \
    --gpu_ids 0


# python $(dirname "$0")/train.py --config configs/mae/convnext_peft.py --work-dir logs/convnext_s_fulltune_cifar --sub_name cifar --launcher none --gpu_ids 0  # --resume_from logs/convnext_s_correlated_shareA_indep_redu16/iter_16000.pth
