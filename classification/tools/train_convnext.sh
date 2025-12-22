#!/usr/bin/env bash

CONFIG_PATH="configs/mae/convnext_peft.py"
BASE_WORKDIR="logs/convnext_s_coloranew_vtab"
TRAIN_SCRIPT="$(dirname "$0")/train.py" 

# List of VTAB-1k classification task names (no class numbers)
TASKS=(
    cifar
    caltech101
    dtd
    oxford_flowers102
    oxford_iiit_pet
    svhn
    sun397
    patch_camelyon
    eurosat
    resisc45
    diabetic_retinopathy
    clevr_count
    clevr_dist
    dmlab
    kitti
    dsprites_loc
    dsprites_ori
    smallnorb_azi
    smallnorb_ele
)

# Loop over each task and run 3 times
for TASK in "${TASKS[@]}"; do
    echo "========================"
    echo "Training on task: $TASK"
    echo "========================"
    for RUN in {1..3}; do
        echo "----- Run $RUN -----"
        WORKDIR="${BASE_WORKDIR}/${TASK}/run${RUN}"
        PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
        python "$TRAIN_SCRIPT" \
            --config "$CONFIG_PATH" \
            --work-dir "$WORKDIR" \
            --sub_name "$TASK" \
            --launcher none \
            --gpu_ids 0
    done
done


# python $(dirname "$0")/train.py --config configs/mae/convnext_peft.py --work-dir logs/convnext_s_fulltune_cifar --sub_name cifar --launcher none --gpu_ids 0  # --resume_from logs/convnext_s_correlated_shareA_indep_redu16/iter_16000.pth
