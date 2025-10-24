#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_detect.py $CONFIG --launcher pytorch ${@:3} --work-dir logs/convnext_xl_fulltune_rerun # --resume-from logs/convnext_xl_fulltune/epoch_36.pth