#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29501}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py --config $CONFIG --launcher pytorch ${@:3} --work-dir logs/convnext_xl_fulltune_ourpretrain_896_lr8e_5_rerun  #  --resume_from logs/convnext_xl_fulltune_ourpretrain_896_lr8e_5/iter_144000.pth
