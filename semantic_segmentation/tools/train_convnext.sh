#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/train.py --config configs/mae/convnext_b_peft.py --work-dir logs/convnext_s_colora --launcher none --gpu_ids 0
