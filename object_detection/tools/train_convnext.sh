#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/train_detect.py --config configs/convnext/convnext_peft.py --work-dir logs/convnext_s_hira --launcher none --gpu_ids 0