#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/train.py --config configs/mae/convnext_linear_probe.py --work-dir logs/convnext_dinov3_pretrain_linear_probe --launcher none --gpu_ids 0
