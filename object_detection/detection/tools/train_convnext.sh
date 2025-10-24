#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/train_detect.py --config configs/convnext/convnext_s_fulltune.py --work-dir logs/convnext_s_fulltune --launcher none --gpu_ids 0   # --resume_from logs/convnext_s_correlated_shareA/iter_32000.pth