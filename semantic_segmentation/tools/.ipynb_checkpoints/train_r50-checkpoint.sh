#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/train.py --config configs/mae/res50_linear_probe_512_stride_80k_ade20k.py --work-dir logs/resnet_imgnet_pretrain_linear_probe --launcher none --gpu_ids 0 --resume_from logs/resnet_dino_pretrain_linear_probe/iter_8000.pth --resume_from logs/resnet_dino_pretrain_linear_probe/iter_8000.pth