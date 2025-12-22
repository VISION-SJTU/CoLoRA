#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py  configs/mae/convnext_peft.py logs/convnext_s_fulltune/iter_160000.pth --launcher none --eval mIoU
