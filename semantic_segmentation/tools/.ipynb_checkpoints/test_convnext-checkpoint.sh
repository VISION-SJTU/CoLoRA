#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py  configs/mae/convnext_.py logs/convnext_s_fulltune/iter_160000.pth --launcher none --eval mIoU --aug-test # --resume_from logs/convnext_s_correlated_shareA_indep_redu16/iter_16000.pth
