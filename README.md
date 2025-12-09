# CoLoRA
This is the code implementation for the CoLoRA (NeurIPS 2025)

## Build environment
> Since this work contains classfication, segmentation and object detection, we provide a detail guide to build environment. Following these instructions:

```
# install torch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# install tap
pip install typed-argument-parser
# install openmim, mmclassification, mmdet, and mmsegmentation
pip install -U openmim
mim install mmcv-full==1.7.0
mim install mmsegmentation==0.30.0
mim install mmdet==2.28.2 
mim install mmcls=0.25.0
# install nvidia apex (download the official repo for apex and use the official command to install. If any error, try the below command)
pip install -v --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
# install timm
pip install timm --no-deps
# enabling half-precision for mmcv
sudo vim /opt/conda/lib/python3.7/site-packages/apex/normalization/fused_layer_norm.py (line 18, force return False)

```