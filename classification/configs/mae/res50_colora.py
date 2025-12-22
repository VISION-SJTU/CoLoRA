# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Xiang Li
# Based on timm, mmseg, setr, xcit, swin, beit and mae code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# https://github.com/microsoft/unilm/tree/master/beit
# 
# --------------------------------------------------------'
# reproduce the semantic segmentation results 48.15 (mIoU) for ViT-base in MAE

_base_ = [
     '../_base_/datasets/vtab1k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='ImageClassifier',
    pretrained=None,
    backbone=dict(
        type='ResNetCoLoRA',
        res50_pretrained="../mae_segmentation/pretrain/resnet50-11ad3fa6.pth",
        use_lora=True, 
        use_adapter=False,
        adaptive_weight=False, 
        conv1_lora=False,
        lora_reduction=8,
        use_filter=True
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=2048,
        # loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ),
)

optimizer = dict(_delete_=True, type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructorRes50', 
                 paramwise_cfg=dict(num_layers=16, layer_decay_rate=0.9))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=32)

runner = dict(type='EpochBasedRunnerAmp', max_epochs=100)

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)