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
_base_ = ['../_base_/datasets/ade20k.py', '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (512, 512)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ResNetFreeze',
        res50_pretrained="pretrain/resnet50-11ad3fa6.pth",
        # res50_pretrained="../",
        embed_dim=1024,
        use_lora=False, 
        use_adapter=False,
        adaptive_weight=False, 
        conv1_lora=False,
        lora_reduction=1,
        use_filter=False
    ),
    decode_head=dict(
        type='BNHead',
        in_channels=[2048],
        in_index=[0],
        input_transform='resize_concat',
        channels=2048,
        dropout_ratio=0,
        num_classes=150,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))

optimizer = dict(_delete_=True, type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructorRes50', 
                 paramwise_cfg=dict(num_layers=16, layer_decay_rate=0.99))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=16)

runner = dict(type='IterBasedRunnerAmp')

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
