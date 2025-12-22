# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    '../_base_/models/upernet_convnext.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)

model = dict(
    backbone=dict(
        type='ConvNeXtPiSSA',
        in_chans=3,
        depths=[3, 3, 27, 3], 
        dims=[192, 384, 768, 1536], 
        drop_path_rate=0.3,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        use_lora=True,
        lora_reduction=20, # pissa 20
        use_adapter=False,
        use_filter=False,
        adaptive_weight=False,
        pretrained_ckpt="pretrain/convnext_large_22k_224.pth"
    ),
    # neck=dict(
    #     type='FPN',
    #     in_channels=[96, 192, 384, 768],
    #     out_channels=1024,
    #     num_outs=4,
    #     norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    #     act_cfg=dict(type='ReLU')  # optional, defaults to ReLU
    # ),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        num_classes=150,
        # channels=1024, # for program
    ),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=150,
        # channels=1024, # for program
    ), 
    # test_cfg = dict(mode='slide', crop_size=crop_size, stride=(512, 512)),
    test_cfg = dict(mode='whole'),
)

optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW', 
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.9,
                                'decay_type': 'stage_wise',
                                'num_layers': 12,
        })

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)

runner = dict(type='IterBasedRunnerAmp')

# custom_hooks = [
#     dict(type='MergeWeightHook',interval=50)
# ]

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
