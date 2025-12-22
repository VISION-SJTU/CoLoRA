# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='ImageClassifier',
    pretrained=None,
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768], 
        drop_path_rate=0.2,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=1536,
        # loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    )
    )