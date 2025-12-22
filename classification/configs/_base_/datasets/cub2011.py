# dataset settings
# MambaPEFT official
dataset_type = 'Cub2011Dataset'

data_root = '/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/dataset/cub2011'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),        # 归一化
    dict(type='ImageToTensor', keys=['img']),      # image 转为 torch.Tensor
    dict(type='ToTensor', keys=['gt_label']),      # gt_label 转为 torch.Tensor
    dict(type='Collect', keys=['img', 'gt_label']) # 决定数据中哪些键应该传递给检测器的流程
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),        # 归一化
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=64,    # 单个 GPU 的 Batch size
    workers_per_gpu=2,     # 单个 GPU 的 线程数
    train=dict(            # 训练数据信息
        type=dataset_type,                  # 数据集名称
        data_root=data_root,
        train=True,
        pipeline=train_pipeline),           # 数据集需要经过的 数据流水线
    val=dict(              # 验证数据集信息
        type=dataset_type,
        data_root=data_root,
        train=False,
        pipeline=test_pipeline),
    test=dict(             # 测试数据集信息
        type=dataset_type,
        data_root=data_root,
        train=False,
        pipeline=test_pipeline)
)
