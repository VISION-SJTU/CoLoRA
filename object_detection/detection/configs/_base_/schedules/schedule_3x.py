# optimizer
# https://github.com/SysCV/bdd100k-models/blob/0935a8a3eb4c7442efdce2a8ce4b93fbe585be15/det/configs/_base_/schedules/schedule_3x.py
optimizer = dict(type="SGD", lr=0.04, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[24, 33],
)
runner = dict(type="EpochBasedRunner", max_epochs=36)