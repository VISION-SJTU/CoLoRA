# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001


# optimizer
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(by_epoch=True)# , interval=10)
evaluation = dict(interval=1, metric='accuracy', metric_options={'topk': (1,)})
