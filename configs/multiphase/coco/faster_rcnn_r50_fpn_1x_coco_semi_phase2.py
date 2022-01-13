_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn_RR.py',
    '../../_base_/datasets/coco_detection_sup.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train = dict(
        type='ConcatDataset',
        datasets=[
           dict(type='RepeatDataset',
                times=4,
                dataset = dict(type=dataset_type,
                    ann_file=data_root + 'annotations/instances_valminusminival2014.json',
                    img_prefix=data_root + 'images/val2014/',
                    pipeline=train_pipeline)),
           dict(type=dataset_type,
                ann_file='labels/coco115k_trainval_pl_phase2.json',
                img_prefix=data_root + 'images/train2014/',
                pipeline=train_pipeline),
        ]),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_minival2014.json',
        img_prefix=data_root + 'images/val2014/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_minival2014.json',
        img_prefix=data_root + 'images/val2014/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
model = dict(roi_head=dict(labeled_sign=data_root + 'images/val2014'))

# optimizer
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[5, 6])
runner = dict(type='EpochBasedRunner', max_epochs=7)
