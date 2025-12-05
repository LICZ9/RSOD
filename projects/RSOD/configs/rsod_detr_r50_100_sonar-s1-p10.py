# Copyright (c) OpenMMLab. All rights reserved.
custom_imports = dict(
    imports=[
        'projects.RSOD.rsod',                                
        'mmdet.models.losses.reliability_aware_L1_loss',       
        'mmdet.models.losses.reliability_aware_loss',  
    ],
    allow_failed_imports=False
)

_base_ = [
    '../../../configs/_base_/default_runtime.py',
    'rsod_sonar_detection.py'
]

# DETR detector configuration
detector = dict(
    type='DETR',
    num_queries=100,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=None,
        num_outs=1),
    encoder=dict(  # DetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)))),
    decoder=dict(  # DetrTransformerDecoder
        num_layers=6,
        layer_cfg=dict(  # DetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True))),
        return_intermediate=True),
    positional_encoding=dict(num_feats=128, normalize=True),
    bbox_head=dict(
        type='DETRHead',
        num_classes=10,  # sonar dataset has 10 classes
        embed_dims=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=1.),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100))

# MixPL semi-supervised model configuration
model = dict(
    _delete_=True,
    type='RSOD', 
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector['data_preprocessor']),
    semi_train_cfg=dict(
        compile=True,  
        max_iters=240000,
        temp=0.2,              
        min_scale=0.8,         
        max_scale=1.2,  
        least_num=1,
        cache_size=8,
        mixup=True,
        mosaic=True,
        mosaic_shape=[(400, 400), (800, 800)],
        mosaic_weight=0.5,
        erase=True,
        erase_patches=(1, 20),
        erase_ratio=(0, 0.1),
        erase_thr=0.7,
        cls_pseudo_thr=0.5,  # Lower threshold for DETR
        reliability_threshold=0.3,  # Lower threshold for DETR
        iou_threshold=0.2,
        freeze_teacher=False,  # Allow teacher to learn
        sup_weight=1.0,
        unsup_weight=1.0,  # Reduced unsupervised weight for DETR
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

# 10% coco train2017 is set as labeled dataset
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset
labeled_dataset.ann_file = 'annotations/instances_01train.json'
unlabeled_dataset.ann_file = 'annotations/instances_01unlabeled.json'
labeled_dataset.data_prefix = dict(img='train/')
unlabeled_dataset.data_prefix = dict(img='train/')

train_dataloader = dict(
    batch_size=2,  # Reduced batch size for DETR
    num_workers=4,
    sampler=dict(batch_size=2, source_ratio=[1, 3]),
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# training schedule for 240k iterations
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=240000, val_interval=4000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# learning rate policy - longer warmup for DETR
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.0001, by_epoch=False, begin=0, end=5000),
    dict(
        type='MultiStepLR',
        begin=5000,
        end=240000,
        by_epoch=False,
        milestones=[160000, 200000],
        gamma=0.1)
]

# optimizer - AdamW for DETR
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

log_processor = dict(by_epoch=False)
custom_hooks = [dict(type='MeanTeacherHook', momentum=0.0002, gamma=4)]
resume = True

# Enable automatic mixed precision training
optim_wrapper.update(dict(type='AmpOptimWrapper'))

# Load interval for saving checkpoints
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=4000, save_best='auto'))