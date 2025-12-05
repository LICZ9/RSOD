custom_imports = dict(
    imports=[
        'projects.RSOD.rsod',                                 
        'mmdet.models.losses.reliability_aware_L1_loss',       
        'mmdet.models.losses.reliability_aware_loss',          
    ],
    allow_failed_imports=False
)

_base_ = [
    '../../../configs/_base_/models/faster-rcnn_r50_fpn.py', '../../../configs/_base_/default_runtime.py',
    'rsod_sonar_detection.py'
]



detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    bgr_to_rgb=False,
    pad_size_divisor=32)
detector.backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=False),
    norm_eval=True,
    style='caffe',
    init_cfg=dict(
        type='Pretrained',
        # checkpoint='../ckpt/resnet50_msra-5891d200.pth'))
        checkpoint='open-mmlab://resnet50_v1c'))  # 使用 MMDetection 提供的权重

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
        cls_pseudo_thr=0.7,
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=2.0,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

# 10% coco train2017 is set as labeled dataset
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset
labeled_dataset.ann_file = 'annotations/instances_05train.json'
unlabeled_dataset.ann_file = 'annotations/instances_05unlabeled.json'
labeled_dataset.data_prefix = dict(img='train/')
unlabeled_dataset.data_prefix = dict(img='train/')

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(batch_size=4, source_ratio=[1, 3]),
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# training schedule for 180k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=240000, val_interval=4000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=20, norm_type=2)
)

log_processor = dict(by_epoch=False)
custom_hooks = [dict(type='MeanTeacherHook', momentum=0.0002, gamma=4)]
resume = True
