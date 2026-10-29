# Use a single stage detector as RPN

Region proposal network (RPN) is a submodule in [Faster R-CNN](https://arxiv.org/abs/1506.01497), which generates proposals for the second stage of Faster R-CNN. Most two-stage detectors in MMDetection use [`RPNHead`](../../../mmdet/models/dense_heads/rpn_head.py) to generate proposals as RPN. However, any single-stage detector can serve as an RPN since their bounding box predictions can also be regarded as region proposals and thus be refined in the R-CNN. Therefore, MMDetection v3.0 supports that.

To illustrate the whole process, here we give an example of how to use an anchor-free single-stage model [FCOS](../../../mmdet/.mim/configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py) as an RPN in [Faster R-CNN](../../../mmdet/.mim/configs/faster_rcnn/faster-rcnn_r50_fpn_fcos-rpn_1x_coco.py).

The outline of this tutorial is as below:

1. Use `FCOSHead` as an `RPNHead` in Faster R-CNN
2. Evaluate proposals
3. Train the customized Faster R-CNN with pre-trained FCOS

Run the commands below in a Bash terminal from the project root. The examples use eight GPUs; adjust the `8` argument to the number available. Output paths are relative to the project root.

## Use `FCOSHead` as an `RPNHead` in Faster R-CNN

The project provides `mmdet/.mim/configs/faster_rcnn/faster-rcnn_r50_fpn_fcos-rpn_1x_coco.py`, which can be used directly by the training command in this section. It uses the FCOS `bbox_head` settings for the Faster R-CNN `rpn_head`, retains the FCOS neck strides of `[8,16,32,64,128]`, and updates `featmap_strides` of `bbox_roi_extractor` accordingly. Warmup lasts 1000 iterations instead of 500 so the learning rate increases more slowly. The relevant configuration is:

```python
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    # copied from mmdet/.mim/configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py
    neck=dict(
        start_level=1,
        add_extra_convs='on_output',  # use P5
        relu_before_extra_convs=True),
    rpn_head=dict(
        _delete_=True,  # ignore the unused old settings
        type='FCOSHead',
        num_classes=1,  # num_classes = 1 for rpn, if num_classes > 1, it will be set to 1 in TwoStageDetector automatically
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    roi_head=dict(  # update featmap_strides due to the strides in neck
        bbox_roi_extractor=dict(featmap_strides=[8, 16, 32, 64, 128])))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),  # Slowly increase lr, otherwise loss becomes NAN
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
```

Then, we could use the following command to train our customized model. For more training commands, please refer to [here](train.md).

```bash
bash tools/dist_train.sh \
    mmdet/.mim/configs/faster_rcnn/faster-rcnn_r50_fpn_fcos-rpn_1x_coco.py \
    8 \
    --work-dir ./work_dirs/faster-rcnn_r50_fpn_fcos-rpn_1x_coco
```

## Evaluate proposals

To evaluate proposal quality, create the `configs/rpn/` directory and save the following configuration as `configs/rpn/fcos-rpn_r50_fpn_1x_coco.py`. Its `rpn_head` uses the FCOS `bbox_head` settings, and its base configurations refer to this project's `mmdet/.mim/configs/_base_/` directory.

```python
_base_ = [
    '../../mmdet/.mim/configs/_base_/models/rpn_r50_fpn.py', '../../mmdet/.mim/configs/_base_/datasets/coco_detection.py',
    '../../mmdet/.mim/configs/_base_/schedules/schedule_1x.py', '../../mmdet/.mim/configs/_base_/default_runtime.py'
]

val_evaluator = dict(metric='proposal_fast')
test_evaluator = val_evaluator

model = dict(
    # copied from mmdet/.mim/configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py
    neck=dict(
        start_level=1,
        add_extra_convs='on_output',  # use P5
        relu_before_extra_convs=True),
    rpn_head=dict(
        _delete_=True,  # ignore the unused old settings
        type='FCOSHead',
        num_classes=1,  # num_classes = 1 for rpn, if num_classes > 1, it will be set to 1 in RPN automatically
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
```

Suppose we have the checkpoint `./work_dirs/faster-rcnn_r50_fpn_fcos-rpn_1x_coco/epoch_12.pth` after training, then we can evaluate the quality of proposals with the following command.

```bash
bash tools/dist_test.sh \
    configs/rpn/fcos-rpn_r50_fpn_1x_coco.py \
    ./work_dirs/faster-rcnn_r50_fpn_fcos-rpn_1x_coco/epoch_12.pth \
    8
```

## Train the customized Faster R-CNN with pre-trained FCOS

This example uses `FCOSHead` as the Faster R-CNN `rpn_head` and loads pre-trained FCOS weights. Create the `configs/faster_rcnn/` directory and save the following configuration as `configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_fcos-rpn_1x_coco.py`. The pre-trained model uses the Caffe variant of ResNet50, so the pixel mean and standard deviation in `data_preprocessor` must match it.

```python
_base_ = [
    '../../mmdet/.mim/configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../../mmdet/.mim/configs/_base_/datasets/coco_detection.py',
    '../../mmdet/.mim/configs/_base_/schedules/schedule_1x.py', '../../mmdet/.mim/configs/_base_/default_runtime.py'
]

model = dict(
    data_preprocessor=dict(
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False),
    backbone=dict(
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe',
        init_cfg=None),  # the checkpoint in ``load_from`` contains the weights of backbone
    neck=dict(
        start_level=1,
        add_extra_convs='on_output',  # use P5
        relu_before_extra_convs=True),
    rpn_head=dict(
        _delete_=True,  # ignore the unused old settings
        type='FCOSHead',
        num_classes=1,  # num_classes = 1 for rpn, if num_classes > 1, it will be set to 1 in TwoStageDetector automatically
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    roi_head=dict(  # update featmap_strides due to the strides in neck
        bbox_roi_extractor=dict(featmap_strides=[8, 16, 32, 64, 128])))

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'
```

The command for training is as below.

```bash
bash tools/dist_train.sh \
    configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_fcos-rpn_1x_coco.py \
    8 \
    --work-dir ./work_dirs/faster-rcnn_r50-caffe_fpn_fcos-rpn_1x_coco
```
