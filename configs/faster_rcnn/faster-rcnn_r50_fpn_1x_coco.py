# 默认配置
# _base_ = [
#     '../_base_/models/faster-rcnn_r50_fpn.py',
#     '../_base_/datasets/coco_detection.py',
#     '../_base_/schedules/schedule_1x.py',
#     '../_base_/default_runtime.py'
# ]

_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/sonar_mnist_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
