# Strong Baselines

<!-- [OTHERS] -->

We train Mask R-CNN with large-scale jitter and longer schedule as strong baselines.
The modifications follow those in [Detectron2](https://github.com/facebookresearch/detectron2/tree/master/configs/new_baselines).

## Results and Models

| Backbone |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP |                                       Config                                       |         Download         |
| :------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :--------------------------------------------------------------------------------: | :----------------------: |
| R-50-FPN | pytorch |   50e   |          |                |        |         |    [config](./mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-50e_coco.py)     | Not available |
| R-50-FPN | pytorch |  100e   |          |                |        |         |    [config](./mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-100e_coco.py)    | Not available |
| R-50-FPN |  caffe  |  100e   |          |                |  44.7  |  40.4   | [config](./mask-rcnn_r50-caffe_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-100e_coco.py) | Not available |
| R-50-FPN |  caffe  |  400e   |          |                |        |         | [config](./mask-rcnn_r50-caffe_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-400e_coco.py) | Not available |

## Notice

When using large-scale jittering, there are sometimes empty proposals in the box and mask heads during training.
These configurations use `MMSyncBN` to handle empty tensors. Install
`mmcv==2.1.0` and `mmengine==0.10.6` by following this project's
[installation instructions](../../../../README.md#installation).
