# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os # Added import os
import os.path as osp
from multiprocessing import Pool

import mmcv
import numpy as np
import torch # Added import
from mmengine.config import Config, DictAction
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.structures import InstanceData, PixelData
from mmengine.utils import ProgressBar, check_file_exist, mkdir_or_exist

from mmdet.datasets import get_loading_pipeline
from mmdet.evaluation import eval_map
from mmdet.models.layers import multiclass_nms
from mmdet.registry import DATASETS, RUNNERS
from mmdet.structures import DetDataSample
from mmdet.utils import replace_cfg_vals, update_data_root
from mmdet.visualization import DetLocalVisualizer
import cv2
def bbox_map_eval(det_result, annotation, nproc=4):
    """Evaluate mAP of single image det result.

    Args:
        det_result (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotation (dict): Ground truth annotations where keys of
             annotations are:

            - bboxes: numpy array of shape (n, 4)
            - labels: numpy array of shape (n, )
            - bboxes_ignore (optional): numpy array of shape (k, 4)
            - labels_ignore (optional): numpy array of shape (k, )

        nproc (int): Processes used for computing mAP.
            Default: 4.

    Returns:
        float: mAP
    """

    # use only bbox det result
    if isinstance(det_result, tuple):
        bbox_det_result = [det_result[0]]
    else:
        bbox_det_result = [det_result]
    # mAP
    # iou_thrs = np.linspace(
    #     .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    iou_thrs = [0.5] # Calculate mAP@0.5 (mAP50)

    processes = []
    workers = Pool(processes=nproc)
    for thr in iou_thrs:
        p = workers.apply_async(eval_map, (bbox_det_result, [annotation]), {
            'iou_thr': thr,
            'logger': 'silent',
            'nproc': 1
        })
        processes.append(p)

    workers.close()
    workers.join()

    mean_aps = []
    for p in processes:
        mean_aps.append(p.get()[0])

    return sum(mean_aps) / len(mean_aps)


class ResultVisualizer:
    """Display and save evaluation results.

    Args:
        show (bool): Whether to show the image. Default: True.
        wait_time (float): Value of waitKey param. Default: 0.
        score_thr (float): Minimum score of bboxes to be shown.
           Default: 0.
        runner (:obj:`Runner`): The runner of the visualization process.
    """

    def __init__(self, show=False, wait_time=0, score_thr=0, runner=None, nms_iou_thr=0.1):
        self.show = show
        self.wait_time = wait_time
        self.score_thr = score_thr
        self.nms_iou_thr = nms_iou_thr
        
        # 创建自定义颜色调色板，增大不同类别间的颜色差距（10个类，不使用红色）
        custom_palette = [
            (0, 255, 0),      # 绿色
            (0, 0, 255),      # 蓝色
            (255, 255, 0),    # 黄色
            (255, 0, 255),    # 洋红
            (0, 255, 255),    # 青色
            (255, 128, 0),    # 橙色
            (255, 255, 255),    # 白色
            (255, 215, 0),    # 金色
            (138, 43, 226),   # 蓝紫色
            (32, 178, 170)    # 浅海绿
        ]
        
        self.visualizer = DetLocalVisualizer()
        # 设置自定义调色板
        self.visualizer.dataset_meta = {'palette': custom_palette}
        
        self.runner = runner
        self.evaluator = runner.test_evaluator

    def _save_image_gts_results(self,
                                dataset,
                                results,
                                performances,
                                out_dir=None,
                                task='det'):
        """Display or save image with groung truths and predictions from a
        model.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Object detection or panoptic segmentation
                results from test results pkl file.
            performances (dict): A dict contains samples's indices
                in dataset and model's performance on them.
            out_dir (str, optional): The filename to write the image.
                Defaults: None.
            task (str): The task to be performed. Defaults: 'det'
        """
        mkdir_or_exist(out_dir)

        index, performance = performances # performances is now a single tuple
        data_info = dataset[index]
        data_info['gt_instances'] = data_info['instances']

        # calc save file path
        filename = data_info['img_path']
        # Ensure unique filename for each image by appending the index
        original_basename = osp.basename(filename)
        name_part, ext_part = osp.splitext(original_basename)
        # Use img_id for unique filename
        img_id = data_info['img_id']
        unique_out_basename = f"{name_part}_{img_id}{ext_part}"
        out_file = osp.join(out_dir, unique_out_basename)

        if task == 'det':
            gt_instances = InstanceData()
            gt_instances.bboxes = [
                d['bbox'] for d in data_info['instances']
            ]
            gt_instances.labels = [
                d['bbox_label'] for d in data_info['instances']
            ]

            pred_instances = InstanceData()
            pred_instances.bboxes = results[index]['pred_instances'][
                'bboxes']
            pred_instances.labels = results[index]['pred_instances'][
                'labels']
            pred_instances.scores = results[index]['pred_instances'][
                'scores']

            data_samples = DetDataSample()
            data_samples.pred_instances = pred_instances
            data_samples.gt_instances = gt_instances

        elif task == 'seg':
            gt_panoptic_seg = PixelData()
            gt_panoptic_seg.sem_seg = [
                d['gt_seg_map'] for d in data_info['gt_instances']
            ]

            pred_panoptic_seg = PixelData()
            pred_panoptic_seg.sem_seg = results[index][
                'pred_panoptic_seg']['sem_seg']

            data_samples = DetDataSample()
            data_samples.pred_panoptic_seg = pred_panoptic_seg
            data_samples.gt_panoptic_seg = gt_panoptic_seg

        img = mmcv.imread(filename, channel_order='rgb')
        # 对检测结果应用NMS处理
        if task == 'det' and 'pred_instances' in data_samples:
            original_pred_instances = data_samples.pred_instances
            bboxes = original_pred_instances.bboxes
            scores = original_pred_instances.scores
            labels = original_pred_instances.labels

            # 1. 应用score_thr
            score_thr = 0.50 # 与 train.py 中的 score_thr 保持一致
            valid_mask = scores > score_thr
            bboxes = bboxes[valid_mask]
            scores = scores[valid_mask]
            labels = labels[valid_mask]

            # 2. 使用multiclass_nms进行NMS处理
            if bboxes.numel() > 0:
                # 获取类别数量
                if hasattr(self.visualizer, 'dataset_meta') and self.visualizer.dataset_meta is not None and 'classes' in self.visualizer.dataset_meta:
                    num_classes = len(self.visualizer.dataset_meta['classes'])
                else:
                    num_classes = labels.max().item() + 1 if labels.numel() > 0 else 1
                
                # 构建2D的multi_scores张量，形状为(n, #class+1)，最后一列为背景类
                multi_scores = torch.zeros((len(bboxes), num_classes + 1), device=scores.device)
                # 将scores填入对应的类别位置
                multi_scores[torch.arange(len(bboxes)), labels] = scores
                
                # 构建NMS配置
                nms_cfg = dict(type='nms', iou_threshold=self.nms_iou_thr) # 使用 self.nms_iou_thr
                max_per_img = 100
                
                # 使用multiclass_nms
                det_bboxes, det_labels = multiclass_nms(
                    bboxes,
                    multi_scores,
                    score_thr=0.1, # 与 train.py 中的 score_thr 保持一致
                    nms_cfg=nms_cfg,
                    max_num=max_per_img
                )
                
                final_bboxes = det_bboxes[:, :4]
                final_scores = det_bboxes[:, 4]
                final_labels = det_labels
            else:
                # 如果没有框，则创建空的张量
                final_bboxes = torch.empty((0, 4), device=bboxes.device)
                final_scores = torch.empty((0,), device=scores.device)
                final_labels = torch.empty((0,), device=labels.device, dtype=torch.long)

            new_pred_instances = InstanceData()
            new_pred_instances.bboxes = final_bboxes
            new_pred_instances.scores = final_scores
            new_pred_instances.labels = final_labels
            
            # 将处理后的预测结果赋值给data_samples
            data_samples.pred_instances = new_pred_instances
            


            # 获取 visualizer 绘制后的图像
            # drawn_img = self.visualizer.get_image() # 使用原始图像进行绘制
            drawn_img = img.copy()

            # 手动在 drawn_img 上绘制文本
            if 'pred_instances' in data_samples:
                pred_instances = data_samples.pred_instances
                bboxes = pred_instances.bboxes.cpu().numpy()
                labels = pred_instances.labels.cpu().numpy()
                scores = pred_instances.scores.cpu().numpy()
                
                # 获取类别名称
                class_names = self.visualizer.dataset_meta.get('classes', None)

                for i in range(len(bboxes)):
                    bbox = bboxes[i]
                    label_idx = labels[i]
                    score = scores[i]

                    if score < self.score_thr:
                        continue

                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # 获取颜色
                    color = self.visualizer.dataset_meta['palette'][label_idx % len(self.visualizer.dataset_meta['palette'])]
                    color_bgr = color # Convert RGB to BGR for OpenCV

                    text_bg_color = color_bgr
                    text_color = (0, 0, 0) # 设置文本颜色为黑色

                    # 绘制边界框
                    cv2.rectangle(drawn_img, (x1, y1), (x2, y2), color_bgr, 2) # 2是线宽

                    text = ''
                    if class_names and label_idx < len(class_names):
                        text = class_names[label_idx]
                    text += f' {score:.2f}'
                    
                    font_scale = 0.5 # 缩小字体
                    thickness = 1 # 可调整
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    
                    # 计算文本大小以确定位置
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    
                    # 将文本放在框的上方，底部与框的上边缘对齐
                    text_x = x1
                    # 初始文本背景框的左上角和右下角坐标 (假设在边界框上方)
                    bg_x1 = x1
                    bg_y1 = y1 - (text_height + baseline) # 背景框的顶部
                    bg_x2 = x1 + text_width
                    bg_y2 = y1 # 背景框的底部，与边界框顶部对齐

                    # 初始文本的绘制位置 (基线)
                    text_x = x1
                    text_y = bg_y2 - baseline 

                    # 检查文本背景框是否超出图像顶部
                    img_h, img_w, _ = drawn_img.shape
                    if bg_y1 < 0: # 如果背景框顶部超出图像顶部
                        # 将文本和背景框移动到边界框内部
                        bg_y1 = y1 # 背景框顶部与边界框顶部对齐
                        bg_y2 = y1 + (text_height + baseline) # 背景框底部
                        text_y = bg_y2 - baseline # 文本基线在背景框内部
                        # Ensure text_y is within the image bounds if y1 is very large
                        if text_y > img_h:
                            text_y = img_h - baseline # Fallback to bottom of image if box is too large
                            bg_y1 = img_h - (text_height + baseline)
                            bg_y2 = img_h
                    
                    # 裁剪背景矩形到图像边界内 (再次裁剪以防调整后仍超出)
                    bg_x1 = max(0, bg_x1)
                    bg_y1 = max(0, bg_y1)
                    bg_x2 = min(img_w, bg_x2)
                    bg_y2 = min(img_h, bg_y2)

                    # 绘制文本背景
                    if bg_x1 < bg_x2 and bg_y1 < bg_y2:
                        cv2.rectangle(drawn_img, (bg_x1, bg_y1), (bg_x2, bg_y2), text_bg_color, -1)

                    # 绘制文本
                    # 确保 text_color 在此处已定义
                    cv2.putText(drawn_img, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)



            # 保存修改后的图像
            if out_file is not None:
                # 将RGB图像转换为BGR格式用于保存
                drawn_img_bgr = cv2.cvtColor(drawn_img, cv2.COLOR_RGB2BGR)
                mmcv.imwrite(drawn_img_bgr, out_file)
            elif self.show:
                self.visualizer.show(drawn_img, win_name='image', wait_time=self.wait_time)



    def evaluate_and_show(self,
                          dataset,
                          results,
                          topk=20,
                          show_dir='work_dir'):
        """Evaluate and show results.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Object detection or panoptic segmentation
                results from test results pkl file.
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20.
            show_dir (str, optional): The filename to write the image.
                Default: 'work_dir'
        """

        # 保留自定义调色板，同时合并dataset的metainfo
        if hasattr(self.visualizer, 'dataset_meta') and self.visualizer.dataset_meta is not None:
            # 合并dataset.metainfo，但保留自定义的palette
            merged_meta = dataset.metainfo.copy()
            merged_meta['palette'] = self.visualizer.dataset_meta['palette']
            self.visualizer.dataset_meta = merged_meta
        else:
            self.visualizer.dataset_meta = dataset.metainfo

        # Calculate and print mAP50
        all_pred_results = []
        all_gt_annotations = []
        for i in range(len(dataset)):
            pred_instances = results[i]['pred_instances']
            # Ensure pred_instances is a list of lists per class for bbox_map_eval
            # Assuming pred_instances['bboxes'] and pred_instances['labels'] are available
            # And that your model outputs results for all classes, even if empty
            # You might need to adjust this part based on your actual 'results' structure
            num_classes = len(dataset.metainfo['classes']) # Or however you get num_classes
            class_preds_list = [[] for _ in range(num_classes)]
            for bbox, score, label in zip(pred_instances['bboxes'], pred_instances['scores'], pred_instances['labels']):
                class_preds_list[label].append(np.hstack([bbox.cpu().numpy(), score.cpu().numpy()]))
            
            processed_class_preds = []
            for preds in class_preds_list:
                if preds:
                    processed_class_preds.append(np.vstack(preds))
                else:
                    processed_class_preds.append(np.empty((0, 5), dtype=np.float32))
            all_pred_results.append(processed_class_preds)

            data_info = dataset[i]
            gt_bboxes = [d['bbox'] for d in data_info['instances']]
            gt_labels = [d['bbox_label'] for d in data_info['instances']]
            gt_bboxes_np = np.array(gt_bboxes).astype(np.float32)
            if gt_bboxes_np.ndim == 1 and gt_bboxes_np.size == 0:
                gt_bboxes_np = gt_bboxes_np.reshape(-1, 4)
            annotation = {
                'bboxes': gt_bboxes_np,
                'labels': np.array(gt_labels).astype(np.int64)
            }
            all_gt_annotations.append(annotation)

        if all_pred_results and all_gt_annotations:
            mean_ap, _ = eval_map(
                all_pred_results,
                all_gt_annotations,
                iou_thr=0.5, # For mAP50
                dataset=dataset.metainfo['classes'],
                logger='silent'
            )
            print(f'\nmAP@0.5 (mAP50): {mean_ap:.4f}')

        # Visualize all samples
        all_samples_dir = osp.abspath(osp.join(show_dir, 'all_visualizations'))
        os.makedirs(all_samples_dir, exist_ok=True)

        for i in range(len(dataset)):
            if 'pred_panoptic_seg' in results[i].keys():
                # For panoptic segmentation, visualize all samples
                self._save_image_gts_results(
                    dataset, results, (i, 0), all_samples_dir, task='seg') # Passing dummy performance for compatibility
            elif 'pred_instances' in results[i].keys():
                # For object detection, visualize all samples
                self._save_image_gts_results(
                    dataset, results, (i, 0), all_samples_dir, task='det') # Passing dummy performance for compatibility
            else:
                raise 'expect \'pred_panoptic_seg\' or \'pred_instances\' \
                    in dict result'

    def detection_evaluate(self, dataset, results, eval_fn=None): # Removed topk
        """Evaluation for object detection.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Object detection results from test
                results pkl file.
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20.
            eval_fn (callable, optional): Eval function, Default: None.

        Returns:
            tuple: A tuple contains good samples and bad samples.
                good_mAPs (dict[int, float]): A dict contains good
                    samples's indices in dataset and model's
                    performance on them.
                bad_mAPs (dict[int, float]): A dict contains bad
                    samples's indices in dataset and model's
                    performance on them.
        """

        if eval_fn is None:
            eval_fn = bbox_map_eval
        else:
            assert callable(eval_fn)

        prog_bar = ProgressBar(len(results))
        _mAPs = {}
        data_info = {}
        for i, (result, ) in enumerate(zip(results)):

            # self.dataset[i] should not call directly
            # because there is a risk of mismatch
            data_info = dataset.prepare_data(i)
            data_info['bboxes'] = data_info['gt_bboxes'].tensor
            data_info['labels'] = data_info['gt_bboxes_labels']

            pred = result['pred_instances']
            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            dets = []
            for label in range(len(dataset.metainfo['classes'])):
                index = np.where(pred_labels == label)[0]
                pred_bbox_scores = np.hstack(
                    [pred_bboxes[index], pred_scores[index].reshape((-1, 1))])
                dets.append(pred_bbox_scores)
            mAP = eval_fn(dets, data_info)

            _mAPs[i] = mAP
            prog_bar.update()
        # Return all mAPs, no topk selection
        return _mAPs, [] # Returning empty list for bad_mAPs for compatibility

    def panoptic_evaluate(self, dataset, results): # Removed topk
        """Evaluation for panoptic segmentation.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Panoptic segmentation results from test
                results pkl file.
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20.

        Returns:
            tuple: A tuple contains good samples and bad samples.
                good_pqs (dict[int, float]): A dict contains good
                    samples's indices in dataset and model's
                    performance on them.
                bad_pqs (dict[int, float]): A dict contains bad
                    samples's indices in dataset and model's
                    performance on them.
        """
        pqs = {}
        prog_bar = ProgressBar(len(results))

        for i in range(len(results)):
            data_sample = {}
            for k in dataset[i].keys():
                data_sample[k] = dataset[i][k]

            for k in results[i].keys():
                data_sample[k] = results[i][k]

            self.evaluator.process([data_sample])
            metrics = self.evaluator.evaluate(1)

            pqs[i] = metrics['coco_panoptic/PQ']
            prog_bar.update()

        # Return all pqs, no topk selection
        return pqs, [] # Returning empty list for bad_pqs for compatibility


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test pkl result')
    parser.add_argument(
        'show_dir', help='directory where painted images will be saved')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=0,
        help='the interval of show (s), 0 is block')
    parser.add_argument(
        '--topk',
        default=float('inf'),
        type=int,
        help='saved Number of the highest topk '
        'and lowest topk after index sorting. Set to a very large number to output all results')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0,
        help='score threshold (default: 0.)')
    parser.add_argument(
        '--nms-iou-thr',
        type=float,
        default=0.1, # 确保与 train.py 一致
        help='NMS IoU threshold for filtering overlapping boxes (default: 0.5)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    check_file_exist(args.prediction_path)

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    cfg.test_dataloader.dataset.test_mode = True

    cfg.test_dataloader.pop('batch_size', 0)
    if cfg.train_dataloader.dataset.type in ('MultiImageMixDataset',
                                             'ClassBalancedDataset',
                                             'RepeatDataset'):
        cfg.test_dataloader.dataset.pipeline = get_loading_pipeline(
            cfg.train_dataloader.dataset.dataset.pipeline)
    elif cfg.train_dataloader.dataset.type in ('ConcatDataset', ):
        cfg.test_dataloader.dataset.pipeline = get_loading_pipeline(
            cfg.train_dataloader.dataset.datasets[0].pipeline)
    else:
        cfg.test_dataloader.dataset.pipeline = get_loading_pipeline(
            cfg.train_dataloader.dataset.pipeline)
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    outputs = load(args.prediction_path)

    cfg.work_dir = args.show_dir
    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    result_visualizer = ResultVisualizer(args.show, args.wait_time,
                                         args.show_score_thr, runner, args.nms_iou_thr)
    result_visualizer.evaluate_and_show(
        dataset, outputs, topk=args.topk, show_dir=args.show_dir)


if __name__ == '__main__':
    main()

