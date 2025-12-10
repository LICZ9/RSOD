# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional, Tuple
import torch.nn as nn
import torch
from torch import Tensor

from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                     reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.visualization import DetLocalVisualizer
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from mmdet.models.detectors import SemiBaseDetector
from mmdet.structures.bbox import bbox_project,bbox_overlaps
from torch.nn import functional as F
import numpy as np
import math
import os.path as osp



@MODELS.register_module()
class RSOD(SemiBaseDetector):
    """Base class for semi-supervised detectors."""

    def __init__(self,
                 detector: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            detector=detector,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.cache_inputs = []
        self.cache_data_samples = []
        self.reliability_calculator = EnhancedReliabilityCalculator(
            temp=semi_train_cfg.get('temp', 0.2),
            min_scale=semi_train_cfg.get('min_scale', 0.8),
            max_scale=semi_train_cfg.get('max_scale', 1.2)
        )
    def merge_reliable_instances(self, batch_inputs, batch_data_samples, reliability_scores, threshold=0.3):
        for idx in range(len(batch_data_samples)):
            assert len(batch_data_samples[idx].gt_instances) == len(reliability_scores[idx]), \
                f"第{idx}组 gt_instances 数量={len(batch_data_samples[idx].gt_instances)}, reliability_scores 数量={len(reliability_scores[idx])}"
        if len(batch_data_samples) <= 1:
            return batch_inputs, batch_data_samples, reliability_scores

        target_img = batch_inputs[0].clone()
        target_sample = batch_data_samples[0]

        merged_instances = target_sample.gt_instances.clone()
        current_boxes = merged_instances.bboxes

        merged_reliability_scores = [reliability_scores[0].clone()]

        for idx in range(1, len(batch_data_samples)):
            curr_img = batch_inputs[idx]
            curr_instances = batch_data_samples[idx].gt_instances
            curr_scores = reliability_scores[idx]

            reliable_mask = curr_scores > threshold
            if not reliable_mask.any():
                continue
            curr_reliable_boxes = curr_instances.bboxes[reliable_mask]
            curr_reliable_instances = curr_instances[reliable_mask]
            curr_reliable_scores = curr_scores[reliable_mask]
            if len(curr_reliable_boxes) == 0:
                continue

            enhanced_boxes, enhanced_img_regions = self._apply_random_augmentation(
                curr_img, curr_reliable_boxes, curr_reliable_instances
            )

            if len(current_boxes) > 0:
                ious = bbox_overlaps(enhanced_boxes, current_boxes)
                non_overlap_threshold = self.semi_train_cfg.get('non_overlap_threshold', 0.3)
                non_overlap_mask = (ious.max(dim=1)[0] < non_overlap_threshold)
            else:
                non_overlap_mask = torch.ones(len(enhanced_boxes), dtype=torch.bool)
            
            if not non_overlap_mask.any():
                continue

            for i, (box, img_region) in enumerate(zip(enhanced_boxes[non_overlap_mask], enhanced_img_regions)):
                if img_region is not None:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(target_img.shape[-1], x2), min(target_img.shape[-2], y2)
                    if x2 > x1 and y2 > y1:
                        region_h, region_w = y2 - y1, x2 - x1
                        if img_region.shape[-2:] != (region_h, region_w):
                            img_region = F.interpolate(
                                img_region.unsqueeze(0), 
                                size=(region_h, region_w), 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze(0)
                        target_img[:, y1:y2, x1:x2] = img_region
        
            enhanced_reliable_instances = curr_reliable_instances[non_overlap_mask]
            enhanced_reliable_instances.bboxes = enhanced_boxes[non_overlap_mask]
            
            if len(enhanced_reliable_instances) > 0:
                merged_instances = merged_instances.cat([merged_instances, enhanced_reliable_instances])
                current_boxes = merged_instances.bboxes
                enhanced_reliable_scores = curr_reliable_scores[non_overlap_mask]
                merged_reliability_scores[0] = torch.cat([merged_reliability_scores[0], enhanced_reliable_scores])
    
        batch_inputs[0] = target_img
        batch_data_samples[0].gt_instances = merged_instances
        
        
        for idx in range(1, len(reliability_scores)):
            merged_reliability_scores.append(reliability_scores[idx])
    
        return batch_inputs, batch_data_samples, merged_reliability_scores

    def _apply_random_augmentation(self, img, boxes, instances):
        enhanced_boxes = boxes.clone()
        enhanced_img_regions = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[-1], x2), min(img.shape[-2], y2)
            
            if x2 <= x1 or y2 <= y1:
                enhanced_img_regions.append(None)
                continue

            img_region = img[:, y1:y2, x1:x2].clone()

            aug_type = torch.randint(0, 2, (1,)).item()  # 0: 缩放, 1: 旋转
            
            if aug_type == 0:  
                scale_factor = (0.95 + torch.rand(1) * 0.1).item()

                original_h, original_w = img_region.shape[-2:]
                new_h, new_w = int(original_h * scale_factor), int(original_w * scale_factor)
                
                if new_h > 0 and new_w > 0:
                    scaled_region = F.interpolate(
                        img_region.unsqueeze(0),
                        size=(new_h, new_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    new_half_w, new_half_h = new_w / 2, new_h / 2
                    
                    new_x1 = max(0, center_x - new_half_w)
                    new_y1 = max(0, center_y - new_half_h)
                    new_x2 = min(img.shape[-1], center_x + new_half_w)
                    new_y2 = min(img.shape[-2], center_y + new_half_h)
                    
                    enhanced_boxes[i] = torch.tensor([new_x1, new_y1, new_x2, new_y2], 
                                                    device=box.device, dtype=box.dtype)
                    enhanced_img_regions.append(scaled_region)
                else:
                    enhanced_img_regions.append(img_region)

                from torchvision.transforms import functional as TF
                import math
                

                rotation_angle = (torch.rand(1).item() - 0.5) * 50  
                angle_rad = math.radians(rotation_angle)

                if img_region.max() <= 1.0:
                    img_region_pil = TF.to_pil_image(img_region)
                else:
                    img_region_pil = TF.to_pil_image(img_region / 255.0)

                rotated_pil = TF.rotate(img_region_pil, rotation_angle, expand=True)
                rotated_region = TF.to_tensor(rotated_pil)

                new_h, new_w = rotated_region.shape[1], rotated_region.shape[2]

                orig_h, orig_w = img_region.shape[1], img_region.shape[2]
                center_x = box[0] + orig_w / 2
                center_y = box[1] + orig_h / 2

                new_x1 = center_x - new_w / 2
                new_y1 = center_y - new_h / 2
                new_x2 = center_x + new_w / 2
                new_y2 = center_y + new_h / 2

                rotated_box = box.clone()
                rotated_box[0] = new_x1
                rotated_box[1] = new_y1
                rotated_box[2] = new_x2
                rotated_box[3] = new_y2
                
                enhanced_boxes[i] = rotated_box
                enhanced_img_regions.append(rotated_region)
                
        return enhanced_boxes, enhanced_img_regions

    def calculate_corner_distance_loss(self, batch_unsup_inputs, batch_unsup_data_samples, filtered_reliability_scores, reliability_threshold=0.3, iou_threshold=0.2, k=2):

        with torch.no_grad():
            student_results = self.student(batch_unsup_inputs, batch_unsup_data_samples, mode='predict')
        
        total_loss = 0.0
        valid_pairs = 0

        for i, (student_result, data_sample, reliability_scores) in enumerate(
            zip(student_results, batch_unsup_data_samples, filtered_reliability_scores)
        ):
            student_bboxes = student_result.pred_instances.bboxes

            teacher_bboxes = data_sample.gt_instances.bboxes
            teacher_reliability = reliability_scores

            reliable_mask = teacher_reliability >= reliability_threshold
            if not reliable_mask.any():
                continue
                
            reliable_teacher_bboxes = teacher_bboxes[reliable_mask]
            
            if len(reliable_teacher_bboxes) == 0 or len(student_bboxes) == 0:
                continue

            from mmdet.structures.bbox import bbox_overlaps
            ious = bbox_overlaps(student_bboxes, reliable_teacher_bboxes)

            max_ious, matched_student_indices = ious.max(dim=0)
            
           
            valid_matches = max_ious > self.semi_train_cfg.get('corner_loss_iou_threshold', 0.1)  
            
            if not valid_matches.any():
                continue
            

            for j, (teacher_idx, student_idx) in enumerate(
                zip(torch.where(valid_matches)[0], matched_student_indices[valid_matches])
            ):
                teacher_box = reliable_teacher_bboxes[teacher_idx]
                student_box = student_bboxes[student_idx]
                
                corner_loss = self._compute_corner_loss(student_box, teacher_box)
                total_loss += corner_loss
                valid_pairs += 1
        

        if valid_pairs > 0:
            return total_loss / valid_pairs
        else:
            return torch.tensor(0.0, device=batch_unsup_inputs[0].device, requires_grad=True)
    
    def _compute_corner_loss(self, box1, box2, k=2):


        corners1 = torch.stack([
            torch.stack([box1[0], box1[1]]),  # 左上
            torch.stack([box1[2], box1[1]]),  # 右上
            torch.stack([box1[2], box1[3]]),  # 右下
            torch.stack([box1[0], box1[3]])   # 左下
        ])
        
        corners2 = torch.stack([
            torch.stack([box2[0], box2[1]]),  # 左上
            torch.stack([box2[2], box2[1]]),  # 右上
            torch.stack([box2[2], box2[3]]),  # 右下
            torch.stack([box2[0], box2[3]])   # 左下
        ])

        corner_distances_sq = torch.sum((corners1 - corners2) ** 2, dim=1)
        numerator = torch.sum(corner_distances_sq)

        min_x = torch.min(box1[0], box2[0])
        max_x = torch.max(box1[2], box2[2])
        min_y = torch.min(box1[1], box2[1])
        max_y = torch.max(box1[3], box2[3])
        
        l_c = torch.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

        epsilon = 1e-8
        l_c = torch.max(l_c, torch.tensor(epsilon, device=l_c.device))

        corner_loss = numerator / (l_c ** k)
        return corner_loss

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        losses = dict()
        losses.update(**self.loss_by_gt_instances(
            multi_batch_inputs['sup'], multi_batch_data_samples['sup']))
        origin_batch_pseudo_data_samples, batch_info = self.get_pseudo_instances(
            multi_batch_inputs['unsup_teacher'], multi_batch_data_samples['unsup_teacher'])

        with torch.no_grad():
        

            reliability_scores = self.reliability_calculator(
                self.teacher, 
                multi_batch_inputs['unsup_teacher'],
                origin_batch_pseudo_data_samples
            )
    

        for data_sample, rel in zip(origin_batch_pseudo_data_samples, reliability_scores ):
            assert len(rel) == len(data_sample.gt_instances), \
                f"维度不匹配: 可靠性分数数={len(rel)}, 实例数={len(data_sample.gt_instances)}"
            data_sample.gt_instances.reliability = rel

        multi_batch_data_samples['unsup_student'] = self.project_pseudo_instances(
            origin_batch_pseudo_data_samples, multi_batch_data_samples['unsup_student'])
        filtered_reliability_scores = [
            data_sample.gt_instances.reliability 
            for data_sample in multi_batch_data_samples['unsup_student']
        ]

        batch_unsup_inputs = copy.deepcopy(multi_batch_inputs['unsup_student'])
        batch_unsup_data_samples = copy.deepcopy(multi_batch_data_samples['unsup_student'])
        batch_unsup_inputs, batch_unsup_data_samples, filtered_reliability_scores = self.merge_reliable_instances(
            batch_unsup_inputs, 
            batch_unsup_data_samples,
            filtered_reliability_scores,
            threshold=self.semi_train_cfg.get('reliability_threshold', 0.3)
        )
        
        corner_loss = self.calculate_corner_distance_loss(
            batch_unsup_inputs,
            batch_unsup_data_samples,
            filtered_reliability_scores,
            reliability_threshold=self.semi_train_cfg.get('reliability_threshold', 0.3),
            iou_threshold=self.semi_train_cfg.get('iou_threshold', 0.2),
            k=self.semi_train_cfg.get('corner_loss_k', 2)
        )

        corner_loss_weight = self.semi_train_cfg.get('corner_loss_weight', 0.5)
        losses['corner_loss'] = corner_loss * corner_loss_weight
        
        batch_unsup_inputs, batch_unsup_data_samples = self.merge(
            *zip(*list(map(self.erase, *self.split(batch_unsup_inputs, batch_unsup_data_samples)))))

        sample_size = len(multi_batch_data_samples['unsup_student'])
        mixup_idxs = np.random.choice(range(sample_size), sample_size, replace=False)
        mosaic_idxs = np.random.choice(range(4), 4, replace=False) + sample_size
        if self.semi_train_cfg.mixup and len(self.cache_inputs) == self.semi_train_cfg.cache_size:
            dst_inputs_list, batch_dst_data_samples = self.split(
                batch_unsup_inputs, batch_unsup_data_samples)
            img_shapes = [tuple(batch_unsup_inputs.shape[-2:])]*batch_unsup_inputs.shape[0]
            src_inputs_list, batch_src_data_samples = self.get_batch(mixup_idxs, img_shapes)

            batch_unsup_inputs, batch_unsup_data_samples = self.merge(*self.mixup(
                dst_inputs_list, batch_dst_data_samples,
                src_inputs_list, batch_src_data_samples))

        if self.semi_train_cfg.mixup:
            losses.update(**rename_loss_dict('mixup_', self.loss_by_pseudo_instances(
                batch_unsup_inputs, batch_unsup_data_samples)))
        else:
            losses.update(**self.loss_by_pseudo_instances(
                batch_unsup_inputs, batch_unsup_data_samples))

        if self.semi_train_cfg.mosaic and len(self.cache_inputs) == self.semi_train_cfg.cache_size:
            if len(self.semi_train_cfg.mosaic_shape) == 1:
                img_shapes = [self.semi_train_cfg.mosaic_shape[0]] * 4
            else:
                mosaic_shape = self.semi_train_cfg.mosaic_shape
                mosaic_h = np.random.randint(
                    min(mosaic_shape[0][0], mosaic_shape[1][0]), max(mosaic_shape[0][0], mosaic_shape[1][0]))
                mosaic_w = np.random.randint(
                    min(mosaic_shape[0][1], mosaic_shape[1][1]), max(mosaic_shape[0][1], mosaic_shape[1][1]))
                img_shapes = [(mosaic_h, mosaic_w)] * 4
            src_inputs_list, batch_src_data_samples = self.get_batch(mosaic_idxs, img_shapes)
            mosaic_inputs, mosaic_data_samples = self.mosaic(src_inputs_list, batch_src_data_samples)
            mosaic_losses = self.loss_by_pseudo_instances(mosaic_inputs, mosaic_data_samples)
            losses.update(**rename_loss_dict('mosaic_', reweight_loss_dict(mosaic_losses, self.semi_train_cfg.mosaic_weight)))
        losses.update(**self.loss_by_pseudo_instances(
            batch_unsup_inputs, batch_unsup_data_samples))
        

        
        self.update_cache(multi_batch_inputs['unsup_student'], multi_batch_data_samples['unsup_student'])
        return losses

    def merge(self, inputs_list, batch_data_samples):
        batch_size = len(inputs_list)
        h, w = 0, 0
        for i in range(batch_size):
            img_h, img_w = batch_data_samples[i].img_shape
            h, w = max(h, img_h), max(w, img_w)
        h, w = max(h, math.ceil(h / 32) * 32), max(w, math.ceil(w / 32) * 32)
        batch_inputs = torch.zeros((batch_size, 3, h, w)).to(self.data_preprocessor.device)
        for i in range(batch_size):
            img_h, img_w = batch_data_samples[i].img_shape
            batch_inputs[i, :, :img_h, :img_w] = inputs_list[i]
            batch_data_samples[i].set_metainfo({'batch_input_shape': (h, w)})
            batch_data_samples[i].set_metainfo({'pad_shape': (h, w)})
        return batch_inputs, batch_data_samples

    def split(self, batch_inputs, batch_data_samples):
        inputs_list = []
        for i in range(len(batch_inputs)):
            inputs = batch_inputs[i]
            data_samples = batch_data_samples[i]
            img_h, img_w = data_samples.img_shape
            inputs_list.append(inputs[..., :img_h, :img_w])
            data_samples.pop('batch_input_shape')
            data_samples.pop('pad_shape')
        return inputs_list, batch_data_samples

    def update_cache(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        inputs_list, batch_data_samples = self.split(batch_inputs, batch_data_samples)
        cache_size = self.semi_train_cfg.cache_size
        self.cache_inputs.extend(inputs_list)
        self.cache_data_samples.extend(batch_data_samples)
        self.cache_inputs = self.cache_inputs[-cache_size:]
        self.cache_data_samples = self.cache_data_samples[-cache_size:]

    def get_cache(self, idx, img_shape):
        inputs = copy.deepcopy(self.cache_inputs[idx])
        data_samples = copy.deepcopy(self.cache_data_samples[idx])
        inputs, data_samples = self.erase(*self.flip(*self.resize(inputs, data_samples, img_shape)))
        return inputs, data_samples

    def get_batch(self, rand_idxs, img_shapes):
        inputs_list, batch_data_samples = [], []
        for i in range(len(rand_idxs)):
            inputs, data_samples = self.get_cache(rand_idxs[i], img_shapes[i])
            inputs_list.append(inputs)
            batch_data_samples.append(data_samples)
        return inputs_list, batch_data_samples

    def resize(self, inputs, data_samples, img_shape):
        scale = min(img_shape[0] / data_samples.img_shape[0], img_shape[1] / data_samples.img_shape[1])
        inputs = F.interpolate(inputs.unsqueeze(0), scale_factor=scale).squeeze(0)
        data_samples.pop('img_shape')
        data_samples.pop('scale_factor')
        img_h, img_w = inputs.shape[-2:]
        data_samples.set_metainfo({'img_shape': (img_h, img_w)})
        ori_h, ori_w = data_samples.ori_shape
        data_samples.set_metainfo({'scale_factor': (img_w / ori_w, img_h / ori_h)})
        hm = data_samples.pop('homography_matrix')
        matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
        data_samples.set_metainfo({'homography_matrix': matrix @ hm})
        data_samples.gt_instances.bboxes *= scale
        data_samples.gt_instances.bboxes[:, 0::2].clamp_(0, img_w)
        data_samples.gt_instances.bboxes[:, 1::2].clamp_(0, img_h)
        return inputs, filter_gt_instances([data_samples])[0]

    def flip(self, inputs, data_samples):
        inputs = inputs.flip(-1)
        img_h, img_w = data_samples.img_shape
        hm = data_samples.pop('homography_matrix')
        matrix = np.array([[-1, 0, img_w], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        data_samples.set_metainfo({'homography_matrix': matrix @ hm})
        flip_flag = data_samples.pop('flip')
        if flip_flag is True:
            data_samples.pop('flip_direction')
            data_samples.set_metainfo({'flip': False})
        else:
            data_samples.set_metainfo({'flip': True})
            data_samples.set_metainfo({'flip_direction': 'horizontal'})
        bboxes = copy.deepcopy(data_samples.gt_instances.bboxes)
        data_samples.gt_instances.bboxes[:, 2] = img_w - bboxes[:, 0]
        data_samples.gt_instances.bboxes[:, 0] = img_w - bboxes[:, 2]
        return inputs, data_samples

    def erase(self, inputs, data_samples):
        def _get_patches(img_shape):
            patches = []
            n_patches = np.random.randint(
                self.semi_train_cfg.erase_patches[0], self.semi_train_cfg.erase_patches[1])
            for _ in range(n_patches):
                ratio = np.random.random() * \
                        (self.semi_train_cfg.erase_ratio[1] - self.semi_train_cfg.erase_ratio[0]) + \
                        self.semi_train_cfg.erase_ratio[0]
                ph, pw = int(img_shape[0] * ratio), int(img_shape[1] * ratio)
                px1 = np.random.randint(0, img_shape[1] - pw)
                py1 = np.random.randint(0, img_shape[0] - ph)
                px2, py2 = px1 + pw, py1 + ph
                patches.append([px1, py1, px2, py2])
            return torch.tensor(patches).to(self.data_preprocessor.device)
        erase_patches = _get_patches(data_samples.img_shape)
        for patch in erase_patches:
            px1, py1, px2, py2 = patch
            inputs[:, py1:py2, px1:px2] = 0
        bboxes = data_samples.gt_instances.bboxes
        left_top = torch.maximum(bboxes[:, None, :2], erase_patches[:, :2])
        right_bottom = torch.minimum(bboxes[:, None, 2:], erase_patches[:, 2:])
        wh = torch.clamp(right_bottom - left_top, 0)
        inter_areas = wh[:, :, 0] * wh[:, :, 1]
        bbox_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        bboxes_erased_ratio = inter_areas.sum(-1) / (bbox_areas + 1e-7)
        valid_inds = bboxes_erased_ratio < self.semi_train_cfg.erase_thr
        data_samples.gt_instances = data_samples.gt_instances[valid_inds]
        return inputs, data_samples

    def mixup(self, dst_inputs_list, batch_dst_data_samples, src_inputs_list, batch_src_data_samples, noise_std=0.01):
        batch_size = len(dst_inputs_list)
        mixup_inputs_list, batch_mixup_data_samples = [], []
        for i in range(batch_size):
            dst_inputs, dst_data_samples = dst_inputs_list[i], batch_dst_data_samples[i]
            src_inputs, src_data_samples = src_inputs_list[i], batch_src_data_samples[i]
            dst_shape, src_shape = dst_inputs.shape[-2:], src_inputs.shape[-2:]
            mixup_shape = (max(dst_shape[0], src_shape[0]), max(dst_shape[1], src_shape[1]))
            d_x1 = np.random.randint(mixup_shape[1] - dst_shape[1] + 1)
            d_y1 = np.random.randint(mixup_shape[0] - dst_shape[0] + 1)
            d_x2, d_y2 = d_x1 + dst_shape[1], d_y1 + dst_shape[0]
            s_x1 = np.random.randint(mixup_shape[1] - src_shape[1] + 1)
            s_y1 = np.random.randint(mixup_shape[0] - src_shape[0] + 1)
            s_x2, s_y2 = s_x1 + src_shape[1], s_y1 + src_shape[0]
            mixup_inputs = dst_inputs.new_zeros((3, mixup_shape[0], mixup_shape[1]))
            mixup_inputs[:, d_y1:d_y2, d_x1:d_x2] += dst_inputs * 0.5
            mixup_inputs[:, s_y1:s_y2, s_x1:s_x2] += src_inputs * 0.5


            img_meta = dst_data_samples.metainfo
            img_meta['img_shape'] = mixup_shape
            mixup_data_samples = DetDataSample(metainfo=img_meta)
            dst_gt_instances = copy.deepcopy(dst_data_samples.gt_instances)
            dst_gt_instances.bboxes[:, ::2] += d_x1
            dst_gt_instances.bboxes[:, 1::2] += d_y1
            src_gt_instances = copy.deepcopy(src_data_samples.gt_instances)
            src_gt_instances.bboxes[:, ::2] += s_x1
            src_gt_instances.bboxes[:, 1::2] += s_y1
            mixup_data_samples.gt_instances = dst_gt_instances.cat([dst_gt_instances, src_gt_instances])
            mixup_inputs_list.append(mixup_inputs)
            batch_mixup_data_samples.append(mixup_data_samples)
        return mixup_inputs_list, batch_mixup_data_samples

    def mosaic(self, inputs_list, batch_data_samples, noise_std=0.01):
        batch_size = len(inputs_list)
        h, w = 0, 0
        for i in range(batch_size):
            img_h, img_w = batch_data_samples[i].img_shape
            h, w = max(h, img_h), max(w, img_w)
        h, w = max(h, math.ceil(h / 16) * 16), max(w, math.ceil(w / 16) * 16)
        mosaic_inputs = inputs_list[0].new_zeros((1, 3, h * 2, w * 2))
        img_meta = batch_data_samples[0].metainfo
        img_meta['batch_input_shape'] = (h * 2, w * 2)
        img_meta['img_shape'] = (h * 2, w * 2)
        img_meta['pad_shape'] = (h * 2, w * 2)
        mosaic_data_samples = [DetDataSample(metainfo=img_meta)]
        mosaic_instances = []
        for i in range(batch_size):
            data_samples_i = copy.deepcopy(batch_data_samples[i])
            gt_instances_i = data_samples_i.gt_instances
            h_i, w_i = data_samples_i.img_shape
            if i == 0:
                mosaic_inputs[0, :, h - h_i:h, w - w_i:w] += inputs_list[i]
                gt_instances_i.bboxes[:, ::2] += w - w_i
                gt_instances_i.bboxes[:, 1::2] += h - h_i
            elif i == 1:
                mosaic_inputs[0, :, h - h_i:h, w:w + w_i] += inputs_list[i]
                gt_instances_i.bboxes[:, ::2] += w
                gt_instances_i.bboxes[:, 1::2] += h - h_i
            elif i == 2:
                mosaic_inputs[0, :, h:h + h_i, w - w_i:w] += inputs_list[i]
                gt_instances_i.bboxes[:, ::2] += w - w_i
                gt_instances_i.bboxes[:, 1::2] += h
            else:
                mosaic_inputs[0, :, h:h + h_i, w:w + w_i] += inputs_list[i]
                gt_instances_i.bboxes[:, ::2] += w
                gt_instances_i.bboxes[:, 1::2] += h
            mosaic_instances.append(gt_instances_i)

        mosaic_data_samples[0].gt_instances = mosaic_instances[0].cat(mosaic_instances)
        return mosaic_inputs, mosaic_data_samples

class EnhancedReliabilityCalculator(nn.Module):
    def __init__(self, 
                temp=0.2,
                min_scale=0.8,
                max_scale=1.2):
        super().__init__()  
        self.temp = temp              # 温度参数
        self.scale_factors = [min_scale, max_scale]

        
    def _transform_bboxes(self, data_sample, transform_info):

        pred_instances = data_sample.pred_instances
        bboxes = pred_instances.bboxes.clone()
        if transform_info['flip']:
            W = transform_info['ori_shape'][1]
            bboxes[:, [0, 2]] = W - bboxes[:, [2, 0]]
        scale_factor = transform_info['scale_factor']
        bboxes = bboxes / scale_factor
        
        bboxes[:, 0::2].clamp_(0, transform_info['ori_shape'][1])
        bboxes[:, 1::2].clamp_(0, transform_info['ori_shape'][0])
        
        return bboxes

    def forward(self, teacher, inputs, data_samples):
        orig_pred = teacher(inputs, data_samples, mode='predict')
        
        aug_preds = []
        transform_infos = []

        flip_inputs = inputs.flip(-1)
        flip_pred = teacher(flip_inputs, data_samples, mode='predict')
        for ds in data_samples:
            transform_infos.append({
                'flip': True,
                'scale_factor': 1.0,
                'ori_shape': ds.ori_shape
            })
        aug_preds.append(flip_pred)

        scale_factor = torch.empty(1).uniform_(*self.scale_factors).item()
        scaled_inputs = F.interpolate(
            inputs, 
            scale_factor=scale_factor, 
            mode='bilinear',
            align_corners=False,
            recompute_scale_factor=True)
        scale_pred = teacher(scaled_inputs, data_samples, mode='predict')
        for ds in data_samples:
            transform_infos.append({
                'flip': False,
                'scale_factor': scale_factor,
                'ori_shape': ds.ori_shape
            })
        aug_preds.append(scale_pred)
        
        aligned_aug_bboxes = []
        for pred_list, t_info in zip(aug_preds, transform_infos):
            for data_sample in pred_list:
                aligned_bbox = self._transform_bboxes(data_sample, t_info)
                new_instance =data_sample.clone()
                new_instance.pred_instances.bboxes = aligned_bbox
                aligned_aug_bboxes.append(new_instance)

        batch_reliability = []
        for data_sample in data_samples:
            instances = data_sample.gt_instances
            orig_bbox = instances.bboxes
            orig_score = instances.scores
            orig_labels = instances.labels
            total_consistency = torch.zeros_like(orig_score)
            valid_count = torch.zeros_like(orig_score)
            for aug_inst in aligned_aug_bboxes:
                aug_pred_instances = aug_inst.pred_instances
                if len(aug_pred_instances) == 0 or len(orig_bbox) == 0:
                    continue
                iou_matrix = bbox_overlaps(orig_bbox, aug_pred_instances.bboxes)
                if iou_matrix.size(1) == 0:
                    continue

                max_iou, match_idx = torch.max(iou_matrix, dim=1)
                consistency = max_iou

                matched_scores = aug_pred_instances.scores[match_idx]
                matched_labels = aug_pred_instances.labels[match_idx]
                same_class_mask = (orig_labels == matched_labels)
                score_diff = torch.where(
                    same_class_mask,
                    1 - (orig_score - matched_scores).abs(),
                    torch.zeros_like(orig_score)  
                )    
                
                total_consistency += consistency * score_diff
                valid_count += 1
            avg_consistency = total_consistency / (valid_count + 1e-7)
            reliability =avg_consistency
            reliability = torch.sigmoid(reliability / self.temp)
            batch_reliability.append(reliability)
        
        return  batch_reliability