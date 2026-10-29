# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional
import torch.nn as nn
import torch
from torch import Tensor

from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                     reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors import SemiBaseDetector
from mmdet.structures.bbox import bbox_project,bbox_overlaps
from torch.nn import functional as F
import numpy as np
import math



@MODELS.register_module()
class RSOD(SemiBaseDetector):
    """Semi-supervised detector with reliability-weighted pseudo labels."""

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
        self._get_cache_size()
        self.cache_inputs = []
        self.cache_data_samples = []
        self.reliability_calculator = EnhancedReliabilityCalculator(
            temp=semi_train_cfg.get('temp', 0.6),
            min_scale=semi_train_cfg.get('min_scale', 0.8),
            max_scale=semi_train_cfg.get('max_scale', 1.2)
        )
    def merge_reliable_instances(self,
                                 batch_inputs,
                                 batch_data_samples,
                                 reliability_scores,
                                 threshold=0.5):
        if (len(batch_inputs) != len(batch_data_samples)
                or len(reliability_scores) != len(batch_data_samples)):
            raise ValueError('Inputs, samples and reliability must match')
        for sample, scores in zip(batch_data_samples, reliability_scores):
            if scores.ndim != 1 or len(sample.gt_instances) != len(scores):
                raise ValueError(
                    'Each instance must have one reliability score')
        if len(batch_data_samples) <= 1:
            return batch_inputs, batch_data_samples, reliability_scores

        target_img = batch_inputs[0].clone()
        target_sample = batch_data_samples[0]
        target_h, target_w = target_sample.img_shape[:2]
        if not (0 < target_h <= target_img.shape[-2]
                and 0 < target_w <= target_img.shape[-1]):
            raise ValueError(
                'Target img_shape must fit inside its input tensor')
        merged_instances = target_sample.gt_instances.clone()
        current_boxes = merged_instances.bboxes
        merged_reliability_scores = [reliability_scores[0].clone()]
        overlap_threshold = self.semi_train_cfg.get('non_overlap_threshold',
                                                    0.2)
        score_threshold = self.semi_train_cfg.get('cls_pseudo_thr', 0.7)
        occlusion_threshold = self.semi_train_cfg.get(
            'paste_occlusion_threshold', 0.3)
        if not 0 <= occlusion_threshold <= 1:
            raise ValueError('Paste occlusion threshold must be in [0, 1]')

        area_dtype = (torch.float64 if current_boxes.dtype == torch.float64
                      else torch.float32)
        visible_regions = current_boxes.to(dtype=area_dtype).clone()
        visible_regions[:, 0::2].clamp_(0, target_w)
        visible_regions[:, 1::2].clamp_(0, target_h)
        box_areas = (visible_regions[:, 2:] -
                     visible_regions[:, :2]).clamp_min(0).prod(-1)
        region_owners = torch.arange(
            len(current_boxes), device=current_boxes.device)
        valid_regions = box_areas > 0
        visible_regions = visible_regions[valid_regions]
        region_owners = region_owners[valid_regions]

        for idx in range(1, len(batch_data_samples)):
            curr_img = batch_inputs[idx]
            src_h, src_w = batch_data_samples[idx].img_shape[:2]
            if not (0 < src_h <= curr_img.shape[-2]
                    and 0 < src_w <= curr_img.shape[-1]):
                raise ValueError(
                    'Source img_shape must fit inside its input tensor')
            curr_img = curr_img[:, :src_h, :src_w]
            curr_instances = batch_data_samples[idx].gt_instances
            curr_scores = reliability_scores[idx]
            reliable_mask = curr_scores > threshold
            if score_threshold is not None:
                reliable_mask &= curr_instances.scores > score_threshold
            if not reliable_mask.any():
                continue
            curr_reliable_instances = curr_instances[reliable_mask]
            curr_reliable_scores = curr_scores[reliable_mask]
            enhanced_boxes, enhanced_img_regions = self._apply_random_augmentation(
                curr_img, curr_reliable_instances.bboxes,
                curr_reliable_instances)
            if (len(enhanced_boxes) != len(curr_reliable_instances) or
                    len(enhanced_img_regions) != len(curr_reliable_instances)):
                raise ValueError(
                    'Augmented boxes and image regions must match instances')

            for i, (box, region) in enumerate(
                    zip(enhanced_boxes, enhanced_img_regions)):
                if region is None or not torch.isfinite(box).all():
                    continue
                left, top, right, bottom = map(int, box.tolist())
                if right <= left or bottom <= top:
                    continue
                if (region.ndim != 3 or region.shape[0] != target_img.shape[0]
                        or region.shape[-2:] != (bottom - top, right - left)):
                    raise ValueError(
                        'Each image region must fit its placement box')

                x1, y1 = max(0, left), max(0, top)
                x2, y2 = min(target_w, right), min(target_h, bottom)
                if x2 <= x1 or y2 <= y1:
                    continue
                visible_box = box.new_tensor([x1, y1, x2, y2])
                if len(current_boxes) > 0:
                    ious = bbox_overlaps(visible_box[None].float(),
                                         current_boxes.float())
                    if ious.max() >= overlap_threshold:
                        continue

                patch_box = visible_box.to(visible_regions)
                intersection_lt = torch.maximum(visible_regions[:, :2],
                                                patch_box[:2])
                intersection_rb = torch.minimum(visible_regions[:, 2:],
                                                patch_box[2:])
                intersecting = (intersection_rb > intersection_lt).all(-1)
                next_regions, next_owners = visible_regions, region_owners
                if intersecting.any():
                    if occlusion_threshold == 0:
                        continue
                    # Keep disjoint visible pieces so overlap is counted once.
                    rectangles = visible_regions[intersecting]
                    left_top = intersection_lt[intersecting]
                    right_bottom = intersection_rb[intersecting]
                    pieces = torch.stack([
                        torch.stack((rectangles[:, 0], rectangles[:, 1],
                                     left_top[:, 0], rectangles[:, 3]), -1),
                        torch.stack((right_bottom[:, 0], rectangles[:, 1],
                                     rectangles[:, 2], rectangles[:, 3]), -1),
                        torch.stack((left_top[:, 0], rectangles[:, 1],
                                     right_bottom[:, 0], left_top[:, 1]), -1),
                        torch.stack((left_top[:, 0], right_bottom[:, 1],
                                     right_bottom[:, 0], rectangles[:, 3]), -1)
                    ], dim=1).reshape(-1, 4)
                    piece_owners = region_owners[intersecting].repeat_interleave(
                        4)
                    nonempty = (pieces[:, 2:] > pieces[:, :2]).all(-1)
                    next_regions = torch.cat(
                        [visible_regions[~intersecting], pieces[nonempty]])
                    next_owners = torch.cat(
                        [region_owners[~intersecting], piece_owners[nonempty]])
                    remaining_areas = box_areas.new_zeros(len(box_areas))
                    remaining_areas.index_add_(
                        0, next_owners,
                        (next_regions[:, 2:] - next_regions[:, :2]).prod(-1))
                    occluded_ratio = (
                        (box_areas - remaining_areas).clamp_min(0) /
                        box_areas.clamp_min(torch.finfo(area_dtype).tiny))
                    if ((occluded_ratio > 0)
                            & (occluded_ratio >= occlusion_threshold)).any():
                        continue

                # Crop the corresponding pixels when a patch crosses the edge.
                visible_region = region[:, y1 - top:y2 - top,
                                        x1 - left:x2 - left]
                target_img[:, y1:y2, x1:x2] = visible_region.to(target_img)
                instance = curr_reliable_instances[i:i + 1].clone()
                instance.bboxes = visible_box[None]
                merged_instances = merged_instances.cat(
                    [merged_instances, instance])
                current_boxes = merged_instances.bboxes
                merged_reliability_scores[0] = torch.cat([
                    merged_reliability_scores[0], curr_reliable_scores[i:i + 1]
                ])
                # A pasted object's visibility starts when it is inserted.
                visible_regions = torch.cat([next_regions, patch_box[None]])
                region_owners = torch.cat([
                    next_owners, region_owners.new_tensor([len(box_areas)])
                ])
                box_areas = torch.cat([
                    box_areas, (patch_box[2:] - patch_box[:2]).prod().reshape(1)
                ])

        batch_inputs[0] = target_img
        batch_data_samples[0].gt_instances = merged_instances
        merged_reliability_scores.extend(reliability_scores[1:])
        return batch_inputs, batch_data_samples, merged_reliability_scores

    def _apply_random_augmentation(self, img, boxes, instances):
        enhanced_boxes = boxes.clone()
        enhanced_img_regions = [None] * len(boxes)

        for i, box in enumerate(boxes):
            if (not torch.isfinite(box).all() or box[2] <= box[0]
                    or box[3] <= box[1]):
                continue
            x1 = max(0, math.floor(box[0].item()))
            y1 = max(0, math.floor(box[1].item()))
            x2 = min(img.shape[-1], math.ceil(box[2].item()))
            y2 = min(img.shape[-2], math.ceil(box[3].item()))
            if x2 <= x1 or y2 <= y1:
                continue

            region = img[:, y1:y2, x1:x2].clone()
            if region.dtype not in (torch.float32, torch.float64):
                region = region.float()
            aug_type = torch.randint(0, 2, (1, )).item()
            if aug_type == 0:
                scale_factor = 0.95 + torch.rand(1).item() * 0.1
                new_h = max(1, int(round(region.shape[-2] * scale_factor)))
                new_w = max(1, int(round(region.shape[-1] * scale_factor)))
                region = F.interpolate(
                    region.unsqueeze(0),
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False).squeeze(0)
            else:
                from torchvision.transforms import InterpolationMode
                from torchvision.transforms import functional as TF

                angle = (torch.rand(1).item() - 0.5) * 50
                # Keep normalized pixel values and channel order throughout.
                region = TF.rotate(
                    region,
                    angle,
                    interpolation=InterpolationMode.BILINEAR,
                    expand=True,
                    fill=0.)

            new_h, new_w = region.shape[-2:]
            left = math.floor((x1 + x2 - new_w) / 2)
            top = math.floor((y1 + y2 - new_h) / 2)
            enhanced_boxes[i] = box.new_tensor(
                [left, top, left + new_w, top + new_h])
            enhanced_img_regions[i] = region.to(dtype=img.dtype)

        return enhanced_boxes, enhanced_img_regions

    def calculate_corner_distance_loss(self,
                                       batch_unsup_inputs,
                                       batch_unsup_data_samples,
                                       filtered_reliability_scores,
                                       reliability_threshold=0.5,
                                       iou_threshold=0.1,
                                       k=2):
        batch_size = len(batch_unsup_data_samples)
        if (len(batch_unsup_inputs) != batch_size
                or len(filtered_reliability_scores) != batch_size):
            raise ValueError('Inputs, samples and reliability must match')
        if not 0 <= iou_threshold <= 1 or k <= 0:
            raise ValueError('IoU threshold must be in [0, 1] and k positive')

        total_loss = batch_unsup_inputs.new_zeros(())
        for parameter in self.student.parameters():
            if parameter.requires_grad:
                total_loss = parameter.reshape(-1)[:1].sum() * 0
                break
        if batch_size == 0:
            return total_loss

        # Prediction writes into data samples; keep that graph out of the cache.
        prediction_samples = copy.deepcopy(batch_unsup_data_samples)
        student_results = self.student.predict(
            batch_unsup_inputs, prediction_samples, rescale=False)
        if len(student_results) != batch_size:
            raise ValueError('Student predictions must match the batch size')
        valid_pairs = 0

        for student_result, data_sample, reliability_scores in zip(
                student_results, batch_unsup_data_samples,
                filtered_reliability_scores):
            student_bboxes = student_result.pred_instances.bboxes.float()
            teacher_bboxes = data_sample.gt_instances.bboxes.detach().to(
                student_bboxes)
            if (reliability_scores.ndim != 1
                    or len(reliability_scores) != len(teacher_bboxes)):
                raise ValueError(
                    'Each teacher box must have one reliability score')
            reliable_mask = reliability_scores.detach().to(
                teacher_bboxes.device) > reliability_threshold
            teacher_bboxes = teacher_bboxes[reliable_mask]
            if len(teacher_bboxes) == 0 or len(student_bboxes) == 0:
                continue

            # Matching chooses indices; the selected student boxes keep gradients.
            with torch.no_grad():
                ious = bbox_overlaps(student_bboxes.detach(), teacher_bboxes)
                max_ious, matched_student_indices = ious.max(dim=0)
                valid_matches = max_ious > iou_threshold
            if valid_matches.any():
                matched_boxes = student_bboxes[
                    matched_student_indices[valid_matches]]
                pair_losses = self._compute_corner_loss(
                    matched_boxes, teacher_bboxes[valid_matches], k=k)
                total_loss = total_loss + pair_losses.sum()
                valid_pairs += pair_losses.numel()

        return total_loss / max(valid_pairs, 1)

    def _compute_corner_loss(self, box1, box2, k=2):
        box1, box2 = box1.float(), box2.float()
        numerator = 2 * (box1 - box2).square().sum(dim=-1)
        enclosing_wh = (
            torch.maximum(box1[..., 2:], box2[..., 2:]) -
            torch.minimum(box1[..., :2], box2[..., :2]))
        diagonal_sq = enclosing_wh.square().sum(dim=-1).clamp_min(1e-16)
        return numerator / diagonal_sq.pow(k / 2)

    def loss_by_pseudo_instances(self,
                                 batch_inputs: Tensor,
                                 batch_data_samples: SampleList,
                                 batch_info: Optional[dict] = None) -> dict:
        if len(batch_inputs) != len(batch_data_samples):
            raise ValueError('Inputs and samples must have equal batch size')
        batch_data_samples = filter_gt_instances(
            batch_data_samples,
            score_thr=self.semi_train_cfg.get('cls_pseudo_thr', 0.7))
        valid_indices = [
            i for i, sample in enumerate(batch_data_samples)
            if len(sample.gt_instances) >= self.semi_train_cfg.least_num
        ]
        if valid_indices and len(valid_indices) < len(batch_data_samples):
            batch_inputs = batch_inputs[valid_indices]
            batch_data_samples = [batch_data_samples[i] for i in valid_indices]
        unsup_weight = self.semi_train_cfg.unsup_weight if valid_indices else 0.
        losses = self.student.loss(batch_inputs, batch_data_samples)
        reliability_scores = [
            sample.gt_instances.reliability for sample in batch_data_samples
        ]
        losses['loss_corner'] = self.calculate_corner_distance_loss(
            batch_inputs,
            batch_data_samples,
            reliability_scores,
            reliability_threshold=self.semi_train_cfg.get(
                'reliability_threshold', 0.5),
            iou_threshold=self.semi_train_cfg.get(
                'corner_loss_iou_threshold',
                self.semi_train_cfg.get('iou_threshold', 0.1)),
            k=self.semi_train_cfg.get('corner_loss_k', 2))
        return rename_loss_dict('unsup_',
                                reweight_loss_dict(losses, unsup_weight))

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        cache_size = self._get_cache_size()
        cache_count = len(self.cache_inputs)
        if cache_count != len(self.cache_data_samples):
            raise ValueError('Image and annotation cache lengths must match')
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
        batch_unsup_inputs, batch_unsup_data_samples, _ = self.merge_reliable_instances(
            batch_unsup_inputs, 
            batch_unsup_data_samples,
            filtered_reliability_scores,
            threshold=self.semi_train_cfg.get('reliability_threshold', 0.5)
        )
        
        batch_unsup_inputs, batch_unsup_data_samples = self.merge(
            *zip(*list(map(self.erase, *self.split(batch_unsup_inputs, batch_unsup_data_samples)))))

        sample_size = len(multi_batch_data_samples['unsup_student'])
        cache_ready = cache_size > 0 and cache_count >= cache_size
        use_mixup = (self.semi_train_cfg.mixup and cache_ready
                     and 0 < sample_size <= cache_count)
        use_mosaic = (self.semi_train_cfg.mosaic and cache_ready
                      and cache_count >= 4)
        cache_order = (np.random.permutation(cache_count)
                       if use_mixup or use_mosaic else None)
        if use_mixup:
            mixup_idxs = cache_order[:sample_size]
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

        if use_mosaic:
            mosaic_start = sample_size if use_mixup else 0
            mosaic_idxs = np.roll(cache_order, -mosaic_start)[:4]
            img_shapes = [self._sample_mosaic_shape()] * 4
            src_inputs_list, batch_src_data_samples = self.get_batch(mosaic_idxs, img_shapes)
            mosaic_inputs, mosaic_data_samples = self.mosaic(src_inputs_list, batch_src_data_samples)
            mosaic_losses = self.loss_by_pseudo_instances(mosaic_inputs, mosaic_data_samples)
            losses.update(**rename_loss_dict('mosaic_', reweight_loss_dict(mosaic_losses, self.semi_train_cfg.mosaic_weight)))

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

    def _sample_mosaic_shape(self):
        """Sample each resize bound independently, keeping equal bounds fixed."""
        shapes = self.semi_train_cfg.get('mosaic_shape')
        message = ('mosaic_shape must contain one or two (height, width) '
                   'pairs of positive integers')
        if not isinstance(shapes, (list, tuple)) or len(shapes) not in (1, 2):
            raise ValueError(message)
        for shape in shapes:
            if not isinstance(shape, (list, tuple)) or len(shape) != 2:
                raise ValueError(message)
            if any(isinstance(value, (bool, np.bool_))
                   or not isinstance(value, (int, np.integer))
                   or value <= 0 for value in shape):
                raise ValueError(message)
        if len(shapes) == 1:
            return tuple(int(value) for value in shapes[0])
        sampled = []
        for first, second in zip(*shapes):
            lower, upper = sorted((int(first), int(second)))
            sampled.append(lower if lower == upper else
                           int(np.random.randint(lower, upper)))
        return tuple(sampled)

    def _get_cache_size(self):
        cache_size = self.semi_train_cfg.get('cache_size', 8)
        if (isinstance(cache_size, (bool, np.bool_))
                or not isinstance(cache_size, (int, np.integer))
                or cache_size < 0):
            raise ValueError('cache_size must be a non-negative integer')
        return int(cache_size)

    def update_cache(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        cache_size = self._get_cache_size()
        if len(self.cache_inputs) != len(self.cache_data_samples):
            raise ValueError('Image and annotation cache lengths must match')
        if len(batch_inputs) != len(batch_data_samples):
            raise ValueError('Inputs and samples must have equal batch size')
        if cache_size == 0:
            self.cache_inputs.clear()
            self.cache_data_samples.clear()
            return
        inputs_list, batch_data_samples = self.split(batch_inputs, batch_data_samples)
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
        if (not isinstance(img_shape, (list, tuple)) or len(img_shape) != 2
                or any(isinstance(value, (bool, np.bool_))
                       or not isinstance(value, (int, np.integer))
                       or value <= 0 for value in img_shape)):
            raise ValueError('Resize bounds must be two positive integers')
        source_shape = data_samples.get('img_shape', ())
        if (inputs.ndim != 3 or not isinstance(source_shape, (list, tuple))
                or len(source_shape) < 2
                or any(isinstance(value, (bool, np.bool_))
                       or not isinstance(value, (int, np.integer))
                       for value in source_shape[:2])):
            raise ValueError('img_shape must describe a valid input image')
        source_h, source_w = map(int, source_shape[:2])
        if not (0 < source_h <= inputs.shape[-2]
                and 0 < source_w <= inputs.shape[-1]):
            raise ValueError('img_shape must fit inside the input tensor')
        bound_h, bound_w = map(int, img_shape)
        img_h = max(1, min(bound_h, source_h * bound_w // source_w))
        img_w = max(1, min(bound_w, source_w * bound_h // source_h))
        image = inputs[:, :source_h, :source_w]
        if image.dtype in (torch.float16, torch.bfloat16):
            image = image.float()
        resized = F.interpolate(
            image.unsqueeze(0), size=(img_h, img_w),
            mode='nearest').squeeze(0).to(dtype=inputs.dtype)

        scale_x, scale_y = img_w / source_w, img_h / source_h
        matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]],
                          dtype=np.float32)
        homography = matrix @ data_samples.homography_matrix
        boxes = data_samples.gt_instances.bboxes
        boxes = boxes * boxes.new_tensor([scale_x, scale_y, scale_x, scale_y])
        boxes[:, 0::2].clamp_(0, img_w)
        boxes[:, 1::2].clamp_(0, img_h)
        ori_h, ori_w = data_samples.ori_shape[:2]
        data_samples.set_metainfo(dict(
            img_shape=(img_h, img_w),
            scale_factor=(img_w / ori_w, img_h / ori_h),
            homography_matrix=homography))
        data_samples.gt_instances.bboxes = boxes
        return resized, filter_gt_instances([data_samples])[0]

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
        if not self.semi_train_cfg.get('erase', True):
            return inputs, data_samples

        def _get_patches(img_shape):
            patches = []
            n_patches = np.random.randint(self.semi_train_cfg.erase_patches[0],
                                          self.semi_train_cfg.erase_patches[1])
            for _ in range(n_patches):
                ratio = np.random.random() * \
                        (self.semi_train_cfg.erase_ratio[1] - self.semi_train_cfg.erase_ratio[0]) + \
                        self.semi_train_cfg.erase_ratio[0]
                ph, pw = int(img_shape[0] * ratio), int(img_shape[1] * ratio)
                px1 = np.random.randint(0, img_shape[1] - pw + 1)
                py1 = np.random.randint(0, img_shape[0] - ph + 1)
                px2, py2 = px1 + pw, py1 + ph
                patches.append([px1, py1, px2, py2])
            return patches

        erase_patches = _get_patches(data_samples.img_shape)
        if not erase_patches:
            return inputs, data_samples
        for patch in erase_patches:
            px1, py1, px2, py2 = patch
            inputs[:, py1:py2, px1:px2] = 0
        bboxes = data_samples.gt_instances.bboxes
        if len(bboxes) == 0:
            return inputs, data_samples
        patches = torch.tensor(erase_patches, device=bboxes.device)
        erased_areas = self._bbox_erased_areas(bboxes, patches)
        coordinates = bboxes.to(dtype=erased_areas.dtype)
        bbox_areas = (coordinates[:, 2:] -
                      coordinates[:, :2]).clamp_min(0).prod(-1)
        bboxes_erased_ratio = (erased_areas / (bbox_areas + 1e-7)).clamp(0, 1)
        valid_inds = bboxes_erased_ratio < self.semi_train_cfg.erase_thr
        data_samples.gt_instances = data_samples.gt_instances[valid_inds]
        return inputs, data_samples

    @staticmethod
    def _bbox_erased_areas(bboxes, patches):
        """Intersect each box with the union of all erased rectangles."""
        boxes = bboxes if bboxes.dtype == torch.float64 else bboxes.float()
        areas = boxes.new_zeros(len(boxes))
        if boxes.numel() == 0 or patches.numel() == 0:
            return areas
        patches = patches.to(boxes).reshape(-1, 4)
        patches = patches[(patches[:, 2] > patches[:, 0])
                          & (patches[:, 3] > patches[:, 1])]
        if len(patches) == 0:
            return areas

        # Patch edges partition the union into disjoint rectangular cells.
        xs = patches[:, [0, 2]].flatten().unique(sorted=True)
        ys = patches[:, [1, 3]].flatten().unique(sorted=True)
        covered_x = ((xs[:-1][None] >= patches[:, 0, None])
                     & (xs[1:][None] <= patches[:, 2, None]))
        covered_y = ((ys[:-1][None] >= patches[:, 1, None])
                     & (ys[1:][None] <= patches[:, 3, None]))
        covered = (covered_y[:, :, None] & covered_x[:, None, :]).any(dim=0)
        widths = (torch.minimum(boxes[:, 2, None], xs[None, 1:]) -
                  torch.maximum(boxes[:, 0, None], xs[None, :-1])).clamp_min(0)
        heights = (
            torch.minimum(boxes[:, 3, None], ys[None, 1:]) -
            torch.maximum(boxes[:, 1, None], ys[None, :-1])).clamp_min(0)
        return (heights[:, :, None] * widths[:, None, :] *
                covered[None]).sum(dim=(1, 2))

    def mixup(self, dst_inputs_list, batch_dst_data_samples, src_inputs_list, batch_src_data_samples):
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

    def mosaic(self, inputs_list, batch_data_samples):
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

    def __init__(self, temp=0.6, min_scale=0.8, max_scale=1.2):
        super().__init__()
        if temp <= 0 or not 0 < min_scale <= max_scale:
            raise ValueError('Temperature and scale bounds must be positive')
        self.temp = temp
        self.scale_factors = [min_scale, max_scale]

    def _build_view(self, inputs, data_samples, scale_factor, flip,
                    pad_size_divisor, pad_value):
        images, view_samples = [], []
        for image, sample in zip(inputs, data_samples):
            img_h, img_w = sample.img_shape[:2]
            if not (0 < img_h <= image.shape[-2]
                    and 0 < img_w <= image.shape[-1]):
                raise ValueError('img_shape must fit inside the input tensor')

            # Transform valid pixels, then restore right/bottom batch padding.
            image = image[:, :img_h, :img_w]
            new_h = max(1, int(img_h * scale_factor))
            new_w = max(1, int(img_w * scale_factor))
            if (new_h, new_w) != (img_h, img_w):
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False).squeeze(0)
            scale_x, scale_y = new_w / img_w, new_h / img_h
            transform = np.diag([scale_x, scale_y, 1]).astype(np.float32)
            view_sample = copy.deepcopy(sample)

            if flip:
                image = image.flip(-1)
                flip_matrix = np.array([[-1, 0, new_w], [0, 1, 0], [0, 0, 1]],
                                       dtype=np.float32)
                transform = flip_matrix @ transform
                direction = (
                    sample.get('flip_direction', None) if sample.get(
                        'flip', False) else None)
                direction = {
                    None: 'horizontal',
                    'horizontal': None,
                    'vertical': 'diagonal',
                    'diagonal': 'vertical'
                }[direction]
                view_sample.set_metainfo(
                    dict(flip=direction is not None, flip_direction=direction))

            homography = sample.homography_matrix
            if isinstance(homography, Tensor):
                homography = homography.detach().cpu().numpy()
            homography = transform @ np.asarray(homography, dtype=np.float32)
            original_scale = sample.get('scale_factor', (1., 1.))
            pad_h = math.ceil(new_h / pad_size_divisor) * pad_size_divisor
            pad_w = math.ceil(new_w / pad_size_divisor) * pad_size_divisor
            view_sample.set_metainfo(
                dict(
                    img_shape=(new_h, new_w),
                    pad_shape=(pad_h, pad_w),
                    scale_factor=(original_scale[0] * scale_x,
                                  original_scale[1] * scale_y),
                    homography_matrix=homography))
            images.append(image)
            view_samples.append(view_sample)

        batch_h = max(sample.pad_shape[0] for sample in view_samples)
        batch_w = max(sample.pad_shape[1] for sample in view_samples)
        view_inputs = inputs.new_full(
            (len(images), inputs.shape[1], batch_h, batch_w), pad_value)
        for i, (image, sample) in enumerate(zip(images, view_samples)):
            view_inputs[i, :, :image.shape[-2], :image.shape[-1]] = image
            sample.set_metainfo(dict(batch_input_shape=(batch_h, batch_w)))
        return view_inputs, view_samples

    def _transform_bboxes(self, data_sample):
        bboxes = data_sample.pred_instances.bboxes.float()
        homography = torch.as_tensor(
            data_sample.homography_matrix,
            device=bboxes.device,
            dtype=bboxes.dtype)
        return bbox_project(bboxes, homography.inverse(),
                            data_sample.ori_shape[:2])

    @torch.no_grad()
    def forward(self, teacher, inputs, data_samples):
        if inputs.ndim != 4 or inputs.shape[0] != len(data_samples):
            raise ValueError(
                'Input batch and data samples must have equal size')
        if not data_samples:
            return []

        preprocessor = getattr(teacher, 'data_preprocessor', None)
        pad_size_divisor = getattr(preprocessor, 'pad_size_divisor', 32)
        pad_value = getattr(preprocessor, 'pad_value', 0)
        scale_factor = torch.empty(1).uniform_(*self.scale_factors).item()
        total_consistency = [
            torch.zeros_like(sample.gt_instances.scores)
            for sample in data_samples
        ]
        views = [(1., True), (scale_factor, False)]

        for view_scale, flip in views:
            view_inputs, view_samples = self._build_view(
                inputs, data_samples, view_scale, flip, pad_size_divisor,
                pad_value)
            results = teacher.predict(view_inputs, view_samples, rescale=False)
            if len(results) != len(data_samples):
                raise ValueError(
                    'Teacher predictions must match the batch size')

            for i, (sample, result) in enumerate(zip(data_samples, results)):
                instances = sample.gt_instances
                predictions = result.pred_instances
                if len(instances) == 0 or len(predictions) == 0:
                    continue

                aligned_bboxes = self._transform_bboxes(result)
                ious = bbox_overlaps(instances.bboxes.float(), aligned_bboxes)
                max_iou, match_idx = ious.max(dim=1)
                orig_score = instances.scores
                orig_labels = instances.labels
                matched_scores = predictions.scores[match_idx]
                matched_labels = predictions.labels[match_idx]
                same_class_mask = (orig_labels == matched_labels)
                score_diff = torch.where(
                    same_class_mask, 1 - (orig_score - matched_scores).abs(),
                    torch.zeros_like(orig_score))
                total_consistency[i] += max_iou * score_diff

        # Average over every view; missing predictions contribute zero.
        return [
            torch.sigmoid(consistency / len(views) / self.temp)
            for consistency in total_consistency
        ]
