from typing import List, Tuple

import torch
from torch import Tensor

from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class ReliabilityAwareRoIHead(StandardRoIHead):
    """Weight positive RoI losses by their assigned instance reliability."""

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        assert len(rpn_results_list) == len(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = unpack_gt_instances(
            batch_data_samples)

        sampling_results = []
        for i, rpn_results in enumerate(rpn_results_list):
            rpn_results.priors = rpn_results.pop('bboxes')
            gt_instances = batch_gt_instances[i]
            assign_result = self.bbox_assigner.assign(
                rpn_results, gt_instances, batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                gt_instances,
                feats=[lvl_feat[i][None] for lvl_feat in x])

            reliability = gt_instances.get('reliability', None)
            if reliability is not None:
                if (not isinstance(reliability, Tensor)
                        or reliability.ndim != 1
                        or reliability.numel() != len(gt_instances)):
                    raise ValueError(
                        'reliability must contain one score per instance')
                if (not torch.isfinite(reliability).all()
                        or (reliability < 0).any() or (reliability > 1).any()):
                    raise ValueError('reliability scores must be in [0, 1]')
                reliability = reliability.detach().to(
                    device=sampling_result.pos_assigned_gt_inds.device)
                sampling_result.pos_reliability = reliability[
                    sampling_result.pos_assigned_gt_inds]
            sampling_results.append(sampling_result)

        losses = dict()
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])

        if self.with_mask:
            mask_results = self.mask_loss(x, sampling_results,
                                          bbox_results['bbox_feats'],
                                          batch_gt_instances)
            losses.update(mask_results['loss_mask'])
        return losses

    def bbox_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        if not any(
                hasattr(res, 'pos_reliability') for res in sampling_results):
            return super().bbox_loss(x, sampling_results)

        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        targets = self.bbox_head.get_targets(sampling_results, self.train_cfg)
        labels, label_weights, bbox_targets, bbox_weights = targets

        # Keep the original sample normalization, including zero-weight RoIs.
        cls_avg_factor = max((label_weights > 0).sum().item(), 1.0)
        bbox_avg_factor = max(bbox_targets.size(0), 1)
        offset = 0
        for res in sampling_results:
            num_pos = res.pos_priors.size(0)
            if hasattr(res, 'pos_reliability'):
                reliability = res.pos_reliability.to(label_weights)
                label_weights[offset:offset + num_pos] *= reliability
                bbox_weights[offset:offset + num_pos] *= reliability[:, None]
            offset += num_pos + res.neg_priors.size(0)

        losses = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            rois,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            reduction_override='none')

        for name, criterion, avg_factor in (
            ('loss_cls', self.bbox_head.loss_cls, cls_avg_factor),
            ('loss_bbox', self.bbox_head.loss_bbox, bbox_avg_factor),
        ):
            if name not in losses:
                continue
            if criterion.reduction == 'mean':
                losses[name] = losses[name].sum() / avg_factor
            elif criterion.reduction == 'sum':
                losses[name] = losses[name].sum()

        bbox_results.update(loss_bbox=losses)
        return bbox_results
