from mmdet.models.roi_heads import StandardRoIHead
from mmdet.registry import MODELS
@MODELS.register_module()
class ReliabilityAwareRoIHead(StandardRoIHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        # 1. 原始前向计算

        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        
        # 2. 提取可靠性权重
        reliability = []
        for res in sampling_results:
            num_pos = len(res.pos_inds)
            sample_rel = res.gt_instances.reliability[res.pos_assigned_gt_inds][:num_pos]
            reliability.append(sample_rel)
        reliability = torch.cat(reliability)
        # 3. 动态加权分类损失
        losses = self.bbox_head.loss(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            labels=gt_labels,
            label_weights=reliability,  # 注入可靠性权重
            bbox_targets=self.get_bbox_targets(sampling_results, gt_bboxes, self.train_cfg),
            bbox_weights=reliability,
            reduction_override='none'
        )
        
        # 4. 加权平均
        losses['loss_cls'] = (losses['loss_cls'] * reliability).sum() / reliability.sum().clamp(min=1e-6)
        losses['loss_bbox'] = (losses['loss_bbox'] * reliability).sum() / reliability.sum().clamp(min=1e-6)
        return losses