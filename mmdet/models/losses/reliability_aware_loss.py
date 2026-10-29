import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class ReliabilityAwareCrossEntropyLoss(nn.Module):
    """Weighted cross entropy with call-time reduction taking precedence."""

    def __init__(self,
                 use_sigmoid=False,
                 reduction='mean',
                 loss_weight=1.0,
                 reduction_override=None):
        super().__init__()
        assert not use_sigmoid, "仅支持Softmax模式"
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError('reduction must be none, mean or sum')
        if reduction_override not in (None, 'none', 'mean', 'sum'):
            raise ValueError(
                'reduction_override must be None, none, mean or sum')
        self.reduction = (reduction if reduction_override is None
                          else reduction_override)
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        if reduction_override not in (None, 'none', 'mean', 'sum'):
            raise ValueError(
                'reduction_override must be None, none, mean or sum')
        reduction = (self.reduction if reduction_override is None
                     else reduction_override)

        # 计算基础损失
        loss = F.cross_entropy(cls_score, label, reduction='none')

        # 应用可靠性权重
        if weight is not None:
            loss = loss * weight

        # 根据reduction参数处理
        if reduction == 'mean':
            if avg_factor is None:
                avg_factor = max(torch.sum(weight > 0).float().item(), 1.0) if weight is not None else loss.numel()
            loss = loss.sum() / avg_factor
        elif reduction == 'sum':
            loss = loss.sum()

        return self.loss_weight * loss
