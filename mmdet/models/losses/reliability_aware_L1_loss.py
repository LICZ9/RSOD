from mmdet.registry import MODELS
import torch.nn as nn
import torch
import torch.nn.functional as F
@MODELS.register_module()
class ReliabilityAwareL1Loss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None,reduction_override=None):
        """
        pred: (N, 4) 预测的边界框偏移量
        target: (N, 4) 目标偏移量
        weight: (N,) 每个样本的可靠性权重
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override or self.reduction
        raw_loss = torch.abs(pred - target)
        
        if weight is None:
            # 没有权重，直接对坐标求和得到 (N,)
            loss = raw_loss.sum(dim=1)
        else:
            # 如果是 (N,) 的样本级权重
            if weight.dim() == 1 and weight.shape[0] == raw_loss.shape[0]:
                # 先对坐标求和，再乘权重
                loss = raw_loss.sum(dim=1) * weight
            # 如果是 (N,4) 的坐标级权重
            elif weight.dim() == 2 and weight.shape == raw_loss.shape:
                # 逐坐标乘权重，再对坐标求和
                loss = (raw_loss * weight).sum(dim=1)
            else:
                raise ValueError(
                    f"Unsupported weight shape {tuple(weight.shape)}; "
                    f"expected (N,) or (N,4) to match pred {tuple(pred.shape)}"
                )
            
        if reduction == 'mean':
            avg = loss.numel() if avg_factor is None else avg_factor
            loss = loss.sum() / avg
        elif self.reduction == 'sum':
            loss = loss.sum()
            
        return self.loss_weight * loss