from mmdet.registry import MODELS
import torch.nn as nn
import torch.nn.functional as F
@MODELS.register_module()
class ReliabilityAwareCrossEntropyLoss(nn.Module):
    def __init__(self, 
                 use_sigmoid=False, 
                 reduction='mean', 
                 loss_weight=1.0,
                 reduction_override=None):  # 新增参数
        super().__init__()
        assert not use_sigmoid, "仅支持Softmax模式"
        self.reduction = reduction
        self.loss_weight = loss_weight
        
        # 显式定义reduction_override处理
        self.reduction_override = reduction_override

    def forward(self, 
                cls_score, 
                label, 
                weight=None, 
                avg_factor=None,
                reduction_override=None):  # 添加参数
        # 处理reduction_override
        reduction = (
            reduction_override if reduction_override else self.reduction
        )
        
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