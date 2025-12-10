# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet.models.necks.fpn import FPN
from mmdet.registry import MODELS


class SEModule(nn.Module):
    """Squeeze-and-Excitation模块"""
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


@MODELS.register_module()
class FineGrainedFPN(FPN):
    """细粒度特征金字塔网络"""
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 attention_type=None,
                 use_dcn=False,
                 dcn_cfg=None,
                 extra_enhancement=False,
                 fusion_method='sum',
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FineGrainedFPN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
            init_cfg=init_cfg)
        
        self.attention_type = attention_type
        self.use_dcn = use_dcn
        self.extra_enhancement = extra_enhancement
        self.fusion_method = fusion_method
        
        # 添加注意力机制
        if self.attention_type == 'se':
            self.attention_modules = nn.ModuleList([
                SEModule(out_channels) for _ in range(len(in_channels))
            ])
        
        # 使用可变形卷积替换普通卷积
        if self.use_dcn:
            self.dcn_cfg = dcn_cfg
            self.dcn_modules = nn.ModuleList()
            for i in range(len(in_channels)):
                self.dcn_modules.append(
                    build_conv_layer(
                        self.dcn_cfg,
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False))
        
        # 额外的特征增强模块
        if self.extra_enhancement:
            self.enhance_convs = nn.ModuleList()
            for i in range(len(in_channels)):
                self.enhance_convs.append(
                    ConvModule(
                        out_channels,
                        out_channels,
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
        
        # 自适应特征融合
        if self.fusion_method == 'adaptive':
            self.fusion_weights = nn.Parameter(
                torch.ones(len(in_channels), dtype=torch.float32))
            self.fusion_bias = nn.Parameter(
                torch.zeros(len(in_channels), dtype=torch.float32))

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # 构建laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 自上而下的路径
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # 在融合前应用注意力
            if hasattr(self, 'attention_modules'):
                laterals[i] = self.attention_modules[i](laterals[i])
                
            if self.fusion_method == 'adaptive':
                # 自适应融合
                weight = F.sigmoid(self.fusion_weights[i])
                prev_shape = laterals[i - 1].shape[2:]
                upsample_feat = F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
                laterals[i - 1] = laterals[i - 1] * (1 - weight) + upsample_feat * weight + self.fusion_bias[i]
            else:
                # 标准融合
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # 构建outputs
        outs = []
        for i in range(used_backbone_levels):
            if self.use_dcn:
                x = self.dcn_modules[i](laterals[i])
            else:
                x = laterals[i]
                
            if self.extra_enhancement:
                x = self.enhance_convs[i](x)
                
            outs.append(self.fpn_convs[i](x))

        # 额外的级别
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(
                        outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                elif self.add_extra_convs == 'on_lateral':
                    outs.append(self.fpn_convs[used_backbone_levels](
                        laterals[-1]))
                elif self.add_extra_convs == 'on_output':
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                else:
                    raise NotImplementedError
                    
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
                        
        return tuple(outs)