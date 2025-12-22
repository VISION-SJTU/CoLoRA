# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES
import math
import numpy as np
from math import floor


class LoRA(nn.Module):
    def __init__(self, in_dim, out_dim, reduction=16,stride=1,kernel_size=1): # wsize larger better
        super(LoRA, self).__init__()
        self.rank = out_dim // reduction
        self.scaling = 1
        self.lora_A = nn.Conv2d(in_dim, self.rank, 1, 1, 0, bias=False)
        self.lora_B = nn.Conv2d(self.rank, out_dim, 1, stride=stride, padding=kernel_size//2, bias=False)
        self._init()
       
    def _init(self):
        # lora_B to zero leads better results
        torch.nn.init.zeros_(self.lora_B.weight)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
    
    def forward(self, x):
        scale_ab1_mean = math.sqrt(self.rank) * (self.lora_A.std() + self.lora_B.std())
        x = self.lora_A(x)
        x = self.lora_B(x)
        x = x * self.scaling / scale_ab1_mean.detach()
        return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, use_lora=False, use_adapter=False, 
                 adaptive_weight=False, lora_reduction=16, use_filter=False, alpha=16, share_ratio=0.5):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # PEFT settings
        self.use_lora = use_lora
        self.use_adapter = use_adapter
        self.adaptive_weight = adaptive_weight
        self.lora_reduction = lora_reduction
        self.use_filter = use_filter
        self.rank =  dim // (self.lora_reduction)
        self.share_rank=int(self.rank*share_ratio)
        self.indep_rank = self.rank - self.share_rank
        self.scaling1 = 1 # math.sqrt(dim // self.rank) 
        self.scaling2 = 1  # 1 / math.sqrt(dim) # math.sqrt(dim // self.rank) 
        self.rho = 0.99
        ## PEFT modules
        self.lora_dwconv = LoRA(dim, dim, reduction=lora_reduction)
        self.lora_pwconv1 = LoRA(dim, 4*dim, reduction=lora_reduction)
        self.lora_pwconv2 = LoRA(4*dim, dim, reduction=lora_reduction)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        ## PEFT branch
        lora_out = self.lora_dwconv(input)
        x = x + lora_out
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        # point convs
        conv_x = x
        x = self.pwconv1(x)
        lora_out = self.lora_pwconv1(conv_x)
        x = x + lora_out
        x = self.act(x)
        conv_x = x
        x = self.pwconv2(x)
        lora_out = self.lora_pwconv2(conv_x)
        x = x + lora_out
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

@BACKBONES.register_module()
class ConvNeXtPEFT(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3], use_lora=False, use_adapter=False, 
                 adaptive_weight=False, lora_reduction=16, use_filter=False, pretrained_ckpt=None):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value, use_lora=use_lora, use_adapter=use_adapter,
                adaptive_weight=adaptive_weight, lora_reduction=lora_reduction, use_filter=use_filter) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)
        ## CNT BASE params
        base_params = self.cnt_params(cnt_lora=False)
        ## INITIALIZE backbone using pretrined weights
        if pretrained_ckpt is not None:
            self.init_weights(pretrained=pretrained_ckpt)
            ## HACK linear layer if load pretrained weights
            # self.hack_linear() # will change number of parameters due to svd decomposition
        ## FREEZE backbone
        self.freeze_backbone()
        self.merge_correlated()
        # self.freeze_conv(type='dwconv')
        ## LOG
        logger = get_root_logger()
        logger.info("==> LORA FINETUNE ENABLED ... ")
        lora_params = self.cnt_params(cnt_lora=True)
        param_ratio = lora_params / base_params * 100
        logger = get_root_logger()
        logger.info("[LoRA setting], base params: {:.2f} M, lora params: {:.2f} M, ratio: {:.2f}%".format(base_params, lora_params, param_ratio))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
    
    def cnt_params(self, cnt_lora=False):
        num_params = 0
        if cnt_lora:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    num_params += param.numel()
        else:
            for name, param in self.named_parameters():
                if "lora" not in name:
                    num_params += param.numel()
        num_params /= 10**6
        return num_params
    
    def freeze_backbone(self):
        freeze_cnt = 0
        for name, param in self.named_parameters():
            if "lora" not in name:
                if "bias" not in name and "norm" not in name: # fix gamma, bias tuning
                    param.requires_grad_(False)
                    freeze_cnt += 1
        logger = get_root_logger()
        logger.info("===> Freeze convnext backbone, total {} params.".format(freeze_cnt))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x