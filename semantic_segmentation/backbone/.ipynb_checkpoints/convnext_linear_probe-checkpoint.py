# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
# from transformers import pipeline
import os
# from safetensors import safe_open
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger


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
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

@BACKBONES.register_module()
class ConvNeXtFreeze(nn.Module):
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
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],
                 last_norm=False, pretrain_ckpt=""):
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
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.last_norm = last_norm
        if self.last_norm: 
            self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.apply(self._init_weights)
        self.load_model(pretrain_ckpt)
        for param in self.parameters():
            param.requires_grad_(False)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def load_model(self, pretrain_ckpt):
        if "dinov3" not in pretrain_ckpt:
            state = torch.load(pretrain_ckpt, map_location="cpu")
            state = state.get("module", state)
            state = state.get("model", state)
            for name, _ in self.named_parameters():
                if name not in state:
                    logger.info(f"Warning: {name} not in pretrain checkpoint")
            self.load_state_dict(state, strict=False)
        else:
            state = torch.load(pretrain_ckpt)
            tran_state = dict()
            for k in state.keys():
                v = state[k]
                if "downsample" in k:
                    new_k = k.replace("downsample_layers.", "")
                    new_k = new_k.replace("stages", "downsample_layers")
                    tran_state[new_k] = v
                elif "stages" in k:
                    k = k.replace("layers.", "")
                    if "depthwise" in k:
                        tran_state[k.replace("depthwise_conv", "dwconv")] = v
                    elif "pointwise" in k:
                        tran_state[k.replace("pointwise_conv", "pwconv")] = v
                    elif "gamma" in k:
                        tran_state[k] = v
                    elif "layer_norm" in k:
                        tran_state[k.replace("layer_norm", "norm")] = v
                elif "layer_norm" in k:
                    tran_state[k.replace("layer_norm", "norm")] = v
                    
            for name, _ in self.named_parameters():
                if name not in tran_state.keys():
                    print(f"Warning: {name} not in pretrain checkpoint")
            self.load_state_dict(tran_state, strict=False)
        print("==> Load pretrain checkpoint from: {}".format(pretrain_ckpt))
    
    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        if self.last_norm:
            x = self.norm(x)
        x = x.detach() # avoid backward
        return [x]
    
    def freeze(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.requires_grad_(False)

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

def convnext_b(pretrain_ckpt=""):
    model = ConvNeXt(depths=[3, 3, 27, 3], drop_path_rate=0.0, dims=[128, 256, 512, 1024])
    if os.path.exists(pretrain_ckpt):
        model.load_model(pretrain_ckpt)
    return model

def convnext_b_norm(pretrain_ckpt="", last_norm=True):
    model = ConvNeXt(depths=[3, 3, 27, 3], drop_path_rate=0.0, dims=[128, 256, 512, 1024], last_norm=True)
    if os.path.exists(pretrain_ckpt):
        model.load_model(pretrain_ckpt)
    return model

def convnext_l(pretrain_ckpt=""):
    model = ConvNeXt(depths=[3, 3, 27, 3], drop_path_rate=0.0, dims=[192, 384, 768, 1536])
    if os.path.exists(pretrain_ckpt):
        model.load_model(pretrain_ckpt)
    return model

# class DINOv3ConvNeXt(nn.Module):
#     def __init__(self, base_dir) -> None:
#         super().__init__()
#         # We use safetensors directory to initialize DINOv3
#         feature_extractor = pipeline(
#             model=base_dir,
#             task="image-feature-extraction", 
#         )
#         self.model = feature_extractor.model
    
#     @torch.no_grad()
#     def forward(self, x):
#         # Following DINOv3, x use imagenet norm
#         b, _, h, w = x.shape
#         feat = self.model(x)
#         feat = feat.last_hidden_state
#         feat = feat[:, 1:, :]
#         feat = feat.view(b, h // 32, w // 32, -1).permute(0, 3, 1, 2).contiguous()
#         return feat

def dinov3_convnext_b(safe_tensors):
    model = ConvNeXt(in_chans=3, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], drop_path_rate=0.0, last_norm=True)
    model.load_model(safe_tensors)
    model.freeze()
    return model

def dinov3_convnext_l(safe_tensors):
    model = ConvNeXt(in_chans=3, depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], drop_path_rate=0.0, last_norm=True)
    model.load_model(safe_tensors)
    model.freeze()
    return model