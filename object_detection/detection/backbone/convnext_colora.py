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
from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES
import math
import numpy as np
from math import floor


class BilateralModule(nn.Module):
    def __init__(self, wsize=11, sigma=0.1, scale_factor=0.1):
        super(BilateralModule, self).__init__()
        self.wsize = wsize
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.gaussian_weight() # [ksize, ksize]
        self.kernel.requires_grad_(False)
    
    def gaussian_weight(self):
        center = self.wsize / 2
        x = (np.arange(self.wsize, dtype=np.float32) - center)
        kernel_1d = np.exp(-(x**2) / (2*self.sigma**2))
        kernel = kernel_1d[..., None] @ kernel_1d[None, ...]
        kernel = torch.from_numpy(kernel)
        kernel = kernel / kernel.sum() # Normalization
        self.kernel = kernel

    def forward(self, x):
        # filtering
        c = x.shape[1]
        shortcut = x
        T = (x.amax(dim=[-2, -1], keepdim=True) - x.amin(dim=[-2, -1], keepdim=True)).detach() + 1e-5 # better
        # T = (x.max() - x.min()).detach() + 1e-5
        gamma = torch.pi / (2*T)
        kernel = self.kernel.view(1, 1, self.wsize, self.wsize).to(x)
        kernel = kernel.repeat(c, 1, 1, 1)
        # pad x
        pad_x = F.pad(x, (self.wsize//2, self.wsize//2, self.wsize//2, self.wsize//2), mode='reflect')
        cos_img = torch.cos(gamma * x).detach()
        sin_img = torch.sin(gamma * x).detach()
        cos_pad = torch.cos(gamma * pad_x).detach()
        sin_pad = torch.sin(gamma * pad_x).detach()
        # filtering
        cos_filt = cos_img * F.conv2d(cos_pad * pad_x, weight=kernel, bias=None, stride=1, padding=0, groups=c)
        sin_filt = sin_img * F.conv2d(sin_pad * pad_x, weight=kernel, bias=None, stride=1, padding=0, groups=c)
        cos_map = cos_img * F.conv2d(cos_pad, weight=kernel, bias=None, stride=1, padding=0, groups=c)
        sin_map = sin_img * F.conv2d(sin_pad, weight=kernel, bias=None, stride=1, padding=0, groups=c)
        x_filter = (cos_filt + sin_filt) / (cos_map + sin_map + 1e-5)
        # print(self.kernel)
        x = shortcut + self.scale_factor * x_filter # only use filtering abla
        # x = x_filter # not well
        return x


class LoRA(nn.Module):
    def __init__(self, in_dim, out_dim, reduction=16,stride=1,kernel_size=1, use_filter=False, wsize=11, sigma=1, scale_factor=0.1, use_depth=False, use_pool=False, alpha=16): # wsize larger better
        super(LoRA, self).__init__()
        self.use_filter=use_filter
        self.use_pool = use_pool
        reduction = 64
        self.rank = out_dim // reduction
        self.scaling = 1 # math.sqrt(out_dim // self.rank)
        self.lora_A = nn.Conv2d(in_dim, self.rank, 1, 1, 0, bias=False)
        if use_depth:
            self.lora_B = nn.Conv2d(self.rank, out_dim, 3, stride=stride, padding=3//2, bias=False, groups=self.rank)
        else:
            self.lora_B = nn.Conv2d(self.rank, out_dim, 1, stride=stride, padding=kernel_size//2, bias=False)
        self._init()
        if self.use_filter:
            self.filtering = BilateralModule(wsize, sigma=sigma, scale_factor=scale_factor)
        # self.ab_norm_ori = torch.sqrt(self.lora_A.weight.detach().square().mean() + self.lora_B.weight.detach().square().mean())

    def _init(self):
        # lora_B to zero leads better results
        torch.nn.init.zeros_(self.lora_B.weight)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
    
    def forward(self, x):
        x = self.lora_A(x)
        if self.use_filter:
            x = self.filtering(x)
        x = self.lora_B(x)
        # with torch.no_grad():
        #     norm_ab = torch.sqrt(self.lora_A.weight.detach().square().mean() + self.lora_B.weight.detach().square().mean())
        x = x * self.scaling
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
                 adaptive_weight=False, lora_reduction=16, use_filter=False, alpha=16, share_ratio=0.6):
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
        if self.use_lora:
            self.lora = LoRA(dim, dim, reduction=self.lora_reduction*2, use_filter=True, wsize=11, sigma=1.0, scale_factor=0.2, use_depth=True, alpha=self.alpha)
        if self.use_lora:
            if self.indep_rank > 0:
                self.lora_pw_B1 = nn.Parameter(torch.empty(4*dim, self.indep_rank))
                self.lora_pw_A1 = nn.Parameter(torch.empty(self.indep_rank, dim))
                self.lora_pw_B2 = nn.Parameter(torch.empty(dim, self.indep_rank))
                self.lora_pw_A2 = nn.Parameter(torch.empty(self.indep_rank, 4*dim))
            if self.share_rank > 0:
                self.lora_pw_share_B = nn.Parameter(torch.empty(dim, self.share_rank))
                self.lora_pw_share_A = nn.Parameter(torch.empty(self.share_rank, dim))
    
    @torch.no_grad()
    def merge_and_reinitialize(self, merge_weights=False):
        """
        Merge weights and reinitialize bias and lora_pw_S
        """
        if self.use_lora:
            if not merge_weights:
                # just initialize
                with torch.cuda.amp.autocast(enabled=False):
                    logger = get_root_logger()
                    logger.info("[Begin to initialize ... ]")
                    if self.share_rank > 0:
                        nn.init.zeros_(self.lora_pw_share_B)
                        nn.init.kaiming_uniform_(self.lora_pw_share_A, a=math.sqrt(5))
                    if self.indep_rank > 0:
                        nn.init.zeros_(self.lora_pw_B1)
                        nn.init.zeros_(self.lora_pw_B2)
                        nn.init.kaiming_uniform_(self.lora_pw_A1, a=math.sqrt(5))
                        nn.init.kaiming_uniform_(self.lora_pw_A2, a=math.sqrt(5))
                    
                    self.ori_weight_1 = self.pwconv1.weight.clone()  # [4c, c]
                    self.ori_weight_2 = self.pwconv2.weight.clone()  # [c, 4c]

                    delta_w1, delta_w2 = self.compute_delta_w()
                    self.pwconv1.weight.data.copy_(self.ori_weight_1 - delta_w1)
                    self.pwconv2.weight.data.copy_(self.ori_weight_2 - delta_w2)
                # merge weights
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        logger = get_root_logger()
                        logger.info("[Begin to merge LoRA weights ...]")
                        # reinitialize
                        # if self.share_rank > 0:
                        #     nn.init.zeros_(self.lora_pw_share_B)
                        #     nn.init.kaiming_uniform_(self.lora_pw_share_A, a=math.sqrt(5))
                        if self.indep_rank > 0:
                            nn.init.zeros_(self.lora_pw_B1)
                            nn.init.zeros_(self.lora_pw_B2)                            
                            nn.init.kaiming_uniform_(self.lora_pw_A1, a=math.sqrt(5))
                            nn.init.kaiming_uniform_(self.lora_pw_A2, a=math.sqrt(5))
                        
                        delta_w1, delta_w2 = self.compute_delta_w()
                        self.pwconv1.weight.data.copy_(self.ori_weight_1 - delta_w1)
                        self.pwconv2.weight.data.copy_(self.ori_weight_2 - delta_w2)
    
    def compute_delta_w(self, scaling=None):
         with torch.cuda.amp.autocast(enabled=False):
            # first compute scale factor
            # ori_weight_1, ori_weight_2 = self.pwconv1.weight.detach(), self.pwconv2.weight.detach()
            scale_w1 = math.sqrt(self.dim) * self.ori_weight_1.std()
            scale_w2 = math.sqrt(self.dim) * self.ori_weight_2.std()
            # scale_w = scale_w1 + scale_w2
            if self.share_rank > 0:
                scale_share = (self.share_rank * self.dim) ** 0.25 * (self.lora_pw_share_A.std() + self.lora_pw_share_B.std())
                tmp_B = torch.matmul(self.ori_weight_2.to(self.lora_pw_share_A.device).detach().transpose(0, 1), self.lora_pw_share_B).detach()
                tmp_A = torch.matmul(self.lora_pw_share_A, self.ori_weight_1.detach().transpose(0, 1).to(self.lora_pw_share_A.device)).detach()
                scale_share_1 = (self.share_rank * self.dim)**0.25 * (tmp_B.std() + self.lora_pw_share_A.std())
                scale_share_2 = (self.share_rank * self.dim)**0.25 * (tmp_A.std() + self.lora_pw_share_B.std())
            if self.indep_rank > 0:
                # variance-stable scale
                # scale_ab1 = (self.indep_rank * self.dim) ** 0.25 * (self.lora_pw_A1.std() + self.lora_pw_B1.std())
                # scale_ab2 = (self.indep_rank * self.dim) ** 0.25 * (self.lora_pw_A2.std() + self.lora_pw_B2.std())
                scale_ab1_mean = math.sqrt(self.indep_rank) * (self.lora_pw_A1.std() + self.lora_pw_B1.std())
                scale_ab2_mean = math.sqrt(self.indep_rank) * (self.lora_pw_A2.std() + self.lora_pw_B2.std())
            ## scaling strategy
            # first compute shared lora
            if self.share_rank > 0:
                lora_share = self.lora_pw_share_B @ self.lora_pw_share_A
                # share_delta1 = torch.matmul(self.ori_weight_2.to(self.lora_pw_share_A.device).detach().transpose(0, 1), lora_share * 1/(scale_w2.detach()*scale_share.detach()))
                # share_delta2 = torch.matmul(lora_share*1/(scale_w1.detach()*scale_share.detach()), self.ori_weight_1.detach().transpose(0, 1).to(self.lora_pw_share_A.device))
                share_delta1 = torch.matmul(self.ori_weight_2.to(self.lora_pw_share_A.device).detach().transpose(0, 1), lora_share * 1/(scale_share_1.detach()))
                share_delta2 = torch.matmul(lora_share*1/(scale_share_2.detach()), self.ori_weight_1.detach().transpose(0, 1).to(self.lora_pw_share_A.device))
            else:
                share_delta1, share_delta2 = 0, 0
            if self.indep_rank > 0:
                indep_delta1 = self.lora_pw_B1 @ self.lora_pw_A1 * 1 / scale_ab1_mean.detach()
                indep_delta2 = self.lora_pw_B2 @ self.lora_pw_A2 * 1 / scale_ab2_mean.detach()
            else:
                indep_delta1, indep_delta2 = 0, 0
            delta_w1 = (share_delta1 + indep_delta1)
            delta_w2 = (share_delta2 + indep_delta2)
            return delta_w1, delta_w2

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        ## PEFT branch
        if self.use_lora:
            lora_out = self.lora(input)
            x = x + lora_out
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        if self.use_lora:
            with torch.cuda.amp.autocast(enabled=False):
                delta_w1, delta_w2 = self.compute_delta_w()
                w1 = self.pwconv1.weight + delta_w1
                w2 = self.pwconv2.weight + delta_w2
                x = F.linear(x, weight=w1, bias=self.pwconv1.bias)
                x = self.act(x)
                x = F.linear(x, weight=w2, bias=self.pwconv2.bias)
                # self.cache_w1, self.cache_w2 = w1.detach(), w2.detach()
                # self.ori_weight_1, self.ori_weight_2 = self.rho * self.ori_weight_1.to(x.device) + (1 - self.rho) * w1.detach(), self.rho * self.ori_weight_2.to(x.device) + (1-self.rho)*w2.detach()
                # self.ori_weight_1, self.ori_weight_2 = self.cache_w1.detach(), self.cache_w2.detach()
                self.ori_weight_1, self.ori_weight_2 = w1.detach(), w2.detach()
        else:
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

@BACKBONES.register_module()
class ConvNeXtCoLoRA(nn.Module):
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
        # self.freeze_backbone()
        # self.merge_correlated()
        # self.freeze_conv(type='dwconv')
        ## LOG
        if use_lora:
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
    
    def merge_correlated(self, merge_weights=False):
        """
        Merge correlated weights for pwconv1 and pwconv2 every K iterations
        """
        for i in range(4):
            stage = self.stages[i]
            for block in stage:
                block.merge_and_reinitialize(merge_weights=merge_weights)

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
        # print("[LoRA setting], base params: {:.2f} M, lora params: {:.2f} M, ratio: {:.2f}%".format(base_params, lora_params, param_ratio))
    
    def hack_linear(self):
        all_modules = dict(self.named_modules())
        for name, module in self.named_modules():
            print(name)
            if "pwconv1" in name:
                # hack
                parent, attr = name.rsplit(".", 1) 
                parent_module = all_modules[parent]
                # svd_module = SVDLinear(layer=module)
                # svd_module = OuterLoRA(layer=module)
                setattr(parent_module, attr, svd_module)
    
    def freeze_backbone(self):
        freeze_cnt = 0
        for name, param in self.named_parameters():
            if "lora" not in name:
                if "bias" not in name and "norm" not in name: # fix gamma, bias tuning
                    param.requires_grad_(False)
                    freeze_cnt += 1
        logger = get_root_logger()
        logger.info("===> Freeze convnext backbone, total {} params.".format(freeze_cnt))
    
    def freeze_conv(self, type='dwconv'):
        freeze_cnt = 0
        # for name, param in self.named_parameters():
        #     if "lora" not in name and "adapt" not in name:
        #         if "downsample" not in name and "bn" not in name:
        #             if type in name or "norm" in name or "gamma" in name:
        #                 param.requires_grad_(False)
        #                 freeze_cnt += 1
        for name, param in self.named_parameters():
            if "pwconv" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        logger = get_root_logger()
        logger.info("===> Freeze convnext {}, total {} params.".format(type, freeze_cnt))

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
