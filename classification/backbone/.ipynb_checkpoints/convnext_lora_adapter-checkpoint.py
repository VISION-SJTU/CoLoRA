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
        T = (x.amax(dim=[-2, -1], keepdim=True) - x.amin(dim=[-2, -1], keepdim=True)).detach() + 1e-5
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
        x = shortcut + self.scale_factor * x_filter
        return x


class Adapter(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1):
        super(Adapter, self).__init__()
        self.adapt = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, padding_mode="reflect", groups=in_channels, stride=stride, bias=True)
        self._init()
    
    def forward(self, x):
        x = self.adapt(x)
        return x
    
    def _init(self):
        torch.nn.init.zeros_(self.adapt.weight)


# Structured Unrestricted-Rank Matrices for Parameter Efficient Fine-tuning
## using circulant matrix for peft
class CircularLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
        Implements a linear layer with a column-circulant weight matrix.
        Instead of storing the full matrix (out_dim × in_dim), we store only
        the first column and compute fast multiplications using FFT.
        
        Args:
            in_dim (int): Input dimension (C)
            out_dim (int): Output dimension (M)
        """
        super().__init__()
        # assert out_dim % in_dim == 0, "Output dimension must be a multiple of input dimension"

        self.in_dim = in_dim
        self.out_dim = out_dim

        # Learnable first column of circulant matrix (shape: [out_dim])
        self.W_first_col = nn.Parameter(torch.zeros(max(in_dim, out_dim)))
        self.W_second_col = nn.Parameter(torch.randn(max(in_dim, out_dim)))

    def forward(self, x):
        """
        Forward pass using FFT-based circulant matrix multiplication.
        Args:
            x (Tensor): Input tensor of shape (batch_size, h, w, in_dim)
        Returns:
            Tensor: Transformed output of shape (batch_size, h, w, out_dim)
        """
        batch_size, h, w, _ = x.shape

        W_first_col = self.W_first_col * self.W_second_col

        # Zero-pad x to match the output dimension
        # Compute FFT of weights and input
        W_fft = torch.fft.fft(W_first_col[:self.in_dim])
        x_fft = torch.fft.fft(x, dim=-1)
        # print(W_fft.shape, x_fft.shape, x.shape)

        # Element-wise multiplication in Fourier space
        result_fft = W_fft * x_fft

        # Inverse FFT to obtain final result
        result = torch.fft.ifft(result_fft, dim=-1).real  # Only take real part
        result = result[..., :self.out_dim]
        return result

class LoRA(nn.Module):
    def __init__(self, in_dim, out_dim, reduction=16,stride=1,kernel_size=1, use_filter=False, wsize=11, sigma=1, scale_factor=0.1, use_depth=False, use_pool=False):
        super(LoRA, self).__init__()
        self.use_filter=use_filter
        self.use_pool = use_pool
        self.lora_A = nn.Conv2d(in_dim, out_dim // reduction, 1, 1, 0, bias=False)
        if use_depth:
            self.lora_B = nn.Conv2d(out_dim//reduction, out_dim, 1, stride=stride, padding=kernel_size//2, bias=False, groups=out_dim//reduction)
        else:
            self.lora_B = nn.Conv2d(out_dim // reduction, out_dim, 1, stride=stride, padding=kernel_size//2, bias=False)
        self._init()
        if self.use_filter:
            self.filtering = BilateralModule(wsize, sigma=sigma, scale_factor=scale_factor)
        if self.use_pool:
            self.pool = nn.MaxPool2d(1)
    
    def _init(self):
        # lora_B to zero leads better results
        torch.nn.init.zeros_(self.lora_B.weight)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        # torch.nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))
    
    def forward(self, x, relation_matrix=None):
        x = self.lora_A(x)
        if self.use_filter:
            x = self.filtering(x)
        if relation_matrix is not None:
            # relation_matrix = F.normalize(relation_matrix, dim=0)
            x = torch.einsum("bchw, cd -> bdhw", x, relation_matrix)
        x = self.lora_B(x)
        return x

class CoeffLayer(nn.Module):
    def __init__(self, dim):
        super(CoeffLayer, self).__init__()
        self.lora_coeff = nn.Parameter(torch.ones(1, 1, 1, dim), requires_grad=True)
        self.lora_bias = nn.Parameter(torch.zeros(1, 1, 1, dim), requires_grad=True)
    
    def forward(self, x):
        # x [n, h, w, c]
        return x * self.lora_coeff + self.lora_bias


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
                 adaptive_weight=False, lora_reduction=16, use_filter=False):
        super().__init__()
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
        ## PEFT modules
        if self.use_lora:
            self.lora = LoRA(dim, dim, reduction=self.lora_reduction, use_filter=self.use_filter, wsize=11, sigma=1.0, scale_factor=0.2, use_depth=True)
            # self.lora_up = LoRA(dim, 4*dim, reduction=dim//2, use_filter=False, use_pool=False, use_depth=False)
            # self.lora_down = LoRA(4*dim, dim, reduction=dim//8, use_filter=False, use_pool=False, use_depth=False)
            # self.lora_pw_B = nn.Parameter(torch.empty(4*dim, dim//self.lora_reduction))
            # self.lora_pw_S = nn.Parameter(torch.zeros(dim//self.lora_reduction, dim))
            # torch.nn.init.kaiming_uniform_(self.lora_pw_B)
            # self.lora_pw_A = nn.Parameter(torch.empty(dim//self.lora_reduction, 4*dim))
            # torch.nn.init.kaiming_uniform_(self.lora_pw_A)
            self.lora_pw_S = nn.Parameter(torch.zeros(dim, dim))
            self.lora_A_bias = nn.Parameter(torch.zeros(1, dim))
            self.lora_B_bias = nn.Parameter(torch.zeros(1, dim))
            # self.lora_adapter = Adapter(dim, kernel_size=7, stride=1)
            # self.lora_up = CircularLinear(dim, 4*dim)
            # self.lora_down = CircularLinear(4*dim, dim)
            # self.lora_coeff = CoeffLayer(4*dim)
            # self.lora_coeff2 = CoeffLayer(dim)
            # self.lora_matrix = nn.Parameter(torch.ones(8, 8), requires_grad=True)
            # torch.nn.init.kaiming_uniform_(self.lora_matrix)
            # self.norm_down = LayerNorm(dim, eps=1e-6)
        if self.use_adapter:
            self.adapter = Adapter(dim, kernel_size=7, stride=1)
        if self.adaptive_weight:
            self.adapt_w = nn.Parameter(torch.ones(1, dim, 1, 1)*0.5, requires_grad=True)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        ## PEFT branch
        if self.use_lora:
            lora_out = self.lora(input)
            if self.adaptive_weight:
                lora_out = lora_out * self.adapt_w
            x = x + lora_out
        if self.use_adapter:
            adapter_out = self.adapter(input)
            if self.adaptive_weight:
                adapter_out = adapter_out * (1 - self.adapt_w)
            x = x + adapter_out
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        conv_x = x# .permute(0, 3, 1, 2)
        x = self.pwconv1(x)
        if self.use_lora:
            lora_out = torch.einsum("bhwc, dc -> bhwd", conv_x, self.lora_pw_S+self.lora_A_bias)
            lora_out = torch.einsum("bhwc, cd -> bhwd", lora_out, self.pwconv2.weight) # [c, 4c]
            # lora_out = self.lora_pw_B(self.lora_pw_S(conv_x))
            # lora_out = self.lora_up(conv_x, self.lora_matrix)
            # lora_out = lora_out.permute(0, 2, 3, 1)
            x = x + lora_out
        x = self.act(x)
        conv_x = x
        x = self.pwconv2(x)
        if self.use_lora:
            lora_out = torch.einsum("bhwc, cd -> bhwd", conv_x, self.pwconv1.weight)
            lora_out = torch.einsum("bhwc, dc -> bhwd", lora_out, self.lora_pw_S + self.lora_B_bias)
            # conv_x = conv_x.permute(0, 3, 1, 2)
            # lora_out = self.lora_pw_S(self.lora_pw_A(conv_x))
            # lora_out = self.lora_down(conv_x, self.lora_matrix.permute(1, 0))
            # lora_out = lora_out.permute(0, 2, 3, 1)
            # lora_out = self.norm_down(lora_out)
            x = x + lora_out
        if self.gamma is not None:
            x = self.gamma * x
        # if self.use_lora:
        #     x = self.lora_coeff2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class OuterLoRA(nn.Module):
    def __init__(self, layer:nn.Linear, rank=16):
        super(OuterLoRA, self).__init__()
        in_features, out_features = layer.in_features, layer.out_features
        self.lora_col = nn.Parameter(torch.empty(out_features, rank))
        self.lora_row = nn.Parameter(torch.zeros(rank, in_features))
        # freeze weight
        self.weight = layer.weight
        self.weight.requires_grad_(False)
        self.bias = layer.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)
        torch.nn.init.kaiming_uniform_(self.lora_col, a=math.sqrt(5))
    
    def forward(self, x, relation_matrix=None):
        # scale = torch.nn.functional.gelu(self.lora_col @ self.lora_row) # up rank
        scale = self.lora_col @ (relation_matrix @ self.lora_row)
        weight = (1 + scale) * self.weight
        out = torch.einsum("bhwc, cd->bhwd", x, weight)
        if self.bias is not None:
            out = out + self.bias
        return out

class RandSVD(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        rank = min(in_features, out_features)
        redu_rank = rank // 4 * 4  # optimization for gpu
        if redu_rank == 0:
            redu_rank = rank
        self.rank = redu_rank
        self.fc_V = nn.Linear(in_features, rank, bias=False)
        self.lora_S = nn.Parameter(torch.zeros((1, rank)), requires_grad=True)
        self.fc_U = nn.Linear(rank, out_features, bias=False)
        self._decompose()
    
    def forward(self, x):
        x = self.fc_V(x)
        x = x.mul(self.lora_S)
        output = self.fc_U(x)
        return output
    
    def _decompose(self):
        layer = nn.Linear(self.in_features, self.out_features, bias=False)
        w = layer.weight
        U, S, V = torch.svd(w)
        S = S[:self.rank].unsqueeze(0) # [1, rank]
        U = U[:, :self.rank] # [m, r]
        V = V[:, :self.rank].permute(1, 0) # [n, r]
        # hack parameters
        # hack V
        self.fc_V.weight.data.copy_(V)
        self.fc_V.weight.requires_grad = False
        # hack U
        self.fc_U.weight.data.copy_(U)
        self.fc_U.weight.requires_grad = False
        print("[SVD Rand initialization] finished !, U: {}, S: {}, V: {}".format(U.shape, S.shape, V.shape))


## SVD 
class SVDLinear(nn.Module):
    def __init__(self, layer:nn.Linear, blocks=32):
        super(SVDLinear, self).__init__()
        in_dim, out_dim = layer.in_features, layer.out_features
        self.blocks = 32
        rank = min(in_dim, out_dim)
        redu_rank = rank // 4 * 4  # optimization for gpu
        if redu_rank == 0:
            redu_rank = rank
        self.rank = redu_rank
        # hack linear layer
        self.fc_V = nn.Linear(in_dim, rank, bias=False)
        self.lora_S = nn.Parameter(torch.ones((1, rank)), requires_grad=True)
        # self.lora_S = nn.Parameter(torch.zeros(self.blocks, self.rank // self.blocks, self.rank // self.blocks)) # [n, r/n, r/n]
        self.fc_U = nn.Linear(rank, out_dim, bias=True if layer.bias is not None else None)
        self._decompose(layer.weight, layer.bias)
        # self.randsvd = RandSVD(in_features=in_dim, out_features=out_dim)
    
    def forward(self, x):
        # x_rand = self.randsvd(x)
        x = self.fc_V(x) # [b, h, w, c]
        b, h, w, c = x.shape
        x = x.mul(self.lora_S)
        # x = x.view(b, h, w, self.blocks, c // self.blocks)
        # x = torch.einsum('bhwnc, ndc -> bhwnd', x, self.lora_S)
        # x = x.reshape(b, h, w, c)
        output = self.fc_U(x)
        return output

    def _decompose(self, w, b=None):
        U, S, V = torch.svd(w)
        S = S[:self.rank] # [1, rank]
        # S = S.view(self.blocks, self.rank // self.blocks) # [n, r/n]
        # S = torch.stack([torch.diag(v) for v in S])
        # print(S)
        U = U[:, :self.rank] # [m, r]
        V = V[:, :self.rank].permute(1, 0) # [n, r]
        # hack parameters
        # hack V
        self.fc_V.weight.data.copy_(V)
        self.fc_V.weight.requires_grad = False
        # hack U
        self.fc_U.weight.data.copy_(U)
        self.fc_U.weight.requires_grad = False
        if self.fc_U.bias is not None:
            self.fc_U.bias.data.copy_(b)
            self.fc_U.bias.requires_grad = False
        # hack S
        self.lora_S.data.copy_(S)
        print("[SVD initialization] finished !, U: {}, S: {}, V: {}".format(U.shape, S.shape, V.shape))


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
        lora_params = self.cnt_params(cnt_lora=True)
        param_ratio = lora_params / base_params * 100
        logger = get_root_logger()
        logger.info("[LoRA setting], base params: {:.2f} M, lora params: {:.2f} M, ratio: {:.2f}%".format(base_params, lora_params, param_ratio))
        ## FREEZE backbone
        self.freeze_backbone()
        # self.freeze_conv(type='dwconv')
        ## LOG
        if use_lora:
            logger = get_root_logger()
            logger.info("==> LORA FINETUNE ENABLED ... ")
        if use_adapter:
            logger = get_root_logger()
            logger.info("==> ADAPTER FINETUNE ENABLED ... ")

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def merge_correlated(self):
        """
        Merge correlated weights for pwconv1 and pwconv2 every K iterations
        """



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
                if "lora" in name:
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
                svd_module = OuterLoRA(layer=module)
                setattr(parent_module, attr, svd_module)

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

    def freeze_backbone(self):
        freeze_cnt = 0
        for name, param in self.named_parameters():
            if "lora" not in name and "adapt" not in name:
                if "downsample" not in name and "norm" not in name:
                    param.requires_grad_(False)
                    freeze_cnt += 1
        logger = get_root_logger()
        logger.info("===> Freeze convnext backbone, total {} params.".format(freeze_cnt))
    
    def freeze_conv(self, type='dwconv'):
        freeze_cnt = 0
        for name, param in self.named_parameters():
            if "lora" not in name and "adapt" not in name:
                if "downsample" not in name and "bn" not in name:
                    if type in name or "norm" in name:
                        param.requires_grad_(False)
                        freeze_cnt += 1
        logger = get_root_logger()
        logger.info("===> Freeze convnext {}, total {} params.".format(type, freeze_cnt))

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