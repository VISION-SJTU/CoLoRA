# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from math import sqrt

import torch
import torch.nn as nn
import os
import timm.models.vision_transformer
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES


@BACKBONES.register_module()
class MAEViT(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, mlp_ratio=4, qkv_bias=True, pretrain_ckpt="", norm_layer=partial(nn.LayerNorm, eps=1e-6), img_size=512, **kwargs):
        super(MAEViT, self).__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        self.intep_size = img_size // 16
        self.load_checkpoint(pretrain_ckpt)
        self.interp_pos_embed()
    
    def interp_pos_embed(self):
        b, l, d = self.pos_embed.shape
        pos_embed = self.pos_embed[:, 1:, :]
        pos_embed = pos_embed.view(b, 14, 14, d).permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = nn.functional.interpolate(pos_embed, size=(self.intep_size, self.intep_size), mode='bilinear', align_corners=True)
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(b, -1, d)  # [b, h*w, c]
        self.pos_embed = nn.Parameter((torch.cat([self.pos_embed[:, [0], :], pos_embed], dim=1)))
    
    def init_weights(self, pretrained=None):
        pass

    def forward_features(self, x):        
        B, c, h, w = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:,1:,:] # [b, l, d]
            outcome = outcome.reshape(B, h//16, w//16, -1).permute(0, 3, 1, 2) # [b, c, h, w]
        outcome = outcome.detach()
        return [outcome]
    
    def forward(self, x):
        return self.forward_features(x)
    
    def load_checkpoint(self, pretrain_ckpt=""):
        logger = get_root_logger()
        if pretrain_ckpt is not None and os.path.exists(pretrain_ckpt):
            logger.info(f"Load checkpoint from {pretrain_ckpt}")
            ckpt = torch.load(pretrain_ckpt)["model"]
            for name, _ in self.named_parameters():
                if name not in ckpt:
                    logger.info(f"Warning: {name} not found in checkpoint")
            self.load_state_dict(ckpt, strict=False)
        else:
            logger.info(f"{pretrain_ckpt} does not exist ...")
        # freeze
        # for param in self.parameters():
        #     param.requires_grad = False
        # self.eval()

def mae_vit_b_16(ckpt_path):
    model = MAEViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=True)["model"]
        for name, _ in model.named_parameters():
            if name not in ckpt:
                print(f"Warning: {name} not found in checkpoint")
        model.load_state_dict(ckpt, strict=False)
    # freeze
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model

def mae_vit_l_16(ckpt_path):
    model = MAEViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=True)["model"]
        for name, _ in model.named_parameters():
            if name not in ckpt:
                print(f"Warning: {name} not found in checkpoint")
        model.load_state_dict(ckpt, strict=False)
    # freeze
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model