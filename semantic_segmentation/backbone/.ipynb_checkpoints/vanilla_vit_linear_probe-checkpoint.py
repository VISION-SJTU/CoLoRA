# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from email.policy import strict
from functools import partial
from math import sqrt

import torch
import torch.nn as nn
import os
import timm.models.vision_transformer
from mmseg.models.builder import BACKBONES
from mmcv.runner import BaseModule


@BACKBONES.register_module()
class VanillaViT(BaseModule):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, model_name="vit_base_patch16_224", img_size=512, pretrained="", **kwargs):
        super(VanillaViT, self).__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        print(dir(self.model), self.model.__class__)
        self.intep_size = img_size // 16
        self.pretrained = pretrained
        self.init_weights()
    
    def init_weights(self):
        self.load_checkpoint(self.pretrained)
        self.interp_pos_embed()
        for param in self.parameters():
            param.requires_grad = False
    
    def interp_pos_embed(self):
        b, l, d = self.model.pos_embed.shape
        pos_embed = self.model.pos_embed[:, 1:, :]
        pos_embed = pos_embed.view(b, 14, 14, d).permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = nn.functional.interpolate(pos_embed, size=(self.intep_size, self.intep_size), mode='bilinear', align_corners=True)
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(b, -1, d)  # [b, h*w, c]
        self.model.pos_embed = nn.Parameter((torch.cat([self.model.pos_embed[:, [0], :], pos_embed], dim=1)))
    
    def load_checkpoint(self, pretrain_ckpt=""):
        if pretrain_ckpt is not None and os.path.exists(pretrain_ckpt):
            ckpt = torch.load(pretrain_ckpt)
            for name, _ in self.named_parameters():
                if name not in ckpt:
                    print(f"Warning: {name} not found in checkpoint")
            self.load_state_dict(ckpt, strict=False)
        else:
            print(f"Warning: {pretrain_ckpt} not exists. ")
        # freeze
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        B,_,h,w = x.shape
        x = self.model.patch_embed(x)

        cls_tokens = self.model.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        for blk in self.model.blocks:
            x = blk(x)
        x = self.model.norm(x)
        outcome = x[:,1:,:] # [b, l, d]
        outcome = outcome.reshape(B, h//16, w//16, -1).permute(0, 3, 1, 2) # [b, c, h, w]
        outcome = outcome.contiguous()
        return [outcome]