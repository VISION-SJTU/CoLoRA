# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from timm.models.layers import trunc_normal_


class Adapter(torch.nn.Module):
    def __init__(self, in_channels, redu_channels, kernel_size, act=torch.nn.ReLU):
        super(Adapter, self).__init__()
        self.channel_adapt = nn.Sequential(
            nn.Conv2d(in_channels, redu_channels, 1, 1, 0),
            act(),
            nn.Conv2d(redu_channels, in_channels, 1, 1, 0)
        )
        self.spatial_adapt = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, padding_mode="reflect",
                                       groups=in_channels)
    
    def forward(self, x):
        x = self.channel_adapt(x)
        x = self.spatial_adapt(x)
        return x

@BACKBONES.register_module()
class ResNet50(torch.nn.Module):
    def __init__(self, res50_pretrained=None, embed_dim=512):
        super(ResNet50, self).__init__()
        self.res50_pretrained = res50_pretrained
        # resnet50 backbone
        self.res50 = create_model("resnet50")
        # obtain resnet 50 channels
        self.fpn_channels = self.res50.feature_info[1:]
        self.fpn_channels = [info["num_chs"] for info in self.fpn_channels]
        # FPN
        self.fpn1 = nn.Sequential(
            nn.Conv2d(self.fpn_channels[0], embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            # nn.GELU(),
        )
        self.fpn2 = nn.Sequential(
            nn.Conv2d(self.fpn_channels[1], embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            # nn.GELU()
            # nn.Identity()
        )
        self.fpn3 = nn.Sequential(
            nn.Conv2d(self.fpn_channels[2], embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            # nn.GELU()
        )
        self.fpn4 = nn.Sequential(
            nn.Conv2d(self.fpn_channels[3], embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            # nn.GELU()
        )
        # initialize
        self.apply(self._init_weights)
        if self.res50_pretrained is not None:
            self.load_res50()
    
    def load_res50(self):
        state = torch.load(self.res50_pretrained, map_location="cpu")
        state = state.get("module", state)
        ckp_dict = dict()
        for name, param in self.res50.named_parameters():
            if name in state and "num_batches_tracked" not in name:
                # load param from state dict (exclude num_batches_tracked)
                ckp_dict[name] = state[name]
            else:
                ckp_dict[name] = param
        for name, param in self.res50.named_buffers():
            if name in state and "num_batches_tracked" not in name:
                # load param from state dict (exclude num_batches_tracked)
                ckp_dict[name] = state[name]
            else:
                ckp_dict[name] = param
        self.res50.load_state_dict(ckp_dict, strict=True)
        logger = get_root_logger()
        logger.info("==> Load resnet50 successfully from {}".format(self.res50_pretrained))
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if "adapter" not in name:
                param.requires_grad_(False)
    
    def forward(self, x):
        """ this forward function is a modified version of `timm.models.resnet.ResNet.forward`
        >>> ResNet.forward
        """
        x = self.res50.conv1(x)
        x = self.res50.bn1(x)
        x = self.res50.act1(x)
        x = self.res50.maxpool(x)
        ls = []
        x = self.res50.layer1(x); ls.append(x)
        x = self.res50.layer2(x); ls.append(x)
        x = self.res50.layer3(x); ls.append(x)
        x = self.res50.layer4(x); ls.append(x)
        features = [self.fpn1(ls[0]), self.fpn2(ls[1]), self.fpn3(ls[2]), self.fpn4(ls[3])]
        # logger = get_root_logger()
        # logger.info("shape of output: {}".format(features[0].shape))
        return features