import torch
import torch.nn as nn
from torch import Tensor
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from timm.models.layers import trunc_normal_
from typing import Any, Callable, List, Optional, Type, Union


class Adapter(nn.Module):
    def __init__(self, in_channels, redu_channels, kernel_size, act=torch.nn.ReLU, stride=1):
        super(Adapter, self).__init__()
        self.channel_adapt = nn.Sequential(
            nn.Conv2d(in_channels, redu_channels, 1, 1, 0, bias=False),
            act(),
            nn.Conv2d(redu_channels, in_channels, 1, 1, 0, bias=False)
        )
        self.spatial_adapt = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, padding_mode="reflect", groups=in_channels, stride=stride, bias=False)
        self._init()
    
    def forward(self, x):
        x = self.channel_adapt(x)
        x = self.spatial_adapt(x)
        return x
    
    def _init(self):
        torch.nn.init_(self.channel_adapt[0].weight)

class LoRA(nn.Module):
    def __init__(self, in_dim, out_dim, rank=16):
        super(LoRA, self).__init__()
        self.lora_A = nn.Conv2d(in_dim, rank, 1, 1, 0, bias=False)
        self.lora_B = nn.Conv2d(rank, out_dim, 1, 1, 0, bias=False)
        self._init()
    
    def _init(self):
        torch.nn.init.zeros_(self.lora_A.weight)
        torch.nn.init.normal_(self.lora_B.weight)
    
    def forward(self, x):
        return self.lora_B(self.lora_A(x))


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_lora=False,
        use_adapter=False,
        lora_rank=16,
        adapter_channels=16
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.use_lora = use_lora
        self.use_adapter = use_adapter
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        if self.use_lora:
            self.lora1 = LoRA(inplanes, width, rank=lora_rank)
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if self.use_adapter:
            self.adapter = Adapter(width, adapter_channels, kernel_size=3, stride=stride)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        if self.use_lora:
            self.lora2 = LoRA(width, planes*self.expansion, rank=lora_rank)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        if self.use_lora:
            lora_out = self.lora1(x)
            out = out + lora_out
        out = self.bn1(out)
        out = self.relu(out)
        
        conv_out = out
        out = self.conv2(out)
        if self.use_adapter:
            adapter_out = self.adapter(conv_out)
            # print(out.shape, adapter_out.shape)
            out = out + adapter_out
        out = self.bn2(out)
        out = self.relu(out)

        conv_out = out
        out = self.conv3(out)
        if self.use_lora:
            lora_out = self.lora2(conv_out)
            out = out + lora_out
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_lora=False, use_adapter=False, lora_rank=16, adapter_channels=16
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.use_lora = use_lora
        self.use_adapter = use_adapter
        self.lora_rank = lora_rank
        self.adapter_channels = adapter_channels
        
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer,use_lora=self.use_lora, use_adapter=self.use_adapter, lora_rank=self.lora_rank, adapter_channels=self.adapter_channels
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    use_lora=self.use_lora, use_adapter=self.use_adapter, lora_rank=self.lora_rank, adapter_channels=self.adapter_channels
                )
            )

        return nn.Sequential(*layers)


@BACKBONES.register_module()
class PEFTResNet(torch.nn.Module):
    def __init__(self, res50_pretrained=None, embed_dim=512, 
                 use_lora=False, use_adapter=False, 
                 lora_rank=16, adapter_channels=16, layers=[3, 4, 6, 3]):
        super(PEFTResNet, self).__init__()
        # lora and adapter
        self.use_lora = use_lora
        self.use_adapter = use_adapter
        self.lora_rank = lora_rank
        self.adapter_channels = adapter_channels
        # backbone
        self.res50_pretrained = res50_pretrained
        # resnet50 backbone
        self.res50 = ResNet(block=Bottleneck, layers=layers, use_lora=use_lora, use_adapter=use_adapter, 
                            lora_rank=lora_rank, adapter_channels=adapter_channels)
        # obtain resnet 50 channels
        self.fpn_channels = [256, 512, 1024, 2048]
        # FPN
        self.fpn1 = nn.Sequential(
            nn.Conv2d(self.fpn_channels[0], embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        self.fpn2 = nn.Sequential(
            nn.Conv2d(self.fpn_channels[1], embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        self.fpn3 = nn.Sequential(
            nn.Conv2d(self.fpn_channels[2], embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        self.fpn4 = nn.Sequential(
            nn.Conv2d(self.fpn_channels[3], embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        # initialize
        if self.res50_pretrained is not None:
            self.load_res50()
        if self.use_lora or self.use_adapter:
            self.freeze_backbone()
        if self.use_lora:
            logger = get_root_logger()
            logger.info("==> LORA FINETUNE ENABLED ... ")
        if self.use_adapter:
            logger = get_root_logger()
            logger.info("==> ADAPTER FINETUNE ENABLED ... ")
    
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
    
    def freeze_backbone(self):
        freeze_cnt = 0
        for name, param in self.res50.named_parameters():
            if "lora" not in name and "adapter" not in name:
                if "downsample" not in name and "bn" not in name:
                    if "weight" in name:
                        # only freeze weight
                        param.requires_grad_(False)
                        freeze_cnt += 1
        logger = get_root_logger()
        logger.info("===> Freeze resnet50 backbone, total {} params.".format(freeze_cnt))

    def forward(self, x):
        """ this forward function is a modified version of `timm.models.resnet.ResNet.forward`
        >>> ResNet.forward
        """
        x = self.res50.conv1(x)
        x = self.res50.bn1(x)
        x = self.res50.relu(x)
        x = self.res50.maxpool(x)
        ls = []
        x = self.res50.layer1(x); ls.append(x)
        x = self.res50.layer2(x); ls.append(x)
        x = self.res50.layer3(x); ls.append(x)
        x = self.res50.layer4(x); ls.append(x)
        features = [self.fpn1(ls[0]), self.fpn2(ls[1]), self.fpn3(ls[2]), self.fpn4(ls[3])]
        return features


# ResNet 18: [2, 2, 2, 2]
# ResNet 34: [3, 4, 6, 3] Basic block
# ResNet 50: [3, 4, 6, 3] Bottleneck
# ResNet101: [3, 4, 23, 3] Bottleneck
# ResNet 152: [3, 8, 36, 3]