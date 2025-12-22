import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmcls.models.builder import BACKBONES
from mmcls.utils import get_root_logger
from timm.models.layers import trunc_normal_
from typing import Any, Callable, List, Optional, Type, Union
import math
import numpy as np

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
        adaptive_weight=False,
        conv1_lora=False,
        lora_reduction=16,
        use_filter=False
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.use_lora = use_lora
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # register params
        self.rank = int(width // lora_reduction)
        self.indep_rank = self.rank // 2
        self.share_rank = self.rank - self.indep_rank
        self.filtering = BilateralModule(wsize=11, sigma=1.0, scale_factor=0.2)
        self.dim = width
        # register lora params
        if self.indep_rank > 0:
            self.lora_B1 = nn.Parameter(torch.empty(width, self.indep_rank))
            self.lora_A1 = nn.Parameter(torch.empty(self.indep_rank, width))
            self.lora_B2 = nn.Parameter(torch.empty(planes * self.expansion, self.indep_rank))
            self.lora_A2 = nn.Parameter(torch.empty(self.indep_rank, width))
        if self.share_rank > 0:
            self.lora_share_B = nn.Parameter(torch.empty(planes * self.expansion, self.share_rank))
            self.lora_share_A = nn.Parameter(torch.empty(self.share_rank, width))
    
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
                        nn.init.zeros_(self.lora_share_B)
                        nn.init.kaiming_uniform_(self.lora_share_A, a=math.sqrt(5))
                    if self.indep_rank > 0:
                        nn.init.zeros_(self.lora_B1)
                        nn.init.zeros_(self.lora_B2)
                        nn.init.kaiming_uniform_(self.lora_A1, a=math.sqrt(5))
                        nn.init.kaiming_uniform_(self.lora_A2, a=math.sqrt(5))
                    
                    self.ori_weight_1 = self.conv2.weight.clone()  # [4c, c]
                    self.ori_weight_2 = self.conv3.weight.clone()  # [c, 4c]

                    delta_w1, delta_w2 = self.compute_delta_w()
                    self.conv2.weight.data.copy_(self.ori_weight_1 - delta_w1)
                    self.conv3.weight.data.copy_(self.ori_weight_2 - delta_w2)
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
                            nn.init.zeros_(self.lora_B1)
                            nn.init.zeros_(self.lora_B2)                            
                            nn.init.kaiming_uniform_(self.lora_A1, a=math.sqrt(5))
                            nn.init.kaiming_uniform_(self.lora_A2, a=math.sqrt(5))
                        
                        delta_w1, delta_w2 = self.compute_delta_w()
                        self.conv2.weight.data.copy_(self.ori_weight_1 - delta_w1)
                        self.conv3.weight.data.copy_(self.ori_weight_2 - delta_w2)
    
    def compute_delta_w(self, scaling=None):
         with torch.cuda.amp.autocast(enabled=False):
            # first compute scale factor
            ori_weight_1, ori_weight_2 = self.ori_weight_1.mean(dim=[-2, -1]), self.ori_weight_2.mean(dim=[-2, -1])
            if self.share_rank > 0:
                scale_share = (self.share_rank * self.dim) ** 0.25 * (self.lora_share_A.std() + self.lora_share_B.std())
                tmp_B = torch.matmul(ori_weight_2.to(self.lora_share_A.device).detach().transpose(0, 1), self.lora_share_B).detach()
                tmp_A = torch.matmul(self.lora_share_A, ori_weight_1.detach().transpose(0, 1).to(self.lora_share_A.device)).detach()
                scale_share_1 = (self.share_rank * self.dim)**0.25 * (tmp_B.std() + self.lora_share_A.std())
                scale_share_2 = (self.share_rank * self.dim)**0.25 * (tmp_A.std() + self.lora_share_B.std())
            if self.indep_rank > 0:
                # variance-stable scale
                scale_ab1_mean = math.sqrt(self.indep_rank) * (self.lora_A1.std() + self.lora_B1.std())
                scale_ab2_mean = math.sqrt(self.indep_rank) * (self.lora_A2.std() + self.lora_B2.std())
            ## scaling strategy
            # first compute shared lora
            if self.share_rank > 0:
                lora_share = self.lora_share_B @ self.lora_share_A
                share_delta1 = torch.matmul(ori_weight_2.to(self.lora_share_A.device).detach().transpose(0, 1), lora_share * 1/(scale_share_1.detach()))
                share_delta2 = torch.matmul(lora_share*1/(scale_share_2.detach()), ori_weight_1.detach().transpose(0, 1).to(self.lora_share_A.device))
            else:
                share_delta1, share_delta2 = 0, 0
            if self.indep_rank > 0:
                indep_delta1 = self.lora_B1 @ self.lora_A1 * 1 / scale_ab1_mean.detach()
                indep_delta2 = self.lora_B2 @ self.lora_A2 * 1 / scale_ab2_mean.detach()
            else:
                indep_delta1, indep_delta2 = 0, 0
            delta_w1 = (share_delta1 + indep_delta1).unsqueeze_(-1).unsqueeze_(-1)
            delta_w2 = (share_delta2 + indep_delta2).unsqueeze_(-1).unsqueeze_(-1)
            return delta_w1, delta_w2

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # compute delta w
        delta_w1, delta_w2 = self.compute_delta_w()
        w1 = self.conv2.weight + delta_w1
        w2 = self.conv3.weight + delta_w2
        # modulate conv2
        self.ori_weight_1, self.ori_weight_2 = w1.detach(), w2.detach()
        out = F.conv2d(out, w1, self.conv2.bias, stride=self.stride, padding=1)
        out = self.filtering(out)
        out = self.bn2(out)
        out = self.relu(out)
        # modulate conv3
        out = F.conv2d(out, w2, self.conv3.bias, stride=1, padding=0)
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
        use_lora=False,
        use_adapter=False,
        adaptive_weight=False,
        conv1_lora=False,
        lora_reduction=16,
        use_filter=False) -> None:
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
        print("==> Use CoLoRA ...")
        self.use_lora = use_lora
        self.use_adapter = False
        self.adaptive_weight=False
        self.conv1_lora=False
        self.lora_reduction = lora_reduction
        self.use_filter = use_filter

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
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer,use_lora=self.use_lora, use_adapter=self.use_adapter, adaptive_weight=self.adaptive_weight, conv1_lora=self.conv1_lora,
                lora_reduction=self.lora_reduction,use_filter=self.use_filter
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
                    use_lora=self.use_lora, use_adapter=self.use_adapter, adaptive_weight=self.adaptive_weight, conv1_lora=self.conv1_lora, lora_reduction=self.lora_reduction,use_filter=self.use_filter
                )
            )

        return nn.Sequential(*layers)


@BACKBONES.register_module()
class ResNetCoLoRA(torch.nn.Module):
    def __init__(self, res50_pretrained=None, embed_dim=512, 
                 use_lora=False, use_adapter=False, adaptive_weight=False,
                 conv1_lora=False, lora_reduction=16, layers=[3, 4, 6, 3],use_filter=False):
        super(ResNetCoLoRA, self).__init__()
        # lora and adapter
        self.use_lora = use_lora
        self.use_adapter = use_adapter
        self.adaptive_weight = adaptive_weight
        self.conv1_lora = conv1_lora
        self.lora_reduction = lora_reduction
        self.use_filter = use_filter
        # backbone
        self.res50_pretrained = res50_pretrained
        # resnet50 backbone
        self.res50 = ResNet(block=Bottleneck, layers=layers, use_lora=self.use_lora, use_adapter=self.use_adapter, adaptive_weight=self.adaptive_weight, conv1_lora=self.conv1_lora, lora_reduction=self.lora_reduction, use_filter=self.use_filter)
        self.norm = nn.BatchNorm2d(2048)
        # obtain resnet 50 channels
        # initialize
        if self.res50_pretrained is not None:
            self.load_res50()
        # cnt base params
        base_params = self.cnt_params(lora=False)/10**6
        self.freeze_backbone()
        self.merge_correlated()
        lora_params = self.cnt_params(lora=True)/10**6
        param_ratio = lora_params / base_params
        logger = get_root_logger()
        logger.info("[LoRA setting], base params: {:.2f} M, lora params: {:.2f} M, ratio: {:.2f}%".format(base_params, lora_params, param_ratio))

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
            if name in state and "num_batches_tracked" not in name: # and "num_batches_tracked" not in name:
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
            if "lora" not in name:
                if "bias" not in name and "bn" not in name: # fix gamma, bias tuning
                    # only freeze weight
                    param.requires_grad_(False)
                    freeze_cnt += 1
        logger = get_root_logger()
        logger.info("===> Freeze resnet50 backbone, total {} params.".format(freeze_cnt))
    
    def cnt_params(self, lora=False):
        cnt = 0
        for name, param in self.named_parameters():
            if lora:
                # cnt convadapter parameters
                if "lora" in name or "bn" in name or "bias" in name:
                    cnt += param.numel()
            else:
                if not "lora" in name:
                    cnt += param.numel()
        return cnt
    
    def merge_correlated(self, merge_weights=False):
        """
        Merge correlated weights for pwconv1 and pwconv2 every K iterations
        """
        for i in range(4):
            stage = getattr(self.res50, "layer{}".format(i+1))
            for layer in stage:
                layer.merge_and_reinitialize(merge_weights=merge_weights)

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
        # features = [self.fpn1(ls[0]), self.fpn2(ls[1]), self.fpn3(ls[2]), self.fpn4(ls[3])]
        return self.norm(ls[-1])

# ResNet 18: [2, 2, 2, 2]
# ResNet 34: [3, 4, 6, 3] Basic block
# ResNet 50: [3, 4, 6, 3] Bottleneck
# ResNet101: [3, 4, 23, 3] Bottleneck
# ResNet 152: [3, 8, 36, 3]