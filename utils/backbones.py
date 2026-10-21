import math
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, VisionEncoderDecoderModel

DINOV3_DEFAULT_CKPT = "facebook/dinov3-vitb16-pretrain-lvd1689m"
TROCR_DEFAULT_CKPT = "microsoft/trocr-base-printed"


class DINOv3Backbone(nn.Module):
    """forward returns a patch-feature grid [B, grid_height, grid_width, feature_dim], the
    contract every backbone here shares."""

    def __init__(self, ckpt: str = DINOV3_DEFAULT_CKPT):
        super().__init__()
        self.model = AutoModel.from_pretrained(ckpt)
        self.feature_dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.patch_size_w = self.patch_size
        self.num_special_tokens = 1 + self.model.config.num_register_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        outputs = self.model(pixel_values=x)
        patch_features = outputs.last_hidden_state[:, self.num_special_tokens:, :]
        grid_height = H // self.patch_size
        grid_width = W // self.patch_size
        return patch_features.view(B, grid_height, grid_width, self.feature_dim)


class TrOCRBackbone(nn.Module):
    """The encoder weights live behind an `encoder.` prefix in the checkpoint, so it is extracted
    via VisionEncoderDecoderModel rather than loaded as a bare ViT. Line images are not square, so
    the pretrained position grid is always interpolated."""

    def __init__(self, repo: str = TROCR_DEFAULT_CKPT):
        super().__init__()
        self.model = VisionEncoderDecoderModel.from_pretrained(repo).encoder
        self.model.pooler = None
        self.feature_dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.patch_size_w = self.patch_size
        self.num_special_tokens = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        outputs = self.model(pixel_values=x, interpolate_pos_encoding=True)
        patch_features = outputs.last_hidden_state[:, self.num_special_tokens:, :]
        grid_height = H // self.patch_size
        grid_width = W // self.patch_size
        return patch_features.view(B, grid_height, grid_width, self.feature_dim)


class SqueezeExcite(nn.Module):

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1)
        self.gate = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=(2, 3), keepdim=True)
        scale = self.act(self.fc1(scale))
        scale = self.gate(self.fc2(scale))
        return x * scale


class DepthSepConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: Tuple[int, int], use_se: bool):
        super().__init__()
        padding = kernel_size // 2
        self.use_residual = tuple(stride) == (1, 1) and in_channels == out_channels
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.dw_act = nn.Hardswish(inplace=True)
        self.se = SqueezeExcite(in_channels) if use_se else nn.Identity()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.pw_act = nn.Hardswish(inplace=True)
        if self.use_residual:
            nn.init.zeros_(self.pw_bn.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.dw_act(self.dw_bn(self.depthwise(x)))
        out = self.se(out)
        out = self.pw_bn(self.pointwise(out))
        if self.use_residual:
            out = out + identity
        return self.pw_act(out)


class CNNBackbone(nn.Module):
    """Vertical total stride is patch_size and horizontal total stride is width_stride, so
    grid_height matches a ViT grid while width keeps the finer resolution CTC needs."""

    def __init__(
        self,
        feature_dim: int = 512,
        patch_size: int = 16,
        width_stride: int = 16,
        in_channels: int = 3,
        base_channels: int = 96,
        stage_depths: Tuple[int, ...] = (1, 2, 4, 6),
    ):
        super().__init__()
        for name, value in (('patch_size', patch_size), ('width_stride', width_stride)):
            if value < 1 or (value & (value - 1)) != 0:
                raise ValueError(f"CNNBackbone {name} must be a power of two >= 1, got {value}")

        num_height_downsamples = int(math.log2(patch_size))
        num_width_downsamples = int(math.log2(width_stride))
        num_stages = max(num_height_downsamples, num_width_downsamples)
        if num_stages < 1:
            raise ValueError("CNNBackbone requires patch_size or width_stride > 1.")

        depths = list(stage_depths)
        if len(depths) < num_stages:
            depths = [1] * (num_stages - len(depths)) + depths
        elif len(depths) > num_stages:
            depths = depths[len(depths) - num_stages:]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.Hardswish(inplace=True),
        )

        blocks = []
        in_ch = base_channels
        out_channels = base_channels
        for stage in range(num_stages):
            stages_left = num_stages - stage
            stride_h = 2 if stages_left <= num_height_downsamples else 1
            stride_w = 2 if stages_left <= num_width_downsamples else 1
            stage_out = feature_dim if stage == num_stages - 1 else out_channels
            in_last_two = stage >= num_stages - 2
            kernel_size = 5 if in_last_two else 3
            for block_idx in range(max(1, depths[stage])):
                is_first = block_idx == 0
                block_in = in_ch if is_first else stage_out
                block_stride = (stride_h, stride_w) if is_first else (1, 1)
                blocks.append(DepthSepConv(block_in, stage_out, kernel_size, block_stride, use_se=in_last_two))
            in_ch = stage_out
            out_channels = min(out_channels * 2, feature_dim)
        self.blocks = nn.Sequential(*blocks)

        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.patch_size_w = width_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return x.permute(0, 2, 3, 1).contiguous()


class GroupedConv(nn.Module):
    """Grouped 3x3 conv with BatchNorm, Hardswish and optional squeeze-excite. Stands in for the
    depthwise + pointwise pair of DepthSepConv.

    The depthwise pair is memory-bound on a GPU: at 128x2048 with 96 channels one DepthSepConv
    (10,464 params) took 6.13 ms against 6.41 ms for a plain 3x3 (83,136 params). Grouping at
    channels // 32, SVTRv2's choice, keeps enough arithmetic intensity to use the tensor cores
    while staying well under a dense conv's parameter count."""

    def __init__(self, in_channels: int, out_channels: int, stride: Tuple[int, int], use_se: bool,
                 group_width: int = 32):
        super().__init__()
        groups = max(1, min(in_channels, out_channels) // group_width)
        while in_channels % groups or out_channels % groups:
            groups -= 1
        self.use_residual = tuple(stride) == (1, 1) and in_channels == out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcite(out_channels) if use_se else nn.Identity()
        self.act = nn.Hardswish(inplace=True)
        if self.use_residual:
            nn.init.zeros_(self.bn.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.se(self.bn(self.conv(x)))
        if self.use_residual:
            out = out + x
        return self.act(out)


class GroupedCNNBackbone(nn.Module):
    """CNNBackbone's contract with GroupedConv blocks and a strided stem. Same output grid:
    [B, target_height / patch_size, W / width_stride, feature_dim].

    The stem halves height but not width. Width sets the CTC frame rate and cannot be recovered
    downstream; height is collapsed to patch_size rows regardless, so taking one of those halvings
    at the stem is close to free. The stem kernel is 5 because going from 3 to 5 cost 0.05 ms in
    the same benchmark. Remaining downsampling is spent on the earliest stages, where each halving
    removes the most work from everything after it."""

    def __init__(
        self,
        feature_dim: int = 512,
        patch_size: int = 16,
        width_stride: int = 8,
        in_channels: int = 3,
        stem_channels: int = 64,
        stage_depths: Tuple[int, ...] = (3, 6, 8),
        group_width: int = 32,
    ):
        super().__init__()
        for name, value in (('patch_size', patch_size), ('width_stride', width_stride)):
            if value < 2 or (value & (value - 1)) != 0:
                raise ValueError(f"GroupedCNNBackbone {name} must be a power of two >= 2, got {value}")

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.Hardswish(inplace=True),
        )

        remaining_h = patch_size // 2
        remaining_w = width_stride
        num_stages = max(int(math.log2(remaining_h)), int(math.log2(remaining_w)))

        depths = list(stage_depths)
        if len(depths) < num_stages:
            depths = [1] * (num_stages - len(depths)) + depths
        elif len(depths) > num_stages:
            depths = depths[len(depths) - num_stages:]

        blocks = []
        in_ch = stem_channels
        for stage in range(num_stages):
            stride = (2 if remaining_h > 1 else 1, 2 if remaining_w > 1 else 1)
            remaining_h //= stride[0]
            remaining_w //= stride[1]
            out_ch = feature_dim if stage == num_stages - 1 else min(stem_channels * 2 ** (stage + 1), feature_dim)
            use_se = stage >= num_stages - 2
            for block_idx in range(max(1, depths[stage])):
                block_in = in_ch if block_idx == 0 else out_ch
                block_stride = stride if block_idx == 0 else (1, 1)
                blocks.append(GroupedConv(block_in, out_ch, block_stride, use_se, group_width))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.patch_size_w = width_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return x.permute(0, 2, 3, 1).contiguous()


def build_backbone(
    name: str,
    dinov3_ckpt: str = DINOV3_DEFAULT_CKPT,
    trocr_ckpt: str = TROCR_DEFAULT_CKPT,
    in_channels: int = 3,
    cnn_feature_dim: int = 512,
    cnn_patch_size: int = 16,
    cnn_width_stride: int = 16,
    cnn_base_channels: int = 96,
    cnn_stage_depths: Tuple[int, ...] = (1, 2, 4, 6),
) -> nn.Module:
    if name == 'dinov3':
        return DINOv3Backbone(dinov3_ckpt)
    if name == 'trocr':
        return TrOCRBackbone(trocr_ckpt)
    if name == 'cnn':
        return CNNBackbone(
            feature_dim=cnn_feature_dim,
            patch_size=cnn_patch_size,
            width_stride=cnn_width_stride,
            in_channels=in_channels,
            base_channels=cnn_base_channels,
            stage_depths=cnn_stage_depths,
        )
    if name == 'cnn_grouped':
        return GroupedCNNBackbone(
            feature_dim=cnn_feature_dim,
            patch_size=cnn_patch_size,
            width_stride=cnn_width_stride,
            in_channels=in_channels,
        )
    raise ValueError(f"Unknown backbone '{name}'; expected 'dinov3', 'trocr', 'cnn' or 'cnn_grouped'.")
