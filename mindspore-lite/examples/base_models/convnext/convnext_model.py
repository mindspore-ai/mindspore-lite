"""Pure-PyTorch reimplementation of ConvNeXt backbone + UPerNet decode head.

Reconstructed from the official ConvNeXt semantic-segmentation checkpoint
``upernet_convnext_tiny_1k_512x512.pth`` (mmseg 0.11.0, ADE20K 150 classes).
Only the inference forward path (backbone + decode_head) is implemented; the
training-only ``auxiliary_head`` is omitted.
"""

import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm that operates on channels-first (N, C, H, W) tensors.

    Delegates to ``F.layer_norm`` via a channels-last permutation so the
    computation is exported as a fused LayerNormalization op (numerically
    stable in fp16) instead of a decomposed mean/var/sqrt subgraph.
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        """Apply layer norm on (N, C, H, W) via channels-last permutation."""
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt residual block: DwConv -> LN -> PWConv -> GELU -> PWConv."""

    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        """Compute the ConvNeXt block residual output."""
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x


class ConvNeXt(nn.Module):
    """ConvNeXt backbone producing multi-scale feature maps."""

    def __init__(self, in_chans=3, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
                 layer_scale_init_value=1e-6, out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.out_indices = out_indices
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[
                ConvNeXtBlock(dims[i], layer_scale_init_value) for _ in range(depths[i])]))
        for i in range(4):
            self.add_module(f"norm{i}", LayerNorm2d(dims[i], eps=1e-6))

    def forward(self, x):
        """Return a list of 4 multi-scale feature maps."""
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                outs.append(getattr(self, f"norm{i}")(x))
        return outs


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm2d + ReLU module (mmseg ConvModule equivalent)."""

    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        """Apply conv, batch norm, and ReLU."""
        return F.relu(self.bn(self.conv(x)), inplace=True)


def _build_pool_matrix(in_size, out_size):
    """Build the (out_size, in_size) adaptive-average-pooling matrix.

    Row i has 1/region_len on columns [start_i, end_i) where
    start_i = floor(i*in/out), end_i = ceil((i+1)*in/out).  Multiplying on
    both spatial dims reproduces ``F.adaptive_avg_pool2d`` exactly while being
    ONNX-exportable (MatMul) even when out does not divide in.
    """
    import math
    mat = torch.zeros(out_size, in_size)
    for i in range(out_size):
        start = (i * in_size) // out_size
        end = math.ceil((i + 1) * in_size / out_size)
        mat[i, start:end] = 1.0 / (end - start)
    return mat


class UPerHead(nn.Module):
    """UPerNet decode head: FPN + PSP, producing per-pixel class logits."""

    def __init__(self, in_channels=(96, 192, 384, 768), channels=512,
                 pool_scales=(1, 2, 3, 6), num_classes=150, dropout_ratio=0.1,
                 align_corners=False, feat_size=16):
        super().__init__()
        self.align_corners = align_corners
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_ch in in_channels[:-1]:
            self.lateral_convs.append(ConvBNReLU(in_ch, channels, 1))
            self.fpn_convs.append(ConvBNReLU(channels, channels, 3, padding=1))
        self.psp_modules = nn.ModuleList([
            nn.Sequential(nn.Identity(), ConvBNReLU(in_channels[-1], channels, 1))
            for _ in pool_scales])
        self.pool_scales = pool_scales
        for i, scale in enumerate(pool_scales):
            self.register_buffer(
                f"pool_mat_{i}", _build_pool_matrix(feat_size, scale))
        self.bottleneck = ConvBNReLU(
            in_channels[-1] + len(pool_scales) * channels, channels, 3, padding=1)
        self.fpn_bottleneck = ConvBNReLU(len(in_channels) * channels, channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)

    def _adaptive_avg_pool(self, x, mat):
        """Apply adaptive average pooling via matrix multiply on H and W."""
        out = torch.matmul(mat, x)
        return torch.matmul(out, mat.t())

    def _psp_forward(self, x):
        """PSP forward: pool to each scale, upsample back, concat, bottleneck."""
        outs = [x]
        for i, module in enumerate(self.psp_modules):
            mat = getattr(self, f"pool_mat_{i}")
            pooled = self._adaptive_avg_pool(x, mat)
            out = module(pooled)
            outs.append(F.interpolate(
                out, size=x.shape[2:], mode="bilinear", align_corners=self.align_corners))
        return self.bottleneck(torch.cat(outs, dim=1))

    def forward(self, inputs):
        """Run UPerNet head on a list of multi-scale feature maps."""
        laterals = [self.lateral_convs[i](inputs[i]) for i in range(len(self.lateral_convs))]
        laterals.append(self._psp_forward(inputs[-1]))
        num_levels = len(laterals)
        for i in range(num_levels - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode="bilinear", align_corners=self.align_corners)
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(num_levels - 1)]
        fpn_outs.append(laterals[-1])
        for i in range(num_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:],
                mode="bilinear", align_corners=self.align_corners)
        output = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        return self.conv_seg(self.dropout(output))


class UPerNetConvNeXt(nn.Module):
    """Full segmentation model: ConvNeXt backbone + UPerNet decode head.

    Output logits are bilinearly upsampled to the input spatial size so the
    exported graph is self-contained for a fixed input resolution.
    """

    def __init__(self, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
                 num_classes=150, align_corners=False, input_size=512):
        super().__init__()
        self.backbone = ConvNeXt(depths=depths, dims=dims, layer_scale_init_value=1e-6)
        self.decode_head = UPerHead(
            in_channels=dims, channels=512, num_classes=num_classes,
            align_corners=align_corners, feat_size=input_size // 32)
        self.align_corners = align_corners

    def forward(self, x):
        """Run backbone + decode head and upsample logits to input size."""
        features = self.backbone(x)
        out = self.decode_head(features)
        return F.interpolate(
            out, size=x.shape[2:], mode="bilinear", align_corners=self.align_corners)


def load_pretrained(model, ckpt_path):
    """Load mmseg checkpoint weights into the pure-PyTorch model."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    own = model.state_dict()
    loaded, skipped = {}, []
    for k, v in state_dict.items():
        if k in own and own[k].shape == v.shape:
            loaded[k] = v
        else:
            skipped.append(k)
    own.update(loaded)
    model.load_state_dict(own, strict=True)
    return len(loaded), skipped
