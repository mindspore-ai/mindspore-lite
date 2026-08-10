# Copyright 2025 The Qwen-Image Team, Wan Team and The HuggingFace Team. All rights reserved.
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We gratefully acknowledge the Wan Team for their outstanding contributions.
# QwenImageVAE is further fine-tuned from the Wan Video VAE to achieve improved performance.
# For more information about the Wan VAE, please refer to:
# - GitHub: https://github.com/Wan-Video/Wan2.1
# - Paper: https://huggingface.co/papers/2503.20314
"""QwenImage VAE model with distributed multi-GPU tiled encode/decode support."""
from __future__ import annotations

import os
from typing import Optional
import math
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.activations import get_activation
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import AutoencoderMixin, DecoderOutput, DiagonalGaussianDistribution

CACHE_T = 2
# Environment variable: when PROMPT_VAE_ENCODE_PARALLEL=1, the encoder uses ranks 0..N-2
# for parallel encoding, leaving rank N-1 dedicated to the prompt text model;
# otherwise all ranks participate in VAE encode/decode parallelism.
PROMPT_VAE_ENCODE_PARALLEL = bool(int(os.environ.get('PROMPT_VAE_ENCODE_PARALLEL', 0)))

class QwenImageCausalConv3d(nn.Conv3d):
    r"""
    A custom 3D causal convolution layer with feature caching support.

    This layer extends the standard Conv3D layer by ensuring causality in the time dimension and handling feature
    caching for efficient inference.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 0
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Set up causal padding
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)

class QwenImageCausalConv2d(nn.Module):
    r"""
    A wrapper that converts QwenImageCausalConv3d into an equivalent native 2D convolution.

    For single-frame (T=1) encode/decode in video VAEs, the temporal dimension of 3D
    convolution collapses to 1. Using native Conv2D avoids unnecessary dimension
    reshuffling overhead and achieves better performance on NPU-like hardware.
    This class auto-converts from existing Conv3d weights via the from_conv3d classmethod,
    preserving numerical equivalence.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int | tuple[int, int]): Kernel size (3D format).
        stride (int | tuple[int, int]): Stride (3D format).
        padding (int | tuple[int, int]): Padding (3D format).
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] = 0,
    ):
        super().__init__()

        # Normalize 2D arguments to 3D format for alignment with original Conv3d weights
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride, ) * 3
        if isinstance(padding, int):
            padding = (padding, ) * 3

        self.kernel_size_3d = kernel_size
        self.stride_3d = stride
        self.padding_3d = padding
        self.in_channels_3d = in_channels
        self.out_channels_3d = out_channels

        # Causal padding layout: (W_left, W_right, H_left, H_right, D_left, D_right)
        self._padding = (padding[2], padding[2], padding[1], padding[1], 2 * padding[0], 0)

        # Retain 3D weight parameters; w2d is lazily built on first forward pass
        self.weight_3d = nn.Parameter(torch.empty(
            out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]
        ))

        self.bias = nn.Parameter(torch.empty(out_channels))
        self.w2d = None

        nn.init.kaiming_uniform_(self.weight_3d, a=math.sqrt(5))
        fan_in = in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_conv3d(cls, conv3d_module: nn.Conv3d):
        """Construct an equivalent CasualConv2d from an existing Conv3d module, copying weights and bias."""
        in_channels = conv3d_module.in_channels
        out_channels = conv3d_module.out_channels
        kernel_size = conv3d_module.kernel_size
        stride = conv3d_module.stride

        if hasattr(conv3d_module, "_padding"):
            conv_pad = conv3d_module._padding
            d_left = conv_pad[4]
            h_pad = conv_pad[2]
            w_pad = conv_pad[0]
            padding = (d_left // 2, h_pad, w_pad)
        else:
            padding = conv3d_module.padding

        obj = cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        obj.weight_3d.data.copy_(conv3d_module.weight.data)
        if conv3d_module.bias is not None:
            obj.bias.data.copy_(conv3d_module.bias.data)
        obj.to(conv3d_module.weight.dtype).to(conv3d_module.weight.device)
        return obj

    def forward(self, x, cache_x=None):
        """Forward pass with optional causal cache support.

        Args:
            x: Input tensor of shape (B, C, T, H, W).
            cache_x: Optional cached features for causal temporal padding.

        Returns:
            Output tensor of shape (B, C_out, 1, H_out, W_out).
        """
        # Cache path: causal padding + multi-frame fusion, flatten 3D input to 2D for conv2d
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)

            padding = list(self._padding)
            padding[4] -= cache_x.shape[2]

            x = F.pad(x, padding)

            bsz, ch, depth, ht, wd = x.shape
            x = x.reshape(bsz, ch * depth, ht, wd)

            if self.w2d is None:
                self.w2d = self.weight_3d.reshape(
                    self.out_channels_3d, -1, self.kernel_size_3d[1], self.kernel_size_3d[2]
                )

            x = F.conv2d(x, self.w2d, self.bias, self.stride_3d[1:])
            x = x.unsqueeze(2)
            return x
        # No-cache path (single frame T=1): take the last frame of kernel for 2D conv
        x = x.squeeze(2)
        if self.w2d is None:
            self.w2d = self.weight_3d[:, :, self.kernel_size_3d[0] - 1, :, :]

        x = F.conv2d(
            x, self.w2d, self.bias,
            stride=self.stride_3d[1:],
            padding=(self.padding_3d[1], self.padding_3d[2]),
        )
        x = x.unsqueeze(2)
        return x
class QwenImageRMS_norm(nn.Module):
    r"""
    A custom RMS normalization layer.

    Args:
        dim (int): The number of dimensions to normalize over.
        channel_first (bool, optional): Whether the input tensor has channels as the first dimension.
            Default is True.
        images (bool, optional): Whether the input represents image data. Default is True.
        bias (bool, optional): Whether to include a learnable bias term. Default is False.
    """

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class QwenImageUpsample(nn.Upsample):
    r"""
    Perform upsampling while ensuring the output tensor has the same data type as the input.

    Args:
        x (torch.Tensor): Input tensor to be upsampled.

    Returns:
        torch.Tensor: Upsampled tensor with the same data type as the input.
    """

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class QwenImageResample(nn.Module):
    r"""
    A custom resampling module for 2D and 3D data.

    Args:
        dim (int): The number of input/output channels.
        mode (str): The resampling mode. Must be one of:
            - 'none': No resampling (identity operation).
            - 'upsample2d': 2D upsampling with nearest-exact interpolation and convolution.
            - 'upsample3d': 3D upsampling with nearest-exact interpolation, convolution, and causal 3D convolution.
            - 'downsample2d': 2D downsampling with zero-padding and convolution.
            - 'downsample3d': 3D downsampling with zero-padding, convolution, and causal 3D convolution.
    """

    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = QwenImageCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = QwenImageCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=None):
        """Forward pass with optional feature caching for causal convolutions."""
        if feat_idx is None:
            feat_idx = [0]
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class QwenImageResidualBlock(nn.Module):
    r"""
    A custom residual block module.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        dropout (float, optional): Dropout rate for the dropout layer. Default is 0.0.
        non_linearity (str, optional): Type of non-linearity to use. Default is "silu".
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_activation(non_linearity)

        # layers
        self.norm1 = QwenImageRMS_norm(in_dim, images=False)
        self.conv1 = QwenImageCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImageCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = QwenImageCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

        # 2D equivalent converters: lazily built from Conv3d weights on first forward pass.
        # When T=1, 3D convolution degenerates to 2D; using native Conv2D yields better performance.
        self.conv_shortcut_2d = None
        self.conv1_2d = None
        self.conv2_2d = None

    def forward(self, x, feat_cache=None, feat_idx=None):
        """Forward pass through the residual block with optional feature caching."""
        if feat_idx is None:
            feat_idx = [0]
        # Apply shortcut connection
        h = self.conv_shortcut(x)

        # First normalization and activation
        x = self.norm1(x)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

            # Replace 3D causal conv with equivalent 2D conv; conv2d outperforms conv3d on NPU
            # x = self.conv1(x, feat_cache[idx])
            if self.conv1_2d is None:
                self.conv1_2d = QwenImageCausalConv2d.from_conv3d(self.conv1)
            x = self.conv1_2d(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            # No-cache path also uses 2D convolution for better hardware performance
            # x = self.conv1(x)
            if self.conv1_2d is None:
                self.conv1_2d = QwenImageCausalConv2d.from_conv3d(self.conv1)
            x = self.conv1_2d(x)
        # Second normalization and activation
        x = self.norm2(x)
        x = self.nonlinearity(x)

        # Dropout
        x = self.dropout(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

            # conv2 also uses 2D convolution instead of 3D
            # x = self.conv2(x, feat_cache[idx])
            if self.conv2_2d is None:
                self.conv2_2d = QwenImageCausalConv2d.from_conv3d(self.conv2)
            x = self.conv2_2d(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            # x = self.conv2(x)
            if self.conv2_2d is None:
                self.conv2_2d = QwenImageCausalConv2d.from_conv3d(self.conv2)
            x = self.conv2_2d(x)

        # Add residual connection
        return x + h


class QwenImageAttentionBlock(nn.Module):
    r"""
    Causal self-attention with a single head.

    Args:
        dim (int): The number of channels in the input tensor.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = QwenImageRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        """Forward pass applying single-head causal self-attention."""
        identity = x
        batch_size, channels, time, height, width = x.size()

        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        # compute query, key, value
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(q, k, v)

        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)

        # output projection
        x = self.proj(x)

        # Reshape back: [(b*t), c, h, w] -> [b, c, t, h, w]
        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        return x + identity


class QwenImageMidBlock(nn.Module):
    """
    Middle block for QwenImageVAE encoder and decoder.

    Args:
        dim (int): Number of input/output channels.
        dropout (float): Dropout rate.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1):
        super().__init__()
        self.dim = dim

        # Create the components
        resnets = [QwenImageResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImageAttentionBlock(dim))
            resnets.append(QwenImageResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None):
        """Forward pass through the middle block with optional feature caching."""
        if feat_idx is None:
            feat_idx = [0]
        # First residual block
        x = self.resnets[0](x, feat_cache, feat_idx)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)

            x = resnet(x, feat_cache, feat_idx)

        return x


class QwenImageEncoder3d(nn.Module):
    r"""
    A 3D encoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temporal_downsample (list of bool): Whether to downsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temporal_downsample=[True, True, False],
        dropout=0.0,
        input_channels=3,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_downsample = temporal_downsample
        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv_in = QwenImageCausalConv3d(input_channels, dims[0], 3, padding=1)

        # downsample blocks
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                self.down_blocks.append(QwenImageResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    self.down_blocks.append(QwenImageAttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temporal_downsample[i] else "downsample2d"
                self.down_blocks.append(QwenImageResample(out_dim, mode=mode))
                scale /= 2.0

        # middle blocks
        self.mid_block = QwenImageMidBlock(out_dim, dropout, non_linearity, num_layers=1)

        # output blocks
        self.norm_out = QwenImageRMS_norm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, z_dim, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None):
        """Forward pass through the 3D encoder with optional feature caching."""
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## downsamples
        for layer in self.down_blocks:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        x = self.mid_block(x, feat_cache, feat_idx)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class QwenImageUpBlock(nn.Module):
    """
    A block that handles upsampling for the QwenImageVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        upsample_mode (str, optional): Mode for upsampling ('upsample2d' or 'upsample3d')
        non_linearity (str): Type of non-linearity to use
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: Optional[str] = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Create layers list
        resnets = []
        # Add residual blocks and attention if needed
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImageResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([QwenImageResample(out_dim, mode=upsample_mode)])

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None):
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor
            feat_cache (list, optional): Feature cache for causal convolutions
            feat_idx (list, optional): Feature index for cache management

        Returns:
            torch.Tensor: Output tensor
        """
        if feat_idx is None:
            feat_idx = [0]
        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache, feat_idx)
            else:
                x = resnet(x)

        if self.upsamplers is not None:
            if feat_cache is not None:
                x = self.upsamplers[0](x, feat_cache, feat_idx)
            else:
                x = self.upsamplers[0](x)
        return x


class QwenImageDecoder3d(nn.Module):
    r"""
    A 3D decoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temporal_upsample (list of bool): Whether to upsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temporal_upsample=[False, True, True],
        dropout=0.0,
        input_channels=3,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_upsample = temporal_upsample

        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # init block
        self.conv_in = QwenImageCausalConv3d(z_dim, dims[0], 3, padding=1)
        # 2D equivalent of conv_in: lazily converted from conv_in's 3D weights on first forward pass
        self.conv_in_2d = None
        # middle blocks
        self.mid_block = QwenImageMidBlock(dims[0], dropout, non_linearity, num_layers=1)

        # upsample blocks
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i > 0:
                in_dim = in_dim // 2

            # Determine if we need upsampling
            upsample_mode = None
            if i != len(dim_mult) - 1:
                upsample_mode = "upsample3d" if temporal_upsample[i] else "upsample2d"

            # Create and add the upsampling block
            up_block = QwenImageUpBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                upsample_mode=upsample_mode,
                non_linearity=non_linearity,
            )
            self.up_blocks.append(up_block)

            # Update scale for next iteration
            if upsample_mode is not None:
                scale *= 2.0

        # output blocks
        self.norm_out = QwenImageRMS_norm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, input_channels, 3, padding=1)
        # 2D equivalent of conv_out: lazily converted from conv_out's 3D weights on first forward pass
        self.conv_out_2d = None
        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None):
        """Forward pass through the 3D decoder with optional feature caching."""
        if feat_idx is None:
            feat_idx = [0]
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            # Replace 3D causal conv with equivalent 2D conv to reduce dimension reshuffling
            # x = self.conv_in(x, feat_cache[idx])
            if self.conv_in_2d is None:
                self.conv_in_2d = QwenImageCausalConv2d.from_conv3d(self.conv_in)
            x = self.conv_in_2d(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            # No-cache path also uses 2D convolution
            # x = self.conv_in(x)
            if self.conv_in_2d is None:
                self.conv_in_2d = QwenImageCausalConv2d.from_conv3d(self.conv_in)
            x = self.conv_in_2d(x)
        ## middle
        x = self.mid_block(x, feat_cache, feat_idx)

        ## upsamples
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache, feat_idx)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            # Output conv also uses 2D convolution
            # x = self.conv_out(x, feat_cache[idx])
            if self.conv_out_2d is None:
                self.conv_out_2d = QwenImageCausalConv2d.from_conv3d(self.conv_out)
            x = self.conv_out_2d(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            # No-cache path also uses 2D convolution
            # x = self.conv_out(x)
            if self.conv_out_2d is None:
                self.conv_out_2d = QwenImageCausalConv2d.from_conv3d(self.conv_out)
            x = self.conv_out_2d(x)
        return x


class AutoencoderKLQwenImage(ModelMixin, AutoencoderMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,

        attn_scales: list[float] = [],
        temporal_downsample: list[bool] = [False, True, True],
        dropout: float = 0.0,
        input_channels: int = 3,
        latents_mean: list[float] = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ],
        latents_std: list[float] = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ],
    ) -> None:
    # fmt: on
        super().__init__()

        self.z_dim = z_dim
        self.temporal_downsample = temporal_downsample
        self.temporal_upsample = temporal_downsample[::-1]

        self.encoder = QwenImageEncoder3d(
            base_dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales,
            self.temporal_downsample, dropout, input_channels
        )
        self.quant_conv = QwenImageCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = QwenImageCausalConv3d(z_dim, z_dim, 1)

        self.decoder = QwenImageDecoder3d(
            base_dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temporal_upsample, dropout, input_channels
        )

        self.spatial_compression_ratio = 2 ** len(self.temporal_downsample)

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = False

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

        # Precompute and cache conv counts for encoder and decoder for clear_cache speedup
        self._cached_conv_counts = {
            "decoder": sum(isinstance(m, QwenImageCausalConv3d) for m in self.decoder.modules())
            if self.decoder is not None
            else 0,
            "encoder": sum(isinstance(m, QwenImageCausalConv3d) for m in self.encoder.modules())
            if self.encoder is not None
            else 0,
        }
        # Initialize distributed communication group: when prompt-parallel mode is enabled,
        # exclude the last rank (reserved for the prompt text model) from the VAE group.
        if PROMPT_VAE_ENCODE_PARALLEL:
            self.vae_group = dist.new_group(range(dist.get_world_size() - 1))
        else:
            self.vae_group = dist.new_group(range(dist.get_world_size()))


    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
    ) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The minimum height required for a sample to be separated into tiles across the height dimension.
            tile_sample_min_width (`int`, *optional*):
                The minimum width required for a sample to be separated into tiles across the width dimension.
            tile_sample_stride_height (`int`, *optional*):
                The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
                no tiling artifacts produced across the height dimension.
            tile_sample_stride_width (`int`, *optional*):
                The stride between two consecutive horizontal tiles. This is to ensure that there are no tiling
                artifacts produced across the width dimension.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width

    def clear_cache(self):
        """Reset feature caches for encoder and decoder causal convolutions."""
        self._conv_num = self._cached_conv_counts["decoder"]
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = self._cached_conv_counts["encoder"]
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def _encode(self, x: torch.Tensor):
        """Encode input tensor into latent representation with tiling support."""
        _, _, num_frame, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        self.clear_cache()
        iter_ = 1 + (num_frame - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)

        enc = self.quant_conv(out)
        self.clear_cache()
        return enc

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        r"""
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded videos. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        """Decode latent tensor back to pixel space with tiling support."""
        _, _, num_frame, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        self.clear_cache()
        x = self.post_quant_conv(z)
        for i in range(num_frame):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)

        out = torch.clamp(out, min=-1.0, max=1.0)
        self.clear_cache()
        if not return_dict:
            return (out,)

        return DecoderOutput(sample=out)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | torch.Tensor:
        r"""
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def _gather_and_blend_tiles(
        self,
        my_tiles,
        my_positions,
        tile_positions,
        use_distributed,
        world_size,
        dist_group,
        blend_height,
        blend_width,
        tile_stride_h,
        tile_stride_w,
        device,
        dtype,
        output_height,
        output_width,
        fallback_bct,
    ):
        """Gather tiles from all ranks via all_gather, then overlap-blend in row-major order."""
        if use_distributed:
            local_info = [(pos, my_tiles[pos].shape) for pos in my_positions]
            gathered_info = [None] * world_size
            if dist_group is not None:
                dist.all_gather_object(gathered_info, local_info, group=dist_group)
            else:
                dist.all_gather_object(gathered_info, local_info)

            all_info = [item for sublist in gathered_info for item in sublist]
            if all_info:
                max_h = max(s[3] for _, s in all_info)
                max_w = max(s[4] for _, s in all_info)
                bsz, ch, num_t = all_info[0][1][:3]
            else:
                max_h = max_w = 1
                bsz, ch, num_t = fallback_bct
            max_tiles = max(len(info) for info in gathered_info)
            flat_size = bsz * ch * num_t * max_h * max_w

            padded_chunks = []
            orig_shape_list = []
            for pos in my_positions:
                tile = my_tiles[pos]
                orig_shape_list.append(tile.shape)
                if tile.shape[-2] != max_h or tile.shape[-1] != max_w:
                    pad_h = max_h - tile.shape[-2]
                    pad_w = max_w - tile.shape[-1]
                    tile = F.pad(tile, (0, pad_w, 0, pad_h))
                padded_chunks.append(tile.reshape(-1))

            dummy = torch.zeros(flat_size, device=device, dtype=dtype)
            for _ in range(max_tiles - len(padded_chunks)):
                padded_chunks.append(dummy.clone())
                orig_shape_list.append(None)
            my_buffer = torch.cat(padded_chunks) if padded_chunks else dummy

            gathered_buffers = [torch.zeros_like(my_buffer) for _ in range(world_size)]
            if dist_group is not None:
                dist.all_gather(gathered_buffers, my_buffer, group=dist_group)
            else:
                dist.all_gather(gathered_buffers, my_buffer)

            gathered_shapes = [None] * world_size
            if dist_group is not None:
                dist.all_gather_object(gathered_shapes, orig_shape_list, group=dist_group)
            else:
                dist.all_gather_object(gathered_shapes, orig_shape_list)

            all_tiles = {}
            for r_idx in range(world_size):
                buf = gathered_buffers[r_idx]
                shapes = gathered_shapes[r_idx]
                for t_idx, (pos, _) in enumerate(gathered_info[r_idx]):
                    shape = shapes[t_idx]
                    if shape is None:
                        continue
                    start = t_idx * flat_size
                    tile = buf[start : start + flat_size].view(bsz, ch, num_t, max_h, max_w)
                    if shape[-2] != max_h or shape[-1] != max_w:
                        tile = tile[:, :, :, : shape[-2], : shape[-1]]
                    all_tiles[pos] = tile
        else:
            all_tiles = my_tiles

        row_keys = sorted({i for i, j in tile_positions})
        result_rows = []
        for i_idx, i in enumerate(row_keys):
            col_keys = sorted(j for pos_i, j in tile_positions if pos_i == i)
            result_row = []
            for j_idx, j in enumerate(col_keys):
                tile = all_tiles[(i, j)]
                if i_idx > 0:
                    prev_i = row_keys[i_idx - 1]
                    tile = self.blend_v(all_tiles[(prev_i, j)], tile, blend_height)
                if j_idx > 0:
                    prev_j = col_keys[j_idx - 1]
                    tile = self.blend_h(all_tiles[(i, prev_j)], tile, blend_width)
                result_row.append(tile[:, :, :, :tile_stride_h, :tile_stride_w])
            result_rows.append(torch.cat(result_row, dim=-1))

        return torch.cat(result_rows, dim=3)[:, :, :, :output_height, :output_width]

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder.

        Args:
            x (`torch.Tensor`): Input batch of videos.

        Returns:
            `torch.Tensor`:
                The latent representation of the encoded videos.
        """
        _, _, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        tile_positions = []
        for i in range(0, height, self.tile_sample_stride_height):
            for j in range(0, width, self.tile_sample_stride_width):
                tile_positions.append((i, j))

        use_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        if use_distributed:
            world_size = dist.get_world_size() - 1 if PROMPT_VAE_ENCODE_PARALLEL else dist.get_world_size()
            rank = dist.get_rank()
            my_positions = tile_positions[rank::world_size]
        else:
            world_size = 0
            my_positions = tile_positions

        my_tiles = {}
        frame_range = 1 + (num_frames - 1) // 4
        for i, j in my_positions:
            self.clear_cache()
            time = []
            for k in range(frame_range):
                self._enc_conv_idx = [0]
                if k == 0:
                    tile = x[:, :, :1, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                else:
                    tile = x[
                        :,
                        :,
                        1 + 4 * (k - 1) : 1 + 4 * k,
                        i : i + self.tile_sample_min_height,
                        j : j + self.tile_sample_min_width,
                    ]
                tile = self.encoder(tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                tile = self.quant_conv(tile)
                time.append(tile)
            my_tiles[(i, j)] = torch.cat(time, dim=2)
        self.clear_cache()

        enc = self._gather_and_blend_tiles(
            my_tiles, my_positions, tile_positions,
            use_distributed, world_size, self.vae_group,
            blend_height, blend_width,
            tile_latent_stride_height, tile_latent_stride_width,
            x.device, x.dtype,
            latent_height, latent_width,
            fallback_bct=(1, 1, 1),
        )
        return enc

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | torch.Tensor:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        tile_positions = []
        for i in range(0, height, tile_latent_stride_height):
            for j in range(0, width, tile_latent_stride_width):
                tile_positions.append((i, j))

        use_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        if use_distributed:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            my_positions = tile_positions[rank::world_size]
        else:
            world_size = 0
            my_positions = tile_positions

        my_tiles = {}
        for i, j in my_positions:
            self.clear_cache()
            time = []
            for k in range(num_frames):
                self._conv_idx = [0]
                tile = z[:, :, k : k + 1, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile, feat_cache=self._feat_map, feat_idx=self._conv_idx)
                time.append(decoded)
            my_tiles[(i, j)] = torch.cat(time, dim=2)
        self.clear_cache()

        all_tiles = self.decode_patch_method(my_positions, my_tiles, use_distributed, world_size, z)

        # Overlap-blend and concatenate all collected tiles in row-major order (decode path)
        row_keys = sorted({i for i, j in tile_positions})
        result_rows = []
        for i_idx, i in enumerate(row_keys):
            col_keys = sorted(j for _i, j in tile_positions if _i == i)
            result_row = []
            for j_idx, j in enumerate(col_keys):
                tile = all_tiles[(i, j)]
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i_idx > 0:
                    prev_i = row_keys[i_idx - 1]
                    tile = self.blend_v(all_tiles[(prev_i, j)], tile, blend_height)
                if j_idx > 0:
                    prev_j = col_keys[j_idx - 1]
                    tile = self.blend_h(all_tiles[(i, prev_j)], tile, blend_width)
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def decode_patch_method(self, my_positions, my_tiles, use_distributed, world_size, z):
        """Gather decoded tiles from all ranks via all_gather for the decode path."""
        if use_distributed:
            # Step 1: all_gather_object to collect tile metadata (position + shape) from all ranks (decode path)
            local_info = [(pos, my_tiles[pos].shape) for pos in my_positions]
            gathered_info = [None] * world_size
            dist.all_gather_object(gathered_info, local_info)

            # Determine max H/W across all tiles for uniform padding (decode path)
            all_info = [item for sublist in gathered_info for item in sublist]
            if all_info:
                max_h = max(s[3] for _, s in all_info)
                max_w = max(s[4] for _, s in all_info)
                b, c, num_t = all_info[0][1][:3]
            else:
                max_h = max_w = 1
                b, c, num_t = z.shape[:3]
            max_tiles = max(len(info) for info in gathered_info)
            flat_size = b * c * num_t * max_h * max_w
            # Pad tiles to uniform size, then flatten for all_gather broadcast (decode path)
            padded_chunks = []
            orig_shape_list = []
            for pos in my_positions:
                tile = my_tiles[pos]
                orig_shape_list.append(tile.shape)
                if tile.shape[-2] != max_h or tile.shape[-1] != max_w:
                    pad_h = max_h - tile.shape[-2]
                    pad_w = max_w - tile.shape[-1]
                    tile = F.pad(tile, (0, pad_w, 0, pad_h))
                padded_chunks.append(tile.reshape(-1))

            dummy = torch.zeros(flat_size, device=z.device, dtype=z.dtype)
            for _ in range(max_tiles - len(padded_chunks)):
                padded_chunks.append(dummy.clone())
                orig_shape_list.append(None)
            my_buffer = torch.cat(padded_chunks) if padded_chunks else dummy
            # all_gather requires equally-sized buffers from every rank (decode path)
            gathered_buffers = [torch.zeros_like(my_buffer) for _ in range(world_size)]
            dist.all_gather(gathered_buffers, my_buffer)

            # all_gather_object carries each tile's original shape for reconstruction (decode path)
            gathered_shapes = [None] * world_size
            dist.all_gather_object(gathered_shapes, orig_shape_list)

            all_tiles = {}
            # Reconstruct each rank's tiles from the gathered buffer using shape metadata (decode path)
            for rank_idx in range(world_size):
                buf = gathered_buffers[rank_idx]
                shapes = gathered_shapes[rank_idx]
                for t_idx, (pos, _) in enumerate(gathered_info[rank_idx]):
                    shape = shapes[t_idx]
                    if shape is None:
                        continue
                    start = t_idx * flat_size
                    tile = buf[start: start + flat_size].view(b, c, num_t, max_h, max_w)
                    if shape[-2] != max_h or shape[-1] != max_w:
                        tile = tile[:, :, :, : shape[-2], : shape[-1]]
                    all_tiles[pos] = tile
        else:
            all_tiles = my_tiles
        return all_tiles

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> DecoderOutput | torch.Tensor:
        """
        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, return_dict=return_dict)
        return dec
