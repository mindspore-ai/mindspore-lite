#!/usr/bin/env python3
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
# ============================================================================
"""
Export CosyVoice2-0.5B model to ONNX format.

Components exported:
  1. LLM Prefill  – full-sequence forward, returns logits + KV cache
  2. LLM Decode   – single-token autoregressive step with KV cache
  3. Flow Encoder  – conformer encoder producing mu / spks / cond / mask
  4. Flow Estimator – CFM estimator network used by Euler sampler
  5. HiFT Vocoder  – mel -> waveform, with bucketed static shapes for Ascend
"""

import sys
import argparse
import gc
import math
import types
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import onnx

try:
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
except ImportError:
    apply_rotary_pos_emb = None


# Set True to route attention through Custom(PromptFlashAttention); see _text_attn_forward.
_ENABLE_PFA = True
# Set True to also patch diffusers Attention in the Flow Estimator with PFA Custom.
_ENABLE_PFA_EST = True


# ---------------------------------------------------------------------------
# CANN Custom: PromptFlashAttention (BNSD layout, GQA-aware, bool mask)
# ---------------------------------------------------------------------------
class _CannPromptFlashAttention(torch.autograd.Function):
    """torch.autograd.Function that exports a CANN PromptFlashAttention Custom op.

    Forward (tracing only) reproduces fp32 softmax+matmul attention so the exported
    ONNX numerics match the original PyTorch reference. The symbolic method emits a
    Custom node so MSLite Ascend can lower it to a fused flash-attention kernel.
    """

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, num_key_value_heads, scale_value):
        """Run fp32 softmax+matmul attention so exported numerics match PyTorch."""
        del ctx
        if int(num_key_value_heads) != int(num_heads):
            repeat = int(num_heads) // int(num_key_value_heads)
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        scale = float(scale_value)
        attn = torch.matmul(query, key.transpose(2, 3)) * scale
        if atten_mask is not None:
            if atten_mask.dtype == torch.bool:
                mask_value = torch.finfo(query.dtype).min
                attn = torch.where(atten_mask, attn, torch.full_like(attn, mask_value))
            else:
                attn = attn + atten_mask
        attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(attn, value)

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, num_key_value_heads, scale_value):
        """Emit a Custom node so MSLite Ascend can lower it to fused flash-attention."""
        # input_index_i follows CANN PromptFlashAttention REG_OP slot order:
        #   0 query, 1 key, 2 value, 3 pse_shift, 4 atten_mask, ...
        y = g.op(
            "Custom",
            query,
            key,
            value,
            atten_mask,
            type_s="PromptFlashAttention",
            input_names_s=["query", "key", "value", "atten_mask"],
            optional_input_names_s=["atten_mask"],
            output_names_s=["attention_out"],
            output_num_i=1,
            input_index_i=[0, 1, 2, 4],
            num_heads_i=int(num_heads),
            num_key_value_heads_i=int(num_key_value_heads),
            scale_value_f=float(scale_value),
            input_layout_s="BNSD",
            inner_precise_i=1,
        )
        y.setType(query.type())
        return y


def _cann_pfa(query, key, value, atten_mask, num_heads, num_key_value_heads, scale_value):
    """Wrapper that lets us call the symbolic-replaced attention as a plain function."""
    return _CannPromptFlashAttention.apply(
        query, key, value, atten_mask, int(num_heads), int(num_key_value_heads), float(scale_value)
    )


def _patch_diffusers_attention_with_pfa():
    """Monkey-patch diffusers AttnProcessor2_0 so the Flow Estimator attention is exported as PFA Custom.

    The Flow Estimator uses diffusers `Attention` with `AttnProcessor2_0`, which routes through
    `F.scaled_dot_product_attention`. That exports to ONNX as a chain of MatMul/Softmax/MatMul.
    We replace the SDPA call with our `_CannPromptFlashAttention.apply` so the exported graph has
    a single Custom node per attention.

    No mask is needed: estimator attention is non-causal and we already pass `mask` separately
    to the estimator's outer forward (it is consumed by GroupNorm-style masking before attention).
    """
    import diffusers.models.attention_processor as ap

    def patched_call(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, temb=None, *args, **kwargs):
        # pylint: disable=keyword-arg-before-vararg
        # Signature mirrors diffusers AttnProcessor2_0.__call__ exactly so we can
        # monkey-patch it; *args absorbs the occasional positional `scale`.
        del self
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            kwargs.pop("scale", None)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            # SDPA mask shape (B, heads, src, tgt). PFA wants (B, 1, q_len, k_len) bool.
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
            # Reduce across heads (all heads share the same mask in our case).
            head_mask = attention_mask[:, 0:1, :, :].to(torch.bool)
        else:
            head_mask = None

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Layout: BNSD. q/k/v all use num_heads (no GQA in estimator).
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # PFA only takes fp16; cast at boundary.
        orig_dtype = query.dtype
        scale = 1.0 / (head_dim ** 0.5)
        if head_mask is None:
            # All-True mask broadcastable to (B, 1, q_len, k_len).
            head_mask = torch.ones(
                batch_size, 1, query.shape[2], key.shape[2],
                dtype=torch.bool, device=query.device,
            )
        hidden_states = _cann_pfa(
            query.to(torch.float16), key.to(torch.float16), value.to(torch.float16),
            head_mask, attn.heads, attn.heads, scale,
        ).to(orig_dtype)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    ap.AttnProcessor2_0.__call__ = patched_call
    print("[fusion] Patched diffusers AttnProcessor2_0 to use Custom(PromptFlashAttention).")


# ===========================================================================
# HiFT vocoder: deterministic replacements + manual STFT/ISTFT
#
# HiFT (HiFi-GAN + Neural Source Filter + ISTFT) has three export blockers:
#   1. SineGen.random ops: Uniform.sample for phase, torch.randn_like for noise
#   2. SourceModuleHnNSF.random ops: torch.randn_like for noise
#   3. torch.stft / torch.istft with windowing
#
# Strategy:
#   - Replace (1) and (2) with deterministic buffers (zero noise, fixed phase).
#   - Replace (3) with Conv1d/ConvTranspose1d using pre-baked DFT bases.
#     n_fft=16 is small enough that the bases are 16x9 weight matrices.
# ===========================================================================

def deterministic_randn_like(ref: torch.Tensor) -> torch.Tensor:
    """Replacement for torch.randn_like.

    torch.onnx.export bakes `int(ref.numel())` as a Constant, which breaks
    dynamic-shape inference on Ascend. We use torch.zeros_like (fully dynamic).
    Cost: unvoiced frames (where the noise branch dominates) will be silent;
    voiced frames (where noise_std is just 0.003) are unaffected.
    """
    return torch.zeros_like(ref)


def patch_sinegen_deterministic():
    """Monkey-patch SineGen (type='1') AND SineGen2 (type='2') forward methods.

    CosyVoice2 at 24000Hz uses SineGen2 (via SourceModuleHnNSF.sinegen_type='2').
    At 22050Hz it would use SineGen type='1'. We patch both for safety.
    """
    from cosyvoice.hifigan.generator import SineGen, SineGen2

    # ----- SineGen (type='1') -----
    def forward_v1(self, f0):
        f0 = f0.transpose(1, 2)  # (B, T, 1) -> (B, 1, T)
        F_mat = torch.zeros((f0.size(0), self.harmonic_num + 1, f0.size(-1)),
                            device=f0.device, dtype=f0.dtype)
        for i in range(self.harmonic_num + 1):
            F_mat[:, i:i + 1, :] = f0 * (i + 1) / self.sampling_rate

        theta_mat = 2 * math.pi * (torch.cumsum(F_mat, dim=-1) % 1)
        # DETERMINISTIC: zero phase
        phase_vec = torch.zeros((f0.size(0), self.harmonic_num + 1, 1),
                                device=F_mat.device, dtype=F_mat.dtype)
        sine_waves = self.sine_amp * torch.sin(theta_mat + phase_vec)
        uv = (f0 > self.voiced_threshold).to(sine_waves.dtype)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * deterministic_randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves.transpose(1, 2), uv.transpose(1, 2), noise

    SineGen.forward = forward_v1

    # ----- SineGen2 (type='2', used at 24000Hz) -----
    # The original SineGen2._f02sine uses F.interpolate(scale_factor=1/U) and
    # F.interpolate(scale_factor=U) which become ONNX Resize ops that Ascend
    # rejects. Since f0_upsamp is patched (below) to repeat_interleave, f0_long
    # is locally constant within each mel window — we can directly cumsum at
    # long rate and skip the original downsample-then-upsample dance.
    def _f02sine_v2(self, f0_values):
        # f0_values: (B, T_long, harmonic_num+1)
        rad_values = (f0_values / self.sampling_rate) % 1

        if not self.flag_for_pulse:
            # Initial phase noise (matches original SineGen2 for causal=True).
            # rand_ini is a fixed (1, F) buffer; harmonic 0 (fundamental) gets 0.
            if self.training is False and getattr(self, 'causal', False) is True:
                rad_values = rad_values.clone()
                rad_values[:, 0, :] = rad_values[:, 0, :] + self.rand_ini.to(rad_values.device)
            # Direct long-rate cumsum. Since f0 is locally constant within each
            # mel window (f0_upsamp uses repeat_interleave), rad_values is also
            # locally constant; cumsum gives a piecewise-linear phase with the
            # correct slope (2*pi*f0/sr) per sample.
            phase = torch.cumsum(rad_values, dim=1) * 2 * math.pi
            sines = torch.sin(phase)
        else:
            uv = (f0_values > self.voiced_threshold).to(f0_values.dtype)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1 = torch.cat([uv_1[:, :-1, :], torch.ones_like(uv_1[:, -1:, :])], dim=1)
            u_loc = (uv < 1) * (uv_1 > 0)
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            tmp_cumsum_new = torch.zeros_like(tmp_cumsum)
            for idx in range(f0_values.shape[0]):
                mask = u_loc[idx, :, 0]
                temp_sum = tmp_cumsum[idx][mask]
                temp_sum = torch.cat([temp_sum[:1], temp_sum[1:] - temp_sum[:-1]], dim=0)
                tmp_cumsum_new[idx][mask] = temp_sum
            i_phase = torch.cumsum(rad_values - tmp_cumsum_new, dim=1)
            sines = torch.cos(i_phase * 2 * math.pi)
        return sines

    def forward_v2(self, f0):
        # f0: (B, T_long, 1)
        harm_idx = torch.arange(1, self.harmonic_num + 2, 1, dtype=f0.dtype, device=f0.device)
        fn = f0 * harm_idx.view(1, 1, -1)
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = (f0 > self.voiced_threshold).to(sine_waves.dtype)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * deterministic_randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise

    SineGen2._f02sine = _f02sine_v2
    SineGen2.forward = forward_v2
    print("[hift] Patched SineGen.forward and SineGen2._f02sine/forward to be deterministic")


def patch_source_module_deterministic():
    """Monkey-patch SourceModuleHnNSF.forward to drop randn_like."""
    from cosyvoice.hifigan.generator import SourceModuleHnNSF

    def forward(self, x):
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = deterministic_randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv

    SourceModuleHnNSF.forward = forward
    print("[hift] Patched SourceModuleHnNSF.forward to be deterministic")


class ManualSTFT(nn.Module):
    """Strided-windowed DFT implemented as depthwise Conv1d.

    Equivalent to torch.stft(x, n_fft=N, hop_length=H, win_length=N,
                             window=hann, center=True, return_complex=True)
    followed by view_as_real.

    Input:  (B, T_signal)
    Output: real (B, N//2+1, T_frames), imag (B, N//2+1, T_frames)
    """

    def __init__(self, n_fft: int, hop_length: int, window: torch.Tensor):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = True  # torch.stft default
        self.pad_amount = n_fft // 2

        n = torch.arange(n_fft, dtype=torch.float32)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float32).view(-1, 1)
        cos_basis = torch.cos(2 * math.pi * k * n / n_fft)
        sin_basis = -torch.sin(2 * math.pi * k * n / n_fft)

        win = window.view(1, -1)
        real_w = (cos_basis * win).unsqueeze(1)  # (F, 1, N)
        imag_w = (sin_basis * win).unsqueeze(1)

        self.register_buffer("real_w", real_w)
        self.register_buffer("imag_w", imag_w)

    def forward(self, x: torch.Tensor):
        """STFT via Conv1d on pre-baked DFT bases (no windowing at trace time)."""
        if self.center:
            x = F.pad(x, (self.pad_amount, self.pad_amount), mode="reflect")
        x = x.unsqueeze(1)
        real = F.conv1d(x, self.real_w, stride=self.hop_length)
        imag = F.conv1d(x, self.imag_w, stride=self.hop_length)
        return real, imag


class ManualISTFT(nn.Module):
    """Inverse DFT with windowing + overlap-add (ConvTranspose1d).

    Equivalent to torch.istft(spec, n_fft=N, hop_length=H, win_length=N,
                              window=hann, center=True).

    Input: magnitude (B, N//2+1, T_frames), phase (B, N//2+1, T_frames)
    Output: (B, T_signal)
    """

    def __init__(self, n_fft: int, hop_length: int, window: torch.Tensor):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = True
        self.pad_amount = n_fft // 2

        # IDFT: x[n] = (1/N) sum_k X[k] exp(i 2π k n / N)
        # Real part: (1/N) sum_k (Re[X[k]] cos(2πkn/N) - Im[X[k]] sin(2πkn/N))
        # The negative sign is folded into imag_w so that
        #   y = real_conv(mag*cos) + imag_conv(mag*sin)
        # computes the correct IDFT real part.
        n = torch.arange(n_fft, dtype=torch.float32).view(-1, 1)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float32)
        cos_basis = torch.cos(2 * math.pi * k * n / n_fft)  # (N, F)
        sin_basis = -torch.sin(2 * math.pi * k * n / n_fft)  # negative sign folded in

        # Half the contribution of DC (k=0) and Nyquist (k=N/2): the full IDFT
        # counts them once (not twice like the inner bins). The 2/N scale below
        # compensates for the rest.
        half_mask = torch.ones(n_fft // 2 + 1, dtype=torch.float32)
        half_mask[0] = 0.5
        half_mask[-1] = 0.5
        cos_basis = cos_basis * half_mask
        sin_basis = sin_basis * half_mask

        win = window.view(-1, 1)  # (N, 1)
        real_w = (cos_basis * win).t().unsqueeze(1)  # (F, 1, N) for ConvTranspose1d
        imag_w = (sin_basis * win).t().unsqueeze(1)

        self.register_buffer("real_w", real_w)
        self.register_buffer("imag_w", imag_w)

        # COLA normalization: for hann N=16 H=4, each output sample sees 4
        # overlapping frames; we use the steady-state window^2 sum.
        denom = torch.zeros(n_fft, dtype=torch.float32)
        for offset in range(n_fft):
            s = 0.0
            for hop_idx in range(-10, 11):
                pos = offset - hop_idx * hop_length
                if 0 <= pos < n_fft:
                    s += float(window[pos]) ** 2
            denom[offset] = s
        steady_state = float(denom[hop_length:hop_length * 2].mean())
        self.register_buffer("inv_norm", torch.tensor(1.0 / max(steady_state, 1e-8),
                                                      dtype=torch.float32))

    def forward(self, magnitude: torch.Tensor, phase: torch.Tensor):
        """ISTFT via ConvTranspose1d on pre-baked DFT bases (no windowing at trace time)."""
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        real_time = F.conv_transpose1d(real, self.real_w, stride=self.hop_length)
        imag_time = F.conv_transpose1d(imag, self.imag_w, stride=self.hop_length)
        y = (real_time + imag_time) * (2.0 / self.n_fft) * self.inv_norm
        # Slice the (always-1) channel dim. y[:, 0] traces cleanly to a
        # Gather/Slice; squeeze(1) would emit an If node that Ascend converter
        # rejects on dynamic-shape models.
        y = y[:, 0]
        if self.center:
            y = y[..., self.pad_amount:-self.pad_amount]
        return y


def patch_conv_transpose1d_dynamic():
    """Monkey-patch torch.nn.functional.conv_transpose1d with an Ascend-safe impl.

    Ascend's `te_conv2dtranspose` kernel crashes ("MTE write address out of
    range") when called with a dynamic time axis. We replace it with an
    equivalent Conv1d-on-dilated-input formulation:

        ConvTranspose1d(input, w, stride=S, padding=P, bias=B) ≡
            Conv1d(pad(dilate(input, S), K-1), flip(w), stride=1)
                .narrow(2, P, (T-1)*S + K - 2P) + B

    where dilate inserts S-1 zeros after each input sample and flip reverses
    the kernel axis. Output length matches ConvTranspose1d exactly:
    (T-1)*S + K - 2*padding + output_padding.

    Supports the subset we use: padding >= 0, output_padding=0, dilation=1,
    groups=1, bias optional.
    """
    _orig = F.conv_transpose1d

    def _patched(x, weight, bias=None, stride=1, padding=0, output_padding=0,
                 groups=1, dilation=1):
        # Normalize tuple/int variants for stride, padding, output_padding
        S = stride[0] if isinstance(stride, (list, tuple)) else int(stride)
        P = padding[0] if isinstance(padding, (list, tuple)) else int(padding)
        OP = output_padding[0] if isinstance(output_padding, (list, tuple)) else int(output_padding)
        G = groups[0] if isinstance(groups, (list, tuple)) else int(groups)
        D = dilation[0] if isinstance(dilation, (list, tuple)) else int(dilation)
        if OP != 0 or G != 1 or D != 1 or x.dim() != 3 or weight.dim() != 3:
            return _orig(x, weight, bias=bias, stride=stride, padding=padding,
                         output_padding=output_padding, groups=groups, dilation=dilation)
        # x: (B, in_c, T), weight: (in_c, out_c, K)
        _, _, K = weight.shape

        # 1. Dilate: dilated[m] = x[m // S] if m % S == 0 else 0.
        #    Implementation: Tile x by S along last axis (giving each sample
        #    repeated S times: tiled[:, :, 0..S-1] = x[:, :, 0]; ...), then
        #    multiply by a periodic [1, 0, ..., 0] mask built via arange+mod.
        #    We avoid (B, in_c, T, 1) * (1, 1, 1, S) broadcast because Ascend's
        #    Mul infersshape rejects it on dynamic T after fusing the unsqueeze
        #    and reshape into one op. Tile + element-wise Mul on matching
        #    (B, in_c, T*S) shapes has no broadcast to reject.
        T_dyn = x.shape[2]
        tiled = x.repeat_interleave(S, dim=-1)  # (B, in_c, T*S) with repeats
        positions = torch.arange(T_dyn * S, device=x.device)  # (T*S,)
        mask = (positions % S == 0).to(x.dtype)  # 1 at multiples of S
        dilated = tiled * mask.view(1, 1, -1)  # (B, in_c, T*S), broadcast only on leading axes

        # 2. Asymmetric pad so Conv1d output length exactly matches
        #    ConvTranspose1d output length: pad_left = K-1-P (skips the P
        #    "left edge" samples that ConvTranspose1d would crop), pad_right =
        #    K-S-P (compensates for the stride trailing zeros). Result:
        #    output length = T*S + (2K-S-2P-1) - K + 1 = T*S + K - S - 2P
        #                  = (T-1)*S + K - 2P  ✓ matches ConvTranspose1d.
        pad_left = K - 1 - P
        pad_right = K - S - P
        dilated_padded = F.pad(dilated, (pad_left, pad_right))

        # 3. Transpose in/out channels (Conv1d weight is (out, in, K);
        #    ConvTranspose1d weight is (in, out, K)) and flip kernel axis.
        weight_flipped = weight.permute(1, 0, 2).flip(dims=[2]).contiguous()

        # 4. Conv1d, stride=1. Output length: (T-1)*S + K - 2P (matches
        #    ConvTranspose1d with padding P). No slicing needed.
        output = F.conv1d(dilated_padded, weight_flipped, stride=1)

        # 5. Add bias if present (broadcasts over time).
        if bias is not None:
            output = output + bias.view(1, -1, 1)
        return output

    F.conv_transpose1d = _patched
    print("[hift] Patched F.conv_transpose1d with Conv1d-on-dilated-input (Ascend dynamic-safe)")


def patch_hift_stft_istft(hift):
    """Replace _stft/_istft with ManualSTFT/ManualISTFT.

    Also patches f0_upsamp (nn.Upsample -> repeat_interleave) because nn.Upsample
    emits a Resize op with dynamic scales that Ascend's dynamic-shape compile
    can't handle.
    """
    n_fft = hift.istft_params["n_fft"]
    hop_len = hift.istft_params["hop_len"]
    window = hift.stft_window.float()

    stft_op = ManualSTFT(n_fft, hop_len, window).to(next(hift.parameters()).device)
    istft_op = ManualISTFT(n_fft, hop_len, window).to(next(hift.parameters()).device)

    hift._manual_stft = stft_op
    hift._manual_istft = istft_op

    def _stft(self, x):
        return self._manual_stft(x)

    def _istft(self, magnitude, phase):
        return self._manual_istft(magnitude, phase)

    hift._stft = types.MethodType(_stft, hift)
    hift._istft = types.MethodType(_istft, hift)

    upsample_factor = int(hift.f0_upsamp.scale_factor)

    def f0_upsamp_fwd(x, _factor=upsample_factor):
        return x.repeat_interleave(_factor, dim=-1)

    hift.f0_upsamp.forward = f0_upsamp_fwd
    print(f"[hift] Patched _stft/_istft (n_fft={n_fft}, hop={hop_len}) and "
          f"f0_upsamp (factor={upsample_factor}) with export-friendly ops")


class HiFTWrapper(nn.Module):
    """Thin wrapper around HiFTGenerator.inference for ONNX export.

    Input:  mel  (B, T_mel, 80)  -- mel features in (B, T, C) layout
    Output: wav  (B, 1, T_wav)   -- waveform
    """

    def __init__(self, hift: nn.Module):
        super().__init__()
        self.hift = hift

    def forward(self, mel):
        """Run HiFTGenerator.inference with channels-first mel and emit (B, 1, T) wav."""
        speech_feat = mel.transpose(1, 2).contiguous()
        speech, _ = self.hift.inference(speech_feat)
        return speech.unsqueeze(1) if speech.dim() == 2 else speech


# ---------------------------------------------------------------------------
# Helper: additive causal + padding mask (legacy non-fused path)
# ---------------------------------------------------------------------------
def _make_additive_causal_mask(attention_mask, q_len, k_len, past_len, dtype):
    mask_value = torch.finfo(dtype).min
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    causal = causal.to(dtype) * mask_value
    causal = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len)
    padding = (1.0 - attention_mask.to(dtype)) * mask_value
    padding = padding[:, None, None, :]
    return causal + padding


# ---------------------------------------------------------------------------
# Helper: boolean causal + padding mask for PFA (True = attend, False = masked)
# ---------------------------------------------------------------------------
def _make_bool_causal_mask(attention_mask, q_len, k_len, past_len):
    """Build a boolean [B, num_heads, q_len, k_len] mask (True=allowed).

    PFA's bool atten_mask convention: True means "attend to this position", False means
    "masked out". We broadcast across heads (num_heads dim is implicit, PFA reuses the
    same mask for every head).
    """
    batch = attention_mask.shape[0]
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = ar_k[None, :] <= (past_len + ar_q[:, None])  # [q_len, k_len]
    causal = causal[None, None, :, :].expand(batch, 1, q_len, k_len)
    padding = attention_mask.to(torch.bool)[:, None, None, :]  # [B, 1, 1, k_len]
    return (causal & padding).to(torch.bool)


def _sanitize_onnx_zero_dims(onnx_path: Path) -> None:
    """Sanitize ONNX model to replace dim_value=0 with dim_param (dynamic dimension).

    MSLite Ascend runtime cannot handle tensors with fixed zero dimensions.
    This function converts zero-dimensional fixed shapes to dynamic parameters.
    """
    model = onnx.load(str(onnx_path))

    def sanitize_value_info(vi):
        if not vi.type.HasField("tensor_type"):
            return
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            return
        for i, d in enumerate(tt.shape.dim):
            if d.HasField("dim_value") and int(d.dim_value) == 0 and not d.HasField("dim_param"):
                d.dim_param = f"{vi.name}_dim{i}"
                d.ClearField("dim_value")

    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        sanitize_value_info(vi)

    onnx.save(model, str(onnx_path))


# ---------------------------------------------------------------------------
# Helper: Qwen2 attention forward (GQA + KV-cache)
# ---------------------------------------------------------------------------
def _text_attn_forward(attn_mod, hidden_states, position_embeddings, attention_mask,
                       past_key, past_value, bool_mask=None):
    """Qwen2 attention forward with GQA + KV-cache (export-friendly).

    When the module-level `_ENABLE_PFA` flag is on AND a boolean mask is supplied,
    the softmax+matmul is routed through CANN Custom(PromptFlashAttention). The q/k/v
    are cast to FP16 around PFA (PFA only supports fp16/bf16/int8), then back to FP32
    so the rest of the layer (o_proj, residual, MLP) stays in FP32 for numerical
    stability (matches the config.ini force_fp32 behavior).
    """
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = attn_mod.q_proj(hidden_states).view(hidden_shape)
    key_states = attn_mod.k_proj(hidden_states).view(hidden_shape)
    value_states = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    cos, sin = position_embeddings
    if apply_rotary_pos_emb is not None:
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key is not None:
        key_states = torch.cat([past_key, key_states], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)

    scaling = getattr(attn_mod, "scaling", 1.0 / (head_dim ** 0.5))

    if _ENABLE_PFA and bool_mask is not None:
        # Fused path: cast q/k/v to fp16, let PFA handle GQA + softmax internally.
        q16 = query_states.to(torch.float16)
        k16 = key_states.to(torch.float16)
        v16 = value_states.to(torch.float16)
        attn_output = _cann_pfa(
            q16, k16, v16, bool_mask,
            num_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            scale_value=scaling,
        ).to(hidden_states.dtype)
        # PFA emits BNSD output; collapse heads back into hidden dim.
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
        attn_output = attn_mod.o_proj(attn_output)
        return attn_output, key_states, value_states

    # Legacy path: manual matmul+softmax+matmul, expects additive FP32 mask.
    key_states_for_attn = key_states
    value_states_for_attn = value_states
    if num_kv_heads < num_heads:
        key_states_for_attn = key_states.repeat_interleave(num_heads // num_kv_heads, dim=1)
        value_states_for_attn = value_states.repeat_interleave(num_heads // num_kv_heads, dim=1)

    attn_weights = torch.matmul(query_states, key_states_for_attn.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states_for_attn)
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
    attn_output = attn_mod.o_proj(attn_output)
    return attn_output, key_states, value_states


# ===========================================================================
# LLM Prefill wrapper
# ===========================================================================
class CosyVoice2LlmPrefill(nn.Module):
    """
    Prefill stage for CosyVoice2 LLM.

    Inputs:
        text_ids       [1, text_len]   int64   Qwen2 text token ids (prompt+target)
        speech_ids     [1, speech_len] int64   prompt speech token ids
        attention_mask [1, total_len]  int64   all-ones (total = 2+text_len+speech_len)
        position_ids   [1, total_len]  int64   0 … total_len-1

    Outputs:
        logits         [1, total_len, 6564]  float32
        present_kv     [2*L, 1, 2, total_len, 64]  float32
    """

    def __init__(self, embed_tokens, llm_embedding, speech_embedding, qwen2_model, llm_decoder):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.llm_embedding = llm_embedding
        self.speech_embedding = speech_embedding
        self.qwen2_model = qwen2_model
        self.llm_decoder = llm_decoder

    def forward(self, text_ids, speech_ids, attention_mask, position_ids):
        """Run LLM prefill forward, returning logits and present KV cache."""
        text_emb = self.embed_tokens(text_ids)
        sos_emb = self.llm_embedding.weight[0].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[1].reshape(1, 1, -1)
        speech_emb = self.speech_embedding(speech_ids)
        inputs_embeds = torch.concat([sos_emb, text_emb, task_id_emb, speech_emb], dim=1)

        position_embeddings = self.qwen2_model.rotary_emb(inputs_embeds, position_ids)
        q_len = inputs_embeds.shape[1]
        attn_mask = _make_additive_causal_mask(attention_mask, q_len, q_len, 0, inputs_embeds.dtype)
        bool_mask = _make_bool_causal_mask(attention_mask, q_len, q_len, 0) if _ENABLE_PFA else None

        hidden_states = inputs_embeds
        present = []
        for layer in self.qwen2_model.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn, hidden_states, position_embeddings, attn_mask, None, None,
                bool_mask=bool_mask)
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)
            present.append(pk)
            present.append(pv)

        hidden_states = self.qwen2_model.norm(hidden_states)
        logits = self.llm_decoder(hidden_states)
        present_kv = torch.stack(present, dim=0)
        return logits, present_kv


# ===========================================================================
# LLM Decode wrapper
# ===========================================================================
class CosyVoice2LlmDecode(nn.Module):
    """
    Decode stage for CosyVoice2 LLM (single-token step).

    Inputs:
        speech_id      [1, 1]          int64   generated speech token id
        attention_mask [1, total_len]  int64   all-ones
        position_ids   [1, 1]          int64   current position
        past_key_values [2*L, 1, 2, past_len, 64]  float32

    Outputs:
        logits         [1, 1, 6564]    float32
        present_kv     [2*L, 1, 2, total_len, 64]  float32
    """

    def __init__(self, speech_embedding, qwen2_model, llm_decoder):
        super().__init__()
        self.speech_embedding = speech_embedding
        self.qwen2_model = qwen2_model
        self.llm_decoder = llm_decoder

    def forward(self, speech_id, attention_mask, position_ids, past_key_values):
        """Run one autoregressive decode step, returning logits and updated KV cache."""
        inputs_embeds = self.speech_embedding(speech_id)

        position_embeddings = self.qwen2_model.rotary_emb(inputs_embeds, position_ids)
        past_len = past_key_values.shape[3]
        q_len = 1
        k_len = past_len + q_len
        attn_mask = _make_additive_causal_mask(
            attention_mask, q_len, k_len, past_len, inputs_embeds.dtype)
        bool_mask = _make_bool_causal_mask(
            attention_mask, q_len, k_len, past_len) if _ENABLE_PFA else None

        hidden_states = inputs_embeds
        present = []
        for i, layer in enumerate(self.qwen2_model.layers):
            pk_in = past_key_values[2 * i]
            pv_in = past_key_values[2 * i + 1]
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn, hidden_states, position_embeddings, attn_mask, pk_in, pv_in,
                bool_mask=bool_mask)
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)
            present.append(pk)
            present.append(pv)

        hidden_states = self.qwen2_model.norm(hidden_states)
        logits = self.llm_decoder(hidden_states)
        present_kv = torch.stack(present, dim=0)
        return logits, present_kv


# ===========================================================================
# Flow Encoder wrapper
# ===========================================================================
class CosyVoice2FlowEncoder(nn.Module):
    """
    Flow encoder: speech tokens → mu / spks / cond / mask

    Inputs:
        token       [1, token_len]  int64   combined prompt+target speech tokens
        token_len   [1]             int64   actual token length
        embedding   [1, 192]        float32 speaker embedding
        prompt_feat [1, feat_len, 80] float32 prompt mel features

    Outputs:
        mu    [1, 80, mel_len]   float32  encoder output
        spks  [1, 80]            float32  projected speaker embedding
        cond  [1, 80, mel_len]   float32  conditions (prompt feat + zeros)
        mask  [1, 1, mel_len]    float32  attention mask
    """

    def __init__(self, flow_module):
        super().__init__()
        self.input_embedding = flow_module.input_embedding
        self.spk_embed_affine_layer = flow_module.spk_embed_affine_layer
        self.encoder = flow_module.encoder
        self.encoder_proj = flow_module.encoder_proj
        self.output_size = flow_module.output_size

    def forward(self, token, token_len, embedding, prompt_feat):
        """Run Flow Encoder forward, returning mu/spks/cond/mask for Flow Estimator."""
        embedding = F.normalize(embedding, dim=1)
        spks = self.spk_embed_affine_layer(embedding)

        mask = self._make_non_pad_mask(token_len, token.shape[1]).unsqueeze(-1).to(spks.dtype)
        token_emb = self.input_embedding(torch.clamp(token, min=0)) * mask

        h, _ = self.encoder(token_emb, token_len, streaming=False)
        h = self.encoder_proj(h)

        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - mel_len1
        mel_len = mel_len1 + mel_len2

        mu = h.transpose(1, 2).contiguous()

        cond = torch.zeros([1, mel_len, self.output_size], device=token.device, dtype=h.dtype)
        cond[:, :mel_len1] = prompt_feat
        cond = cond.transpose(1, 2).contiguous()

        # In inference all frames in the single generated sequence are valid.
        # Avoid torch.tensor([mel_len]) here: legacy ONNX export may trace it as
        # the dummy export length and keep the tail masked for longer inputs.
        attn_mask = h.new_ones((h.shape[0], 1, mel_len))

        return mu, spks, cond, attn_mask

    @staticmethod
    def _make_non_pad_mask(lengths, max_len):
        seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
        seq_range_expand = seq_range.unsqueeze(0).expand(lengths.shape[0], max_len)
        seq_length_expand = lengths.unsqueeze(-1)
        return seq_range_expand < seq_length_expand


# ===========================================================================
# Model loading  (direct instantiation, avoids hyperpyyaml dependency chain)
# ===========================================================================
def _build_llm(model_dir: str):
    """Build CosyVoice2 Qwen2LM module (without weights)."""
    from functools import partial
    from cosyvoice.llm.llm import Qwen2LM, Qwen2Encoder
    from cosyvoice.utils.common import ras_sampling

    qwen_path = str(Path(model_dir) / "CosyVoice-BlankEN")
    qwen2_encoder = Qwen2Encoder(pretrain_path=qwen_path)
    sampling_fn = partial(ras_sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1)
    return Qwen2LM(
        llm_input_size=896,
        llm_output_size=896,
        speech_token_size=6561,
        llm=qwen2_encoder,
        sampling=sampling_fn,
        length_normalized_loss=True,
        lsm_weight=0,
        mix_ratio=[5, 15],
    )


def _patch_upsample1d_for_ge_onnx_export():
    """
    CosyVoice Upsample1D uses F.interpolate on (B, C, T), which ONNX exports as
    3D Resize. Ascend GE ResizeNearestNeighborV2 requires 4D input.

    Use (B, C, T, 1) + scale_factor=(stride, 1) + squeeze; numerically equivalent
    to upsampling the time axis only, but traces as 4D Resize for converter_lite.
    """
    from cosyvoice.transformer import upsample_encoder as upsample_enc_mod

    def forward_patched(self, inputs, input_lengths):
        x4 = inputs.unsqueeze(-1)
        y4 = F.interpolate(
            x4,
            scale_factor=(float(self.stride), 1.0),
            mode="nearest",
        )
        outputs = y4.squeeze(-1)
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        outputs = self.conv(outputs)
        return outputs, input_lengths * self.stride

    upsample_enc_mod.Upsample1D.forward = forward_patched


def _build_flow():
    """Build CosyVoice2 CausalMaskedDiffWithXvec module (without weights)."""
    from cosyvoice.flow.flow import CausalMaskedDiffWithXvec
    from cosyvoice.flow.flow_matching import CausalConditionalCFM
    from cosyvoice.flow.decoder import CausalConditionalDecoder
    from omegaconf import DictConfig

    _patch_upsample1d_for_ge_onnx_export()
    from cosyvoice.transformer.upsample_encoder import UpsampleConformerEncoder

    encoder = UpsampleConformerEncoder(
        output_size=512,
        attention_heads=8,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        normalize_before=True,
        input_layer="linear",
        pos_enc_layer_type="rel_pos_espnet",
        selfattention_layer_type="rel_selfattn",
        input_size=512,
        use_cnn_module=False,
        macaron_style=False,
        static_chunk_size=25,
    )
    estimator = CausalConditionalDecoder(
        in_channels=320,
        out_channels=80,
        channels=[256],
        dropout=0.0,
        attention_head_dim=64,
        n_blocks=4,
        num_mid_blocks=12,
        num_heads=8,
        act_fn="gelu",
        static_chunk_size=50,
        num_decoding_left_chunks=-1,
    )
    cfm = CausalConditionalCFM(
        in_channels=240,
        n_spks=1,
        spk_emb_dim=80,
        cfm_params=DictConfig({
            "sigma_min": 1e-06,
            "solver": "euler",
            "t_scheduler": "cosine",
            "training_cfg_rate": 0.2,
            "inference_cfg_rate": 0.7,
            "reg_loss_type": "l1",
        }),
        estimator=estimator,
    )
    return CausalMaskedDiffWithXvec(
        input_size=512,
        output_size=80,
        spk_embed_dim=192,
        output_type="mel",
        vocab_size=6561,
        input_frame_rate=25,
        only_mask_loss=True,
        token_mel_ratio=2,
        pre_lookahead_len=3,
        encoder=encoder,
        decoder=cfm,
    )


def _build_hift():
    """Build HiFT vocoder module (without weights)."""
    from cosyvoice.hifigan.generator import HiFTGenerator
    from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor

    f0_predictor = ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
    return HiFTGenerator(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=24000,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        f0_predictor=f0_predictor,
    )


def _load_weights(llm, flow, hift, model_dir: str, device: str):
    """Load llm/flow/HiFT weights from model_dir."""
    llm_path = str(Path(model_dir) / "llm.pt")
    flow_path = str(Path(model_dir) / "flow.pt")
    hift_path = str(Path(model_dir) / "hift.pt")

    llm.load_state_dict(torch.load(llm_path, map_location=device, weights_only=True), strict=True)
    flow.load_state_dict(torch.load(flow_path, map_location=device, weights_only=True), strict=True)
    hift_state_dict = {
        k.replace("generator.", ""): v
        for k, v in torch.load(hift_path, map_location=device, weights_only=True).items()
    }
    hift.load_state_dict(hift_state_dict, strict=True)


def _load_model(model_dir, model_code_dir, device):
    """Load CosyVoice2-0.5B model components by direct instantiation.

    HiFT deterministic patches must be applied BEFORE _build_hift() so they
    affect any random ops invoked during weight init.
    """
    sys.path.insert(0, str(model_code_dir))
    sys.path.insert(0, str(model_code_dir / "third_party" / "Matcha-TTS"))
    patch_sinegen_deterministic()
    patch_source_module_deterministic()
    patch_conv_transpose1d_dynamic()
    llm = _build_llm(model_dir)
    flow = _build_flow()
    hift = _build_hift()
    _load_weights(llm, flow, hift, model_dir, device)
    patch_hift_stft_istft(hift)

    llm.to(device).float().eval()
    flow.to(device).float().eval()
    hift.to(device).float().eval()

    return llm, flow, hift


# ===========================================================================
# Export functions
# ===========================================================================
def _export_prefill(prefill, dummy_inputs, output_path):
    """Export LLM Prefill to ONNX."""
    print(f"  Exporting prefill → {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            prefill,
            dummy_inputs,
            str(output_path),
            input_names=["text_ids", "speech_ids", "attention_mask", "position_ids"],
            output_names=["logits", "present_key_values"],
            opset_version=17,
            dynamic_axes={
                "text_ids": {0: "batch", 1: "text_len"},
                "speech_ids": {0: "batch", 1: "speech_len"},
                "attention_mask": {0: "batch", 1: "total_len"},
                "position_ids": {0: "batch", 1: "total_len"},
                "logits": {0: "batch", 1: "total_len"},
                "present_key_values": {1: "batch", 3: "total_len"},
            },
            keep_initializers_as_inputs=True,
            do_constant_folding=False,
            dynamo=False,
        )
    _sanitize_onnx_zero_dims(output_path)
    print("  Prefill exported ✓")


def _export_decode(decode, dummy_inputs, output_path):
    """Export LLM Decode to ONNX."""
    print(f"  Exporting decode  → {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            decode,
            dummy_inputs,
            str(output_path),
            input_names=["speech_id", "attention_mask", "position_ids", "past_key_values"],
            output_names=["logits", "present_key_values"],
            opset_version=17,
            dynamic_axes={
                "speech_id": {0: "batch"},
                "attention_mask": {0: "batch", 1: "total_seq_len"},
                "position_ids": {0: "batch"},
                "past_key_values": {1: "batch", 3: "past_seq_len"},
                "logits": {0: "batch"},
                "present_key_values": {1: "batch", 3: "total_seq_len"},
            },
            keep_initializers_as_inputs=True,
            do_constant_folding=False,
            dynamo=False,
        )
    _sanitize_onnx_zero_dims(output_path)
    print("  Decode exported ✓")


def _export_llm(llm, output_dir, device):
    """Export LLM Prefill and Decode to ONNX."""
    print("\n" + "=" * 60)
    print("Exporting LLM (Prefill + Decode)")
    print("=" * 60)

    embed_tokens = llm.llm.model.model.embed_tokens
    llm_embedding = llm.llm_embedding
    speech_embedding = llm.speech_embedding
    qwen2_model = llm.llm.model.model
    llm_decoder = llm.llm_decoder

    num_layers = llm.llm.model.config.num_hidden_layers
    num_kv_heads = llm.llm.model.config.num_key_value_heads
    head_dim = getattr(llm.llm.model.config, "head_dim",
                       llm.llm.model.config.hidden_size // llm.llm.model.config.num_attention_heads)

    # --- Prefill ---
    # Use speech_len=1 with token 0 for export to match MSLite inference behavior.
    # MSLite Ascend runtime cannot handle zero-sized tensors (size=0), so we pad
    # empty speech_ids with a dummy token (0) during inference. To match this, we
    # export with speech_len=1 (token 0) instead of speech_len=0.
    prefill = CosyVoice2LlmPrefill(
        embed_tokens, llm_embedding, speech_embedding, qwen2_model, llm_decoder
    ).to(device).eval()

    text_len = 8
    speech_len = 1  # Use 1 dummy token (0) instead of 0 to avoid MSLite empty tensor issue
    total_len = 2 + text_len + speech_len

    dummy_text_ids = torch.randint(0, 1000, (1, text_len), dtype=torch.int64, device=device)
    dummy_speech_ids = torch.zeros(1, speech_len, dtype=torch.int64, device=device)
    dummy_attn_mask = torch.ones(1, total_len, dtype=torch.int64, device=device)
    dummy_pos_ids = torch.arange(total_len, device=device, dtype=torch.int64).view(1, -1)

    prefill_path = Path(output_dir) / "cosyvoice2_llm_prefill.onnx"
    _export_prefill(prefill, (dummy_text_ids, dummy_speech_ids, dummy_attn_mask, dummy_pos_ids),
                    prefill_path)

    # --- Decode ---
    decode = CosyVoice2LlmDecode(
        speech_embedding, qwen2_model, llm_decoder
    ).to(device).eval()

    dummy_past_len = total_len
    dummy_speech_id = torch.tensor([[0]], dtype=torch.int64, device=device)
    dummy_decode_attn_mask = torch.ones(1, dummy_past_len + 1, dtype=torch.int64, device=device)
    dummy_decode_pos_ids = torch.tensor([[dummy_past_len]], dtype=torch.int64, device=device)
    dummy_past_kv = torch.zeros(
        2 * num_layers, 1, num_kv_heads, dummy_past_len, head_dim,
        dtype=torch.float32, device=device,
    )

    decode_path = Path(output_dir) / "cosyvoice2_llm_decode.onnx"
    _export_decode(decode, (dummy_speech_id, dummy_decode_attn_mask, dummy_decode_pos_ids, dummy_past_kv),
                   decode_path)

    del prefill, decode
    gc.collect()


def _export_flow_encoder(flow, output_dir, device):
    """Export Flow Encoder to ONNX."""
    print("\n" + "=" * 60)
    print("Exporting Flow Encoder")
    print("=" * 60)

    encoder = CosyVoice2FlowEncoder(flow).to(device).eval()

    token_len = 20
    # Use feat_len=1 to match MSLite inference behavior (pad empty prompt with 1 frame).
    # MSLite Ascend runtime cannot handle feat_len=0 (empty prompt_feat).
    feat_len = 1

    dummy_token = torch.randint(0, 6561, (1, token_len), dtype=torch.int64, device=device)
    dummy_token_len = torch.tensor([token_len], dtype=torch.int64, device=device)
    dummy_embedding = torch.randn(1, 192, device=device)
    dummy_prompt_feat = torch.randn(1, feat_len, 80, device=device)

    enc_path = Path(output_dir) / "cosyvoice2_flow_encoder.onnx"
    print(f"  Exporting flow encoder → {enc_path}")
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (dummy_token, dummy_token_len, dummy_embedding, dummy_prompt_feat),
            str(enc_path),
            input_names=["token", "token_len", "embedding", "prompt_feat"],
            output_names=["mu", "spks", "cond", "mask"],
            opset_version=17,
            dynamic_axes={
                "token":       {0: "batch", 1: "token_len"},
                "token_len":   {0: "batch"},
                "embedding":   {0: "batch"},
                "prompt_feat": {0: "batch", 1: "feat_len"},
                "mu":          {0: "batch", 2: "mel_len"},
                "cond":        {0: "batch", 2: "mel_len"},
                "mask":        {0: "batch", 2: "mel_len"},
            },
            keep_initializers_as_inputs=True,
            do_constant_folding=False,
            dynamo=False,
        )
    _sanitize_onnx_zero_dims(enc_path)
    print("  Flow Encoder exported ✓")

    del encoder
    gc.collect()


def _export_flow_estimator(flow, output_dir, device):
    """Export Flow Estimator (CausalConditionalDecoder) to ONNX."""
    print("\n" + "=" * 60)
    print("Exporting Flow Estimator")
    print("=" * 60)

    if _ENABLE_PFA_EST:
        _patch_diffusers_attention_with_pfa()

    estimator = flow.decoder.estimator.to(device).eval()

    seq_len = 20
    batch = 1

    dummy_x = torch.randn(batch, 80, seq_len, device=device)
    dummy_mask = torch.ones(batch, 1, seq_len, device=device)
    dummy_mu = torch.randn(batch, 80, seq_len, device=device)
    dummy_t = torch.rand(batch, device=device)
    dummy_spks = torch.randn(batch, 80, device=device)
    dummy_cond = torch.randn(batch, 80, seq_len, device=device)

    est_path = Path(output_dir) / "cosyvoice2_flow_estimator.onnx"
    print(f"  Exporting flow estimator → {est_path}")
    with torch.no_grad():
        torch.onnx.export(
            estimator,
            (dummy_x, dummy_mask, dummy_mu, dummy_t, dummy_spks, dummy_cond),
            str(est_path),
            input_names=["x", "mask", "mu", "t", "spks", "cond"],
            output_names=["estimator_out"],
            opset_version=17,
            dynamic_axes={
                "x":             {0: "batch", 2: "seq_len"},
                "mask":          {0: "batch", 2: "seq_len"},
                "mu":            {0: "batch", 2: "seq_len"},
                "t":             {0: "batch"},
                "spks":          {0: "batch"},
                "cond":          {0: "batch", 2: "seq_len"},
                "estimator_out": {0: "batch", 2: "seq_len"},
            },
            keep_initializers_as_inputs=True,
            do_constant_folding=False,
            dynamo=False,
        )
    _sanitize_onnx_zero_dims(est_path)
    print("  Flow Estimator exported ✓")

    del estimator
    gc.collect()


def _export_hift(hift, output_dir, opset: int = 17):
    """Export HiFT vocoder as a single dynamic-shape ONNX.

    Output: cosyvoice2_hift.onnx with mel's T_mel axis marked dynamic.
    The corresponding MindIR uses 纯动态（pure dynamic）— no ge.dynamicDims —
    so the infer side calls model.resize() with the actual T_mel each run.
    """
    print("\n" + "=" * 60)
    print("Exporting HiFT vocoder (pure dynamic shape)")
    print("=" * 60)

    wrapper = HiFTWrapper(hift).eval()
    t_mel = 200  # dummy export length; real shape comes from dynamic_axes
    dummy_mel = torch.randn(1, t_mel, 80, dtype=torch.float32)

    with torch.no_grad():
        y = wrapper(dummy_mel)
    print(f"  Forward sanity: mel {tuple(dummy_mel.shape)} -> wav {tuple(y.shape)}")

    out_path = Path(output_dir) / "cosyvoice2_hift.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_mel,),
            str(out_path),
            input_names=["mel"],
            output_names=["wav"],
            dynamic_axes={
                "mel": {0: "batch", 1: "t_mel"},
                "wav": {0: "batch", 2: "t_wav"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Exported HiFT -> {out_path} ({size_mb:.2f} MB, dynamic T_mel)")


# ===========================================================================
# Main
# ===========================================================================
def main():
    """Parse args, load model, export the 5 ONNX sub-models to --output-dir."""
    parser = argparse.ArgumentParser(description="Export CosyVoice2-0.5B to ONNX")
    parser.add_argument(
        "--model-dir", type=str,
        default="/data/llj/models/model_weight/CosyVoice2-0.5B",
        help="Path to CosyVoice2-0.5B weights directory",
    )
    parser.add_argument(
        "--model-code-dir", type=str,
        default="/data/llj/models/model_code/CosyVoice",
        help="Path to CosyVoice source code directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./pfa_fused/cosyvoice2_onnx",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for export (cpu or cuda)",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM export",
    )
    parser.add_argument(
        "--skip-flow", action="store_true",
        help="Skip Flow export",
    )
    parser.add_argument(
        "--skip-hift", action="store_true",
        help="Skip HiFT vocoder export",
    )
    parser.add_argument(
        "--disable-fusion", action="store_true",
        help="Disable CANN Custom(PromptFlashAttention) fusion for LLM attention.",
    )
    parser.add_argument(
        "--disable-fusion-estimator", action="store_true",
        help="Disable patching Flow Estimator's diffusers attention with PFA Custom.",
    )

    args = parser.parse_args()
    global _ENABLE_PFA, _ENABLE_PFA_EST
    _ENABLE_PFA = not bool(args.disable_fusion)
    _ENABLE_PFA_EST = not bool(args.disable_fusion_estimator)
    if _ENABLE_PFA:
        print("[fusion] PromptFlashAttention Custom op enabled for LLM attention.")
    if _ENABLE_PFA_EST:
        print("[fusion] PromptFlashAttention Custom op enabled for Flow Estimator attention.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model dir:    {args.model_dir}")
    print(f"Model code:   {args.model_code_dir}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Device:       {args.device}")

    llm, flow, hift = _load_model(args.model_dir, Path(args.model_code_dir), args.device)

    if not args.skip_llm:
        _export_llm(llm, output_dir, args.device)

    if not args.skip_flow:
        _export_flow_encoder(flow, output_dir, args.device)
        _export_flow_estimator(flow, output_dir, args.device)

    if not args.skip_hift:
        _export_hift(hift, output_dir)

    print("\n" + "=" * 60)
    print(f"Export finished. Files saved in {args.output_dir}")
    print("=" * 60)

    del llm, flow, hift
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
