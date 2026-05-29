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
# WITHOUT WARRANTIES OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Export FireRedASR-AED-L model to ONNX format.

Split into two ONNX files:
  - Encoder: Conformer encoder + cross-attention K/V projection
  - DecoderStep: Single-step Transformer decoder with self-attention KV cache
"""

import argparse
import gc
import os
import sys
from math import sqrt
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn.functional as F


def _add_repo_to_sys_path(repo_dir: str) -> None:
    if not repo_dir:
        return
    repo_dir = os.path.abspath(os.path.expanduser(repo_dir))
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


def _load_model(model_path: str) -> Tuple[torch.nn.Module, Any]:
    package = torch.load(model_path, map_location="cpu", weights_only=False)
    args = package["args"]
    from fireredasr.models.fireredasr_aed import FireRedAsrAed

    model = FireRedAsrAed.from_args(args)
    model.load_state_dict(package["model_state_dict"], strict=True)
    return model, args


class PromptFlashAttentionFunction(torch.autograd.Function):
    """Torch op wrapper to export PromptFlashAttention into ONNX."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, scale_value):
        """Fallback PyTorch implementation used for tracing/export correctness."""

        del ctx, num_heads
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * float(scale_value)
        if atten_mask is not None:
            if atten_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(atten_mask, torch.finfo(attn_weights.dtype).min)
            else:
                attn_weights = attn_weights + atten_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(attn_weights, value)

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, scale_value):
        """ONNX symbolic that emits a PromptFlashAttention node."""

        if atten_mask is not None:
            return g.op(
                "PromptFlashAttention",
                query,
                key,
                value,
                atten_mask,
                num_heads_i=int(num_heads),
                num_key_value_heads_i=int(num_heads),
                scale_value_f=float(scale_value),
                input_layout_s="BNSD",
                next_tokens_i=65536,
                inner_precise_i=1,
            )
        return g.op(
            "PromptFlashAttention",
            query,
            key,
            value,
            num_heads_i=int(num_heads),
            num_key_value_heads_i=int(num_heads),
            scale_value_f=float(scale_value),
            input_layout_s="BNSD",
            next_tokens_i=65536,
            inner_precise_i=1,
        )


class FireRedAsrEncoder(torch.nn.Module):
    """Encoder wrapper that also pre-computes cross-attention K/V for decoder layers."""

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.cross_attn_layers = torch.nn.ModuleList()
        for dec_layer in decoder.layer_stack:
            self.cross_attn_layers.append(dec_layer.cross_attn)

    def forward(self, padded_input: torch.Tensor, input_lengths: torch.Tensor):
        """Runs encoder and returns (enc_out, enc_mask, cross_k, cross_v)."""

        enc_output, _, enc_mask = self.encoder(padded_input, input_lengths, pad=True)
        cross_k_list = []
        cross_v_list = []
        for cross_attn in self.cross_attn_layers:
            bs = enc_output.size(0)
            k = cross_attn.w_ks(enc_output).view(bs, -1, cross_attn.n_head, cross_attn.d_k).transpose(1, 2)
            v = cross_attn.w_vs(enc_output).view(bs, -1, cross_attn.n_head, cross_attn.d_k).transpose(1, 2)
            cross_k_list.append(k.unsqueeze(1))
            cross_v_list.append(v.unsqueeze(1))
        cross_k = torch.cat(cross_k_list, dim=1)
        cross_v = torch.cat(cross_v_list, dim=1)
        return enc_output, enc_mask, cross_k, cross_v


class FireRedAsrDecoderStep(torch.nn.Module):
    """Single-step decoder with optional self-attention PromptFlashAttention fusion."""

    def __init__(self, decoder: torch.nn.Module, enable_attn_fusion: bool):
        super().__init__()
        self.enable_attn_fusion = bool(enable_attn_fusion)
        self.sos_id = decoder.sos_id
        self.eos_id = decoder.eos_id
        self.pad_id = decoder.pad_id
        self.tgt_word_emb = decoder.tgt_word_emb
        self.positional_encoding = decoder.positional_encoding
        self.dropout = decoder.dropout
        self.layer_stack = decoder.layer_stack
        self.tgt_word_prj = decoder.tgt_word_prj
        self.layer_norm_out = decoder.layer_norm_out
        self.scale = decoder.scale

    def forward(
        self,
        ys: torch.Tensor,
        src_mask: torch.Tensor,
        cache_k_self: torch.Tensor,
        cache_v_self: torch.Tensor,
        cross_k: torch.Tensor,
        cross_v: torch.Tensor,
    ):
        """Runs one decoder step and returns (log_probs, new_cache_k_self, new_cache_v_self)."""

        tgt_mask = self._make_tgt_mask(ys)

        dec_input = self.dropout(self.tgt_word_emb(ys) * self.scale + self.positional_encoding(ys))

        new_cache_k_self_list = []
        new_cache_v_self_list = []

        for i, dec_layer in enumerate(self.layer_stack):
            x = dec_input
            residual = x
            x = dec_layer.self_attn_norm(x)

            xq = x[:, -1:, :]
            residual_q = residual[:, -1:, :]
            tgt_mask_step = tgt_mask[:, -1:, :]

            k_new = (
                dec_layer.self_attn.w_ks(x)
                .view(x.size(0), -1, dec_layer.self_attn.n_head, dec_layer.self_attn.d_k)
                .transpose(1, 2)
            )
            v_new = (
                dec_layer.self_attn.w_vs(x)
                .view(x.size(0), -1, dec_layer.self_attn.n_head, dec_layer.self_attn.d_k)
                .transpose(1, 2)
            )

            k_full = torch.cat([cache_k_self[:, i], k_new], dim=2)
            v_full = torch.cat([cache_v_self[:, i], v_new], dim=2)

            q_proj = dec_layer.self_attn.w_qs(xq).view(
                xq.size(0), -1, dec_layer.self_attn.n_head, dec_layer.self_attn.d_k
            ).transpose(1, 2)

            if self.enable_attn_fusion:
                num_heads = int(dec_layer.self_attn.n_head)
                scale_value = 1.0 / sqrt(float(dec_layer.self_attn.d_k))
                q_fp16 = q_proj.to(torch.float16)
                k_fp16 = k_full.to(torch.float16)
                v_fp16 = v_full.to(torch.float16)
                attn_out = PromptFlashAttentionFunction.apply(
                    q_fp16, k_fp16, v_fp16, None, num_heads, scale_value
                )
                attn_out = attn_out.to(q_proj.dtype)
            else:
                mask_expanded = tgt_mask_step.unsqueeze(1) if tgt_mask_step is not None else None
                attn_out = dec_layer.self_attn.attention(q_proj, k_full, v_full, mask=mask_expanded)
            attn_out = attn_out.transpose(1, 2).contiguous().view(xq.size(0), -1, dec_layer.self_attn.d_model)
            attn_out = dec_layer.self_attn.fc(attn_out)
            attn_out = dec_layer.self_attn.dropout(attn_out)
            x = residual_q + attn_out

            new_cache_k_self_list.append(k_full.unsqueeze(1))
            new_cache_v_self_list.append(v_full.unsqueeze(1))

            residual = x
            x = dec_layer.cross_attn_norm(x)
            q_cross = dec_layer.cross_attn.w_qs(x).view(
                x.size(0), -1, dec_layer.cross_attn.n_head, dec_layer.cross_attn.d_k
            ).transpose(1, 2)
            src_mask_expanded = src_mask.unsqueeze(1) if src_mask is not None else None
            cross_attn_out = dec_layer.cross_attn.attention(
                q_cross, cross_k[:, i], cross_v[:, i], mask=src_mask_expanded
            )
            cross_attn_out = cross_attn_out.transpose(1, 2).contiguous().view(
                x.size(0), -1, dec_layer.cross_attn.d_model
            )
            cross_attn_out = dec_layer.cross_attn.fc(cross_attn_out)
            cross_attn_out = dec_layer.cross_attn.dropout(cross_attn_out)
            x = residual + cross_attn_out

            residual = x
            x = dec_layer.mlp_norm(x)
            x = residual + dec_layer.mlp(x)

            dec_input = x

        dec_output = self.layer_norm_out(dec_input)
        logit = self.tgt_word_prj(dec_output[:, -1:])
        log_probs = F.log_softmax(logit, dim=-1)

        new_cache_k_self = torch.cat(new_cache_k_self_list, dim=1)
        new_cache_v_self = torch.cat(new_cache_v_self_list, dim=1)

        return log_probs, new_cache_k_self, new_cache_v_self

    def _make_tgt_mask(self, ys):
        mask = torch.ne(ys, self.pad_id).unsqueeze(1).bool()
        t = ys.size(-1)
        upper = torch.tril(torch.ones(t, t, device=ys.device, dtype=torch.bool))
        return mask & upper


def export_encoder(model: torch.nn.Module, output_path: Path, device: str) -> None:
    dummy_feat = torch.randn(1, 100, 80, dtype=torch.float32, device=device)
    dummy_lengths = torch.tensor([100], dtype=torch.long, device=device)

    encoder = FireRedAsrEncoder(model.encoder, model.decoder).to(device).eval()

    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (dummy_feat, dummy_lengths),
            str(output_path),
            input_names=["padded_input", "input_lengths"],
            output_names=["encoder_outputs", "enc_mask", "cross_k", "cross_v"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
            dynamic_axes={
                "padded_input": {0: "batch", 1: "time"},
                "input_lengths": {0: "batch"},
                "encoder_outputs": {0: "batch", 1: "time"},
                "enc_mask": {0: "batch", 2: "time"},
                "cross_k": {0: "batch", 3: "time"},
                "cross_v": {0: "batch", 3: "time"},
            },
        )


def export_decoder_step(
    model: torch.nn.Module,
    output_path: Path,
    n_layers: int,
    n_head: int,
    d_k: int,
    device: str,
    enable_attn_fusion: bool,
) -> None:
    """Exports decoder_step ONNX (fused or unfused controlled by enable_attn_fusion)."""

    decoder_step = FireRedAsrDecoderStep(model.decoder, enable_attn_fusion=enable_attn_fusion).to(device).eval()

    dummy_ys = torch.tensor([[int(model.decoder.sos_id)]], dtype=torch.long, device=device)
    dummy_src_mask = torch.ones(1, 1, 25, dtype=torch.uint8, device=device)
    dummy_cache_k_self = torch.zeros(1, n_layers, n_head, 1, d_k, dtype=torch.float32, device=device)
    dummy_cache_v_self = torch.zeros(1, n_layers, n_head, 1, d_k, dtype=torch.float32, device=device)
    dummy_cross_k = torch.randn(1, n_layers, n_head, 25, d_k, dtype=torch.float32, device=device)
    dummy_cross_v = torch.randn(1, n_layers, n_head, 25, d_k, dtype=torch.float32, device=device)

    with torch.no_grad():
        torch.onnx.export(
            decoder_step,
            (dummy_ys, dummy_src_mask, dummy_cache_k_self, dummy_cache_v_self, dummy_cross_k, dummy_cross_v),
            str(output_path),
            input_names=["ys", "src_mask", "cache_k_self", "cache_v_self", "cross_k", "cross_v"],
            output_names=["log_probs", "new_cache_k_self", "new_cache_v_self"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
            dynamic_axes={
                "ys": {0: "batch", 1: "tgt_len"},
                "src_mask": {0: "batch", 2: "src_len"},
                "cache_k_self": {0: "batch", 3: "cached_len"},
                "cache_v_self": {0: "batch", 3: "cached_len"},
                "cross_k": {0: "batch", 3: "src_len"},
                "cross_v": {0: "batch", 3: "src_len"},
                "log_probs": {0: "batch"},
                "new_cache_k_self": {0: "batch", 3: "new_cached_len"},
                "new_cache_v_self": {0: "batch", 3: "new_cached_len"},
            },
        )


def _parse_args() -> argparse.Namespace:
    """Parses CLI args for ONNX export."""

    p = argparse.ArgumentParser(description="Export FireRedASR-AED-L to ONNX")
    p.add_argument("--fireredasr-repo", type=str, default="", help="Path to cloned FireRedASR repo.")
    p.add_argument("--model-dir", type=str, required=True, help="Directory containing model.pth.tar")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for ONNX files")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument(
        "--disable-attn-fusion",
        action="store_true",
        help="Disable decoder self-attn fusion (PromptFlashAttention). Enable by default.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _add_repo_to_sys_path(args.fireredasr_repo)

    model_path = os.path.join(os.path.abspath(os.path.expanduser(args.model_dir)), "model.pth.tar")
    output_dir = Path(os.path.abspath(os.path.expanduser(args.output_dir)))
    (output_dir / "onnx_encoder").mkdir(parents=True, exist_ok=True)
    (output_dir / "onnx_decoder").mkdir(parents=True, exist_ok=True)
    enable_attn_fusion = not bool(args.disable_attn_fusion)

    model, model_args = _load_model(model_path)
    model.to(args.device)
    model.eval()

    n_layers = int(model_args.n_layers_dec)
    n_head = int(model_args.n_head)
    d_k = int(model_args.d_model // n_head)

    export_encoder(model, output_dir / "onnx_encoder" / "fireredasr_aed_encoder.onnx", args.device)
    export_decoder_step(
        model,
        output_dir / "onnx_decoder" / "fireredasr_aed_decoder_step.onnx",
        n_layers,
        n_head,
        d_k,
        args.device,
        enable_attn_fusion=enable_attn_fusion,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("ONNX export done.")
    print(f"encoder: {output_dir / 'onnx_encoder' / 'fireredasr_aed_encoder.onnx'}")
    print(f"decoder_step: {output_dir / 'onnx_decoder' / 'fireredasr_aed_decoder_step.onnx'}")
    print(f"attn_fusion: {enable_attn_fusion}")


if __name__ == "__main__":
    main()
