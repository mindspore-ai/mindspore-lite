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
"""

import sys
import argparse
import gc
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import onnx

try:
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
except ImportError:
    apply_rotary_pos_emb = None


# ---------------------------------------------------------------------------
# Helper: additive causal + padding mask
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
                       past_key, past_value):
    """Qwen2 attention forward with GQA + KV-cache (export-friendly)."""
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

    key_states_for_attn = key_states
    value_states_for_attn = value_states
    if num_kv_heads < num_heads:
        key_states_for_attn = key_states.repeat_interleave(num_heads // num_kv_heads, dim=1)
        value_states_for_attn = value_states.repeat_interleave(num_heads // num_kv_heads, dim=1)

    scaling = getattr(attn_mod, "scaling", 1.0 / (head_dim ** 0.5))
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

        hidden_states = inputs_embeds
        present = []
        for layer in self.qwen2_model.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn, hidden_states, position_embeddings, attn_mask, None, None)
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

        hidden_states = inputs_embeds
        present = []
        for i, layer in enumerate(self.qwen2_model.layers):
            pk_in = past_key_values[2 * i]
            pv_in = past_key_values[2 * i + 1]
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn, hidden_states, position_embeddings, attn_mask, pk_in, pv_in)
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
    """Load CosyVoice2-0.5B model components by direct instantiation."""
    sys.path.insert(0, str(model_code_dir))
    sys.path.insert(0, str(model_code_dir / "third_party" / "Matcha-TTS"))
    llm = _build_llm(model_dir)
    flow = _build_flow()
    hift = _build_hift()
    _load_weights(llm, flow, hift, model_dir, device)

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


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Export CosyVoice2-0.5B to ONNX")
    parser.add_argument(
        "--model-dir", type=str,
        default="/Users/apple/git/models/models_weights/CosyVoice2-0.5B",
        help="Path to CosyVoice2-0.5B weights directory",
    )
    parser.add_argument(
        "--model-code-dir", type=str,
        default="/Users/apple/git/models/models_code/CosyVoice",
        help="Path to CosyVoice source code directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./cosyvoice2_onnx",
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

    args = parser.parse_args()
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

    print("\n" + "=" * 60)
    print(f"Export finished. Files saved in {args.output_dir}")
    print("=" * 60)

    del llm, flow, hift
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
