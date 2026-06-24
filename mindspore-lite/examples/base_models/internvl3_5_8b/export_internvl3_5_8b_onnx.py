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
"""Export OpenGVLab/InternVL3_5-8B to ONNX for MindSpore Lite cloud-side inference.

InternVL3.5 is a vision-language model = InternViT vision encoder + pixel-shuffle
+ MLP projector (``extract_feature``) + an autoregressive LLM. For InternVL3.5 the
LLM is a stock ``Qwen3ForCausalLM`` (``model.language_model``), so the LLM export
drives the real Qwen3 layers with a manual forward (QKV fused linear -> QK-norm ->
rotary -> attention with an explicit causal+padding mask -> SwiGLU MLP), mirroring
the verified ``qwen3_8b`` template but emitting **standard ONNX ops**.

Note on attention op: the CANN ``PromptFlashAttention`` Custom op with
``sparse_mode=0`` + an explicit attention mask does NOT apply the mask on this
CANN/converter build (it runs full bidirectional attention -> garbage output). The
eager math is exact (cos=1.0 vs HF), and standard MatMul/Softmax attention converts
and runs correctly (cos=1.0 vs HF), so this exporter uses standard ops. For
fixed-shape Ascend deployment the model is split into three sub-models:

  * vision encoder : InternViT + mlp1 (``extract_feature``),
                     pixel_values[1,3,H,W] -> image_embeds[1, num_img_tokens, hidden]
  * llm prefill    : inputs_embeds (text + visual) + attention_mask + position_ids
                     -> logits + present KV (padded to KV_CACHE_LEN)
  * llm decode     : inputs_embeds[1,1] + attention_mask + position_ids + past KV
                     -> logits + updated KV (scatter into the fixed cache)

The LLM sub-models consume ``inputs_embeds`` (multimodal fusion is done in the
inference script: embed input_ids then replace ``<IMG_CONTEXT>`` positions with the
visual embeds). Exported in float32 with the legacy exporter (opset 18). The same
script works for the 2B/4B/8B and Flash siblings -- only the model id and the
checkpoint-derived shapes differ.
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

KV_CACHE_LEN = 1024

try:
    import torch._dynamo

    torch._dynamo.disable()
except Exception:  # noqa: BLE001 - disable best-effort, ignore failures
    pass

try:
    from transformers import AutoModel
except ImportError:
    print("Error: transformers package not found or version too low.")
    print("Please install the latest version: pip install transformers")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Standard-op primitives (trace to plain ONNX MatMul/Softmax/ScatterElements).
# ---------------------------------------------------------------------------


def _rotate_half(x):
    """Rotate half the hidden dims of the input tensor for rotary embedding."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


def rotary_mul(x, cos4, sin4):
    """Apply rotary position embedding (x*cos + rotate_half(x)*sin)."""
    return (x * cos4) + (_rotate_half(x) * sin4)


def rms_norm(x, gamma, epsilon=1e-6):
    """Return (RMSNorm output, rstd) for parity with the 2-output call site."""
    x_fp32 = x.to(torch.float32)
    var = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + float(epsilon))
    y = (x_fp32 * rstd).to(x.dtype) * gamma
    return y, rstd


def swiglu(x, dim=-1):
    """SwiGLU activation: silu(first half) * second half."""
    d = int(dim)
    if d < 0:
        d = x.dim() + d
    a, b = torch.chunk(x, 2, dim=d)
    return F.silu(a) * b


def _expand_gqa_kv(k, v, num_heads, num_kv_heads):
    """Expand GQA key/value tensors to match num_heads via repeat_interleave."""
    if 0 < num_kv_heads < num_heads:
        rep = num_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return k, v


def _standard_attention(q, k, v, atten_mask, num_heads, num_kv_heads, scale_value):
    """Compute attention over BNSD tensors with an explicit boolean mask."""
    k, v = _expand_gqa_kv(k, v, num_heads, num_kv_heads)
    attn = torch.matmul(q, k.transpose(2, 3)) * float(scale_value)
    if atten_mask is not None:
        m = atten_mask.to(torch.bool)
        if m.dim() == 4 and m.shape[1] == 1:
            m = m.expand(attn.shape[0], attn.shape[1], m.shape[2], m.shape[3])
        attn = attn.masked_fill(m, torch.finfo(attn.dtype).min)
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v)


def prompt_flash_attention(query, key, value, atten_mask, num_heads, scale_value,
                           input_layout, num_key_value_heads, sparse_mode=0,
                           inner_precise=1, pre_tokens=214748647, next_tokens=0):
    """Prefill attention (q/k/v already BNSD) -- standard MatMul/Softmax.

    The CANN Custom-op signature is kept for compatibility, but the body uses plain
    ONNX ops (see module docstring on why the Custom PFA op is avoided here).
    """
    del input_layout, sparse_mode, inner_precise, pre_tokens, next_tokens
    return _standard_attention(query, key, value, atten_mask,
                               num_heads, num_key_value_heads, scale_value)


def incre_flash_attention(query, key, value, atten_mask, num_heads, scale_value,
                          input_layout, num_key_value_heads, block_size=0,
                          inner_precise=1):
    """Decode-step attention (q/k/v already BNSD) -- standard MatMul/Softmax."""
    del input_layout, block_size, inner_precise
    return _standard_attention(query, key, value, atten_mask,
                               num_heads, num_key_value_heads, scale_value)


def scatter(var, indices, updates, reduce="update", axis=-2):
    """Write a single decode KV slot into the fixed cache (axis=2 of a 4D tensor)."""
    del reduce, axis
    bsz, num_heads, _, _ = var.shape
    pos = indices
    if pos.dim() == 2 and pos.shape[-1] == 1:
        pos = pos.squeeze(-1)
    pos = pos.to(torch.long).view(bsz)
    upd = updates[:, :, 0, :] if updates.dim() == 4 and updates.shape[2] == 1 else updates
    out = var.clone()
    b = torch.arange(bsz, device=out.device).view(bsz, 1).expand(bsz, num_heads)
    h = torch.arange(num_heads, device=out.device).view(1, num_heads).expand(bsz, num_heads)
    s = pos.view(bsz, 1).expand(bsz, num_heads)
    out[b, h, s, :] = upd
    return out


# ---------------------------------------------------------------------------
# Qwen3 layer machinery (drives model.language_model.model.layers).
# ---------------------------------------------------------------------------


def _compute_qkv_linear(hidden_states, attn_mod):
    """Compute fused QKV linear projection and return (query, key, value) states."""
    q_w, k_w, v_w = attn_mod.q_proj.weight, attn_mod.k_proj.weight, attn_mod.v_proj.weight
    q_b, k_b, v_b = attn_mod.q_proj.bias, attn_mod.k_proj.bias, attn_mod.v_proj.bias
    w = torch.cat([q_w, k_w, v_w], dim=0)
    b = None if q_b is None else torch.cat([q_b, k_b, v_b], dim=0)
    q_out = int(q_w.shape[0])
    kv_out = int(k_w.shape[0])
    qkv = F.linear(hidden_states, w, b)
    return qkv[..., :q_out], qkv[..., q_out:q_out + kv_out], qkv[..., q_out + kv_out:]


def _apply_rotary_and_cache(cos4, sin4, query_states, key_states, value_states,
                            cache_pos, past_key, past_value):
    """Apply rotary embedding and scatter KV cache updates if past_key is provided."""
    query_states = rotary_mul(query_states, cos4, sin4)
    key_states = rotary_mul(key_states, cos4, sin4)
    if past_key is not None:
        pos = cache_pos
        if pos is None:
            raise RuntimeError("cache_pos is required when past_key_values is provided.")
        if pos.dim() == 2:
            pos = pos[:, -1]
        key_states = scatter(past_key, pos, key_states, reduce="update", axis=-2)
        value_states = scatter(past_value, pos, value_states, reduce="update", axis=-2)
    return query_states, key_states, value_states


def _prefill_attn(query_states, key_states, value_states, attention_mask, scaling,
                  num_heads, num_kv_heads):
    """Compute prefill-phase attention with an explicit causal + padding mask."""
    q = query_states.permute(0, 2, 1, 3)
    k = key_states.permute(0, 2, 1, 3)
    v = value_states.permute(0, 2, 1, 3)
    q_len, kv_len = q.shape[2], k.shape[2]
    ar_q = torch.arange(q_len, device=q.device)
    ar_k = torch.arange(kv_len, device=k.device)
    causal = ar_k[None, :] > ar_q[:, None]
    pad = attention_mask.to(torch.bool).logical_not()
    full_mask = causal[None, None, :, :] | pad[:, None, None, :]
    attn_output = prompt_flash_attention(
        q, k, v, full_mask,
        num_heads=num_heads, scale_value=float(scaling),
        input_layout="BNSD", num_key_value_heads=num_kv_heads)
    return attn_output.permute(0, 2, 1, 3)


def _rms_norm_layer(norm_mod, x):
    """Apply RMS normalization using the standard-op rms_norm."""
    gamma = norm_mod.weight
    eps = getattr(norm_mod, "variance_epsilon", getattr(norm_mod, "eps", 1e-6))
    y, _ = rms_norm(x, gamma, eps)
    return y


def _text_attn_forward(attn_mod, hidden_states, cos4, sin4, attention_mask,
                       cache_pos, past_key, past_value):
    """Run text attention: QKV projection, rotary, attention, output projection."""
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    hidden_shape = (*input_shape, -1, head_dim)

    q_lin, k_lin, v_lin = _compute_qkv_linear(hidden_states, attn_mod)
    query_states = q_lin.view(hidden_shape)
    key_states = k_lin.view(hidden_shape)
    if hasattr(attn_mod, "q_norm"):
        query_states = _rms_norm_layer(attn_mod.q_norm, query_states)
    if hasattr(attn_mod, "k_norm"):
        key_states = _rms_norm_layer(attn_mod.k_norm, key_states)
    value_states = v_lin.view(hidden_shape)
    if past_key is not None:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

    query_states, key_states, value_states = _apply_rotary_and_cache(
        cos4, sin4, query_states, key_states, value_states, cache_pos, past_key, past_value)

    scaling = getattr(attn_mod, "scaling", 1.0 / (head_dim ** 0.5))
    if past_key is None:
        attn_output = _prefill_attn(query_states, key_states, value_states,
                                    attention_mask, scaling, num_heads, num_kv_heads)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
    else:
        pad_mask = attention_mask[:, None, None, :].to(torch.bool).logical_not()
        attn_output = incre_flash_attention(
            query_states, key_states, value_states, pad_mask,
            num_heads=num_heads, scale_value=float(scaling),
            input_layout="BNSD", num_key_value_heads=num_kv_heads)

    if past_key is None:
        attn_output = attn_output.reshape(*input_shape, -1)
    else:
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
    attn_output = attn_mod.o_proj(attn_output)
    return attn_output, key_states, value_states


def _mlp_gate_up_linear(mlp_mod, x):
    """Merge gate_proj and up_proj into a single linear, return (gate, up)."""
    gate_w = mlp_mod.gate_proj.weight
    up_w = mlp_mod.up_proj.weight
    gate_b = mlp_mod.gate_proj.bias
    up_b = mlp_mod.up_proj.bias
    w = torch.cat([gate_w, up_w], dim=0)
    b = None if gate_b is None else torch.cat([gate_b, up_b], dim=0)
    y = F.linear(x, w, b)
    gate_out = int(gate_w.shape[0])
    return y[..., :gate_out], y[..., gate_out:]


def _run_mlp(layer, hidden_states, residual):
    """Run MLP sub-layer with SwiGlu activation, return output hidden states."""
    mlp = layer.mlp
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
        gate, up = _mlp_gate_up_linear(mlp, hidden_states)
        mlp_out = mlp.down_proj(swiglu(torch.cat([gate, up], dim=-1), dim=-1))
        return residual + mlp_out
    return residual + mlp(hidden_states)


def _pad_kv_cache(kv_tensor):
    """Pad KV cache tensor along dim=2 to KV_CACHE_LEN and truncate."""
    pad = kv_tensor.new_zeros(kv_tensor.shape[0], kv_tensor.shape[1],
                              KV_CACHE_LEN, kv_tensor.shape[3])
    return torch.cat([kv_tensor, pad], dim=2)[:, :, :KV_CACHE_LEN, :]


# ---------------------------------------------------------------------------
# Sub-module wrappers.
# ---------------------------------------------------------------------------


class _VisionWrapper(torch.nn.Module):
    """Wrap the InternVL vision pipeline: pixel_values -> image_embeds.

    Uses the model's ``extract_feature`` (InternViT -> select_layer -> drop CLS
    -> pixel_shuffle -> mlp1) so the projected embeds match the LLM input dim.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        """Return projected visual embeddings [1, num_img_tokens, hidden]."""
        feat = self.model.extract_feature(pixel_values)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        return feat


class _InternVLPrefill(torch.nn.Module):
    """InternVL LLM prefill wrapper: inputs_embeds -> logits + padded KV."""

    def __init__(self, llm):
        super().__init__()
        self.model = llm.model
        self.lm_head = llm.lm_head

    def forward(self, inputs_embeds, attention_mask, position_ids):
        """Run prefill over the full multimodal sequence, return logits + KV."""
        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(2) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(2) if sin.dim() == 3 else sin
        hidden_states = inputs_embeds
        present_k, present_v = [], []
        for layer in self.model.layers:
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.input_layernorm, hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn, hidden_states, cos4, sin4, attention_mask, None, None, None)
            pk = _pad_kv_cache(pk)
            pv = _pad_kv_cache(pv)
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.post_attention_layernorm, hidden_states)
            hidden_states = _run_mlp(layer, hidden_states, residual)
            present_k.append(pk)
            present_v.append(pv)
        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)


class _InternVLDecode(torch.nn.Module):
    """InternVL LLM decode wrapper: one token + past KV -> logits + updated KV."""

    def __init__(self, llm):
        super().__init__()
        self.model = llm.model
        self.lm_head = llm.lm_head

    def forward(self, inputs_embeds, attention_mask, position_ids,
                past_key_cache, past_value_cache):
        """Run one decode step, scattering the new KV into the fixed cache."""
        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(1) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(1) if sin.dim() == 3 else sin
        hidden_states = inputs_embeds
        present_k, present_v = [], []
        past_k_layers = past_key_cache.unbind(0)
        past_v_layers = past_value_cache.unbind(0)
        for i, layer in enumerate(self.model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn, hidden_states, cos4, sin4, attention_mask,
                position_ids, past_k_layers[i], past_v_layers[i])
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = _run_mlp(layer, hidden_states, residual)
            present_k.append(pk)
            present_v.append(pv)
        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)


# ---------------------------------------------------------------------------
# Export entry points.
# ---------------------------------------------------------------------------


def _load_model(model_id, dtype):
    """Load the InternVL model with trust_remote_code in the given dtype."""
    return AutoModel.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True,
                                     low_cpu_mem_usage=True).eval()


def _get_kv_cache_config(llm):
    """Return (num_layers, num_kv_heads, head_dim) from the LLM config."""
    cfg = llm.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return num_layers, num_kv_heads, head_dim


def _clear_cache():
    """Release torch caches and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _export_onnx(wrapper, dummy_inputs, input_names, output_names, out_path):
    """Trace a wrapper to ONNX (legacy exporter, opset 18, float32, no folding)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, tuple(dummy_inputs), str(out_path),
            input_names=input_names, output_names=output_names,
            opset_version=18, do_constant_folding=False, dynamo=False)
    _clear_cache()
    print(f"[export] saved {out_path}")


def export_vision(model, output_dir, image_size, dtype):
    """Export the InternViT vision encoder + mlp1 projector."""
    wrapper = _VisionWrapper(model).eval()
    pixel_values = torch.zeros((1, 3, image_size, image_size), dtype=dtype)
    out_path = Path(output_dir) / "internvl_vision.onnx"
    _export_onnx(wrapper, (pixel_values,),
                 input_names=["pixel_values"], output_names=["image_embeds"],
                 out_path=out_path)
    del wrapper
    _clear_cache()


def export_prefill(model, output_dir, num_img_tokens, max_text_len, dtype):
    """Export the LLM prefill sub-model (consumes inputs_embeds)."""
    llm = model.language_model
    num_layers, num_kv_heads, head_dim = _get_kv_cache_config(llm)
    hidden = llm.config.hidden_size
    seq_len = num_img_tokens + max_text_len
    wrapper = _InternVLPrefill(llm).eval()
    inputs_embeds = torch.zeros((1, seq_len, hidden), dtype=dtype)
    attention_mask = torch.ones((1, seq_len), dtype=torch.int64)
    position_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)
    out_path = Path(output_dir) / "internvl_llm_prefill.onnx"
    _export_onnx(wrapper, (inputs_embeds, attention_mask, position_ids),
                 input_names=["inputs_embeds", "attention_mask", "position_ids"],
                 output_names=["logits", "present_key_cache", "present_value_cache"],
                 out_path=out_path)
    del wrapper
    _clear_cache()
    print(f"[export] prefill: hidden={hidden} num_layers={num_layers} "
          f"num_kv_heads={num_kv_heads} head_dim={head_dim} seq={seq_len} kv_len={KV_CACHE_LEN}")


def export_decode(model, output_dir, dtype):
    """Export the LLM single-step decode sub-model with fixed-shape KV cache."""
    llm = model.language_model
    num_layers, num_kv_heads, head_dim = _get_kv_cache_config(llm)
    hidden = llm.config.hidden_size
    wrapper = _InternVLDecode(llm).eval()
    inputs_embeds = torch.zeros((1, 1, hidden), dtype=dtype)
    attention_mask = torch.ones((1, KV_CACHE_LEN), dtype=torch.int64)
    position_ids = torch.zeros((1, 1), dtype=torch.int64)
    past_k = torch.zeros((num_layers, 1, num_kv_heads, KV_CACHE_LEN, head_dim), dtype=dtype)
    past_v = torch.zeros((num_layers, 1, num_kv_heads, KV_CACHE_LEN, head_dim), dtype=dtype)
    out_path = Path(output_dir) / "internvl_llm_decode.onnx"
    _export_onnx(wrapper, (inputs_embeds, attention_mask, position_ids, past_k, past_v),
                 input_names=["inputs_embeds", "attention_mask", "position_ids",
                              "past_key_cache", "past_value_cache"],
                 output_names=["logits", "present_key_cache", "present_value_cache"],
                 out_path=out_path)
    del wrapper
    _clear_cache()
    print(f"[export] decode: num_layers={num_layers} num_kv_heads={num_kv_heads} "
          f"head_dim={head_dim} kv_len={KV_CACHE_LEN}")


def export_embeds(model, output_dir):
    """Dump the LLM input-embedding matrix to .npy for torch-free infer fusion.

    The LLM sub-models consume ``inputs_embeds`` (multimodal fusion is done in the
    inference script), so the token->embed lookup needs the embedding matrix outside
    the ONNX graph. Saving it once as numpy lets the MSLite infer script stay
    torch-free.
    """
    embed = model.language_model.get_input_embeddings().weight.detach().to(torch.float32).cpu().numpy()
    out_path = Path(output_dir) / "embed_weights.npy"
    np.save(out_path, embed)
    print(f"[export] embed_weights {embed.shape} -> {out_path}")


def _parse_args():
    """Parse command-line arguments for the export script."""
    parser = argparse.ArgumentParser(description="Export InternVL3.5-8B submodules to ONNX")
    parser.add_argument("--model-id", default="./InternVL3_5-8B")
    parser.add_argument("--output-dir", default="./internvl3_5_8b_onnx")
    parser.add_argument("--parts", default="vision,prefill,decode,embeds")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--num-img-tokens", type=int, default=256)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    return parser.parse_args()


def main():
    """Parse args, load InternVL, export the requested sub-models to ONNX."""
    args = _parse_args()
    dtype = {"float32": torch.float32, "float16": torch.float16}[args.dtype]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]

    model = _load_model(args.model_id, dtype) if parts else None
    try:
        if "vision" in parts:
            print("[export] vision encoder ...")
            export_vision(model, args.output_dir, args.image_size, dtype)
        if "prefill" in parts:
            print("[export] llm prefill ...")
            export_prefill(model, args.output_dir, args.num_img_tokens, args.max_text_len, dtype)
        if "decode" in parts:
            print("[export] llm decode ...")
            export_decode(model, args.output_dir, dtype)
        if "embeds" in parts:
            print("[export] llm embed weights ...")
            export_embeds(model, args.output_dir)
    finally:
        if model is not None:
            del model
            _clear_cache()
    print(f"[export] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
