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

"""Export Wan2.1-T2V-1.3B-Diffusers submodules to ONNX for MindSpore Lite cloud-side inference.

The text-to-video pipeline is split into three fixed-shape sub-models. Any
resolution (multiples of 16) and frame count (4k+1) can be given via
``--height/--width/--num-frames``; shapes below are the official spec
(480x832, 81 frames = 4k+1, 5s @16fps):

  * text encoder  : UMT5-XXL, input_ids[1,512] + attention_mask[1,512]
                    -> last_hidden_state[1,512,4096]          (fp32, legacy exporter)
  * transformer   : WanTransformer3DModel (DiT, 1.3B),
                    hidden_states[1,16,21,60,104] + timestep[1] +
                    encoder_hidden_states[1,512,4096] -> noise_pred
                    (fp32 dynamo export, then CANN PromptFlashAttention graph surgery)
  * vae decoder   : AutoencoderKLWan.decode. <=13 latent frames: single-pass
                    ``wan_vae_decoder.onnx`` (shape follows --num-frames).
                    >13 latent frames (e.g. the 81-frame spec, 21 latents):
                    ``wan_vae_decoder_13lat.onnx`` segment decoder
                    (latents[1,16,13,60,104] -> video[1,3,49,480,832]); the
                    infer script reconstructs the video from overlapping
                    segments. (fp32 dynamo export)

Transformer attention is first exported as materialised standard ops
(``BatchMatMul + Softmax + BatchMatMul``; numerically identical, Wan attention is
full bidirectional without mask) and then rewritten by :func:`inject_pfa` into 60
CANN built-in ``PromptFlashAttention`` Custom nodes (30 blocks x self/cross). The
flash kernel never materialises the [12, 32760, 32760] QK^T (25.8GB per layer,
109.4GB across the graph) that otherwise OOMs the 310P3 conversion. The mslite
converter maps those Custom nodes onto the built-in ``aclnnPromptFlashAttention``.

The Wan VAE decoder uses a streaming causal-conv cache that mis-traces under both
JIT trace and torch.export; the patches below replace it with an equivalent
stateless single full-sequence pass (numerically identical), plus standard-op
decompositions for ``SDPA`` / ``_upsample_nearest_exact2d`` which have no
converter-friendly lowering.

Outputs (default ``./wan2_1_t2v_1_3b_onnx``):
    wan_text_encoder.onnx            (+ external weight data)
    wan_transformer_pfa.onnx         (PFA injected; external weight data)
    wan_vae_decoder[_13lat].onnx     (+ external weight data)

With ``--tp-size 2 --rank {0,1}`` the transformer is additionally sharded into
two rank graphs (``wan_transformer_pfa_rank{0,1}.onnx``) for the TP2 x CFG
4-chip inference path (see the infer script's ``--tp-size``).

Example:
    python export_wan2_1_t2v_1_3b_onnx.py --model-dir ./Wan2.1-T2V-1.3B-Diffusers
"""

import argparse
import gc
import math
import os
from collections import Counter

import torch
from torch import nn
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers.models.transformers import transformer_wan
from transformers import UMT5EncoderModel

_VAE_SCALE_FACTOR_TEMPORAL = 4
_VAE_SCALE_FACTOR_SPATIAL = 8
_LATENT_CHANNELS = 16
_OPSET = 17
_NUM_HEADS = 12          # Wan DiT is MHA: num_heads == num_kv_heads == 12
_HEAD_DIM = 128
# The VAE is exported once as a 13-latent segment decoder (49 decoded frames);
# the 81-frame video is reconstructed from two overlapping segments (see infer).
_VAE_SEGMENT_LATENTS = 13
_VAE_EXPORT_FRAMES = _VAE_SEGMENT_LATENTS * _VAE_SCALE_FACTOR_TEMPORAL  # 52


# ---------------------------------------------------------------------------
# diffusers monkey-patches (standard-op decompositions for clean conversion).
# ---------------------------------------------------------------------------


def _patch_rmsnorm():
    """Replace ``torch.nn.RMSNorm.forward`` with a standard-op decomposition.

    Wan uses ``torch.nn.RMSNorm`` for q/k-norms; the legacy ONNX exporter has no
    symbolic for ``aten::rms_norm``. The decomposition (fp32 variance -> rsqrt ->
    mul weight) is numerically identical and traces to standard ONNX ops.
    """

    def _forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if getattr(self, "weight", None) is not None:
            hidden_states = hidden_states.to(self.weight.dtype) * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)
        return hidden_states

    torch.nn.RMSNorm.forward = _forward


def _patch_wan_layernorm():
    """Replace ``FP32LayerNorm.forward`` with a standard-op decomposition.

    Wan uses ``FP32LayerNorm`` (an ``nn.LayerNorm`` subclass) for the per-block
    norm1/norm2/norm3 and the output norm. The dynamo ONNX exporter fuses
    ``F.layer_norm`` into a single ``LayerNormalization`` op whose output shape the
    Ascend GE build cannot infer ("context is a null pointer"). Decomposing into
    Mean/Sub/Pow/Mean/Sqrt/Div/Mul/Add (all standard ops, computed in fp32) is
    numerically identical and keeps the graph fully static + standard-op.
    """

    from diffusers.models.normalization import FP32LayerNorm

    def _forward(self, inputs):
        origin_dtype = inputs.dtype
        x = inputs.float()
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) * (x - mean)).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if getattr(self, "weight", None) is not None:
            x = x * self.weight.float()
        if getattr(self, "bias", None) is not None:
            x = x + self.bias.float()
        return x.to(origin_dtype)

    FP32LayerNorm.forward = _forward


def _patch_wan_attention(tp_rank=None, tp_size=None):
    """Replace Wan's attention dispatch with materialised standard-op attention.

    The Wan processor calls ``dispatch_attention_fn(q, k, v, ...)`` with q/k/v in
    layout (batch, seq, num_heads, head_dim) [BSHD]. We transpose to BNSD and run
    explicit ``BatchMatMul + Softmax + BatchMatMul`` attention, then transpose back.
    Wan's self/cross attention are full (no causal / no padding mask), so this is
    numerically identical to diffusers' SDPA. q/k/v projections, RMSNorm
    (norm_q/norm_k) and RoPE stay as the original diffusers code. The softmax is
    computed in fp32 for stability; the CANN PromptFlashAttention Custom nodes
    injected afterwards (:func:`inject_pfa`) replace the whole chain.

    TP (``tp_rank``/``tp_size``): Wan's norm_q/norm_k are *across-heads* RMSNorm
    (1536-dim), which cannot be head-sharded, so q/k/v projections and norms run
    replicated on every rank and only the attention heads are sliced here
    (``[B,S,heads_per_rank,128]``). The PFA Custom nodes are then injected with
    ``num_heads = total_heads // tp_size`` and the row-parallel o_proj / ffn-out
    partial sums are reduced by :func:`inject_allreduce`.
    """

    def _custom_dispatch(query, key, value, attn_mask=None, dropout_p=0.0,
                         is_causal=False, scale=None, enable_gqa=False,
                         attention_kwargs=None, *, backend=None,
                         parallel_config=None):
        del attn_mask, dropout_p, is_causal, enable_gqa, attention_kwargs
        del backend, parallel_config
        head_dim = int(value.shape[-1])
        scale_val = float(scale) if scale is not None else float(1.0 / math.sqrt(head_dim))
        if tp_size is not None and tp_size > 1:
            # value: [B, S, total_heads, D] BSHD -> slice this rank's heads
            heads_per_rank = int(value.shape[2]) // tp_size
            lo = tp_rank * heads_per_rank
            query = query[:, :, lo:lo + heads_per_rank]
            key = key[:, :, lo:lo + heads_per_rank]
            value = value[:, :, lo:lo + heads_per_rank]
        q = query.transpose(1, 2)  # BNSD
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        # fp32 softmax: with the 1/sqrt(d) pre-scale the logits stay small
        # (observed absmax ~13), so the cast is stability-only, not a necessity.
        # NOTE: no explicit .float() cast here -- the export runs in fp32 anyway,
        # and inject_pfa pattern-matches the Mul -> Softmax -> MatMul chain
        # directly (a Cast node in between would break the match).
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale_val, dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2)  # back to BSHD

    transformer_wan.dispatch_attention_fn = _custom_dispatch


# ---------------------------------------------------------------------------
# Transformer: CANN built-in PromptFlashAttention graph surgery.
# ---------------------------------------------------------------------------


def _pfa_scalar(init, prod, name):
    """Resolve a float scalar (initializer or Constant node) by tensor name."""
    import numpy as np
    t = init.get(name)
    if t is not None and t.raw_data:
        return float(np.frombuffer(t.raw_data, dtype=np.float32)[0])
    cn = prod.get(name)
    if cn is not None and cn.op_type == "Constant":
        for a in cn.attribute:
            if a.name == "value" and a.t.raw_data:
                return float(np.frombuffer(a.t.raw_data, dtype=np.float32)[0])
    raise ValueError(f"scale scalar not found: {name}")


def inject_pfa(src, dst, num_heads=_NUM_HEADS, head_dim=_HEAD_DIM, inner_precise=0,
               layout="BNSD"):
    """Rewrite materialised attention chains into CANN PromptFlashAttention Custom nodes.

    Replaces each ``MatMul(q,kT) -> Mul(scale) -> Softmax -> MatMul(,v)`` chain
    with a single Custom node, which the mslite converter maps onto the built-in
    ``aclnnPromptFlashAttention``. This removes the [B,N,Sq,Sk] attention matrix
    entirely -- the change that makes the official 81-frame spec (32760 tokens,
    25.8GB QK^T per layer) representable on 310P3 at all.

    The Custom node takes over the AV matmul's *output name* (whose shape is
    already recorded in the graph's value_info), so the GE build can infer every
    node shape -- a freshly minted Custom output with no value_info entry is what
    made the naive Custom-op path abort with "context is a null pointer".

    ``layout`` selects the PFA input layout. ``"BNSD"`` keeps the layout the dynamo
    graph materialises for the QK^T/AV matmuls (the only layout the 310P built-in
    PFA GE op accepts on CANN 9.0.1); ``"BSHD"`` would feed q/k/v in the
    processor-native layout and take over the back-transpose's output name, but
    the 310P GE op rejects it ("Invalid input layout: BSHD, not support") -- kept
    as an option for future CANN versions. Either way the dynamo graph stores
    q/k/v in a few layouts (self-attn K post-RoPE is a [D,S,N,1] ScatterND quirk,
    cross-attn K/V are BSHD), so per-chain Transpose/Reshape glue is emitted as
    needed.
    """
    import onnx
    from onnx import AttributeProto, helper

    # string attributes are built by hand: helper cannot infer the attribute
    # type for an empty strings list
    def _s1(attr_name, value):
        a = AttributeProto()
        a.name = attr_name
        a.type = AttributeProto.STRING
        a.s = value.encode()
        return a

    def _ss(attr_name, values):
        a = AttributeProto()
        a.name = attr_name
        a.type = AttributeProto.STRINGS
        for v in values:
            a.strings.append(v.encode())
        return a

    m = onnx.load(src, load_external_data=False)
    prod = {}
    for n in m.graph.node:
        for o in n.output:
            prod[o] = n
    init = {i.name: i for i in m.graph.initializer}
    shapes = {}
    for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        shapes[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]

    glue = []          # extra Transpose/Reshape nodes for layout fixes
    if layout == "BNSD":
        k_reshp = helper.make_tensor("pfa_k_reshp_shape", onnx.TensorProto.INT64,
                                     dims=[4], vals=[1, num_heads, -1, head_dim])
    else:
        # BSHD quirk-K reshape: [D,S,N,1] -> T(1,2,0,3) [S,N,D,1] -> [1,S,N,D]
        k_reshp = helper.make_tensor("pfa_k_reshp_shape", onnx.TensorProto.INT64,
                                     dims=[4], vals=[1, -1, num_heads, head_dim])
    m.graph.initializer.append(k_reshp)
    sms = [n for n in m.graph.node if n.op_type == "Softmax"]

    to_remove = set()
    chain_head = {}
    new_nodes = []
    chain_prefix = {}   # chain idx -> prefix used for its glue node names
    for idx, sm in enumerate(sms):
        mul = prod[sm.input[0]]
        mm1 = prod[mul.input[0]]
        av = [n for n in m.graph.node if sm.output[0] in n.input][0]
        assert mul.op_type == "Mul" and mm1.op_type == "MatMul" and av.op_type == "MatMul", \
            f"chain {idx}: unexpected ops {mul.op_type}/{mm1.op_type}/{av.op_type}"
        qin = mm1.input[0]
        # The QK^T MatMul consumes K *already transposed*; the AV MatMul consumes V
        # in BSHD or a dynamo DSND quirk. Build whatever Transpose/Reshape glue
        # each PFA input needs.
        kout = mm1.input[1]           # kT shape [B,N,D,S] (QK^T operand)
        vout = av.input[1]            # v as consumed by AV matmul
        kt = prod.get(kout)
        ksrc = kt.input[0] if (kt is not None and kt.op_type == "Transpose") else kout
        ks = shapes.get(ksrc)
        if ks and len(ks) == 4 and ks[-1] == 1 and ks[0] == head_dim:
            # dynamo self-attn quirk: K stored [D,S,N,1] (post-RoPE ScatterND)
            t = helper.make_node("Transpose", [ksrc], [f"pfa{idx}_kt"],
                                 perm=[2, 1, 0, 3] if layout == "BNSD" else [1, 2, 0, 3])
            glue.append(t)
            r = helper.make_node("Reshape", [f"pfa{idx}_kt", "pfa_k_reshp_shape"],
                                 [f"pfa{idx}_k"])
            glue.append(r)
            kin = f"pfa{idx}_k"
        elif ks and len(ks) == 4 and ks[2] == num_heads and ks[1] != num_heads:
            # BSHD (cross-attn K)
            if layout == "BNSD":
                t = helper.make_node("Transpose", [ksrc], [f"pfa{idx}_kt"], perm=[0, 2, 1, 3])
                glue.append(t)
                kin = f"pfa{idx}_kt"
            else:
                kin = ksrc            # already BSHD
        elif ks and len(ks) == 4 and ks[1] == num_heads and ks[2] != num_heads:
            if layout == "BNSD":
                kin = ksrc            # already BNSD
            else:
                t = helper.make_node("Transpose", [ksrc], [f"pfa{idx}_kt"], perm=[0, 2, 1, 3])
                glue.append(t)
                kin = f"pfa{idx}_kt"  # BNSD -> BSHD
        else:
            kin = ksrc
        # --- V glue: vout consumed by AV matmul.
        vt = prod.get(vout)
        vshape = shapes.get(vout)
        if layout == "BNSD":
            if vshape and len(vshape) == 4 and vshape[1] == num_heads:
                vin = vout                     # BNSD (cross-attn after earlier fix)
            elif vshape and len(vshape) == 4 and vshape[2] == num_heads:
                t = helper.make_node("Transpose", [vout], [f"pfa{idx}_vt"], perm=[0, 2, 1, 3])
                glue.append(t)
                vin = f"pfa{idx}_vt"           # BSHD -> BNSD (self-attn)
            elif vt is not None and vt.op_type == "Transpose":
                vin = vt.input[0]              # transpose source is the BNSD view
            else:
                vin = vout
        else:
            if vshape and len(vshape) == 4 and vshape[2] == num_heads:
                vin = vout                     # already BSHD
            elif vt is not None and vt.op_type == "Transpose":
                vin = vt.input[0]              # transpose source is the BSHD view
            elif vshape and len(vshape) == 4 and vshape[1] == num_heads:
                t = helper.make_node("Transpose", [vout], [f"pfa{idx}_vt"], perm=[0, 2, 1, 3])
                glue.append(t)
                vin = f"pfa{idx}_vt"           # BNSD -> BSHD
            else:
                vin = vout
        # --- Q: BSHD mode wants the pre-transpose tensor (BSHD source).
        if layout == "BSHD":
            qt = prod.get(qin)
            q_src = qt.input[0] if (qt is not None and qt.op_type == "Transpose") else qin
        else:
            q_src = qin
        scale_val = _pfa_scalar(init, prod, mul.input[1])
        chain_prefix[idx] = f"pfa{idx}_"
        # BSHD mode: the chain ends with a back-transpose ([B,N,S,D] -> [B,S,N,D])
        # whose output name the PFA (BSHD out) takes over; the transpose becomes
        # dead and is DCE'd by GE.
        out_name = av.output[0]
        if layout == "BSHD":
            consumers = [n for n in m.graph.node if av.output[0] in n.input
                         and n is not av]
            assert len(consumers) == 1 and consumers[0].op_type == "Transpose", \
                f"chain {idx}: expected single back-transpose after AV matmul"
            out_t = consumers[0]
            perm = list(out_t.attribute)[0].ints
            assert list(perm) == [0, 2, 1, 3], f"chain {idx}: back-transpose perm {list(perm)}"
            out_name = out_t.output[0]
            to_remove.add(out_t.name)
        pfa = helper.make_node(
            "Custom", [q_src, kin, vin], [out_name],
            name=f"PromptFlashAttention_{idx}",
        )
        pfa.attribute.extend([
            _s1("type", "PromptFlashAttention"),
            _ss("input_names", ["query", "key", "value"]),
            _ss("output_names", ["attention_out"]),
            helper.make_attribute("output_num", 1),
            helper.make_attribute("input_index", [0, 1, 2]),
            helper.make_attribute("num_heads", num_heads),
            helper.make_attribute("num_key_value_heads", num_heads),
            helper.make_attribute("scale_value", scale_val),
            _s1("input_layout", layout),
            helper.make_attribute("inner_precise", inner_precise),
        ])
        new_nodes.append((mul.name, pfa))
        chain_head[mul.name] = idx
        to_remove.update((mul.name, mm1.name, sm.name, av.name))

    if len(sms) == 0:
        raise ValueError(f"no materialised attention chain (Softmax) found in {src}")

    final = []
    for n in m.graph.node:
        if n.name in chain_head:
            idx = chain_head[n.name]
            # topology-safe: glue nodes are registered per chain, emit grouped
            for g in [x for x in glue if x.output[0].startswith(chain_prefix[idx])]:
                final.append(g)
            final.append(new_nodes[idx][1])
        elif n.name in to_remove:
            continue
        else:
            final.append(n)
    # fallback: glue nodes claimed by no chain (should not exist)
    emitted = {n.name for n in final} | {n.output[0] for n in final if n.output}
    for g in glue:
        if g.name not in emitted and g.output[0] not in emitted:
            final.append(g)
    del m.graph.node[:]
    m.graph.node.extend(final)

    c = Counter(n.op_type for n in m.graph.node)
    print(f"[export] PFA injection ({layout}): {len(sms)} chains -> "
          f"Custom={c.get('Custom', 0)} Softmax={c.get('Softmax', 0)} "
          f"MatMul={c.get('MatMul', 0)} Transpose={c.get('Transpose', 0)}")
    onnx.save(m, dst)
    return len(sms)


def _shard_model_for_tp(model, rank, tp_size):
    """Shard row-parallel weights in-place for TP export (Megatron-style).

    - attention to_out[0] / ffn net.2: row-parallel -- weight [out, in] sliced on
      the *input* dim to [out, in/tp]. Their bias stays FULL and is added AFTER
      the AllReduce: dynamo decomposes nn.Linear into MatMul + Add(bias), and
      :func:`inject_allreduce` rewires that Add to take AllReduce(partial) -- so
      the bias is applied exactly once on the fully-reduced sum.
    - ffn net[0].proj (GELU wrapper holds fc1): column-parallel -- sliced on the
      *output* dim, bias sliced along.
    - everything else (qkv projections, across-heads RMSNorm, modulation, norms,
      embeddings) is replicated: norm_q/norm_k normalise across ALL heads and
      cannot be sharded mathematically.
    We shard weights in place (no module replacement) so the exported graph stays
    structurally identical to the single-chip graph.
    """
    for block in model.blocks:
        for attn_mod in (block.attn1, block.attn2):
            w = attn_mod.to_out[0].weight.data            # [1536, 1536]
            shard_in = w.shape[1] // tp_size
            attn_mod.to_out[0].weight.data = (
                w[:, rank * shard_in:(rank + 1) * shard_in].contiguous())
        ff = block.ffn
        proj = ff.net[0].proj                             # Linear(1536, 8960, bias) inside GELU
        w0, b0 = proj.weight.data, proj.bias.data         # [8960, 1536]
        shard_out = w0.shape[0] // tp_size
        proj.weight.data = w0[rank * shard_out:(rank + 1) * shard_out].contiguous()
        proj.bias.data = b0[rank * shard_out:(rank + 1) * shard_out].contiguous()
        w2 = ff.net[2].weight.data                        # [1536, 8960]
        ff.net[2].weight.data = w2[:, rank * shard_out:(rank + 1) * shard_out].contiguous()


def inject_allreduce(src, dst, tp_size):
    """Insert Custom(AllReduce, sum) after the row-parallel o_proj / ffn-out matmuls.

    Locates the 3 x n_blocks reduction points by the dynamo bias-Add nodes
    (``model.blocks.{i}.attn{1,2}.to_out.0.bias`` / ``ffn.net.2.bias``): the Add
    consuming the bias takes ``AllReduce(partial_sum)`` as its data input instead
    of the raw partial sum, so the bias is applied exactly once on the reduced
    result. The Custom(AllReduce) nodes are compiled online by GE at inference
    time, once a rank_table (HCCL world) exists.
    """
    import onnx
    from onnx import AttributeProto, helper

    m = onnx.load(src, load_external_data=False)
    init = {i.name: i for i in m.graph.initializer}

    def _ss(attr_name, values):
        a = AttributeProto()
        a.name = attr_name
        a.type = AttributeProto.STRINGS
        for v in values:
            a.strings.append(v.encode())
        return a

    def _s1(attr_name, value):
        a = AttributeProto()
        a.name = attr_name
        a.type = AttributeProto.STRING
        a.s = value.encode()
        return a

    def _is_anchor_bias(name):
        return (".attn1.to_out.0.bias" in name or ".attn2.to_out.0.bias" in name
                or ".ffn.net.2.bias" in name)

    rewired = 0
    final = []
    for n in m.graph.node:
        if n.op_type == "Add":
            bias_in = next((i for i in n.input if i in init and _is_anchor_bias(i)), None)
            if bias_in is not None:
                partial = next(i for i in n.input if i != bias_in)
                ar_out = f"{n.output[0]}_ar"
                ar = helper.make_node("Custom", [partial], [ar_out],
                                      name=f"AllReduce_{rewired}")
                ar.attribute.extend([
                    _s1("type", "AllReduce"),
                    _ss("input_names", ["x"]),
                    _ss("output_names", ["y"]),
                    helper.make_attribute("output_num", 1),
                    helper.make_attribute("input_index", [0]),
                    _s1("op", "sum"),
                    _s1("group", "hccl_world_group"),
                    helper.make_attribute("rank_size", tp_size),
                    helper.make_attribute("fusion", 0),
                ])
                final.append(ar)
                n.input[0] = ar_out        # rewire in place: Add(AR_out, bias)
                rewired += 1
        final.append(n)
    del m.graph.node[:]
    m.graph.node.extend(final)

    c = Counter(x.op_type for x in m.graph.node)
    print(f"[export] AllReduce injection: {rewired} reductions -> "
          f"Custom={c.get('Custom', 0)}")
    onnx.save(m, dst)
    return rewired


# ---------------------------------------------------------------------------
# VAE patches: streaming cache -> stateless single full-sequence pass.
# ---------------------------------------------------------------------------


def _patch_wan_resample_singlepass():
    """Make ``WanResample`` do the full temporal upsample in a single pass.

    The stock ``upsample3d`` path only runs the temporal 2x upsample
    (``time_conv`` + interleave-reshape) on the *second* call of its streaming
    cache protocol; on the first call it just marks the cache ("Rep") and returns
    ``x`` unchanged. The export wrapper runs the decoder as a single full-sequence
    pass (``_VaeWrapper`` with ``feat_cache=None``), so that first call is also the
    only call -- and without this patch the temporal axis is never upsampled (a
    4-latent-frame input decodes to 4 frames instead of 13). We replace ``forward``
    so ``upsample3d`` always does ``time_conv`` (full causal conv over the whole
    sequence) + the 2x interleave, which is exactly what the streaming "Rep"
    branch computes for a full sequence.
    """

    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResample

    def _forward(self, x):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            x = self.time_conv(x)                       # [b, c*2, t, h, w]
            x = x.reshape(b, 2, c, t, h, w)
            x = torch.stack((x[:, 0], x[:, 1]), 3)      # [b, c, t, 2, h, w]
            x = x.reshape(b, c, t * 2, h, w)            # temporal 2x interleave
            # diffusers' streaming decode feeds latent frames one at a time, so the
            # first frame is not duplicated -> 2t-1 output frames per upsample (4->7->13
            # for a 4-latent-frame / 13-video-frame clip). The full-sequence single pass
            # interleaves 2t; drop the g1_0 slot (temporal index 1) to match. The causal
            # time_conv sees the whole sequence, so the kept frames are numerically
            # identical to the streaming decode.
            x = torch.cat([x[:, :, :1], x[:, :, 2:]], dim=2)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)                            # spatial 2x upsample
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
        return x

    WanResample.forward = _forward


def _patch_wan_vae_no_cache():
    """Strip the streaming feat_cache/feat_idx logic from the VAE conv blocks.

    The stock ``WanCausalConv3d`` / ``WanResidualBlock`` forwards mutate a Python
    list (``feat_cache``) and an index counter (``feat_idx``) passed through the
    decoder. ``torch.export`` (dynamo) mis-traces that mutable state and the
    exported VAE collapses to a spatially-flat output (each channel comes out
    uniform). For a single full-sequence pass the cache is never read (every conv
    is called once with ``cache_x=None`` -> full causal conv), so removing the
    cache branches changes nothing numerically but gives dynamo a clean graph.
    """

    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    def _cc_forward(self, x, cache_x=None):
        del cache_x
        # WanCausalConv3d extends nn.Conv3d; pad (asymmetric, causal) then run the conv.
        return torch.nn.Conv3d.forward(self, torch.nn.functional.pad(x, self._padding))

    WanCausalConv3d.forward = _cc_forward

    def _res_forward(self, x, **_kwargs):
        h = self.conv_shortcut(x)
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + h

    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualBlock
    WanResidualBlock.forward = _res_forward


def _patch_wan_vae_attention():
    """Replace SDPA in the VAE mid-block attention with explicit ops.

    The stock ``WanAttentionBlock`` calls ``F.scaled_dot_product_attention``, which the
    legacy ONNX exporter lowers to an ONNX ``If`` node (flash-vs-math backend selection).
    The ``ascend_oriented`` converter cannot lower that ``If``
    ("i: 3 out of range ... ValueNode<If>", conversion aborts). The mid-block attention
    is single-head over the HxW spatial tokens, so spelling it out as
    BatchMatMul + Softmax + BatchMatMul (1/sqrt(C) scaling) is numerically identical and
    traces to standard ops with no control flow.
    """

    from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock

    def _forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1).permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        # explicit single-head attention (replaces F.scaled_dot_product_attention)
        scale = float(1.0 / float(channels) ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)
        # x is [B*T, 1, H*W, C]; index the singleton head axis with [:, 0] instead of
        # squeeze(1) -- torch.squeeze emits a conditional ONNX If ("squeeze only if the
        # dim is size 1") that the ascend_oriented converter cannot lower.
        x = x[:, 0].permute(0, 2, 1).reshape(batch_size * time, channels, height, width)
        x = self.proj(x)
        x = x.view(batch_size, time, channels, height, width).permute(0, 2, 1, 3, 4)
        return x + identity

    WanAttentionBlock.forward = _forward


def _patch_wan_upsample():
    """Replace ``WanUpsample.forward`` with a standard-op 2x nearest-exact decomposition.

    diffusers' WanUpsample delegates to ``nn.Upsample(mode="nearest-exact",
    scale_factor=(2, 2))``, which traces to ``aten::_upsample_nearest_exact2d`` -- an
    internal op with no symbolic in the legacy ONNX exporter (export aborts with
    ``UnsupportedOperatorError`` at opset 17/19). WanUpsample is only ever built with a
    2x2 factor over a 4D ``[N, C, H, W]`` tensor (WanResample reshapes the 5D video to
    4D before upsampling, then back to 5D). For an integer 2x factor, nearest-exact is
    exactly "duplicate each pixel 2x2" -- i.e. ``repeat_interleave(2)`` on H and W --
    which traces to standard ONNX ops and needs no upsample symbolic.
    """

    from diffusers.models.autoencoders.autoencoder_kl_wan import WanUpsample

    def _forward(x):
        dtype = x.dtype
        xf = x.float()
        xf = xf.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        return xf.to(dtype)

    WanUpsample.forward = staticmethod(_forward)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _clear_cache():
    """Release torch caches and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _latent_shape(num_frames, height, width):
    """Compute the fixed latent shape (B, C, F', H', W') for the given video size."""
    num_latent_frames = (num_frames - 1) // _VAE_SCALE_FACTOR_TEMPORAL + 1
    latent_h = height // _VAE_SCALE_FACTOR_SPATIAL
    latent_w = width // _VAE_SCALE_FACTOR_SPATIAL
    return (1, _LATENT_CHANNELS, num_latent_frames, latent_h, latent_w)


def _export_onnx(wrapper, dummy_inputs, input_names, output_names, out_path,
                 opset=_OPSET, constant_folding=False, use_dynamo=False):
    """Trace a wrapper to ONNX (float32) and resolve shapes for the converter.

    Two exporters:

    * **legacy** (``use_dynamo=False``, default) — ``torch.onnx.utils.export``. The
      text encoder uses this; a post-pass runs onnxsim (or ONNX shape inference
      with ``data_prop``) to turn the ``.size()``-driven symbolic dims into
      concrete shapes.
    * **dynamo** (``use_dynamo=True``) — ``torch.onnx.export(dynamo=True)``. The
      transformer and VAE use this: the legacy exporter leaves thousands of
      symbolic dims that onnxsim would normally fold, but both models are large /
      slow to trace, and dynamo produces a fully static graph directly. The
      ``ascend_oriented`` GE build aborts ("context is a null pointer") if any
      intermediate shape stays symbolic, so we must hand it a fully static graph.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if use_dynamo:
        torch.onnx.export(
            wrapper,
            tuple(dummy_inputs),
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamo=True,
            external_data=True,
        )
    else:
        torch.onnx.utils.export(
            wrapper,
            tuple(dummy_inputs),
            out_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=constant_folding,
        )
        # Resolve symbolic dims -> concrete shapes for the ascend_oriented converter.
        # data_prop shape inference alone is not enough for models with dynamic
        # Shape/Gather/Concat->Reshape chains; onnxsim constant-folds them, the way
        # dynamo does for the transformer. onnxsim only fits models <2 GiB (the text
        # encoder), so it is best-effort and falls back to data_prop on failure.
        import onnx
        from onnx import shape_inference
        model = onnx.load(out_path, load_external_data=False)
        try:
            import onnxsim
            model, _ = onnxsim.simplify(model)
        except Exception as exc:  # noqa: BLE001 -- fall back to data_prop only
            print(f"[export] onnxsim skipped ({type(exc).__name__}); using data_prop")
            model = shape_inference.infer_shapes(model, data_prop=True)
        onnx.save(model, out_path)
    _clear_cache()
    print(f"[export] saved {out_path}")


# ---------------------------------------------------------------------------
# Sub-module wrappers.
# ---------------------------------------------------------------------------


class _TextEncoderWrapper(nn.Module):
    """Wrap UMT5EncoderModel to expose input_ids/attention_mask -> last_hidden_state."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        """Return the encoder last hidden state [1, seq_len, 4096]."""
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class _TransformerWrapper(nn.Module):
    """Wrap WanTransformer3DModel to expose hidden_states/timestep/embeds -> noise_pred."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        """Return the predicted noise [1, 16, F', H', W']."""
        return self.model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


class _VaeWrapper(nn.Module):
    """Wrap AutoencoderKLWan decode to expose latents -> video.

    Calls the decode steps directly (post_quant_conv -> decoder -> clamp) instead of
    ``vae.decode``/``_decode``: the streaming ``_decode`` mutates instance state
    (``self._feat_map`` via ``clear_cache``) which ``torch.export`` (dynamo)
    mis-traces and collapses the spatial output. The conv/resample patches already
    ignore the cache, so calling ``decoder(x, feat_cache=None)`` is equivalent and
    gives dynamo a clean, stateless graph.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        """Return the decoded video [1, 3, F, H, W] (latents already denormalised)."""
        vae = self.vae
        x = vae.post_quant_conv(latents)
        out = vae.decoder(x, feat_cache=None, feat_idx=[0], first_chunk=True)
        patch_size = getattr(vae.config, "patch_size", None)
        if patch_size is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
            out = unpatchify(out, patch_size=patch_size)
        return torch.clamp(out, min=-1.0, max=1.0)


# ---------------------------------------------------------------------------
# Export entry points.
# ---------------------------------------------------------------------------


def export_text_encoder(model_dir, output_dir, max_seq_len, dtype):
    """Export the UMT5-XXL text encoder to ONNX."""
    encoder = UMT5EncoderModel.from_pretrained(
        os.path.join(model_dir, "text_encoder"), torch_dtype=dtype
    ).eval()
    wrapper = _TextEncoderWrapper(encoder)
    input_ids = torch.zeros((1, max_seq_len), dtype=torch.int64)
    attention_mask = torch.ones((1, max_seq_len), dtype=torch.int64)
    out_path = os.path.join(output_dir, "wan_text_encoder.onnx")
    _export_onnx(
        wrapper, (input_ids, attention_mask),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"], out_path=out_path,
    )
    del encoder, wrapper
    _clear_cache()


def export_transformer(model_dir, output_dir, height, width, num_frames,
                       max_seq_len, dtype, pfa_inner_precise=0, pfa_layout="BNSD",
                       tp_rank=None, tp_size=None):
    """Export the Wan DiT transformer to ONNX + inject CANN PromptFlashAttention.

    The dynamo exporter first produces a fully static standard-op graph
    (materialised attention, RMSNorm/LayerNorm decomposed); :func:`inject_pfa`
    then rewrites the 60 attention chains into PFA Custom nodes. The PFA Custom
    output replaces the AV matmul's output, so no value_info re-inference is
    needed and the graph stays converter-friendly.

    With ``tp_rank``/``tp_size`` set (e.g. 0/2 and 1/2), row-parallel weights are
    sharded (:func:`_shard_model_for_tp`), attention heads are sliced in the
    patched dispatch, PFA is injected with the per-rank head count and the
    row-parallel partial sums are reduced by :func:`inject_allreduce` -- producing
    one graph per rank (``wan_transformer_pfa_rank{r}.onnx``). The rank graphs
    are converted with ``optimize=none`` (plain MindIR) and compiled online by GE
    at inference time, when a rank_table (HCCL world) is available.
    """
    _patch_rmsnorm()
    _patch_wan_layernorm()
    _patch_wan_attention(tp_rank=tp_rank, tp_size=tp_size)
    model = WanTransformer3DModel.from_pretrained(
        os.path.join(model_dir, "transformer"), torch_dtype=dtype
    ).eval()
    if tp_size is not None and tp_size > 1:
        _shard_model_for_tp(model, tp_rank, tp_size)
    text_dim = model.config.text_dim
    latent_shape = _latent_shape(num_frames, height, width)
    wrapper = _TransformerWrapper(model)
    hidden_states = torch.zeros(latent_shape, dtype=dtype)
    timestep = torch.tensor([950.0], dtype=dtype)
    encoder_hidden_states = torch.zeros((1, max_seq_len, text_dim), dtype=dtype)
    par_suffix = f"_rank{tp_rank}" if tp_size else ""
    out_path = os.path.join(output_dir, f"wan_transformer{par_suffix}.onnx")
    _export_onnx(
        wrapper, (hidden_states, timestep, encoder_hidden_states),
        input_names=["hidden_states", "timestep", "encoder_hidden_states"],
        output_names=["noise_pred"], out_path=out_path,
        # dynamo exporter: produces a fully static graph (0 symbolic dims). The
        # legacy exporter leaves thousands of symbolic dims that onnxsim would
        # fold, but the transformer is >2 GiB so onnxsim cannot run.
        use_dynamo=True,
    )
    del model, wrapper
    _clear_cache()
    injected = os.path.join(output_dir, f"wan_transformer_pfa{par_suffix}.onnx")
    num_heads = _NUM_HEADS // tp_size if tp_size else _NUM_HEADS
    inject_pfa(out_path, injected, num_heads=num_heads,
               inner_precise=pfa_inner_precise, layout=pfa_layout)
    if tp_size is not None and tp_size > 1:
        # PFA replaced the attention bodies in-place; now reduce the row-parallel
        # o_proj / ffn partial sums (bias applied after the reduction).
        inject_allreduce(injected, injected, tp_size)
    os.remove(out_path)   # only the injected graph is used downstream
    return injected


def export_vae(model_dir, output_dir, height, width, num_frames, dtype):
    """Export the Wan VAE decoder to ONNX.

    Videos of at most 13 latent frames decode in a single pass
    (``wan_vae_decoder.onnx``, shape follows ``num_frames``). Longer videos
    would exceed GE tiling memory in one pass, so the VAE is exported once as
    a 13-latent segment decoder (``wan_vae_decoder_13lat.onnx``) and the
    infer script reconstructs the full video from overlapping segments.
    """
    _patch_wan_vae_no_cache()
    _patch_wan_resample_singlepass()
    _patch_wan_vae_attention()
    _patch_wan_upsample()
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_dir, "vae"), torch_dtype=dtype
    ).eval()
    num_latents = _latent_shape(num_frames, height, width)[2]
    if num_latents <= _VAE_SEGMENT_LATENTS:
        export_frames, out_name = num_frames, "wan_vae_decoder.onnx"
    else:
        export_frames, out_name = _VAE_EXPORT_FRAMES, "wan_vae_decoder_13lat.onnx"
    latent_shape = _latent_shape(export_frames, height, width)
    wrapper = _VaeWrapper(vae)
    latents = torch.zeros(latent_shape, dtype=dtype)
    out_path = os.path.join(output_dir, out_name)
    _export_onnx(
        wrapper, (latents,),
        input_names=["latents"], output_names=["video"], out_path=out_path,
        # dynamo exporter: the legacy JIT trace of the full-res 3D causal convs
        # takes ~1h on CPU, while dynamo's torch.export is far faster and also
        # yields a fully static graph.
        use_dynamo=True,
    )
    del vae, wrapper
    _clear_cache()


def _dtype_of(name):
    """Map a dtype name string to a torch dtype."""
    return {"float32": torch.float32, "float16": torch.float16}[name]


def main():
    """Parse arguments and export the requested Wan2.1-T2V-1.3B sub-modules."""
    parser = argparse.ArgumentParser(
        description="Export Wan2.1-T2V-1.3B submodules to ONNX (81-frame official spec)")
    parser.add_argument("--model-dir", required=True,
                        help="Wan2.1-T2V-1.3B-Diffusers weight dir")
    parser.add_argument("--output-dir", default="./wan2_1_t2v_1_3b_onnx",
                        help="ONNX output dir")
    parser.add_argument("--parts", default="text,transformer,vae",
                        help="comma list: text,transformer,vae")
    parser.add_argument("--height", type=int, default=480, help="video height (multiple of 16)")
    parser.add_argument("--width", type=int, default=832, help="video width (multiple of 16)")
    parser.add_argument("--num-frames", type=int, default=81,
                        help="number of video frames (4k+1; official spec 81)")
    parser.add_argument("--max-seq-len", type=int, default=512, help="UMT5 max sequence length")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                        help="export dtype (float32 recommended for converter compatibility)")
    parser.add_argument("--pfa-inner-precise", type=int, default=0, choices=[0, 1, 4],
                        help="PFA compute mode: 0=high precision (fp32 softmax), "
                             "1=high performance, 4=approximate (310P)")
    parser.add_argument("--pfa-layout", default="BNSD", choices=["BSHD", "BNSD"],
                        help="PFA input layout. BNSD is the only layout the 310P "
                             "built-in PFA GE op supports (CANN 9.0.1: 'Invalid "
                             "input layout: BSHD, not support'); BSHD is kept as "
                             "an option for future CANN versions")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="tensor-parallel size (2 = shard the transformer "
                             "into 2 rank graphs for TP2 x CFG 4-chip inference)")
    parser.add_argument("--rank", type=int, default=0,
                        help="rank id to export when --tp-size > 1")
    args = parser.parse_args()

    if args.height % 16 or args.width % 16:
        raise ValueError("height/width must be multiples of 16")
    if (args.num_frames - 1) % 4:
        raise ValueError("num-frames must be 4k+1 (official spec: 81)")
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = _dtype_of(args.dtype)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]

    if "text" in parts:
        print("[export] UMT5 text encoder ...", flush=True)
        export_text_encoder(args.model_dir, args.output_dir, args.max_seq_len, dtype)
    if "transformer" in parts:
        print("[export] Wan transformer (DiT) ...", flush=True)
        export_transformer(args.model_dir, args.output_dir, args.height, args.width,
                           args.num_frames, args.max_seq_len, dtype,
                           pfa_inner_precise=args.pfa_inner_precise,
                           pfa_layout=args.pfa_layout,
                           tp_rank=args.rank if args.tp_size > 1 else None,
                           tp_size=args.tp_size if args.tp_size > 1 else None)
    if "vae" in parts:
        print("[export] Wan VAE decoder ...", flush=True)
        export_vae(args.model_dir, args.output_dir, args.height, args.width,
                   args.num_frames, dtype)

    print("[export] done ->", args.output_dir)


if __name__ == "__main__":
    main()
