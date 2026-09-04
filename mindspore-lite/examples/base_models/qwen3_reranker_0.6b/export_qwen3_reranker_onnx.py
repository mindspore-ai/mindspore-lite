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
Export Qwen3-Reranker-0.6B with PromptFlashAttention and RotaryMul.

The exported model computes only the final token's yes/no logits. The
reference forward is manually unrolled so the legacy exporter can emit CANN
Custom operators without entering the Transformers masking implementation.
"""

import argparse
import os

import torch
import onnx
from transformers import AutoModelForCausalLM, AutoTokenizer

# Final export fusion helpers.

def _make_output_shapes(val, mask_dims=None):
    """Build output_shapes string for ONNX Custom op symbolic.

    mask_dims: list of dimension indices to replace with -1 (dynamic).
    """
    sizes = val.type().sizes()
    if sizes is None:
        return ""
    dims = [int(d) if d is not None else -1 for d in list(sizes)]
    if mask_dims:
        for idx in mask_dims:
            if idx < len(dims):
                dims[idx] = -1
    return ",".join([str(len(dims))] + [str(i) for i in dims])


class _CannRotaryMul(torch.autograd.Function):
    """rotate_half + cos/sin multiply -> Custom(RotaryMul)."""

    @staticmethod
    def forward(ctx, x, r1, r2):
        del ctx
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = torch.cat([-x2, x1], dim=-1)
        y = x * r1 + rotated * r2
        return y

    @staticmethod
    def symbolic(g, x, r1, r2):
        """Emit ONNX Custom node mapping to the CANN RotaryMul op (rotate_half + cos/sin mul)."""
        out_shapes = _make_output_shapes(x, [0, 2])

        y = g.op(
            "Custom",
            x,
            r1,
            r2,
            type_s="RotaryMul",
            input_names_s=["x", "r1", "r2"],
            optional_input_names_s=[],
            output_names_s=["y"],
            output_num_i=1,
            input_index_i=[0, 1, 2],
            output_shapes_s=out_shapes,
        )
        y.setType(x.type())
        return y


class _CannPromptFlashAttention(torch.autograd.Function):
    """QK^T + softmax + V -> Custom(PromptFlashAttention)."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, num_key_value_heads, scale_value):
        """PyTorch reference impl (traced for ONNX export numerics).

        Returns [B, S, N, D] (transposed) to match CANN PFA op output layout.
        """
        del ctx
        if num_key_value_heads < num_heads:
            key = key.repeat_interleave(num_heads // num_key_value_heads, dim=1)
            value = value.repeat_interleave(num_heads // num_key_value_heads, dim=1)
        scale = float(scale_value)
        attn = torch.matmul(query, key.transpose(2, 3)) * scale
        if atten_mask is not None:
            attn = attn + atten_mask
        attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn, value)
        return attn_output

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, num_key_value_heads, scale_value):
        """Emit ONNX Custom node mapping to the CANN PromptFlashAttention op."""
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
            input_index_i=[0, 1, 2, 3],
            num_heads_i=int(num_heads),
            num_key_value_heads_i=int(num_key_value_heads),
            scale_value_f=float(scale_value),
            input_layout_s="BNSD",
            inner_precise_i=0,
        )
        y.setType(query.type())
        return y


def _linear(linear_mod, x):
    out = torch.matmul(x, linear_mod.weight.t())
    if linear_mod.bias is not None:
        out = out + linear_mod.bias
    return out


def _qwen3_rotary_emb_matmul2d(rotary_emb, x, position_ids):
    """Qwen3 rotary cos/sin with matmul-friendly tensor shapes (matches jina_v3)."""
    rope_type = getattr(rotary_emb, "rope_type", "") or getattr(
        rotary_emb, "config", None
    ) and getattr(rotary_emb.config, "rope_type", "")
    if rope_type == "dynamic":
        seq_len = torch.max(position_ids) + 1
        if seq_len > rotary_emb.max_seq_len_cached:
            inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
                rotary_emb.config, x.device, seq_len=seq_len
            )
            rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
            rotary_emb.max_seq_len_cached = seq_len
        if (
            seq_len < rotary_emb.original_max_seq_len
            and rotary_emb.max_seq_len_cached > rotary_emb.original_max_seq_len
        ):
            rotary_emb.original_inv_freq = rotary_emb.original_inv_freq.to(x.device)
            rotary_emb.register_buffer(
                "inv_freq", rotary_emb.original_inv_freq, persistent=False
            )
            rotary_emb.max_seq_len_cached = rotary_emb.original_max_seq_len

    inv_freq = rotary_emb.inv_freq.to(device=x.device, dtype=torch.float32)
    position_ids_f = position_ids.to(dtype=torch.float32)
    bsz, seq_len = position_ids_f.shape
    freqs = (position_ids_f.reshape(-1, 1) @ inv_freq.reshape(1, -1)).reshape(
        bsz, seq_len, -1
    )
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * rotary_emb.attention_scaling
    sin = emb.sin() * rotary_emb.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _make_bool_causal_mask(attention_mask, q_len, k_len, past_len):
    """Boolean causal mask (True=masked) for CANN PromptFlashAttention.

    Uses arithmetic ops (Sub, Add) instead of logical (Not, Or) for ONNX→Ascend
    ACL compatibility. Logical ops cause graph partition failures.
    """
    device = attention_mask.device
    ar_q = torch.arange(q_len, device=device)
    ar_k = torch.arange(k_len, device=device)
    causal = (ar_k[None, :] > (past_len + ar_q[:, None]))  # [q_len, k_len]
    # padding: 1 where masked (pad positions), 0 where real
    padding = 1.0 - attention_mask.to(torch.float)  # Sub + Cast (avoid Not)
    # mask: causal OR padding → use arithmetic: (causal.float() + padding) > 0
    mask_val = causal.to(torch.float)[None, None, :, :] + padding[:, None, None, :]  # Add (avoid Or)
    mask = mask_val > 0.5  # Greater → bool
    mask = mask.expand(attention_mask.shape[0], 1, q_len, k_len)
    return mask


def _cann_attn_forward(
    attn_mod,
    hidden_states,
    position_embeddings,
    bool_mask,
    past_key=None,
    past_value=None,
):
    """Attention forward using RotaryMul and PromptFlashAttention Custom ops."""
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = _linear(attn_mod.q_proj, hidden_states).view(
        hidden_shape
    )
    key_states = _linear(attn_mod.k_proj, hidden_states).view(
        hidden_shape
    )
    if hasattr(attn_mod, "q_norm"):
        query_states = attn_mod.q_norm(query_states)
    if hasattr(attn_mod, "k_norm"):
        key_states = attn_mod.k_norm(key_states)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = (
        _linear(attn_mod.v_proj, hidden_states)
        .view(hidden_shape)
        .transpose(1, 2)
    )

    cos, sin = position_embeddings
    while cos.dim() < 4:
        cos = cos.unsqueeze(1)
    while sin.dim() < 4:
        sin = sin.unsqueeze(1)
    query_states = _CannRotaryMul.apply(query_states, cos, sin)
    key_states = _CannRotaryMul.apply(key_states, cos, sin)

    if past_key is not None:
        key_states = torch.cat([past_key, key_states], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)

    scaling = getattr(attn_mod, "scaling", 1.0 / (head_dim**0.5))
    attn_output = _CannPromptFlashAttention.apply(
        query_states, key_states, value_states, bool_mask,
        int(num_heads), int(num_kv_heads), float(scaling),
    )

    # Both paths return [B,N,S,D], need transpose to [B,S,N,D]
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = _linear(attn_mod.o_proj, attn_output)
    return attn_output, key_states, value_states


def _cann_mlp_forward(mlp_mod, hidden_states):
    gate = _linear(mlp_mod.gate_proj, hidden_states)
    up = _linear(mlp_mod.up_proj, hidden_states)
    return _linear(mlp_mod.down_proj, torch.nn.functional.silu(gate) * up)


def _cann_add_rms_norm(residual, hidden_states, norm_mod):
    x = residual + hidden_states
    y = norm_mod(x)
    return y, x


class Qwen3RerankerFused(torch.nn.Module):
    """Qwen3-Reranker with CANN fused Custom ops (PFA + RotaryMul).

    Manually unrolls the Qwen3Model forward so attention/RoPE route through
    CANN Custom operators. Keeps slice_last (last-token hidden state -> lm_head)
    so the output is [batch, 1, vocab], matching the baseline reranker I/O.
    """

    def __init__(self, model, slice_last=True, slice_lm_head=True,
                 yes_id=None, no_id=None):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers
        self.norm = model.model.norm
        self.rotary_emb = model.model.rotary_emb
        self.num_layers = len(self.layers)
        self.slice_last = slice_last

        # lm_head weight slice: reranker only consumes logits[yes_id] and
        # logits[no_id]. With tie_word_embeddings=True, lm_head weight equals
        # embed_tokens.weight ([vocab, hidden]). Slice it to the [yes, no] two
        # rows so lm_head becomes [1,1,hidden] x [hidden,2] -> [1,1,2]:
        #   - compute: 155.3M MAC -> 2K MAC (~76000x)
        #   - weight read: ~311MB(fp16) -> 4KB (bandwidth-bound GEMV, the
        #     lm_head is the largest single MatMul per forward at M=1)
        #   - D2H: ~607KB -> 8B
        # Bit-identical to full lm_head then index those 2 logits (same weight
        # rows, same fp16 accumulation). embed_tokens stays full-vocab for input
        # embedding; the 2-row head is a clone, decoupled from the tie.
        # Output convention: row 0 = yes, row 1 = no (infer reads positionally).
        self.slice_lm_head = slice_lm_head
        if slice_lm_head:
            if yes_id is None or no_id is None:
                raise ValueError(
                    "slice_lm_head=True requires yes_id and no_id from the "
                    "tokenizer (tokenizer.convert_tokens_to_ids('yes'/'no'))."
                )
            hidden = model.config.hidden_size
            two_row = (
                model.model.embed_tokens.weight[[yes_id, no_id], :]
                .clone()
                .to(torch.float32)
            )
            self.lm_head = torch.nn.Linear(hidden, 2, bias=False)
            with torch.no_grad():
                self.lm_head.weight.copy_(two_row)
            self._head_yes_pos = 0
            self._head_no_pos = 1
        else:
            self.lm_head = model.lm_head

    def forward(self, input_ids, attention_mask):
        """Fused forward: embeddings -> layers (PFA) -> norm -> slice_last -> lm_head."""
        seq_len = input_ids.shape[1]

        # position_ids from attention_mask (left-padding safe): real tokens get
        # 0,1,2,...; pad positions get 0.
        position_ids = attention_mask.to(torch.long).cumsum(dim=1) - 1
        position_ids = torch.where(
            attention_mask.to(torch.bool),
            position_ids,
            torch.zeros_like(position_ids),
        )

        inputs_embeds = self.embed_tokens(input_ids)
        position_embeddings = _qwen3_rotary_emb_matmul2d(
            self.rotary_emb, inputs_embeds, position_ids
        )

        # Boolean causal mask (True=masked), built with arithmetic ops for
        # ONNX->Ascend ACL compatibility (logical ops cause graph partition
        # failures). Handles padding via attention_mask.
        bool_mask = _make_bool_causal_mask(attention_mask, seq_len, seq_len, 0)

        hidden_states = inputs_embeds
        residual = hidden_states
        hidden_states = self.layers[0].input_layernorm(hidden_states)

        for i, layer in enumerate(self.layers):
            attn_out, _, _ = _cann_attn_forward(
                layer.self_attn,
                hidden_states,
                position_embeddings,
                bool_mask,
                None,
                None,
            )
            hidden_states, residual = _cann_add_rms_norm(
                residual,
                attn_out,
                layer.post_attention_layernorm,
            )

            mlp_out = _cann_mlp_forward(
                layer.mlp,
                hidden_states,
            )
            if i < self.num_layers - 1:
                hidden_states, residual = _cann_add_rms_norm(
                    residual,
                    mlp_out,
                    self.layers[i + 1].input_layernorm,
                )
            else:
                hidden_states, _ = _cann_add_rms_norm(
                    residual, mlp_out, self.norm
                )

        if self.slice_last:
            # left padding: last real token is at the rightmost position.
            hidden_states = hidden_states[:, -1:, :]
        logits = self.lm_head(hidden_states)
        return logits


def _parse_args():
    """
    Parse command-line arguments for the export script.
    """
    parser = argparse.ArgumentParser(
        description="Export Qwen3-Reranker-0.6B (fused PFA+RotaryMul) to ONNX"
    )
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--output-dir", type=str, default="./onnx")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--no-slice-last",
        action="store_true",
        help="Disable slice_last (output full [batch, seq, vocab]).",
    )
    parser.add_argument(
        "--no-slice-lm-head",
        action="store_true",
        help=(
            "Disable lm_head weight slice. By default the lm_head weight is "
            "sliced to the [yes, no] two rows (output [batch,1,2]); pass this "
            "to keep the full-vocab lm_head (output [batch,1,vocab])."
        ),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="qwen3_reranker_0.6b.onnx",
        help="Output ONNX file name.",
    )
    return parser.parse_args()


def _load_model_and_tokenizer(args):
    print(f"Loading model from {args.model_id}")
    # float32: MindSpore Lite converter rejects FLOAT16 type decls (data_type:10).
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float32, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    return model, tokenizer


def _create_dummy_inputs(args, tokenizer):
    """Dummy input padded to a 128-multiple (smallest gear) for clean tracing."""
    prefix = (
        "<|im_start|>system\nJudge whether the Document meets "
        "the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    dummy_text = "<Instruct>: test\n<Query>: test\n<Document>: test"
    dummy_ids = prefix_tokens + tokenizer.encode(dummy_text, add_special_tokens=False) + suffix_tokens

    # pad dummy to a 128-multiple (aligned with converter ge.dynamicDims gears,
    # all of which are 128-multiples) so PFA traces on an aligned seq length.
    gear = 128
    pad_to = max(gear, ((len(dummy_ids) + gear - 1) // gear) * gear)
    pad_to = min(pad_to, args.max_length)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    # left pad to keep last real token at rightmost (matches infer-time padding).
    dummy_ids = [pad_id] * (pad_to - len(dummy_ids)) + dummy_ids

    dummy_input_ids = torch.tensor([dummy_ids], dtype=torch.long)
    dummy_attention_mask = torch.ones_like(dummy_input_ids)
    # left-padded zeros in attention_mask
    pad_len = pad_to - (
        len(prefix_tokens)
        + len(tokenizer.encode(dummy_text, add_special_tokens=False))
        + len(suffix_tokens)
    )
    if pad_len > 0:
        dummy_attention_mask[0, :pad_len] = 0
    return dummy_input_ids, dummy_attention_mask


def _remove_isnan_nodes(onnx_model):
    """Replace Where(IsNaN(x), default, x) with Identity(x); drop IsNaN nodes."""
    isnan_nodes = [n for n in onnx_model.graph.node if n.op_type == "IsNaN"]
    if not isnan_nodes:
        print("No IsNaN nodes found in the model")
        return onnx_model
    print(f"Found {len(isnan_nodes)} IsNaN nodes, replacing with Identity...")
    remove = set()
    add = []
    for isnan_node in isnan_nodes:
        isnan_output = isnan_node.output[0]
        for node in onnx_model.graph.node:
            if node.op_type == "Where" and isnan_output in node.input:
                if len(node.input) == 3 and node.input[0] == isnan_output:
                    identity = onnx.helper.make_node(
                        "Identity",
                        inputs=[node.input[2]],
                        outputs=[node.output[0]],
                        name=node.name + "_identity",
                    )
                    add.append(identity)
                    remove.add(node.name)
        remove.add(isnan_node.name)
    new_nodes = [n for n in onnx_model.graph.node if n.name not in remove]
    new_nodes.extend(add)
    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(new_nodes)
    print(f"Removed {len(isnan_nodes)} IsNaN nodes")
    return onnx_model


def _export_to_onnx(model, output_path, dummy_input_ids, dummy_attention_mask):
    """
    Export the wrapped model to ONNX with dynamic axes on seq_len.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size"},  # slice_last -> [batch, 1, vocab]
    }
    print(f"Exporting fused model to {output_path} (dynamo=False, opset 17)")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    print("Export done; removing IsNaN nodes & rewriting external data...")
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx_model = _remove_isnan_nodes(onnx_model)
    data_path = output_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)
    onnx.save_model(
        onnx_model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(output_path) + ".data",
        size_threshold=1024,
        convert_attribute=True,
    )
    print(f"Fused model saved to {output_path}")


def _print_custom_stats(onnx_path):
    """
    Print Custom op counts and summary stats for the exported ONNX graph.
    """
    m = onnx.load(onnx_path, load_external_data=False)
    from collections import Counter
    ops = Counter(n.op_type for n in m.graph.node)
    custom = ops.get("Custom", 0)
    if custom:
        type_cnts = Counter()
        for n in m.graph.node:
            if n.op_type != "Custom":
                continue
            for a in n.attribute:
                if a.name == "type":
                    type_cnts[a.s.decode("utf-8")] += 1
                    break
        print(f"Custom op counts: {dict(type_cnts)}")
    else:
        print("No Custom ops found")
    print(f"Softmax: {ops.get('Softmax', 0)}  MatMul: {ops.get('MatMul', 0)}  IsNaN: {ops.get('IsNaN', 0)}")


def main():
    args = _parse_args()
    slice_last = not args.no_slice_last
    slice_lm_head = not args.no_slice_lm_head
    model, tokenizer = _load_model_and_tokenizer(args)
    # yes/no token ids for the 2-row lm_head slice (reranker scoring target).
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    print(f"lm_head slice: yes_id={yes_id}, no_id={no_id}, enabled={slice_lm_head}")
    wrapper = Qwen3RerankerFused(
        model,
        slice_last=slice_last,
        slice_lm_head=slice_lm_head,
        yes_id=yes_id,
        no_id=no_id,
    ).to(args.device).eval()
    dummy_input_ids, dummy_attention_mask = _create_dummy_inputs(args, tokenizer)
    print(
        f"dummy input_ids shape: {tuple(dummy_input_ids.shape)}, "
        f"slice_last={slice_last}, slice_lm_head={slice_lm_head}"
    )

    output_path = os.path.join(args.output_dir, args.output_name)
    _export_to_onnx(wrapper, output_path, dummy_input_ids, dummy_attention_mask)
    _print_custom_stats(output_path)
    print("\nFused export completed successfully!")
    print(f"ONNX model saved to: {output_path}")


if __name__ == "__main__":
    main()
