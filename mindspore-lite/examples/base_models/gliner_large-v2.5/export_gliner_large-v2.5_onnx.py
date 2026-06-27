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

"""Export gliner_large-v2.5 PyTorch model to ONNX.

The upstream GLiNER package ships an ONNX-friendly ``_SmallOpLSTM`` (no
``nn.LSTM`` / ``pack_padded_sequence``), but that implementation iterates the
sequence with a Python ``for t in range(seq_len)`` loop. JIT tracing unrolls
that loop to the dummy batch's word count, which then becomes a hard cap on
the runtime sequence length: any input longer than the dummy's word count is
silently truncated, and shorter inputs leak stale state across the unused
positions. To get a model that actually honors the dynamic ``sequence_length``
axis, we swap ``_SmallOpLSTM`` for a native ``nn.LSTM`` (C++ implementation,
no Python loop, traces cleanly to a single dynamic-shape ONNX subgraph).

The published ``gliner-community/gliner_large-v2.5`` checkpoint was saved
with native ``nn.LSTM`` weights (``weight_ih_l0`` / ``weight_hh_l0`` etc.),
so swapping in ``nn.LSTM`` also removes the need for any weight-name
remapping — the checkpoint keys line up directly with the new module.

On top of that fix, three export blockers remain:

1. ``UniEncoderSpanModel.forward`` uses ``torch.einsum("BLKD,BCD->BLKC", ...)``
   for span-vs-prompt scoring. Einsum is a common conversion blocker on Ascend,
   so we rewrite it as ``matmul`` + ``reshape`` ahead of time.
2. ``_fit_length`` uses Python ``if target_len == L:`` / ``if target_len > L:``
   to pick between slicing and padding. JIT tracing bakes the branch taken at
   trace time, so when the dummy batch has fewer words than a real batch the
   graph silently picks the wrong branch and the downstream gather fails with
   out-of-range indices. We replace it with a branchless pre-pad + dynamic
   slice that always works regardless of runtime shape.
3. DeBERTa-v2's disentangled attention uses a ``@torch.jit.script``-decorated
   ``make_log_bucket_position``. When traced, that scripted function emits one
   ONNX ``If`` subgraph per attention layer (24 in DeBERTa-large) plus a
   ``Sign`` op — both rejected by MSLite's Ascend converter. We precompute the
   bucketed relative-position matrix once at export time as a plain tensor and
   slice it at runtime, eliminating every ``If``/``Sign`` from the graph.

The script then delegates to ``GLiNER.export_to_onnx`` which already wires up
the dummy batch, the dynamic axes, and the I/O spec.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn
from gliner import GLiNER
from gliner.modeling.base import GLiNERBaseOutput, UniEncoderSpanModel
from gliner.modeling.layers import LstmSeq2SeqEncoder
import gliner.modeling.base as gbase
import gliner.modeling.utils as gutils
import transformers.models.deberta_v2.modeling_deberta_v2 as dv2
from transformers.models.deberta_v2.modeling_deberta_v2 import make_log_bucket_position

DEFAULT_MODEL_DIR = "gliner_large-v2.5"
DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx")

# Idempotency registry: each _patch_* helper checks ``key in _PATCHED`` so
# re-invoking export in the same process is a no-op instead of re-wrapping.
_PATCHED = set()

# Safe upper bound for words_embedding length after _fit_length.
# max_len=768 in gliner_config.json, so target_W <= ~768 + max_width.
_FIT_LENGTH_MAX_PAD = 1024

# Safe upper bound for the relative-position buffer. The gliner_config has
# max_len=768 and the prompt adds ~10 tokens, so 1024 is comfortably larger.
_REL_POS_MAX_SEQ = 1024
# Single-element mutable holder so _patch_relative_position can populate the
# buffer without a ``global`` declaration.
_REL_POS_BUFFER_HOLDER = [None]


def _fit_length_dynamic(embedding, mask, target_len):
    """Branchless, ONNX-friendly replacement for ``UniEncoderSpanModel._fit_length``.

    The stock implementation picks between slicing and padding with a Python
    ``if``, which JIT tracing bakes as a single branch. When the runtime
    ``target_len`` falls in the other branch the graph silently produces the
    wrong shape and the downstream ``GatherElements`` fails.

    We always pre-pad with ``_FIT_LENGTH_MAX_PAD`` zeros along dim=1 and then
    slice to ``target_len``. ``target_len`` is computed at runtime from
    ``span_idx.size(1) // max_width`` and propagates as a dynamic value
    through the ONNX graph, so the slice adjusts correctly for any input.
    """
    b = embedding.size(0)
    d = embedding.size(-1)
    zeros_emb = torch.zeros(b, _FIT_LENGTH_MAX_PAD, d, dtype=embedding.dtype, device=embedding.device)
    emb_padded = torch.cat([embedding, zeros_emb], dim=1)
    zeros_mask = torch.zeros(b, _FIT_LENGTH_MAX_PAD, dtype=mask.dtype, device=mask.device)
    mask_padded = torch.cat([mask, zeros_mask], dim=1)
    return emb_padded[:, :target_len], mask_padded[:, :target_len]


def _patch_lstm_seq2seq_forward() -> int:
    """Monkey-patch ``LstmSeq2SeqEncoder.forward`` to call native ``nn.LSTM``.

    The stock implementation passes ``lengths=lengths`` to ``self.lstm`` (which
    is a ``_SmallOpLSTM`` accepting that kwarg) and then slices
    ``output[:, :max_len]`` with ``max_len = int(lengths.max().item())``. After
    ``_swap_in_native_lstm`` swaps in a native ``nn.LSTM`` (which has no
    ``lengths`` parameter), we must drop the kwarg; and the
    ``int(...max().item())`` slice must be replaced with ``x.size(1)`` so JIT
    tracing keeps the sequence dimension symbolic instead of baking the dummy
    batch's max word count as a constant (which would silently truncate
    longer real inputs at inference time).

    We also pre-allocate ``h0`` / ``c0`` as static zero tensors sized to the
    native ``nn.LSTM``'s expected ``(num_layers * num_directions, batch=1,
    hidden_size)`` shape. With ``hidden=None`` PyTorch traces the zero init as
    a ``ConstantOfShape`` whose shape is itself derived from the LSTM input
    via ``Shape``/``Gather``/``Concat`` — and Ascend's multi-batch compiler
    rejects the resulting 4-dynamic-dim LSTM output. Pre-allocated constant
    initial states eliminate that shape-construction chain entirely.
    """
    if "lstm_forward" in _PATCHED:
        return 0

    def patched_forward(self, x, mask, hidden=None, lengths=None):
        del mask, lengths  # unused; native nn.LSTM doesn't accept lengths
        if hidden is None:
            lstm = self.lstm
            n_dirs = 2 if lstm.bidirectional else 1
            n_layers = lstm.num_layers
            h = torch.zeros(n_layers * n_dirs, x.size(0), lstm.hidden_size,
                            dtype=x.dtype, device=x.device)
            c = torch.zeros(n_layers * n_dirs, x.size(0), lstm.hidden_size,
                            dtype=x.dtype, device=x.device)
            hidden = (h, c)
        output, _ = self.lstm(x, hidden)
        return output[:, : x.size(1)]

    LstmSeq2SeqEncoder.forward = patched_forward
    _PATCHED.add("lstm_forward")
    return 1


def _precompute_rel_pos_buffer(bucket_size: int, max_position: int) -> torch.Tensor:
    """Precompute ``make_log_bucket_position`` output for all rel_pos in [-N+1, N-1].

    DeBERTa-v2's disentangled attention uses a JIT-scripted
    ``make_log_bucket_position`` that, when traced, emits one ONNX ``If``
    subgraph per attention layer (24 in DeBERTa-large) plus a ``Sign`` op.
    MSLite's Ascend converter rejects both. We sidestep this entirely by
    computing the bucketed relative-position matrix ONCE at export time as a
    plain tensor — the bucketing depends only on ``q - k`` and the layer's
    bucket_size / max_position, not on token content — and then slicing the
    cached buffer at runtime to the actual ``[query_size, key_size]`` shape.
    """
    n = _REL_POS_MAX_SEQ
    q_ids = torch.arange(n, dtype=torch.long)
    k_ids = torch.arange(n, dtype=torch.long)
    rel_pos = q_ids[:, None] - k_ids[None, :]
    return make_log_bucket_position(rel_pos, bucket_size, max_position).to(torch.long)


def _find_disentangled_attention(model):
    """Return the first ``DisentangledSelfAttention`` module, or None."""
    for module in model.modules():
        if module.__class__.__name__ == "DisentangledSelfAttention":
            return module
    return None


def _patch_relative_position(attn_module) -> int:
    """Replace ``build_relative_position`` with a slice into a precomputed buffer."""
    if "rel_pos" in _PATCHED:
        return 0
    if attn_module is None:
        return 0

    _REL_POS_BUFFER_HOLDER[0] = _precompute_rel_pos_buffer(
        attn_module.position_buckets, attn_module.max_relative_positions
    )

    def patched_build_relative_position(query_layer, key_layer, bucket_size=-1, max_position=-1):
        del bucket_size, max_position  # already baked into the buffer
        query_size = query_layer.size(-2)
        key_size = key_layer.size(-2)
        return _REL_POS_BUFFER_HOLDER[0][:query_size, :key_size].unsqueeze(0).to(query_layer.device)

    dv2.build_relative_position = patched_build_relative_position
    _PATCHED.add("rel_pos")
    return 1


def _patch_build_rpos() -> int:
    """Replace JIT-scripted ``build_rpos`` with an identity (key_size == query_size in self-attn)."""
    if "build_rpos" in _PATCHED:
        return 0

    def patched_build_rpos(query_layer, key_layer, relative_pos, position_buckets=-1, max_relative_positions=-1):
        del query_layer, key_layer, position_buckets, max_relative_positions
        return relative_pos

    dv2.build_rpos = patched_build_rpos
    _PATCHED.add("build_rpos")
    return 1


def _patch_transpose_for_scores(attn_module) -> int:
    """Replace ``transpose_for_scores`` with a Python-int-only reshape."""
    if "transpose_for_scores" in _PATCHED:
        return 0
    if attn_module is None:
        return 0
    heads = attn_module.num_attention_heads
    hidden_size = getattr(attn_module, "all_head_size", None)
    if hidden_size is None:
        hidden_size = getattr(getattr(attn_module, "query_proj", None), "in_features", None)
    if hidden_size is None:
        return 0
    head_dim_py = hidden_size // heads

    def patched_transpose_for_scores(self, x, attention_heads):
        del self, attention_heads  # captured at patch time; matches heads_py
        # (B, S, hidden) → (B, S, heads, head_dim); -1 absorbs S.
        x = x.view(-1, x.size(1), heads, head_dim_py)
        x = x.permute(0, 2, 1, 3).contiguous()
        # (B, heads, S, head_dim) → (-1, S, head_dim) flattens B*heads.
        return x.view(-1, x.size(2), head_dim_py)

    dv2.DisentangledSelfAttention.transpose_for_scores = patched_transpose_for_scores
    _PATCHED.add("transpose_for_scores")
    return 1


def _patched_extract_prompt_features(class_token_index, token_embeds, input_ids, attention_mask,
                                     batch_size, embed_dim, embed_ent_token=True):
    """Force ``.max()`` scalars to rank-0 via ``.reshape(())`` so downstream Concat infers."""
    class_token_mask = input_ids == class_token_index
    num_class_tokens = torch.sum(class_token_mask, dim=-1, keepdim=True)
    max_embed_dim = num_class_tokens.max().reshape(())
    aranged_class_idx = torch.arange(max_embed_dim, dtype=attention_mask.dtype, device=token_embeds.device).expand(
        batch_size, -1
    )
    batch_indices, target_class_idx = torch.where(aranged_class_idx < num_class_tokens)
    _, class_indices = torch.where(class_token_mask)
    if not embed_ent_token:
        class_indices = class_indices + 1
    prompts_embedding = torch.zeros(
        batch_size, max_embed_dim, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device
    )
    prompts_embedding[batch_indices, target_class_idx] = token_embeds[batch_indices, class_indices]
    prompts_embedding_mask = (aranged_class_idx < num_class_tokens).to(attention_mask.dtype)
    return prompts_embedding, prompts_embedding_mask


def _patched_extract_prompt_features_and_word_embeddings(class_token_index, token_embeds, input_ids,
                                                         attention_mask, text_lengths, words_mask,
                                                         embed_ent_token=True, **kwargs):
    """Same rank-0 fix applied to ``extract_prompt_features_and_word_embeddings``."""
    del kwargs
    batch_size, _, embed_dim = token_embeds.shape
    max_text_length = text_lengths.max().reshape(())
    prompts_embedding, prompts_embedding_mask = _patched_extract_prompt_features(
        class_token_index, token_embeds, input_ids, attention_mask, batch_size, embed_dim, embed_ent_token
    )
    words_embedding, mask = gutils.extract_word_embeddings(
        token_embeds, words_mask, attention_mask, batch_size, max_text_length, embed_dim, text_lengths
    )
    return prompts_embedding, prompts_embedding_mask, words_embedding, mask


def _patch_extract_prompt_features() -> int:
    """Patch prompt-feature extractors in both ``gliner.modeling.utils`` and ``base``."""
    if "extract_prompt_features" in _PATCHED:
        return 0

    gutils.extract_prompt_features = _patched_extract_prompt_features
    gutils.extract_prompt_features_and_word_embeddings = _patched_extract_prompt_features_and_word_embeddings
    gbase.extract_prompt_features = _patched_extract_prompt_features
    gbase.extract_prompt_features_and_word_embeddings = _patched_extract_prompt_features_and_word_embeddings
    _PATCHED.add("extract_prompt_features")
    return 1


def _patch_deberta_build_relative_position(model) -> int:
    """Apply all four DeBERTa-side patches needed for Ascend conversion.

    Patches:
    1. ``build_relative_position`` → slice into precomputed bucket buffer (Sign op)
    2. ``build_rpos`` → identity (If/Range ops in JIT-scripted branch)
    3. ``transpose_for_scores`` → Python-int-only reshape (Concat rank mismatch)
    4. ``extract_prompt_features`` → ``.max().reshape(())`` for rank-0 scalars
    """
    attn_module = _find_disentangled_attention(model)
    n1 = _patch_relative_position(attn_module)
    n2 = _patch_build_rpos()
    n3 = _patch_transpose_for_scores(attn_module)
    n4 = _patch_extract_prompt_features()
    return n1 + n2 + n3 + n4


def _patch_uni_encoder_forward() -> int:
    """Monkey-patch ``UniEncoderSpanModel.forward`` for clean ONNX export.

    Two changes vs. upstream:
    - Replace ``einsum("BLKD,BCD->BLKC", ...)`` with ``matmul`` after a
      reshape/transpose. Einsum is a common Ascend conversion blocker.
    - Replace ``_fit_length`` with ``_fit_length_dynamic`` so the model
      handles any runtime ``span_idx`` shape (see its docstring for why).
    """
    if "uni_encoder_forward" in _PATCHED:
        return 0

    def patched_forward(self, input_ids=None, attention_mask=None, words_embedding=None,
                        mask=None, prompts_embedding=None, prompts_embedding_mask=None,
                        words_mask=None, text_lengths=None, span_idx=None, span_mask=None,
                        labels=None, **kwargs):
        del words_embedding, mask, prompts_embedding, prompts_embedding_mask, labels, kwargs

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_representations(
            input_ids, attention_mask, text_lengths, words_mask
        )

        target_w = span_idx.size(1) // self.config.max_width
        words_embedding, mask = _fit_length_dynamic(words_embedding, mask, target_w)

        span_idx = span_idx * span_mask.unsqueeze(-1)
        span_rep = self.span_rep_layer(words_embedding, span_idx)

        # During inference labels is None, so target_c == prompts_embedding.size(1)
        # and _fit_length is a no-op. Skip the call entirely to avoid baking a
        # Python int into the graph.
        prompts_embedding = self.prompt_rep_layer(prompts_embedding)

        b, length, k, d = span_rep.shape
        c = prompts_embedding.size(1)
        span_rep_flat = span_rep.reshape(b, length * k, d)
        prompts_t = prompts_embedding.transpose(1, 2)
        scores = torch.matmul(span_rep_flat, prompts_t).reshape(b, length, k, c)

        return GLiNERBaseOutput(
            logits=scores,
            loss=None,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            words_embedding=words_embedding,
            mask=mask,
        )

    UniEncoderSpanModel.forward = patched_forward
    _PATCHED.add("uni_encoder_forward")
    return 1


def _swap_in_native_lstm(model, checkpoint_dir) -> int:
    """Replace ``_SmallOpLSTM`` with a native ``nn.LSTM`` loaded from the checkpoint.

    The upstream ``LstmSeq2SeqEncoder`` ships with ``_SmallOpLSTM`` — a Python
    loop implementation of the BiLSTM. JIT tracing unrolls that loop to the
    dummy batch's word count, which then becomes a hard cap on the runtime
    sequence length (longer inputs are silently truncated at inference time).
    A native ``nn.LSTM`` has no Python loop and traces cleanly to a single
    dynamic-shape ONNX subgraph.

    The published ``gliner_large-v2.5`` checkpoint was saved with native
    ``nn.LSTM`` weights (``weight_ih_l0`` / ``weight_hh_l0`` etc.), so we can
    construct an ``nn.LSTM`` with the same hyperparameters as the original
    ``_SmallOpLSTM``, load the checkpoint keys directly (no name remapping),
    and substitute it in place via ``rnn.lstm = nn_lstm``.

    Numerically this is equivalent to ``_SmallOpLSTM`` at runtime because the
    collator pads every batch to its own ``lengths.max()``, so
    ``x.size(1) == lengths.max()`` and there is no real padding in the input —
    every position is active, which is exactly the case where the masked
    ``_SmallOpLSTM`` and the unmasked native ``nn.LSTM`` agree.
    """
    rnn = getattr(model.model, "rnn", None)
    if rnn is None or not hasattr(rnn, "lstm"):
        return 0
    small_lstm = rnn.lstm
    if isinstance(small_lstm, nn.LSTM):
        return 0  # already swapped (e.g. by a previous call in the same process)

    state_path = _find_checkpoint(checkpoint_dir)
    if state_path is None:
        return 0
    state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
    nn_keys = [k for k in state_dict if k.startswith("rnn.lstm.") and "_l" in k]
    if not nn_keys:
        return 0

    nn_lstm = nn.LSTM(
        input_size=small_lstm.input_size,
        hidden_size=small_lstm.hidden_size,
        num_layers=small_lstm.num_layers,
        bidirectional=small_lstm.bidirectional,
        batch_first=True,
    )
    nn_state = {k[len("rnn.lstm."):]: state_dict[k] for k in nn_keys}
    nn_lstm.load_state_dict(nn_state, strict=True)
    nn_lstm.eval()
    rnn.lstm = nn_lstm  # in-place swap; _SmallOpLSTM no longer reachable
    return len(nn_state)


def _find_checkpoint(checkpoint_dir) -> Path:
    """Return the .bin/.safetensors checkpoint path inside ``checkpoint_dir``."""
    for name in ("pytorch_model.bin", "model.safetensors"):
        path = Path(checkpoint_dir) / name
        if path.exists():
            return path
    return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export gliner_large-v2.5 to ONNX")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Path to gliner checkpoint")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="Output directory for ONNX")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


def main() -> None:
    """Load GLiNER, swap in native nn.LSTM, patch einsum/_fit_length, export to ONNX."""
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] loading GLiNER from {args.model_dir}", flush=True)
    model = GLiNER.from_pretrained(args.model_dir, load_tokenizer=True)
    model.eval()

    n_lstm_weights = _swap_in_native_lstm(model, args.model_dir)
    print(f"[export] swapped in native nn.LSTM (loaded {n_lstm_weights} tensors)", flush=True)

    n_patched = _patch_uni_encoder_forward()
    print(f"[export] patched UniEncoderSpanModel.forward (matmul + dynamic fit_length): {n_patched}",
          flush=True)

    n_lstm_patched = _patch_lstm_seq2seq_forward()
    print(f"[export] patched LstmSeq2SeqEncoder.forward (native nn.LSTM call, symbolic max_len): "
          f"{n_lstm_patched}", flush=True)

    n_rel_pos_patched = _patch_deberta_build_relative_position(model)
    print(f"[export] patched DeBERTa build_relative_position (precomputed bucket buffer): "
          f"{n_rel_pos_patched}", flush=True)

    print(f"[export] exporting ONNX to {save_dir}/model.onnx (opset={args.opset})", flush=True)
    model.export_to_onnx(
        save_dir=save_dir,
        onnx_filename="model.onnx",
        opset=args.opset,
    )
    print("[export] done. files:", sorted(os.listdir(save_dir)), flush=True)


if __name__ == "__main__":
    sys.exit(main())
