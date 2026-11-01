# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Qwen2.5-0.5B ONNX exporter for the mslite-llm NNRT (Kirin NPU) runtime.

Ported from ``mindspore-lite-llm/llm_converter/qwen2.5/qwen_2_5_exporter.py``
and adapted to the graph I/O contract consumed by the mslite-llm NNRT executor
(``lite_llm/src/executor/nnrt/nnrt_executor.cc``):

    inputs:  [valid_seq_len, lmhead_idx, rope_cos, rope_sin, inputs_embeds,
              attention_mask, embedding_weight] + past_key_i/past_val_i (per layer)
    outputs: [logits]   (past_key_i/past_val_i are writable inputs updated in place)

Contract differences vs the reference exporter:

* ``kvcache_mask`` input removed.  KV update happens inside the fused
  ``MsTransRopeScatterNDUpdate`` operator at ``pos=valid_seq_len``.
* ``embedding_weight`` (tied lm_head weight) is inserted as a graph input at
  index 6 by ``apply_shared_weight`` (see ``utils/quant/quantize.py``), which
  matches the executor's 7-input shape check.
* Fused inference ops (same contract as Qwen3): ``MsTransRopeScatterNDUpdate``
  performs BSND->BNSD, RoPE and the in-place packed-NZ cache update and returns
  only ``rope_q``; ``MsAttention`` fuses scores + mask + softmax + value.  The
  graph exposes ``logits`` as its sole output and keeps ``past_key_i`` /
  ``past_val_i`` as writable inputs.  ``MsRmsNorm`` / ``MsAddRmsNorm`` (post-
  export fusion) and the quantization ops are unchanged.

External step (not part of this package): compile the exported ONNX to a
Kirin ``.omc`` offline model, e.g. with ``omg --model=... --framework=5
--target=omc --platform=kirinx90 --dynamic_dims="128;1"`` (see
``export_omc_chunked.py`` in the sibling converter for the exact
``--input_shape`` string).
"""

import json
import logging
import os
from typing import Optional

import numpy as np
import onnx
import torch
from onnxslim import slim
from torch.onnx import OperatorExportTypes
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from utils.onnx_postprocess import (
    _save_onnx,
    duplicate_shared_initializers,
    fuse_add_rmsnorm,
    validate_contract,
)
from utils.export_quant import LiteTurboConfig, ModelConfig, QuantizationConfig
from utils.export_quant import (
    apply_quant,
    apply_shared_weight,
    EmbeddingAsset,
)

from utils.quantization import QuantType

from .qwen2_5_wrapper import Qwen2NnrtWrapper, resolve_head_dim

logger = logging.getLogger(__name__)

DEVICE = "cpu"

# MsAttention cache/tile alignment (same values as ``qwen3_exporter.py:75-76``).
ATTENTION_CACHE_ALIGNMENT = 64
ATTENTION_TILE_ALIGNMENT = 16
dtype = torch.float16

# ─────────────────────────────────────────────────────────────────────────────
# Part 3: exporter
# ─────────────────────────────────────────────────────────────────────────────


class Qwen2Onnx:
    """Export a Qwen2.5-0.5B HF model to the mslite-llm NNRT ONNX contract."""

    #: Only the Qwen2.5-0.5B architecture is validated (matches the NNRT gear
    #: shapes and the proven omc contract).
    REQUIRED_ARCH = {
        "num_hidden_layers": 24,
        "intermediate_size": 4864,
        "max_position_embeddings": 32768,
        "hidden_size": 896,
    }

    def __init__(self):
        self.model = None
        self.config = None
        self.tokenizer = None
        self.num_layers = 0
        self.hidden_size = 0
        self.num_kv_heads = 0
        self.model_name = "qwen2.5-0.5b"

    def load(self, model_path, layers=24):
        """Load the HF model in fp16 and validate the Qwen2.5-0.5B architecture.

        ``model_path`` may be a HF directory or a ``.gguf`` file.  GGUF
        skeletons are dequantized by transformers via the ``gguf_file=`` kwarg
        (requires transformers >= 4.57); the real Q4_0 weights are injected
        afterwards by ``gguf_loader`` -- but the q/k/v attention biases and
        layer norms are REAL and stay in the exported graph.
        """
        is_gguf = os.path.isfile(model_path) and model_path.endswith(".gguf")
        if is_gguf:
            # transformers routes GGUF loading by (directory, gguf_file=) pairs.
            self.model = AutoModelForCausalLM.from_pretrained(
                os.path.dirname(os.path.abspath(model_path)),
                gguf_file=os.path.basename(model_path),
                trust_remote_code=True,
                device_map=DEVICE,
                dtype=dtype,
                attn_implementation="eager",
            )
            self.config = self.model.config
        else:
            self.config = AutoConfig.from_pretrained(model_path)
            self.config._attn_implementation = "eager"  # pylint: disable=W0212
            self.config.num_hidden_layers = layers
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                config=self.config,
                device_map=DEVICE,
                dtype=dtype,
                attn_implementation="eager",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.config.model_type != "qwen2":
            raise ValueError(f"Model type must be qwen2, got {self.config.model_type}")
        for key, expected in self.REQUIRED_ARCH.items():
            if getattr(self.config, key, None) != expected:
                raise ValueError(
                    f"Error: the model at '{model_path}' is not a Qwen2.5-0.5B model "
                    f"({key}={getattr(self.config, key, None)}, expected {expected})."
                )
        self.model = self.model.eval()

        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        self.num_kv_heads = self.model.config.num_key_value_heads
        logger.info("model loaded: %s", model_path)

    def export(self, model_path, max_seq_len=1024, chunk_size=128):
        """Trace the logits-only graph; KV caches are updated in place."""
        # Same resolution rule as ``Qwen2OpSet`` so the traced rope/cache shapes and
        # the operator attribute cannot drift when the config sets ``head_dim`` explicitly.
        head_dim = resolve_head_dim(self.model.config)
        if max_seq_len <= 0 or max_seq_len % ATTENTION_CACHE_ALIGNMENT or head_dim % ATTENTION_TILE_ALIGNMENT:
            raise ValueError("MsAttention requires KV capacity divisible by 64 and head_dim divisible by 16")
        if chunk_size <= 0 or chunk_size > max_seq_len or (
                chunk_size != 1 and chunk_size % ATTENTION_TILE_ALIGNMENT):
            raise ValueError("MsAttention requires a prefill chunk divisible by 16, or decode chunk 1")

        input_names = [
            "valid_seq_len",
            "lmhead_idx",
            "rope_cos",
            "rope_sin",
            "inputs_embeds",
            "attention_mask",
        ]
        kv_names = [(f"past_key_{i}", f"past_val_{i}") for i in range(self.num_layers)]
        kv_names = [name for kv in kv_names for name in kv]
        input_names = input_names + kv_names
        out_kv_names = [(f"out_key_{i}", f"out_val_{i}") for i in range(self.num_layers)]
        out_kv_names = [name for kv in out_kv_names for name in kv]

        valid_seq_len = torch.tensor([0], dtype=torch.int32).to(DEVICE)
        lmhead_idx = torch.tensor([0], dtype=torch.int32).to(DEVICE)

        rope_cos = torch.zeros((1, chunk_size, head_dim), device=DEVICE, dtype=dtype)
        rope_sin = torch.zeros((1, chunk_size, head_dim), device=DEVICE, dtype=dtype)

        past_key_or_value = torch.zeros(
            (1, self.num_kv_heads, max_seq_len, head_dim), device=DEVICE, dtype=dtype
        )
        past_key_values = [[past_key_or_value] * 2] * self.num_layers

        inputs_embeds = torch.zeros((1, chunk_size, self.hidden_size), device=DEVICE, dtype=dtype)
        attention_mask = torch.zeros(1, 1, chunk_size, max_seq_len, dtype=dtype)

        inputs = (
            None,  # input_ids (not an input: embedding lookup is CPU-side)
            valid_seq_len,
            lmhead_idx,
            rope_cos,
            rope_sin,
            inputs_embeds,
            attention_mask,
            past_key_values,
        )
        # Trace the NNRT wrapper (Ms* ops + submodule refs) instead of the HF
        # model — the wrapper's forward emits ``custom::`` ONNX nodes directly,
        # so no transformers monkey-patch is needed.
        wrapper = Qwen2NnrtWrapper(self.model, self.config)
        # The fused cache op is inference-only and rejects grad-carrying inputs;
        # eval() does not disable autograd and the legacy exporter traces with
        # grad enabled, so disable grad at the export boundary.
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                inputs,
                model_path,
                input_names=input_names,
                do_constant_folding=True,
                output_names=["logits", *out_kv_names],
                opset_version=18,
                operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
                dynamo=False,  # legacy TorchScript exporter: required for Ms* custom symbolic ops
            )

        # MsTransRopeScatterNDUpdate writes past_key_i/past_val_i in place, so the
        # KV outputs returned by the wrapper only alias those writable inputs.
        # Drop them before slimming/quantizing: the NNRT runtime reuses the input
        # buffers and expects logits as the sole graph output.
        traced = onnx.load(model_path)
        traced_outputs = [out.name for out in traced.graph.output]
        expected_outputs = ["logits", *out_kv_names]
        if traced_outputs != expected_outputs:
            raise ValueError(
                f"traced graph outputs {traced_outputs} do not match the expected "
                f"logits + per-layer KV aliases {expected_outputs}; refusing to drop outputs"
            )
        del traced.graph.output[1:]
        _save_onnx(traced, model_path)

        new_model = slim(model_path, skip_fusion_patterns=["FusionGemm"])
        _save_onnx(new_model, model_path)

        fuse_add_rmsnorm(model_path, model_path)

        new_model = onnx.load(model_path)
        duplicate_shared_initializers(new_model)
        _save_onnx(new_model, model_path)
        logger.info("Export + slim + add-rmsnorm fusion done: %s", model_path)

    def embedding_weight_save(self, embedding_weight_save_path, embedding_asset: EmbeddingAsset):
        """Save the shared graph payload without repeating quantization."""
        embedding_asset.save(embedding_weight_save_path)
        logger.info("Saved embedding weight to %s", embedding_weight_save_path)

    def rope_sin_cos_save(self, cos_path, sin_path, seq_len):
        """Save the RoPE cos/sin constants (fp16, [seq_len, head_dim] flattened).

        RoPE lives on ``Qwen2Model.rotary_emb``; forward takes position_ids and
        returns [batch, seq_len, head_dim].
        """
        input_embed = torch.rand(1, seq_len, self.hidden_size, dtype=torch.float16).to(DEVICE)
        position_ids = torch.arange(0, seq_len).unsqueeze(0)

        rotary_layer = self.model.model.rotary_emb
        rope_cos, rope_sin = rotary_layer(input_embed, position_ids)
        rope_cos = rope_cos[0]
        rope_sin = rope_sin[0]

        rope_cos = rope_cos.detach().numpy().astype(np.float16)
        rope_sin = rope_sin.detach().numpy().astype(np.float16)

        rope_cos.flatten().tofile(cos_path)
        rope_sin.flatten().tofile(sin_path)
        logger.info("Saved rope cos/sin to %s / %s", cos_path, sin_path)

    @staticmethod
    def attention_mask_save(attention_mask_path, max_seq_len):
        """Save the causal attention mask (fp16, [1,1,max_seq_len,max_seq_len])."""
        mask = torch.full((max_seq_len, max_seq_len), torch.finfo(dtype).min)
        mask_cond = torch.arange(mask.size(-1))
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        attention_mask = mask[None, None, :, :].expand(1, 1, max_seq_len, max_seq_len)
        attention_mask.to(torch.float16).detach().numpy().tofile(attention_mask_path)
        logger.info("Saved attention mask to %s", attention_mask_path)

    def build_config(self, max_length, chunk_size, embedding_quant_config, decoder_quant_config):
        """Build the packager-consumable model config (architecture/generation/assets/npu)."""
        arch = self.model.config
        model_name = os.path.basename(getattr(arch, "_name_or_path", "") or "") or getattr(
            self, "model_name", "qwen2.5-0.5b"
        )
        return {
            "model_name": model_name,
            "architecture": {
                "num_layers": int(arch.num_hidden_layers),
                "hidden_size": int(arch.hidden_size),
                "intermediate_size": int(arch.intermediate_size),
                "num_heads": int(arch.num_attention_heads),
                "num_kv_heads": int(arch.num_key_value_heads),
                "head_dim": resolve_head_dim(arch),
                "vocab_size": int(arch.vocab_size),
                "max_position_embeddings": int(arch.max_position_embeddings),
                "rope_theta": float(getattr(arch, "rope_theta", 10000.0)),
                "norm_eps": float(getattr(arch, "rms_norm_eps", 1e-6)),
                "tie_word_embeddings": int(bool(arch.tie_word_embeddings)),
            },
            "generation": {
                "stop_token_ids": [
                    int(t)
                    for t in ([arch.eos_token_id] if getattr(arch, "eos_token_id", None) is not None else [])
                ],
                "suppress_token_ids": [],
            },
            "npu": {
                "max_length": int(max_length),
                "chunk_size": int(chunk_size),
                "embedding_quant": embedding_quant_config.asdict() if embedding_quant_config.is_quant else None,
                "decoder_quant": decoder_quant_config.asdict() if decoder_quant_config.is_quant else None,
                **({"q4_0_weight_layout": "q4_0_nzf_compact_phase4"}
                   if embedding_quant_config.is_quant and embedding_quant_config.quant_method == QuantType.Q4_0
                   else {}),
            },
            "sampling": LiteTurboConfig(
                max_length=max_length,
                chunk_size=chunk_size,
                vocab_size=int(arch.vocab_size),
                hidden_size=int(arch.hidden_size),
                num_attention_heads=int(arch.num_attention_heads),
                num_key_value_heads=int(arch.num_key_value_heads),
                eos_id=int(arch.eos_token_id) if getattr(arch, "eos_token_id", None) is not None else -1,
                scale_gp_size=embedding_quant_config.group_size if embedding_quant_config.is_quant else 32,
                embedding_quant=bool(embedding_quant_config.is_quant),
            ).asdict(),
        }


def export_qwen2_5(
    model_dir: str,
    output_dir: str,
    max_length: int = 1024,
    chunk_size: int = 128,
    embedding_quant: Optional[str] = None,
    decoder_quant: Optional[str] = None,
    layers: int = 24,
    model_name: str = "qwen2.5-0b5",
    onnx_name: str = "qwen2_5_0b5.onnx",
):
    """Export a Qwen2.5-0.5B model to the mslite-llm NNRT contract.

    Produces, under ``output_dir``:
      * ``<onnx_name>`` — the ONNX graph (custom Ms* ops, 7-input contract)
      * ``embedding.bin`` / ``embedding_quant.bin`` — embedding weight
      * ``rope_cos.bin`` / ``rope_sin.bin`` — RoPE constants
      * ``attention_mask.bin`` — precomputed causal mask
      * ``qwen2_5_config.json`` — packager-consumable config

    Returns the path to ``qwen2_5_config.json``.
    """
    embedding_quant_config = QuantizationConfig(embedding_quant)
    decoder_quant_config = QuantizationConfig(decoder_quant)
    embedding_quant = embedding_quant_config.quant_method
    decoder_quant = decoder_quant_config.quant_method
    if max_length <= 0 or chunk_size <= 0 or max_length % chunk_size != 0:
        raise ValueError(f"max_length {max_length} must be a positive multiple of chunk_size {chunk_size}")
    if max_length % chunk_size != 0:
        raise ValueError("max_length must be a multiple of chunk_size (NNRT chunked prefill contract)")

    os.makedirs(output_dir, exist_ok=True)

    exporter = Qwen2Onnx()
    exporter.load(model_dir, layers)
    exporter.model_name = model_name

    onnx_path = os.path.join(output_dir, onnx_name)
    exporter.export(onnx_path, max_seq_len=max_length, chunk_size=chunk_size)

    if embedding_quant_config.is_quant or decoder_quant_config.is_quant:
        model_config = ModelConfig(
            max_length=max_length,
            chunk_size=chunk_size,
            vocab_size=exporter.config.vocab_size,
            hidden_size=exporter.config.hidden_size,
            num_attention_heads=exporter.config.num_attention_heads,
            num_key_value_heads=exporter.config.num_key_value_heads,
            eos_id=exporter.config.eos_token_id,
            embedding_quant=embedding_quant_config,
            decoder_quant=decoder_quant_config,
        )
        path, name = os.path.split(onnx_path)
        name, ext = os.path.splitext(name)
        quant_model_path = os.path.join(path, name + "_quant" + ext)
        apply_quant(onnx_path, quant_model_path, model_config)
        onnx_path = quant_model_path  # the .omc is compiled from the quantized graph

    embedding_bin = os.path.join(output_dir, "embedding_quant.bin" if embedding_quant else "embedding.bin")
    model = onnx.load(onnx_path)
    embedding_asset = apply_shared_weight(model)
    _save_onnx(model, onnx_path)

    validate_contract(
        onnx_path,
        exporter.num_layers,
        embedding_quant=embedding_quant_config.is_quant,
        embedding_format=embedding_quant_config.embedding_format,
        inplace_cache=True,
    )

    config = exporter.build_config(max_length, chunk_size, embedding_quant_config, decoder_quant_config)

    # Standalone packager-consumable fragments (same content as qwen2_5_config.json).
    with open(os.path.join(output_dir, "architecture.json"), "w", encoding="utf-8") as f:
        json.dump(config["architecture"], f, indent=2)
    with open(os.path.join(output_dir, "generation_policy.json"), "w", encoding="utf-8") as f:
        json.dump(config["generation"], f, indent=2)

    exporter.embedding_weight_save(embedding_bin, embedding_asset)

    cos_path = os.path.join(output_dir, "rope_cos.bin")
    sin_path = os.path.join(output_dir, "rope_sin.bin")
    exporter.rope_sin_cos_save(cos_path, sin_path, max_length)

    mask_path = os.path.join(output_dir, "attention_mask.bin")
    exporter.attention_mask_save(mask_path, max_length)

    config["assets"] = {
        "embedding": os.path.basename(embedding_bin),
        "rope_sin": os.path.basename(sin_path),
        "rope_cos": os.path.basename(cos_path),
        "attention_mask": os.path.basename(mask_path),
    }
    config["onnx"] = os.path.basename(onnx_path)

    config_path = os.path.join(output_dir, "qwen2_5_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("Qwen2.5 export complete. Config: %s", config_path)
    return config_path
