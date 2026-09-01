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
    outputs: [logits, out_key_i/out_val_i]   (KV updated in place on device)

Contract differences vs the reference exporter:

* ``kvcache_mask`` input removed.  KV update is a pure ``MsScatterND`` at
  ``pos=valid_seq_len`` (executor: "device: past[:,:,pos:pos+L]=state").
* ``embedding_weight`` (tied lm_head weight) is inserted as a graph input at
  index 6 by ``apply_shared_weight`` (see ``utils/quant/quantize.py``), which
  matches the executor's 7-input shape check.
* Custom ops stay unchanged: ``MsRotaryPosEmb``, ``MsGroupMatmul``,
  ``MsAddSoftmax``, ``MsRmsNorm``, ``MsAddRmsNorm`` (post-export fusion),
  ``MsScatterND``, plus ``MsQuant4N0Group32`` / ``MsQuant4N0Group128`` /
  ``MsQuant2N0Group32`` when weight quantization is enabled.

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
    quantize_weight_g128_4bit_nz,
    quantize_weight_g32_4bit_nd,
)

from .qwen2_5_wrapper import Qwen2NnrtWrapper

logger = logging.getLogger(__name__)

device = "cpu"
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
                device_map=device,
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
                device_map=device,
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
        """Export the ONNX graph (NNRT 7-input + interleaved KV contract)."""
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads

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

        valid_seq_len = torch.tensor([1], dtype=torch.int32).to(device)
        lmhead_idx = torch.tensor([0], dtype=torch.int32).to(device)

        rope_cos = torch.zeros((1, chunk_size, head_dim), device=device, dtype=dtype)
        rope_sin = torch.zeros((1, chunk_size, head_dim), device=device, dtype=dtype)

        past_key_or_value = torch.zeros(
            (1, self.num_kv_heads, max_seq_len, head_dim), device=device, dtype=dtype
        )
        past_key_values = [[past_key_or_value] * 2] * self.num_layers

        inputs_embeds = torch.zeros((1, chunk_size, self.hidden_size), device=device, dtype=dtype)
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

        new_model = slim(model_path, skip_fusion_patterns=["FusionGemm"])
        _save_onnx(new_model, model_path)

        fuse_add_rmsnorm(model_path, model_path)

        new_model = onnx.load(model_path)
        duplicate_shared_initializers(new_model)
        _save_onnx(new_model, model_path)
        logger.info("Export + slim + add-rmsnorm fusion done: %s", model_path)

    def embedding_weight_save(self, embedding_weight_save_path=None, embedding_quantize_config=None):
        """Save the input embedding weight (fp16 raw / W4A8 / W4A16 quantized)."""
        embedding_layer = self.model.get_input_embeddings()
        weight = embedding_layer.weight.detach().numpy().astype(np.float16)
        if embedding_quantize_config == "W4A8":
            weight_4bit = quantize_weight_g128_4bit_nz(weight.T)
            weight_4bit.tofile(embedding_weight_save_path)
        elif embedding_quantize_config == "W4A16":
            weight_4bit_gp32 = quantize_weight_g32_4bit_nd(weight.T)
            weight_4bit_gp32.tofile(embedding_weight_save_path)
        else:
            weight.flatten().tofile(embedding_weight_save_path)
        logger.info("Saved embedding weight to %s", embedding_weight_save_path)

    def rope_sin_cos_save(self, cos_path, sin_path, seq_len):
        """Save the RoPE cos/sin constants (fp16, [seq_len, head_dim] flattened).

        RoPE lives on ``Qwen2Model.rotary_emb``; forward takes position_ids and
        returns [batch, seq_len, head_dim].
        """
        input_embed = torch.rand(1, seq_len, self.hidden_size, dtype=torch.float16).to(device)
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

    def attention_mask_save(self, attention_mask_path, max_seq_len):
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
                "head_dim": int(arch.hidden_size // arch.num_attention_heads),
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

    embedding_quant_config = QuantizationConfig(embedding_quant)
    decoder_quant_config = QuantizationConfig(decoder_quant)

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
    else:
        model = onnx.load(onnx_path)
        apply_shared_weight(model)  # inserts embedding_weight input at index 6
        _save_onnx(model, onnx_path)

    validate_contract(onnx_path, exporter.num_layers, embedding_quant=embedding_quant_config.is_quant)

    config = exporter.build_config(max_length, chunk_size, embedding_quant_config, decoder_quant_config)

    # Standalone packager-consumable fragments (same content as qwen2_5_config.json).
    with open(os.path.join(output_dir, "architecture.json"), "w", encoding="utf-8") as f:
        json.dump(config["architecture"], f, indent=2)
    with open(os.path.join(output_dir, "generation_policy.json"), "w", encoding="utf-8") as f:
        json.dump(config["generation"], f, indent=2)

    embedding_bin = os.path.join(output_dir, "embedding_quant.bin" if embedding_quant else "embedding.bin")
    exporter.embedding_weight_save(embedding_bin, embedding_quant)

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
