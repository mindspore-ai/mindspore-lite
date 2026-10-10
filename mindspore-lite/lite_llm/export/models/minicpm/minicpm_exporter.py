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
"""MiniCPM-2B ONNX exporter for the mslite-llm NNRT (Kirin NPU) runtime.

Same NNRT graph I/O contract as ``qwen2_5_exporter`` (7 inputs + interleaved
per-layer KV), instantiated for the MiniCPM architecture:

* ``scale_emb`` (12) is applied inside ``MiniCpmNnrtWrapper`` at the graph
  entry (embedding lookup itself stays CPU-side).
* RoPE lives on each attention (LLaMA<=4.4x layout,
  ``layers[i].self_attn.rotary_emb``, old-style ``forward(x, seq_len)``
  returning ``[S, D]``) instead of the model-level rotary of Qwen2.5
  (new-style ``forward(x, position_ids)`` returning ``[B, S, D]``);
  ``rope_sin_cos_save`` handles both.

Loading caveat (see README): transformers removed the built-in ``minicpm``
model after 4.5x, so both the HF path and the GGUF ``gguf_file=`` path
require ``trust_remote_code=True`` with a repo that ships its own
``modeling_minicpm.py`` (only validated on the 4.5x series).
"""

import logging
import os
from typing import Optional

import numpy as np
import torch
from torch.onnx import OperatorExportTypes
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from models._base.export_utils import (
    NNRT_INPUT_NAMES,
    ExportArtifacts,
    NnrtTraceSpec,
    build_quant_request,
    collect_kv_names,
    dump_json,
    finalize_onnx_graph,
    make_input_placeholders,
    prepare_weights_for_compile,
)
from utils.onnx_postprocess import validate_contract
from utils.export_quant import LiteTurboConfig, quantize_weight_g128_4bit_nz

from .minicpm_wrapper import MiniCpmNnrtWrapper

logger = logging.getLogger(__name__)

DEVICE = "cpu"
dtype = torch.float16


class MiniCpmOnnx:
    """Export a MiniCPM-2B HF model to the mslite-llm NNRT ONNX contract."""

    #: Only the MiniCPM-2B architecture is validated (skeleton shapes + GGUF
    #: tensor layout of the reference model).
    REQUIRED_ARCH = {
        "num_hidden_layers": 40,
        "intermediate_size": 9216,
        "max_position_embeddings": 2048,
        "hidden_size": 2304,
    }

    def __init__(self):
        """Declare lazily-loaded model artifacts; load() fills them in."""
        self.model: Optional[AutoModelForCausalLM] = None
        self.config: Optional[AutoConfig] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.num_layers: int = 0
        self.hidden_size: int = 0
        self.num_kv_heads: int = 0
        self.model_name: str = "minicpm-2b"

    def load(self, model_path, layers=40):
        """Load the HF model in fp16 and validate the MiniCPM-2B architecture."""
        is_gguf = os.path.isfile(model_path) and model_path.endswith(".gguf")
        if is_gguf:
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
            self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
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
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self.config.model_type != "minicpm":
            raise ValueError(f"Model type must be minicpm, got {self.config.model_type}")
        for key, expected in self.REQUIRED_ARCH.items():
            if getattr(self.config, key, None) != expected:
                raise ValueError(
                    f"Error: the model at '{model_path}' is not a MiniCPM-2B model "
                    f"({key}={getattr(self.config, key, None)}, expected {expected})."
                )
        self.model = self.model.eval()

        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        self.num_kv_heads = self.model.config.num_key_value_heads
        logger.info("model loaded: %s", model_path)

    def _rotary(self):
        """Locate the RoPE module: model-level (Qwen/LLaMA>=4.4x) or per-attention (MiniCPM)."""
        inner = self.model.model
        rotary = getattr(inner, "rotary_emb", None)
        if rotary is None:
            rotary = inner.layers[0].self_attn.rotary_emb
        return rotary

    def export(self, model_path, max_seq_len=1024, chunk_size=128):
        """Export the ONNX graph (NNRT 7-input + interleaved KV contract)."""
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads

        kv_names, out_kv_names = collect_kv_names(self.num_layers)
        input_names = NNRT_INPUT_NAMES + kv_names

        spec = NnrtTraceSpec(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            chunk_size=chunk_size,
        )
        inputs = make_input_placeholders(spec)
        wrapper = MiniCpmNnrtWrapper(self.model, self.config)
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

        finalize_onnx_graph(model_path)
        logger.info("Export + slim + add-rmsnorm fusion done: %s", model_path)

    def embedding_weight_save(self, embedding_weight_save_path=None, embedding_quantize_config=None):
        """Save the input embedding weight (fp16 raw / W4A8 / W4A16 quantized)."""
        embedding_layer = self.model.get_input_embeddings()
        weight = embedding_layer.weight.detach().numpy().astype(np.float16)
        if embedding_quantize_config == "W4A8":
            weight_4bit = quantize_weight_g128_4bit_nz(weight.T)
            weight_4bit.tofile(embedding_weight_save_path)
        elif embedding_quantize_config == "W4A16":
            from torch_custom.ms_quant4_n0_group32 import MsQuant4N0Group32

            weight_4bit_gp32 = MsQuant4N0Group32.quantize_weight_g32_4bit(weight.T)
            weight_4bit_gp32.tofile(embedding_weight_save_path)
        else:
            weight.flatten().tofile(embedding_weight_save_path)
        logger.info("Saved embedding weight to %s", embedding_weight_save_path)

    def rope_sin_cos_save(self, cos_path, sin_path, seq_len):
        """Save the RoPE cos/sin constants (fp16, [seq_len, head_dim] flattened).

        Handles both rotary layouts: new-style ``forward(x, position_ids)``
        returning ``[batch, seq_len, head_dim]`` and old-style (MiniCPM /
        LLaMA<=4.4x) ``forward(x, seq_len)`` returning ``[seq_len, head_dim]``.
        """
        input_embed = torch.rand(1, seq_len, self.hidden_size, dtype=torch.float16).to(DEVICE)
        rotary_layer = self._rotary()
        try:
            position_ids = torch.arange(0, seq_len).unsqueeze(0)
            rope_cos, rope_sin = rotary_layer(input_embed, position_ids)
            rope_cos = rope_cos[0]
            rope_sin = rope_sin[0]
        except TypeError:
            # Old-style per-attention rotary: forward(x, seq_len=None) -> [S, D].
            rope_cos, rope_sin = rotary_layer(input_embed, seq_len=seq_len)

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
            self, "model_name", "minicpm-2b"
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
                "scale_emb": float(getattr(arch, "scale_emb", 1) or 1),
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
                   if embedding_quant_config.is_quant and embedding_quant_config.quant_method == "W4A16"
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


def export_minicpm(
    model_dir: str,
    output_dir: str,
    max_length: int = 1024,
    chunk_size: int = 128,
    embedding_quant: Optional[str] = None,
    decoder_quant: Optional[str] = None,
    layers: int = 40,
    model_name: str = "minicpm-2b",
    onnx_name: str = "minicpm_2b.onnx",
):
    """Export a MiniCPM-2B model to the mslite-llm NNRT contract.

    Produces, under ``output_dir``:
      * ``<onnx_name>`` — the ONNX graph (custom Ms* ops, 7-input contract)
      * ``embedding.bin`` / ``embedding_quant.bin`` — embedding weight
      * ``rope_cos.bin`` / ``rope_sin.bin`` — RoPE constants
      * ``attention_mask.bin`` — precomputed causal mask
      * ``minicpm_config.json`` — packager-consumable config

    Returns the path to ``minicpm_config.json``.
    """
    if max_length <= 0 or chunk_size <= 0 or max_length % chunk_size != 0:
        raise ValueError(f"max_length {max_length} must be a positive multiple of chunk_size {chunk_size}")

    os.makedirs(output_dir, exist_ok=True)

    exporter = MiniCpmOnnx()
    exporter.load(model_dir, layers)
    exporter.model_name = model_name

    artifacts = ExportArtifacts(
        onnx=os.path.join(output_dir, onnx_name),
        embedding=os.path.join(output_dir, "embedding_quant.bin" if embedding_quant else "embedding.bin"),
        rope_cos=os.path.join(output_dir, "rope_cos.bin"),
        rope_sin=os.path.join(output_dir, "rope_sin.bin"),
        mask=os.path.join(output_dir, "attention_mask.bin"),
        config=os.path.join(output_dir, "minicpm_config.json"),
    )
    exporter.export(artifacts.onnx, max_seq_len=max_length, chunk_size=chunk_size)

    request = build_quant_request(embedding_quant, decoder_quant, max_length, chunk_size)
    artifacts.onnx = prepare_weights_for_compile(artifacts.onnx, exporter.config, request)

    validate_contract(artifacts.onnx, exporter.num_layers, embedding_quant=request.embedding_quant.is_quant)

    config = exporter.build_config(max_length, chunk_size, request.embedding_quant, request.decoder_quant)

    # Standalone packager-consumable fragments (same content as minicpm_config.json).
    dump_json(os.path.join(output_dir, "architecture.json"), config["architecture"])
    dump_json(os.path.join(output_dir, "generation_policy.json"), config["generation"])

    exporter.embedding_weight_save(artifacts.embedding, embedding_quant)
    exporter.rope_sin_cos_save(artifacts.rope_cos, artifacts.rope_sin, max_length)
    exporter.attention_mask_save(artifacts.mask, max_length)

    config["assets"] = {
        "embedding": os.path.basename(artifacts.embedding),
        "rope_sin": os.path.basename(artifacts.rope_sin),
        "rope_cos": os.path.basename(artifacts.rope_cos),
        "attention_mask": os.path.basename(artifacts.mask),
    }
    config["onnx"] = os.path.basename(artifacts.onnx)

    dump_json(artifacts.config, config)
    logger.info("MiniCPM export complete. Config: %s", artifacts.config)
    return artifacts.config
