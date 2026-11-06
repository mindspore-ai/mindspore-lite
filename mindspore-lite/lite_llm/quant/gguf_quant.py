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
"""Local HF -> pure Q4_0 or Q3_0 GGUF, independent of the export CLI.

Q4_0 supports dense Qwen2/Qwen3 checkpoints. Q3_0 keeps the existing tied
Qwen2.5-0.5B contract (private GGUF type 41). Embeddings remain Q4_0;
normalization weights and biases are F32. No importance-based type promotion.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .generation_policy import GGUF_GENERATION_CONFIG_KEY, load_generation_policy
from .q3_0 import BLOCK_BYTES as Q3_0_BLOCK_BYTES, BLOCK_SIZE as Q3_0_BLOCK_SIZE
from .q3_0 import quantize_q3_0_blocks, register_private_q3_0_type

SUPPORTED_MODEL = {
    "model_type": "qwen2", "num_hidden_layers": 24, "hidden_size": 896,
    "intermediate_size": 4864, "num_attention_heads": 14,
    "num_key_value_heads": 2, "vocab_size": 151936,
}


def validate_config(config, quant_type="q4_0"):
    """Validate the explicitly supported HF model families before conversion."""
    if quant_type not in ("q4_0", "q3_0"):
        raise ValueError("quant_type must be q4_0 or q3_0")
    if quant_type == "q4_0":
        if getattr(config, "model_type", None) not in ("qwen2", "qwen3"):
            raise ValueError("q4_0 supports dense Qwen2/Qwen3 HF checkpoints only")
        return
    mismatches = [name for name, expected in SUPPORTED_MODEL.items()
                  if getattr(config, name, None) != expected]
    if mismatches or not getattr(config, "tie_word_embeddings", False):
        raise ValueError("only tied Qwen2.5-0.5B is supported; invalid fields: " + ", ".join(mismatches))


def _gguf_name(name):
    """Map supported Qwen2 HF tensor names to their GGUF counterparts."""
    replacements = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
        "model.layers.": "blk.",
        ".self_attn.q_proj.": ".attn_q.",
        ".self_attn.k_proj.": ".attn_k.",
        ".self_attn.v_proj.": ".attn_v.",
        ".self_attn.o_proj.": ".attn_output.",
        ".self_attn.q_norm.": ".attn_q_norm.",
        ".self_attn.k_norm.": ".attn_k_norm.",
        ".mlp.gate_proj.": ".ffn_gate.",
        ".mlp.up_proj.": ".ffn_up.",
        ".mlp.down_proj.": ".ffn_down.",
        ".input_layernorm.": ".attn_norm.",
        ".post_attention_layernorm.": ".ffn_norm.",
    }
    for source, target in replacements.items():
        name = name.replace(source, target)
    return name


def _checkpoint_files(model_dir):
    """Resolve local HF shards, preferring safetensors over pickle checkpoints."""
    root = Path(model_dir).resolve()
    for filename in ("model.safetensors.index.json", "model.safetensors",
                     "pytorch_model.bin.index.json", "pytorch_model.bin"):
        source = root / filename
        if not source.is_file():
            continue
        if filename.endswith(".json"):
            with source.open(encoding="utf-8") as stream:
                names = sorted(set(json.load(stream)["weight_map"].values()))
        else:
            names = [filename]
        paths = [(root / name).resolve() for name in names]
        if not paths or any(not path.is_relative_to(root) or not path.is_file() for path in paths):
            raise ValueError("HF checkpoint index contains missing or out-of-directory shards")
        return paths
    raise ValueError("HF directory must contain safetensors or pytorch_model.bin weights")


def _checkpoint_tensors(model_dir):
    """Read one tensor/shard at a time, without constructing an HF model."""
    import torch
    from safetensors import safe_open

    for path in _checkpoint_files(model_dir):
        if path.suffix == ".safetensors":
            with safe_open(path, framework="pt", device="cpu") as shard:
                for name in shard.keys():
                    yield name, shard.get_tensor(name).to(torch.float32).numpy()
        else:
            shard = torch.load(path, map_location="cpu", weights_only=True)
            for name, tensor in shard.items():
                yield name, tensor.to(torch.float32).numpy()


def _add_tokenizer(writer, config, tokenizer, model_dir):
    """Keep tokenizer vocabulary, merges and generation IDs in the container."""
    from gguf import TokenType

    vocab = tokenizer.get_vocab()
    reverse = {index: token for token, index in vocab.items()}
    tokens = [reverse.get(index, f"[PAD{index}]") for index in range(config.vocab_size)]
    special = set(tokenizer.all_special_tokens)
    added = tokenizer.get_added_vocab()
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_types([
        TokenType.CONTROL if token in special else
        TokenType.USER_DEFINED if token in added else TokenType.NORMAL
        for token in tokens
    ])
    with (Path(model_dir) / "tokenizer.json").open(encoding="utf-8") as stream:
        merges = json.load(stream)["model"].get("merges", [])
    writer.add_token_merges([" ".join(merge) if isinstance(merge, list) else merge for merge in merges])
    if tokenizer.bos_token_id is not None:
        writer.add_bos_token_id(tokenizer.bos_token_id)
    writer.add_eos_token_id(tokenizer.eos_token_id)
    policy = load_generation_policy(tokenizer, model_dir)
    writer.add_string(GGUF_GENERATION_CONFIG_KEY, json.dumps({
        "eos_token_id": policy["stop_token_ids"],
        "suppress_tokens": policy["suppress_token_ids"],
    }))
    if tokenizer.pad_token_id is not None:
        writer.add_pad_token_id(tokenizer.pad_token_id)
    writer.add_add_bos_token(bool(getattr(tokenizer, "add_bos_token", False)))
    if tokenizer.chat_template:
        writer.add_chat_template(tokenizer.chat_template)


def _add_metadata(writer, config, tokenizer, model_dir):
    """Record architecture metadata needed for a self-contained skeleton."""
    writer.add_name(Path(model_dir).name)
    writer.add_context_length(config.max_position_embeddings)
    writer.add_embedding_length(config.hidden_size)
    writer.add_block_count(config.num_hidden_layers)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.num_attention_heads)
    writer.add_head_count_kv(config.num_key_value_heads)
    writer.add_rope_dimension_count(getattr(config, "head_dim", None) or
                                    config.hidden_size // config.num_attention_heads)
    writer.add_rope_freq_base(config.rope_theta)
    writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    # general.file_type is a GGUFFileType, not the private tensor type 41.
    # Omit it for the type-41 extension; each tensor records its true type.
    writer.add_quantization_version(2)
    _add_tokenizer(writer, config, tokenizer, model_dir)


def _add_weights(writer, model_dir, quant_type="q4_0", tie_word_embeddings=True):
    """Write uniformly quantized matrices; Q3 uses a standard Q4 embedding.

    gguf's canonical Q4_0 quantizer emits 32-value / 18-byte blocks. Do not
    promote any projections to Q4_1/Q6_K, including output or large matrices.
    """
    from gguf import GGMLQuantizationType
    from gguf.quants import quantize

    if quant_type not in ("q4_0", "q3_0"):
        raise ValueError("quant_type must be q4_0 or q3_0")
    q3_type = register_private_q3_0_type() if quant_type == "q3_0" else None
    for name, array in _checkpoint_tensors(model_dir):
        name = _gguf_name(name)
        if name == "output.weight" and tie_word_embeddings:
            # Tied lm_head is reconstructed from token_embd by the consumer.
            continue
        if not np.isfinite(array).all():
            raise ValueError(f"HF tensor {name} contains non-finite values")
        if array.ndim == 2:
            if array.shape[1] % 32:
                raise ValueError(f"HF tensor {name} requires K divisible by 32")
            if quant_type == "q4_0" or name == "token_embd.weight":
                data = quantize(np.ascontiguousarray(array), GGMLQuantizationType.Q4_0)
                wire_type = GGMLQuantizationType.Q4_0
            else:
                data = quantize_q3_0_blocks(np.ascontiguousarray(array.T))
                # Keep logical [K,N] dimensions, not a flattened [K*N] tensor.
                data = data.reshape(array.shape[0], array.shape[1] // Q3_0_BLOCK_SIZE * Q3_0_BLOCK_BYTES)
                wire_type = q3_type
            writer.add_tensor(name, data, raw_dtype=wire_type)
        elif array.ndim == 1:
            # llama.cpp --pure keeps normalization weights/biases in F32.
            writer.add_tensor(name, np.ascontiguousarray(array, dtype=np.float32))
        else:
            raise ValueError(f"Unsupported HF tensor {name} with shape {array.shape}")


def export_gguf(model_dir, output, quant_type="q4_0"):
    """Write pure local HF weights without a DDK or a transformers model."""
    import gguf
    from transformers import AutoConfig, AutoTokenizer

    if not Path(model_dir).is_dir():
        raise ValueError("--model must name a local HF model directory")
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False, local_files_only=True)
    validate_config(config, quant_type)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False, local_files_only=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(output), config.model_type, use_temp_file=True)
    try:
        _add_metadata(writer, config, tokenizer, model_dir)
        _add_weights(writer, model_dir, quant_type, getattr(config, "tie_word_embeddings", False))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
    finally:
        writer.close()
    return str(output)


def build_parser():
    """Build the standalone HF-to-GGUF quantization CLI."""
    parser = argparse.ArgumentParser(
        prog="mslite-llm-quantize-gguf",
        description="Quantize a local HF model to pure Q4_0/Q3_0 GGUF (no DDK required).",
        epilog="Q4_0: dense Qwen2/Qwen3. Q3_0: tied Qwen2.5-0.5B (private GGUF type 41).",
    )
    parser.add_argument("--model", required=True, help="Local HF model directory")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--quant-type", choices=("q4_0", "q3_0"), default="q4_0",
                        help="Matrix quantization (default: q4_0); Q3 embedding stays Q4_0")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


def main(argv=None):
    """Console entry point; upstream quantization only, never .omc/.msl export."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    result = export_gguf(args.model, args.output, args.quant_type)
    logging.getLogger(__name__).info("wrote %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
