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
"""Compile the quantized ONNX to a Kirin .omc offline model via DDK omg.

Requires the CANN/DDK environment. The omg tool is located inside the DDK tree
relative to the ``DDK_PATH`` env var (``$DDK_PATH/tools/tools_omg/omg``) and is
*not* expected to be on PATH; source the DDK env first, e.g.
``source $DDK/tools/tools_ascendc/set_ascendc_env.sh``. The Ms* custom ops must
already be registered in the DDK; install them from the vendored custom_ops
operator library (``../custom_ops``, see the README appendix) via
``build.py --ops ... --install``.
"""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

# Candidate omg locations relative to DDK_PATH, newest layout first.
_OMG_RELATIVE_CANDIDATES = (
    os.path.join("tools", "tools_omg", "omg"),  # ddk_0820+ (version wrapper)
    os.path.join("tools", "tools_omg", "master", "omg"),  # ddk_0820+ (binary)
    os.path.join("tools", "n", "omg"),  # older CANN Kit layout
)


def resolve_omg(ddk_path=None):
    """Return the DDK omg tool path under ``DDK_PATH``.

    The DDK env script only adds the ascendc/compiler dirs to PATH, so omg must
    be located relative to ``DDK_PATH`` instead of assumed to be on PATH. Raise
    a clear error when ``DDK_PATH`` is unset or contains no known omg location.
    """
    if ddk_path is None:
        ddk_path = (os.environ.get("DDK_PATH") or "").strip()
    if not ddk_path:
        raise RuntimeError(
            "DDK_PATH is not set; source the DDK environment first, e.g. "
            "`source $DDK/tools/tools_ascendc/set_ascendc_env.sh`"
        )
    ddk_root = os.path.realpath(ddk_path)
    for rel in _OMG_RELATIVE_CANDIDATES:
        omg = os.path.join(ddk_root, rel)
        if os.path.isfile(omg):
            return omg
    raise FileNotFoundError(
        "no omg tool under DDK_PATH=%s (tried %s)"
        % (ddk_root, ", ".join(_OMG_RELATIVE_CANDIDATES))
    )


def _omg_invocation(omg):
    """How to invoke omg: version wrapper scripts may lack the exec bit."""
    try:
        with open(omg, "rb") as f:
            if f.read(2) == b"#!":
                return ["bash", omg]
    except OSError:
        pass
    return [omg]


def embedding_weight_elems(vocab_size, hidden_size, quant):
    """Byte-element count of the embedding_weight graph input.

    W4A16 (g32): vocab * (hidden/2 + hidden/32*2)
    W4A8  (g128): ceil(vocab,16) * (hidden/2 + hidden/128*4)
    """
    if quant in (None, "", "FP16"):
        return vocab_size * hidden_size
    if quant == "W4A16":
        return vocab_size * (hidden_size // 2 + hidden_size // 32 * 2)
    if quant == "W4A8":
        ceil_v = (vocab_size + 15) // 16 * 16
        return ceil_v * (hidden_size // 2 + hidden_size // 128 * 4)
    raise ValueError(f"quant {quant} not supported")


def build_omg_command(onnx_path, omc_path, config, max_seq_len, chunk_sizes, embedding_quant,
                         platform="kirin9020", omg=None):
    """Build the omg command for a Qwen2.5-0.5B NNRT graph."""
    if omg is None:
        omg = resolve_omg()
    vocab_size = config["vocab_size"]
    num_layers = config.get("num_layers", config.get("num_hidden_layers"))
    hidden_size = config["hidden_size"]
    num_kv_heads = config.get("num_kv_heads", config.get("num_key_value_heads"))
    num_heads = config.get("num_heads", config.get("num_attention_heads"))
    head_dim = hidden_size // num_heads

    emb_elems = embedding_weight_elems(vocab_size, hidden_size, embedding_quant)

    parts = ["valid_seq_len:1", "lmhead_idx:1"]
    parts += [f"rope_cos:1,-1,{head_dim}", f"rope_sin:1,-1,{head_dim}"]
    parts += [f"inputs_embeds:1,-1,{hidden_size}"]
    parts += [f"attention_mask:1,1,-1,{max_seq_len}"]
    parts += [f"embedding_weight:{emb_elems}"]
    for i in range(num_layers):
        parts += [f"past_key_{i}:1,{num_kv_heads},{max_seq_len},{head_dim}"]
        parts += [f"past_val_{i}:1,{num_kv_heads},{max_seq_len},{head_dim}"]
    input_shape = ";".join(parts)

    dynamic_dims = ";".join(",".join([str(c)] * 4) for c in chunk_sizes)

    cmd = _omg_invocation(omg) + [
        f"--model={onnx_path}",
        "--framework=5",
        f"--output={omc_path}",
        "--target=omc",
        f"--platform={platform}",
        f"--input_shape={input_shape}",
        f"--dynamic_dims={dynamic_dims}",
    ]
    return cmd


def compile_omc(onnx_path, config, max_seq_len=1024, chunk_sizes=(128,), embedding_quant="W4A16",
                   omc_path=None, platform="kirin9020"):
    """Compile onnx -> .omc. Returns the .omc path."""
    if omc_path is None:
        omc_path = os.path.splitext(onnx_path)[0]
    cmd = build_omg_command(onnx_path, omc_path, config, max_seq_len, chunk_sizes, embedding_quant, platform)
    logger.info("Running omg: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return omc_path + ".omc"
