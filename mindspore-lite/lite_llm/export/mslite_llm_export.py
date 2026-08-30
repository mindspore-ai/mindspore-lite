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
"""One-click LLM -> single-file ``.msl`` exporter (qwen2_5 / qwen3).

Orchestrates the internal pipeline modules (all internal API, no CLI):

    models/qwen2_5/qwen2_5_exporter.py     Qwen2.5-0.5B HF/GGUF -> ONNX skeleton
    models/qwen2_5/qwen2_5_gguf_loader.py  Qwen2.5 GGUF Q4_0 raw weight injection
    models/qwen3/qwen3_exporter.py         MiniMind-3 (Qwen3 dense) HF/GGUF -> ONNX skeleton
    models/qwen3/qwen3_gguf_loader.py      MiniMind-3 GGUF Q4_0 raw weight injection
    utils/omc_compiler.py                  DDK omg -> .omc (unconditional)
    utils/export_tokenizer.py              tokenizer -> vocab.bin + policy (含 chat template IR)
    utils/export_quant.py                  quant configs + weight packing + graph quantization
    utils/msl_pack.py                      artifacts -> single-file .msl (v1) + pack/unpack

Interface (see ``README.md`` in this directory):

    mslite-llm-export --target TARGET --model MODEL --output OUTPUT [options]

Preconditions (documented in README): ``pip install -r ../requirements.txt``,
CANN DDK sourced (omg resolved from ``DDK_PATH``, Ms* ops installed).
The final .msl is produced by the bundled Python ``msl_pack`` (no external
packing tool).
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile

# Package root: export/ must be importable so ``models.*`` / ``utils.*`` resolve.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position  # package root injected on sys.path above
from models.qwen2_5.qwen2_5_exporter import export_qwen2_5  # noqa: E402
from models.qwen2_5.qwen2_5_gguf_loader import gguf_loader as qwen2_5_gguf_loader  # noqa: E402
from models.qwen3.qwen3_exporter import export_qwen3  # noqa: E402
from models.qwen3.qwen3_gguf_loader import gguf_loader as qwen3_gguf_loader  # noqa: E402
from utils.omc_compiler import compile_omc, resolve_omg  # noqa: E402
from utils.msl_pack import build_single_file_msl  # noqa: E402
from utils.export_quant import QuantizationConfig  # noqa: E402
from utils.export_tokenizer import export_tokenizer  # noqa: E402

logger = logging.getLogger(__name__)

SUPPORTED_TARGETS = ("kirin9020",)

# Model metadata architecture -> MODEL_TYPES key.
MODEL_TYPE_BY_ARCH = {"qwen2": "qwen2_5", "qwen3": "qwen3"}

# GGUF Q4_0 == W4A16 group-32 quantization (no calibration data needed).
GGUF_QUANT = "W4A16"

# Pinned ChatML template for the Qwen2.5 / Qwen3 template family.
# Real HF/GGUF tokenizer metadata carries the full tool-call Jinja template,
# which is deliberately outside the v1 restricted-IR subset ({% if tools %}
# folds to false, no implicit system injection, no tool branches).  The v1
# runtime interprets exactly this IR, so the template is pinned at export time
# instead of being taken verbatim from model metadata.
QWEN_CHAT_TEMPLATE = (
    "{%- if tools %}\n"
    "    {{- '<|im_start|>system\\n' }}\n"
    "{%- endif %}\n"
    "{%- for message in messages %}\n"
    "    {{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + "
    "'<|im_end|>' + '\\n' }}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n"
    "{%- endif %}\n"
)

# NNRT omg dynamic-dims decode gear (prefill gear is ``args.chunk_size``).
DECODE_GEAR = 1

# Per-model-type pipeline knobs.
MODEL_TYPES = {
    "qwen2_5": {
        "exporter": export_qwen2_5,
        "gguf_loader": qwen2_5_gguf_loader,
        "layers": 24,
        "onnx_name": "qwen2_5_0b5.onnx",
        "quant_name": "qwen2_5_0b5_quant.onnx",
        "gguf_name": "qwen2_5_0b5_gguf.onnx",
        "model_name": "qwen2.5-0.5b",
    },
    "qwen3": {
        "exporter": export_qwen3,
        "gguf_loader": qwen3_gguf_loader,
        "layers": 8,
        "onnx_name": "minimind3_qwen3.onnx",
        "quant_name": "minimind3_qwen3_quant.onnx",
        "gguf_name": "minimind3_qwen3_gguf.onnx",
        "model_name": "minimind-3-qwen3",
    },
}


def detect_model_kind(model):
    """Return ``"gguf"`` or ``"hf"`` based on the ``--model`` argument."""
    if os.path.isfile(model):
        if model.endswith(".gguf"):
            return "gguf"
        raise ValueError(f"--model is a file but not a .gguf: {model}")
    if os.path.isdir(model):
        return "hf"
    raise ValueError(f"--model does not exist: {model}")


def detect_model_type(model):
    """Detect the MODEL_TYPES key from model metadata (config.json / GGUF).

    Validates the layer count against the reference model too: the pipeline
    only supports the two pre-validated sizes, and a wrong layer count would
    silently produce a corrupt skeleton/weight mix otherwise.
    """
    if os.path.isfile(model) and model.endswith(".gguf"):
        from gguf import GGUFReader

        reader = GGUFReader(model)
        try:
            arch = reader.fields["general.architecture"].parts[-1].tobytes().decode("utf-8")
            layers = int(reader.fields[f"{arch}.block_count"].parts[-1])
        except KeyError as exc:
            raise ValueError(f"GGUF metadata missing required field: {exc}") from exc
    else:
        with open(os.path.join(model, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        arch = config.get("model_type")
        layers = config.get("num_hidden_layers")
        if arch is None or layers is None:
            raise ValueError("HF config.json missing model_type or num_hidden_layers")

    key = MODEL_TYPE_BY_ARCH.get(arch)
    if key is None:
        raise ValueError(f"unsupported model architecture: {arch}")
    expected = MODEL_TYPES[key]["layers"]
    if layers != expected:
        raise ValueError(
            f"unsupported model size: {arch} with {layers} layers "
            f"(expected {expected}; only the reference model is supported)"
        )
    return key


def run_pipeline(args, work_dir):
    """Execute the full export pipeline inside ``work_dir``. Returns the .msl path."""
    model_kind = detect_model_kind(args.model)
    model_type = detect_model_type(args.model)
    logger.info("model kind: %s (%s), type: %s", model_kind, args.model, model_type)
    mt = MODEL_TYPES[model_type]

    # ── Step 1: skeleton export (HF/GGUF -> ONNX + assets) ────────────────
    if model_kind == "gguf":
        # Skeleton from GGUF (transformers dequantizes for the graph), weights
        # quantized to W4A16 placeholders; real Q4_0 weights injected in step 2.
        mt["exporter"](
            model_dir=args.model,
            output_dir=work_dir,
            max_length=args.max_length,
            chunk_size=args.chunk_size,
            embedding_quant=GGUF_QUANT,
            decoder_quant=GGUF_QUANT,
            layers=mt["layers"],
            model_name=mt["model_name"],
        )
        quant_onnx = os.path.join(work_dir, mt["quant_name"])
        if not os.path.exists(quant_onnx):
            raise RuntimeError(f"expected quantized skeleton not produced: {quant_onnx}")
        onnx_path = os.path.join(work_dir, mt["gguf_name"])
        embedding_bin = os.path.join(work_dir, "embedding_weight.bin")
        mt["gguf_loader"](
            gguf_path=args.model,
            onnx_input_path=quant_onnx,
            onnx_output_path=onnx_path,
            embedding_weight_save_path=embedding_bin,
            layers=mt["layers"],
            embedding_quantize_config=GGUF_QUANT,
            decoder_quantize_config=GGUF_QUANT,
        )
        embedding_quant = GGUF_QUANT
    else:
        mt["exporter"](
            model_dir=args.model,
            output_dir=work_dir,
            max_length=args.max_length,
            chunk_size=args.chunk_size,
            embedding_quant=None,
            decoder_quant=None,
            layers=mt["layers"],
            model_name=mt["model_name"],
        )
        onnx_path = os.path.join(work_dir, mt["onnx_name"])
        embedding_bin = os.path.join(work_dir, "embedding.bin")
        embedding_quant = None

    # ── Step 2: omg compile (ONNX -> .omc, unconditional) ─────────────────
    arch_path = os.path.join(work_dir, "architecture.json")
    with open(arch_path, encoding="utf-8") as f:
        architecture = json.load(f)

    omc_path = compile_omc(
        onnx_path=onnx_path,
        config=architecture,
        max_seq_len=args.max_length,
        chunk_sizes=(DECODE_GEAR, args.chunk_size),  # decode=1 / prefill=chunk_size
        embedding_quant=embedding_quant,
        omc_path=os.path.join(work_dir, "model"),
        platform=args.target,
    )

    # ── Step 3: tokenizer -> vocab.bin ────────────────────────────────────
    # Chat template is pinned to the canonical ChatML IR; real GGUF
    # metadata templates (tool-call Jinja) are outside the v1 IR subset.
    tokenizer_dir = os.path.join(work_dir, "tokenizer")
    vocab_path = export_tokenizer(
        model_dir=args.model,
        output_dir=tokenizer_dir,
        chat_template=QWEN_CHAT_TEMPLATE,
    )
    policy_path = os.path.join(tokenizer_dir, "generation_policy.json")
    with open(policy_path, encoding="utf-8") as f:
        generation_policy = json.load(f)

    # ── Step 4: package into single-file .msl ─────────────────────────────
    package_name = os.path.splitext(os.path.basename(args.output))[0]
    npu_config = {
        "max_length": args.max_length,
        "chunk_size": args.chunk_size,
        "embedding_quant": embedding_quant is not None,
        # W4A16 g32 group size (QuantizationConfig derives it from the method).
        "scale_gp_size": QuantizationConfig(embedding_quant).group_size,
    }

    result = build_single_file_msl(
        omc_path=omc_path,
        vocab_path=vocab_path,
        embedding_path=embedding_bin,
        rope_cos=os.path.join(work_dir, "rope_cos.bin"),
        rope_sin=os.path.join(work_dir, "rope_sin.bin"),
        attention_mask=os.path.join(work_dir, "attention_mask.bin"),
        architecture=architecture,
        npu_config=npu_config,
        generation_policy=generation_policy,
        package_name=package_name,
        output_path=args.output,
    )
    logger.info("exported %s", result)
    return result


def build_parser():
    """Build the CLI argument parser for mslite-llm-export."""
    parser = argparse.ArgumentParser(
        prog="mslite-llm-export",
        description="One-click LLM (Qwen2.5-0.5B / MiniMind-3 Qwen3, GGUF Q4_0 or HF) -> single-file .msl",
    )
    parser.add_argument("--target", default="kirin9020",
                        help=f"Deployment chip (default: kirin9020), one of {list(SUPPORTED_TARGETS)}")
    parser.add_argument("--model", required=True, help="GGUF file (.gguf) or HF model directory")
    parser.add_argument("--output", required=True, help="Output single-file .msl path")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length (default: 1024)")
    parser.add_argument("--chunk-size", type=int, default=64, help="Prefill chunk size (default: 64)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


def _infer_ddk_root(omg_path, target):
    """Walk up from the omg binary to the DDK root (has ``tools/platform/<target>``)."""
    d = os.path.dirname(os.path.abspath(omg_path))
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "tools", "platform", target)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent
    return None


def _log_ddk_env(target):
    """Log DDK / omg environment facts early so misconfiguration fails fast.

    ``omg`` dlopens the DDK platform libraries (``libai_npucore_itf.so``) and
    the Ms* custom op library (``libcustom_op.so``) only at the final compile
    step; when the DDK env was not sourced those dirs are missing from
    LD_LIBRARY_PATH and the failure surfaces minutes after the export started.
    Printing them up front lets a developer confirm the setup before the slow
    pipeline runs. omg is resolved relative to ``DDK_PATH``, not assumed to be
    on PATH.
    """
    omg = None
    omg_error = ""
    try:
        omg = resolve_omg()
    except (RuntimeError, FileNotFoundError) as exc:
        omg_error = str(exc)
    logger.info("DDK environment (target=%s):", target)
    logger.info("  DDK_PATH=%s", os.environ.get("DDK_PATH") or "(not set)")
    logger.info("  ASCEND_HOME_PATH=%s", os.environ.get("ASCEND_HOME_PATH") or "(not set)")
    logger.info("  omg=%s", omg or "(not resolved from DDK_PATH)")
    logger.info("  LD_LIBRARY_PATH=%s", os.environ.get("LD_LIBRARY_PATH") or "(not set)")
    if omg is None:
        logger.warning(
            "omg not resolved from DDK_PATH: %s; source the DDK env first "
            "(e.g. `source $DDK/tools/tools_ascendc/set_ascendc_env.sh`)",
            omg_error,
        )
        return

    ddk_root = _infer_ddk_root(omg, target)
    if ddk_root is None:
        logger.warning("  (could not infer DDK root from omg location)")
        return
    logger.info("  inferred DDK root: %s", ddk_root)

    platform = os.path.join(ddk_root, "tools", "platform", target)
    checks = [
        ("runtime", os.path.join(platform, "lib64"), "libai_npucore_itf.so"),
        ("custom ops", os.path.join(platform, "lib64"), "libcustom_op.so"),
        (
            "custom ops tiling",
            os.path.join(platform, "customize", "op_impl", "ai_core", "tbe", "op_tiling"),
            "libcustom_op.so",
        ),
    ]
    ldl = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    missing = []
    for label, lib_dir, lib_name in checks:
        lib = os.path.join(lib_dir, lib_name)
        ok = os.path.isfile(lib) and lib_dir in ldl
        missing.append(not ok)
        if ok:
            logger.info("  [OK] %s %s (%s)", label, lib_name, lib)
        else:
            hint = "file absent" if not os.path.isfile(lib) else "dir not in LD_LIBRARY_PATH"
            logger.warning("  [MISSING] %s %s (%s)", label, lib_name, hint)

    if any(missing):
        logger.warning(
            "DDK libraries not fully available: source the DDK env before exporting, "
            "e.g. `source $DDK/tools/tools_ascendc/set_ascendc_env.sh`"
        )


def main(argv=None):
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.target not in SUPPORTED_TARGETS:
        raise ValueError(f"--target {args.target} not supported; choose from {list(SUPPORTED_TARGETS)}")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if args.max_length % args.chunk_size != 0:
        logger.warning(
            "--max-length (%d) is not a multiple of --chunk-size (%d); the prefill "
            "tail chunk will be padded. Prefer a multiple for optimal NPU throughput.",
            args.max_length,
            args.chunk_size,
        )

    # Fail fast on DDK misconfiguration: omg only errors at the final compile
    # step, minutes after the export started.
    _log_ddk_env(args.target)

    output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(output_dir, exist_ok=True)

    work_dir = tempfile.mkdtemp(prefix=".msl_export_", dir=output_dir)
    try:
        result = run_pipeline(args, work_dir)
        print(result)
        return result
    finally:
        # Intermediate artifacts (ONNX / .omc / assets) are scratch; the .msl
        # is self-contained.  Remove the work dir to keep the tree clean.
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
