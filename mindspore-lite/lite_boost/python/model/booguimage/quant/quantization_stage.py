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
"""Offline BooguImage quantization: one public call from model to package."""

import importlib
import json
import logging
import os
import sys
import time

import torch
from torch import nn
import yaml

from lite_boost.model.booguimage.quant.package_w8a8 import PackageW8A8
from lite_boost.model.booguimage.quant.smooth_quant import DEFAULT_ALPHA, quantize_fused_ffn, quantize_linear

logger = logging.getLogger(__name__)
QUANT_ALGS = ("W8A8SQFusedFFN", "W8A8SQFusedFFNFull")
COUT_ALIGN = 256
CALIB_RESOLUTION = 1024
CALIB_NUM_INFERENCE_STEPS = 4


_QKVO_PROC_LEAVES = (
    "img_to_q", "img_to_k", "img_to_v",
    "instruct_to_q", "instruct_to_k", "instruct_to_v",
    "img_out", "instruct_out",
)

_QKVO_SIMPLE_ATTN_LISTS = (
    "noise_refiner", "ref_image_refiner", "context_refiner",
    "single_stream_layers",
)

EXPECTED_QKVO = 256

_DIT_FFN_BLOCKS = (
    ("noise_refiner", ("feed_forward",)),
    ("ref_image_refiner", ("feed_forward",)),
    ("context_refiner", ("feed_forward",)),
    ("double_stream_layers", ("img_feed_forward", "instruct_feed_forward")),
    ("single_stream_layers", ("feed_forward",)),
)

_TEXT_MLP_LEAVES = ("gate_proj", "up_proj")


def _collect_dit_ffn_module_targets(transformer):
    """Yield ``(block, ffn_name, path, ffn)`` for every DiT FFN module."""
    for list_name, ffn_names in _DIT_FFN_BLOCKS:
        blocks = getattr(transformer, list_name, None) or []
        for block_idx, block in enumerate(blocks):
            for ffn_name in ffn_names:
                ffn = getattr(block, ffn_name, None)
                if not isinstance(ffn, nn.Module):
                    continue
                if all(isinstance(getattr(ffn, leaf, None), nn.Linear)
                       for leaf in ("linear_1", "linear_2", "linear_3")):
                    yield (block, ffn_name,
                           f"{list_name}.{block_idx}.{ffn_name}", ffn)


def _validate_fused_ffn_targets(targets):
    """Fast-fail structure checks for the fused swap (pre-calibration)."""
    for unused_block, unused_ffn_name, path, ffn in targets:
        linear_1, linear_3 = ffn.linear_1, ffn.linear_3
        if linear_1.weight.shape != linear_3.weight.shape:
            raise ValueError(
                f"{path}: fused FFN requires linear_1/linear_3 of equal "
                f"shape, got {tuple(linear_1.weight.shape)} vs "
                f"{tuple(linear_3.weight.shape)}"
            )
        if linear_1.bias is not None or linear_3.bias is not None:
            raise ValueError(
                f"{path}: fused FFN requires bias-free linear_1/linear_3 "
                f"(the packed matmul carries no bias)"
            )
        inner_dim = linear_1.weight.shape[0]
        if (2 * inner_dim) % COUT_ALIGN:
            raise ValueError(
                f"{path}: packed cout {2 * inner_dim} is not aligned to "
                f"{COUT_ALIGN}; the fused matmul would drop out of the top "
                f"throughput tier"
            )


def _collect_text_targets(mllm):
    """Yield ``(parent, leaf, path, linear)`` for text-encoder gate/up_proj."""
    lang = getattr(mllm, "language_model", None)
    layers = getattr(lang, "layers", None) if lang is not None else None
    for layer_idx, layer in enumerate(layers or []):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for leaf in _TEXT_MLP_LEAVES:
            linear = getattr(mlp, leaf, None)
            if isinstance(linear, nn.Linear):
                yield mlp, leaf, f"language_model.layers.{layer_idx}.mlp.{leaf}", linear


def _append_attn_qkv_targets(targets, attn, prefix):
    """Append to_q/k/v leaves of ``attn`` (None for deleted joint qkv)."""
    for leaf in ("to_q", "to_k", "to_v"):
        linear = getattr(attn, leaf, None)
        if isinstance(linear, nn.Linear):
            targets.append((attn, leaf, f"{prefix}.{leaf}", linear))


def _append_attn_to_out_target(targets, attn, prefix):
    """Append the to_out.0 leaf; the parent is a ModuleList (leaf ``"0"``)."""
    to_out = getattr(attn, "to_out", None)
    linear = to_out[0] if to_out is not None and len(to_out) > 0 else None
    if isinstance(linear, nn.Linear):
        targets.append((to_out, "0", f"{prefix}.to_out.0", linear))


def _collect_dit_attn_targets(transformer):
    """Yield ``(parent, leaf, path, linear)`` for every DiT qkvo projection.

    Cover the single-stream/refiner blocks (plain ``block.attn``), the
    double-stream img_self_attn (full qkv+out), and the joint
    img_instruct_attn (its to_q/k/v are deleted at block init — the
    projections live on the attention processor instance; only to_out
    stays on the module). The caller gates the total against
    ``EXPECTED_QKVO``.
    """
    targets = []
    for list_name in _QKVO_SIMPLE_ATTN_LISTS:
        blocks = getattr(transformer, list_name, None) or []
        for block_idx, block in enumerate(blocks):
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            prefix = f"{list_name}.{block_idx}.attn"
            _append_attn_qkv_targets(targets, attn, prefix)
            _append_attn_to_out_target(targets, attn, prefix)

    blocks = getattr(transformer, "double_stream_layers", None) or []
    for block_idx, block in enumerate(blocks):
        img_attn = getattr(block, "img_self_attn", None)
        if img_attn is not None:
            prefix = f"double_stream_layers.{block_idx}.img_self_attn"
            _append_attn_qkv_targets(targets, img_attn, prefix)
            _append_attn_to_out_target(targets, img_attn, prefix)
        jattn = getattr(block, "img_instruct_attn", None)
        if jattn is None:
            continue
        jprefix = f"double_stream_layers.{block_idx}.img_instruct_attn"
        _append_attn_to_out_target(targets, jattn, jprefix)
        processor = getattr(jattn, "processor", None)
        for leaf in _QKVO_PROC_LEAVES:
            linear = getattr(processor, leaf, None) if processor is not None else None
            if isinstance(linear, nn.Linear):
                targets.append((processor, leaf,
                                f"{jprefix}.processor.{leaf}", linear))
    return targets


def load_calib_prompts_file(path):
    """Read a calibration prompt list from a JSON file.

    The file must contain a non-empty JSON array of non-empty strings.
    Relative paths are resolved against the process CWD; the offline
    package builder absolutizes them against its YAML directory first.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("Quant.calib_prompts must be a JSON file path")
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "r", encoding="utf-8") as stream:
        prompts = json.load(stream)
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(
            f"Quant.calib_prompts file {path} must contain a non-empty "
            f"JSON array of prompt strings"
        )
    for prompt in prompts:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Quant.calib_prompts entries must be non-empty strings, "
                f"got {prompt!r}"
            )
    return prompts


class _ActAbsmaxCollector:
    """Running per-input-channel activation absmax via forward pre-hooks.

    Mathematically equivalent to buffering every calibration input and
    taking the absmax at build time, but it keeps one fp32 ``[cin]`` vector
    per module instead of tens of GB of bf16 activations. Modules that only
    ever see empty inputs stay unmarked so they are left in BF16.
    """

    def __init__(self):
        self._handles = []
        self._absmax = {}

    def _hook(self, module, inputs):
        """Pre-hook updating the module's per-input-channel absmax."""
        x = inputs[0]
        if x.numel() == 0:
            return
        cin = x.shape[-1]
        current = x.reshape(-1, cin).abs().amax(dim=0).float().cpu()
        previous = self._absmax.get(id(module))
        if previous is None:
            self._absmax[id(module)] = current
        else:
            self._absmax[id(module)] = torch.maximum(previous, current)

    def attach(self, modules):
        for module in modules:
            self._handles.append(module.register_forward_pre_hook(self._hook))

    def detach(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def get(self, module):
        """fp32 ``[cin]`` absmax, or ``None`` when no non-empty input was seen."""
        return self._absmax.get(id(module))


def _default_t2i_kwargs(prompt, seed, device):
    """BooguImageTurboPipeline T2I call arguments for one calibration pass."""
    return {
        "instruction": prompt,
        "negative_instruction": "",
        "empty_instruction": "",
        "height": CALIB_RESOLUTION,
        "width": CALIB_RESOLUTION,
        "max_input_image_pixels": CALIB_RESOLUTION * CALIB_RESOLUTION,
        "max_input_image_side_length": 2 * CALIB_RESOLUTION,
        "num_inference_steps": CALIB_NUM_INFERENCE_STEPS,
        "text_guidance_scale": 1.0,
        "image_guidance_scale": 1.0,
        "empty_instruction_guidance_scale": 0.0,
        "use_dmd_student_inference": True,
        "dmd_conditioning_sigma": 0.001,
        "generator": torch.Generator(device).manual_seed(seed),
        "device": device,
    }


def calibrate(pipe, hook_modules, prompts):
    """Run one real T2I generation per calibration prompt."""
    collector = _ActAbsmaxCollector()
    collector.attach(hook_modules)
    device = next(pipe.transformer.parameters()).device
    started = time.time()
    try:
        with torch.inference_mode():
            for i, prompt in enumerate(prompts):
                _ = pipe(**_default_t2i_kwargs(prompt, 42 + i, device))
                if device.type == "npu":
                    torch.npu.synchronize()
                logger.info("W8A8 calibration %d/%d done (%.1fs elapsed)",
                            i + 1, len(prompts), time.time() - started)
    finally:
        collector.detach()
    return collector


def _read_recipe(config):
    """Validate a YAML recipe before loading the model or touching the device."""
    config = os.path.abspath(os.path.expanduser(os.fspath(config)))
    with open(config, "r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    quant = document.get("Quant") if isinstance(document, dict) else None
    if not isinstance(quant, dict) or quant.get("alg") not in QUANT_ALGS:
        raise ValueError(f"Quant.alg must be one of {QUANT_ALGS}")
    alpha = float(quant.get("alpha", DEFAULT_ALPHA))
    if not 0 < alpha <= 1:
        raise ValueError("Quant.alpha must be in (0, 1]")
    prompts_path = quant.get("calib_prompts")
    if isinstance(prompts_path, str):
        prompts_path = os.path.expanduser(prompts_path)
        if not os.path.isabs(prompts_path):
            prompts_path = os.path.join(os.path.dirname(config), prompts_path)
    return {"alg": quant["alg"], "alpha": alpha,
            "prompts": load_calib_prompts_file(prompts_path)}


def _collect_targets(pipe, alg):
    """Enumerate fused FFNs, text Linears and optional attention projections."""
    if pipe.__class__.__name__ != "BooguImageTurboPipeline":
        raise ValueError("Quantization supports BooguImageTurboPipeline only")
    ffn = list(_collect_dit_ffn_module_targets(pipe.transformer))
    _validate_fused_ffn_targets(ffn)
    text = list(_collect_text_targets(getattr(pipe, "mllm", None)))
    qkvo = []
    if alg == "W8A8SQFusedFFNFull":
        qkvo = _collect_dit_attn_targets(pipe.transformer)
        if len(qkvo) != EXPECTED_QKVO:
            raise ValueError(f"Expected {EXPECTED_QKVO} qkvo projections; found {len(qkvo)}")
    if not ffn:
        raise ValueError("No BooguImage FFN targets found")
    return {"ffn": ffn, "text": text, "qkvo": qkvo}


def _linear_meta(linear, path, component, pad_output):
    """Record original dimensions, without any hardware padding decisions."""
    return {"path": path, "component": component, "kind": "linear",
            "in_features": int(linear.in_features),
            "out_features": int(linear.out_features),
            "pad_output": pad_output}


class _BuildResult:
    """Offline tensor data and source records; never contains runtime modules."""

    def __init__(self, recipe):
        self.records = []
        self.modules = []
        self.tensors = {}
        self.targets = []
        self.stats = {"alg": recipe["alg"], "alpha": recipe["alpha"],
                      "calib_prompts": recipe["prompts"], "kept_bf16_paths": []}

    def add(self, target, meta, tensors):
        """Append a canonical tensor record, retaining source information for pruning."""
        parent, attr, path, original = target
        meta["tensors"] = sorted(tensors)
        self.modules.append(meta)
        self.records.append((parent, attr, original, meta["kind"], path))
        self.tensors.update({f"{path}#{key}": tensor for key, tensor in tensors.items()})


def _quantize_ffn(target, collector, recipe, result):
    """Quantize a complete gate/up FFN and optionally its down projection."""
    unused_parent, unused_attr, path, ffn = target
    act = collector.get(ffn)
    if act is None:
        result.stats["kept_bf16_paths"].append(path)
        return
    tensors = quantize_fused_ffn(ffn, act, recipe["alpha"])
    inner_dim, dim = ffn.linear_1.weight.shape
    full = recipe["alg"] == "W8A8SQFusedFFNFull"
    meta = {"path": path, "component": "transformer", "kind": "ffn",
            "dim": dim, "inner_dim": inner_dim, "has_int8_l2": full}
    if full:
        down_act = collector.get(ffn.linear_2)
        if down_act is None:
            raise ValueError(f"{path}: missing linear_2 calibration statistics")
        down = quantize_linear(ffn.linear_2, down_act, recipe["alpha"])
        tensors.update({f"linear_2.{key}": value for key, value in down.items()})
        meta["l2_out_features"] = ffn.linear_2.out_features
        meta["l2_in_features"] = ffn.linear_2.in_features
    result.add(target, meta, tensors)
    leaves = (ffn.linear_1, ffn.linear_3, ffn.linear_2) if full else (ffn.linear_1, ffn.linear_3)
    for linear in leaves:
        linear.to("cpu")


def _quantize_linears(targets, collector, recipe, result):
    """Quantize text or attention projections without replacing model modules."""
    for target in targets:
        unused_parent, unused_attr, path, linear = target
        act = collector.get(linear)
        if act is None:
            result.stats["kept_bf16_paths"].append(path)
            continue
        component = "mllm" if path.startswith("language_model.") else "transformer"
        meta = _linear_meta(linear, path, component, component == "transformer")
        result.add(target, meta, quantize_linear(linear, act, recipe["alpha"]))
        device_type = linear.weight.device.type
        linear.to("cpu")
        if device_type == "npu":
            torch.npu.empty_cache()


def _build_result(targets, collector, recipe):
    """Create the two supported schemes as unpadded offline tensor data."""
    result = _BuildResult(recipe)
    for kind, entries in targets.items():
        component = "mllm" if kind == "text" else "transformer"
        result.targets.extend({"component": component, "path": entry[2]}
                              for entry in entries)
    for target in targets["ffn"]:
        _quantize_ffn(target, collector, recipe, result)
    _quantize_linears(targets["text"] + targets["qkvo"], collector, recipe, result)
    kept = result.stats["kept_bf16_paths"]
    result.stats.update(n_swapped=len(result.modules), n_kept_bf16=len(kept))
    result.stats["n_l2_int8"] = sum(meta.get("has_int8_l2", False) for meta in result.modules)
    result.stats["n_qkvo_int8"] = sum(meta["kind"] == "linear" and
                                        meta["component"] == "transformer"
                                        for meta in result.modules)
    return result


def _load_pipeline(model_path, device, boogu_path):
    """Load the source model and apply the same NPU patches used by inference."""
    importlib.import_module("torch_npu")
    if not device.startswith("npu:") or not torch.npu.is_available():
        raise ValueError("Offline quantization requires an available npu:<index> device")
    if boogu_path:
        sys.path.insert(0, os.path.abspath(os.path.expanduser(boogu_path)))
    os.environ["device"] = device
    pipeline_module = importlib.import_module("boogu.pipelines.boogu.pipeline_boogu_turbo")
    pipe = pipeline_module.BooguImageTurboPipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    pipe.to(device)
    from lite_boost import BoostManager
    BoostManager(pipe)
    return pipe


class BooguImageQuantizer:
    """Public offline entry point; no test scripts or user orchestration required."""

    @staticmethod
    def quantize(model_path, output, config, device="npu:0", boogu_path=None):
        """Calibrate, quantize and write a standalone package; return its manifest.

        Config is a YAML path with Quant.alg, Quant.alpha and Quant.calib_prompts.
        Model and output directories must be separate; existing packages are refused.
        """
        recipe = _read_recipe(config)
        model_path = os.path.realpath(os.path.expanduser(os.fspath(model_path)))
        output = os.path.realpath(os.path.expanduser(os.fspath(output)))
        if not os.path.isdir(model_path):
            raise ValueError(f"Source model directory not found: {model_path}")
        same_drive = os.path.splitdrive(model_path)[0] == os.path.splitdrive(output)[0]
        if same_drive and os.path.commonpath((model_path, output)) in (model_path, output):
            raise ValueError("Model and output directories must not overlap")
        if os.path.exists(output) and (not os.path.isdir(output) or os.listdir(output)):
            raise ValueError(f"Output must be a new or empty directory: {output}")
        pipe = _load_pipeline(model_path, device, boogu_path)
        targets = _collect_targets(pipe, recipe["alg"])
        hooks = [entry[3] for group in targets.values() for entry in group]
        if recipe["alg"] == "W8A8SQFusedFFNFull":
            hooks.extend(target[3].linear_2 for target in targets["ffn"])
        collector = calibrate(pipe, hooks, recipe["prompts"])
        result = _build_result(targets, collector, recipe)
        return PackageW8A8(pipe, model_path, output).write(result)
