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
"""Online-only package loading, padded W8A8 Linear and fused FFN execution."""

import json
import logging
import os
import time
from types import SimpleNamespace

import torch
from torch import nn
import torch_npu

logger = logging.getLogger(__name__)
QUANT_ALGS = ("W8A8SQFusedFFN", "W8A8SQFusedFFNFull")
PAD_MULTIPLE = 256
COUT_ALIGN = 256
_ZERO_I8_BUFFERS = {}


def _shared_zero_i8(rows, delta, device):
    """``[rows, delta]`` int8 zeros (shared buffer, sliced view)."""
    key = (str(device), delta)
    buf = _ZERO_I8_BUFFERS.get(key)
    if buf is None or buf.shape[0] < rows:
        # The initial row count covers the largest expected token count
        # (1024x1024 T2I is ~4.2k rows); the buffer only ever grows.
        buf = torch.zeros(max(rows, 8448), delta, dtype=torch.int8,
                          device=device)
        _ZERO_I8_BUFFERS[key] = buf
    return buf[:rows]


def dynamic_quant_int8(x):
    """Per-token dynamic int8 quantization (bf16 in).

    Returns:
        ``(x_int8 [m, k] int8, scale fp32 [m])``.
    """
    x_int8, scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
    return x_int8, scale.reshape(-1).to(torch.float32)


def quant_matmul_w8a8(x_int8, qweight, weight_scale, pertoken_scale, bias):
    """int8 x int8 matmul with bf16 output via ``torch_npu.npu_quant_matmul``.

    Args:
        x_int8: int8 ``[m, k]`` activations.
        qweight: int8 ``[k, n]`` transposed weight (``[cin, cout]``).
        weight_scale: fp32 ``[n]`` per-output-channel scale.
        pertoken_scale: fp32 ``[m]`` per-token scale.
        bias: bf16 ``[n]`` or ``None``.
    """
    return torch_npu.npu_quant_matmul(
        x_int8,
        qweight,
        scale=weight_scale.to(torch.float32),
        pertoken_scale=pertoken_scale.to(torch.float32),
        bias=bias,
        output_dtype=torch.bfloat16,
    )


def _check_linear_tensors(tensors, meta):
    """Require canonical, unpadded tensors before allocating runtime buffers."""
    cin, cout = meta["in_features"], meta["out_features"]
    if not isinstance(cin, int) or not isinstance(cout, int) or min(cin, cout) <= 0:
        raise ValueError("Linear dimensions must be positive integers")
    expected = {"qweight": ((cin, cout), torch.int8),
                "weight_scale": ((cout,), torch.float32),
                "inv_smooth_scale": ((cin,), torch.bfloat16)}
    if "bias" in tensors:
        expected["bias"] = ((cout,), torch.bfloat16)
    if set(tensors) != set(expected):
        raise ValueError("Linear tensor names do not match the package format")
    for name, (shape, dtype) in expected.items():
        value = tensors[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f"Invalid unpadded linear tensor: {name}")
        if value.device.type == "meta":
            raise ValueError(f"Unmaterialized linear tensor: {name}")
        if name != "qweight" and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite linear tensor: {name}")
    if (tensors["weight_scale"] <= 0).any() or (tensors["inv_smooth_scale"] <= 0).any():
        raise ValueError("Linear scales must be positive")


class W8A8QuantLinear(nn.Module):
    """Online Linear; output padding is created once when loading the package."""

    def __init__(self, tensors, meta):
        super().__init__()
        _check_linear_tensors(tensors, meta)
        self.in_features = meta["in_features"]
        self.out_features = meta["out_features"]
        width = self.out_features
        if meta.get("pad_output"):
            width = ((width + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        self.n_pad = width - self.out_features
        for name, value in tensors.items():
            if self.n_pad and name != "inv_smooth_scale":
                shape = (self.in_features, width) if name == "qweight" else (width,)
                padded = value.new_zeros(shape)
                padded[..., :self.out_features] = value
                value = padded
            self.register_buffer(name, value.contiguous())
        if "bias" not in tensors:
            self.bias = None

    @property
    def weight(self):
        """Expose the runtime weight device for Boogu attention processors."""
        return self.qweight

    def forward(self, x, structure_transform=None):
        """W8A8 matmul on the n-padded cout, sliced back to the original."""
        orig_shape = x.shape
        if x.numel() == 0:
            # Mirror F.linear for empty inputs (the aclnn int8 matmul
            # rejects empty tensors): the T2I empty-instruction and
            # ref_image_refiner paths feed empty activations every step.
            return torch.empty(*orig_shape[:-1], self.out_features,
                               dtype=x.dtype, device=x.device)
        x2d = x.reshape(-1, orig_shape[-1]).contiguous()
        x2d = x2d * self.inv_smooth_scale
        x_int8, pertoken_scale = dynamic_quant_int8(x2d)
        out = quant_matmul_w8a8(x_int8, self.qweight, self.weight_scale,
                                pertoken_scale, self.bias)
        if self.n_pad:
            out = out[:, :self.out_features]
        return out.reshape(*orig_shape[:-1], self.out_features)


class Int8FusedFFN(nn.Module):
    """Online fused gate/up FFN; k padding is absent from the stored tensors."""

    def __init__(self, tensors, meta, linear_2):
        super().__init__()
        self.dim = meta["dim"]
        self.inner_dim = meta["inner_dim"]
        expected = {"qw13": ((self.dim, 2 * self.inner_dim), torch.int8),
                    "w13_scale": ((2 * self.inner_dim,), torch.float32),
                    "inv_smooth13": ((self.dim,), torch.bfloat16)}
        if min(self.dim, self.inner_dim) <= 0 or set(tensors) != set(expected):
            raise ValueError("Invalid fused FFN dimensions or tensor names")
        for name, (shape, dtype) in expected.items():
            value = tensors[name]
            if tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f"Invalid unpadded fused FFN tensor: {name}")
            if name != "qw13" and (not torch.isfinite(value).all() or (value <= 0).any()):
                raise ValueError(f"Invalid fused FFN scale: {name}")
        padded_dim = ((self.dim + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        self.k_pad = padded_dim - self.dim
        weight = tensors["qw13"].new_zeros((padded_dim, 2 * self.inner_dim))
        weight[:self.dim] = tensors["qw13"]
        self.register_buffer("qw13", weight)
        self.register_buffer("w13_scale", tensors["w13_scale"].contiguous())
        self.register_buffer("inv_smooth13", tensors["inv_smooth13"].contiguous())
        self.linear_2 = linear_2

    def forward(self, x):
        """Run fused gate/up, SwiGLU and the down projection."""
        orig_shape = x.shape
        if x.numel() == 0:
            # The FFN preserves the feature dim, so an empty input (T2I
            # ref_image_refiner) is also the empty output.
            return x
        x2d = x.reshape(-1, orig_shape[-1]).contiguous()
        x2d = x2d * self.inv_smooth13
        x_int8, pertoken_scale = dynamic_quant_int8(x2d)
        if self.k_pad:
            zeros = _shared_zero_i8(x_int8.shape[0], self.k_pad,
                                    x_int8.device)
            x_int8 = torch.cat([x_int8, zeros], dim=-1)
        h = quant_matmul_w8a8(x_int8, self.qw13, self.w13_scale,
                              pertoken_scale, None)
        y = torch_npu.npu_swiglu(h, dim=-1)
        out = self.linear_2(y)
        return out.reshape(orig_shape)


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
    """Validate the supported fused FFN layout."""
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


def _pkg_pipeline_modules(pipe):
    """Iterate neural network components of a Diffusers pipeline.

    A pipeline is a container, not an nn.Module. Its registered components
    include modules as well as tokenizers, processors and schedulers.
    """
    return ((name, component) for name, component in pipe.components.items()
            if isinstance(component, torch.nn.Module))


def _pkg_check_materialized(pipe):
    """Reject unresolved meta parameters and buffers in any component."""
    for component_name, component in _pkg_pipeline_modules(pipe):
        for tensors in (component.named_parameters(), component.named_buffers()):
            for name, tensor in tensors:
                if tensor.device.type == "meta":
                    raise ValueError(
                        "package left unresolved meta tensors in the pipeline: "
                        f"{component_name}.{name}")


def _expected_paths(pipe, alg):
    """Validate the online model layout independently of the offline implementation."""
    ffn = list(_collect_dit_ffn_module_targets(pipe.transformer))
    _validate_fused_ffn_targets(ffn)
    text = list(_collect_text_targets(getattr(pipe, "mllm", None)))
    qkvo = []
    if alg == "W8A8SQFusedFFNFull":
        qkvo = _collect_dit_attn_targets(pipe.transformer)
        if len(qkvo) != EXPECTED_QKVO:
            raise ValueError(f"Expected {EXPECTED_QKVO} online qkvo projections; found {len(qkvo)}")
    paths = {(component, path): (parent, attr, module)
             for component, targets in (("transformer", ffn + qkvo), ("mllm", text))
             for parent, attr, path, module in targets}
    return paths


def _load_manifest(package_dir):
    """Online format validation; no dependency on the offline package writer."""
    with open(os.path.join(package_dir, "manifest.json"), "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("Package manifest must be an object")
    if manifest.get("format_version") != 3 or manifest.get("tensor_layout") != "unpadded_kn":
        raise ValueError("Package format_version must be 3 with tensor_layout unpadded_kn")
    if manifest.get("alg") not in QUANT_ALGS:
        raise ValueError(f"Package alg must be one of {QUANT_ALGS}")
    for field in ("modules", "targets", "shards", "kept_bf16"):
        if not isinstance(manifest.get(field), list):
            raise ValueError(f"Package {field} must be a list")
    return manifest


def _check_structure(manifest, paths):
    """Reject duplicate, missing or overlapping quantized/retained targets."""
    declared = [(entry["component"], entry["path"]) for entry in manifest["targets"]]
    modules = [(entry["component"], entry["path"]) for entry in manifest["modules"]]
    kept_names = manifest["kept_bf16"]
    if any(not isinstance(name, str) for name in kept_names):
        raise ValueError("Invalid retained BF16 path")
    kept = [("mllm" if name.startswith("language_model.") else "transformer", name)
            for name in kept_names]
    if len(set(declared)) != len(declared) or set(declared) != set(paths):
        raise ValueError("Package targets do not match the online model layout")
    if (len(set(modules)) != len(modules) or len(set(kept)) != len(kept)
            or set(modules) & set(kept) or set(modules) | set(kept) != set(paths)):
        raise ValueError("Package modules and retained BF16 must exactly cover model targets")
    _check_module_policies(manifest, paths)


def _check_module_policies(manifest, paths):
    """Keep module kinds and padding policies consistent with the chosen scheme."""
    full = manifest["alg"] == "W8A8SQFusedFFNFull"
    for meta in manifest["modules"]:
        original = paths[(meta["component"], meta["path"])][2]
        fused = hasattr(original, "linear_1") and hasattr(original, "linear_3")
        if meta.get("kind") != ("ffn" if fused else "linear"):
            raise ValueError("Package module kind does not match the model target")
        if fused and meta.get("has_int8_l2") is not full:
            raise ValueError("Package FFN quantization does not match its declared algorithm")
        if not fused and meta.get("pad_output") is not (meta["component"] == "transformer"):
            raise ValueError("Package Linear padding policy does not match its target")


def _load_tensors(package_dir, shard_names):
    """Read plain tensors while rejecting duplicate keys and shard path escapes."""
    if not shard_names or len(set(shard_names)) != len(shard_names):
        raise ValueError("Package shards must be a nonempty list without duplicates")
    tensors = {}
    root = os.path.realpath(os.path.join(package_dir, "int8"))
    for filename in shard_names:
        if not isinstance(filename, str) or os.path.basename(filename) != filename:
            raise ValueError("Package shard must be a file name")
        path = os.path.realpath(os.path.join(root, filename))
        if os.path.commonpath((root, path)) != root:
            raise ValueError("Package shard escapes the int8 directory")
        shard = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(shard, dict) or set(shard) & set(tensors):
            raise ValueError("Invalid shard or duplicated tensor keys")
        if any(not isinstance(value, torch.Tensor) for value in shard.values()):
            raise ValueError("Package shard contains non-tensor values")
        tensors.update(shard)
    return tensors


def _module_tensors(meta, tensors):
    """Resolve exactly the tensor names declared for one module."""
    names = meta["tensors"]
    if not isinstance(names, list) or len(set(names)) != len(names):
        raise ValueError("Invalid module tensor list")
    return {name: tensors[f"{meta['path']}#{name}"] for name in names}


def _build_module(meta, tensors, original):
    """Create runtime buffers from canonical package tensors, then pad them."""
    values = _module_tensors(meta, tensors)
    if meta["kind"] == "linear":
        if (original.in_features, original.out_features) != (meta["in_features"], meta["out_features"]):
            raise ValueError(f"Linear shape mismatch: {meta['path']}")
        return W8A8QuantLinear(values, meta).eval()
    if meta["kind"] != "ffn" or original.linear_1.weight.shape != (meta["inner_dim"], meta["dim"]):
        raise ValueError(f"FFN shape mismatch: {meta['path']}")
    linear_2 = original.linear_2
    if meta["has_int8_l2"]:
        down = {name[len("linear_2."):]: value for name, value in values.items()
                if name.startswith("linear_2.")}
        down_meta = {"in_features": meta["l2_in_features"],
                     "out_features": meta["l2_out_features"], "pad_output": True}
        if (linear_2.in_features, linear_2.out_features) != (down_meta["in_features"], down_meta["out_features"]):
            raise ValueError(f"Down projection shape mismatch: {meta['path']}")
        linear_2 = W8A8QuantLinear(down, down_meta).eval()
    fused = {name: value for name, value in values.items() if not name.startswith("linear_2.")}
    return Int8FusedFFN(fused, meta, linear_2).eval()


def apply_w8a8_sq_offline(pipe, package_dir):
    """Online deployment before pipe.to(device); never calibrates or quantizes weights."""
    if pipe.__class__.__name__ != "BooguImageTurboPipeline":
        raise ValueError("W8A8 deployment supports BooguImageTurboPipeline only")
    manifest = _load_manifest(package_dir)
    paths = _expected_paths(pipe, manifest["alg"])
    _check_structure(manifest, paths)
    tensors = _load_tensors(package_dir, manifest["shards"])
    expected_names = [f"{meta['path']}#{name}" for meta in manifest["modules"] for name in meta["tensors"]]
    if len(set(expected_names)) != len(expected_names) or set(expected_names) != set(tensors):
        raise ValueError("Package tensor names do not exactly match its module declarations")
    started = time.time()
    pending = []
    for meta in manifest["modules"]:
        parent, attr, original = paths[(meta["component"], meta["path"])]
        quant = _build_module(meta, tensors, original)
        pending.append((parent, attr, quant))
    for parent, attr, quant in pending:
        setattr(parent, attr, quant)
    _pkg_check_materialized(pipe)
    stats = {**manifest["counts"], "alg": manifest["alg"],
             "package_dir": os.path.abspath(package_dir),
             "kept_bf16_paths": manifest["kept_bf16"],
             "load_time_s": time.time() - started}
    handle = SimpleNamespace(stats=stats)
    setattr(pipe, "_lite_boost_w8a8_handle", handle)
    return handle
