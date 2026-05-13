"""Export Qwen3-TTS ONNX models (talker KV, speech decoder, code predictor) in one entry script."""

from __future__ import annotations

import argparse
import inspect
import importlib
import json
import os
import sys
import types
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

@lru_cache(maxsize=None)
def _import_module(name: str) -> Any:
    """Import a module by name with caching."""
    return importlib.import_module(name)


def _optional_module(name: str) -> Any | None:
    """Import a module if available, otherwise return None."""
    try:
        return _import_module(name)
    except (ImportError, ModuleNotFoundError):
        return None


def _require_module(name: str) -> Any:
    """Import a required module or raise a clear runtime error."""
    mod = _optional_module(name)
    if mod is None:
        raise RuntimeError(f"Missing dependency: {name!r}.")
    return mod


_TORCH = _optional_module("torch")
_ONNX = _optional_module("onnx")

_TORCH_NPU = None
if os.environ.get("QWEN3_TTS_ENABLE_TORCH_NPU", "0") == "1":
    _TORCH_NPU = _optional_module("torch_npu")

if _TORCH is None:

    class _AutogradFunction:
        pass

    class _NNModule:
        pass

else:
    _AutogradFunction = _TORCH.autograd.Function
    _NNModule = _TORCH.nn.Module

onnx_helper = getattr(_ONNX, "helper", None)
onnx_numpy_helper = getattr(_ONNX, "numpy_helper", None)
torch = _TORCH
onnx = _ONNX


class CustomRoatryMul(_AutogradFunction):
    """Custom rotary multiplication operator for ONNX export."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        del ctx
        dim = int(x.shape[-1])
        half = dim // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)

    @staticmethod
    def symbolic(g: torch.Graph, x, cos, sin):
        return g.op(
            "Custom",
            x,
            cos,
            sin,
            input_names_s=["x", "r1", "r2"],
            output_names_s=["y"],
            type_s="RotaryMul",
        )


class CustomIncreFlashAttention(_AutogradFunction):
    """Custom incremental flash attention operator for ONNX export."""

    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        atten_mask: torch.Tensor,
        num_heads: int,
        scale_value: float,
        input_layout: str = "BNSD",
        num_key_value_heads: int = 0,
    ):
        """Fallback forward used during tracing and shape inference."""
        del ctx, key, value, atten_mask, num_heads, scale_value, input_layout, num_key_value_heads
        return query

    @staticmethod
    def symbolic(
        g: torch.Graph,
        query,
        key,
        value,
        atten_mask,
        num_heads: int,
        scale_value: float,
        input_layout: str = "BNSD",
        num_key_value_heads: int = 0,
    ):
        """Export a Custom node for incremental flash attention."""
        input_names = [
            "query",
            "key",
            "value",
            "atten_mask",
        ]
        optional_input_names = [
            "atten_mask",
        ]
        return g.op(
            "Custom",
            query,
            key,
            value,
            atten_mask,
            input_index_i=[0, 1, 2, 3],
            input_names_s=input_names,
            optional_input_names_s=optional_input_names,
            output_names_s=["attention_out"],
            type_s="IncreFlashAttention",
            num_heads_i=int(num_heads),
            scale_value_f=float(scale_value),
            input_layout_s=str(input_layout),
            num_key_value_heads_i=int(num_key_value_heads),
        )


class CustomAddRmsNorm(_AutogradFunction):
    """Custom fused Add + RMSNorm operator for ONNX export."""

    @staticmethod
    def forward(
        ctx: Any,
        x1: torch.Tensor,
        x2: torch.Tensor,
        gamma: torch.Tensor,
        epsilon: float = 1e-6,
    ):
        """Compute Add + RMSNorm with an optional NPU fused kernel."""
        del ctx
        if _TORCH_NPU is not None and x1.device.type == "npu":
            return _TORCH_NPU.npu_add_rms_norm(x1, x2, gamma, epsilon=float(epsilon))
        x = x1 + x2
        rstd = torch.rsqrt(x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True) + float(epsilon))
        y = (x.to(torch.float32) * rstd).to(x.dtype) * gamma
        rstd_full = rstd.expand_as(x).to(torch.float32)
        return y, rstd_full, x

    @staticmethod
    def symbolic(g: torch.Graph, x1, x2, gamma, epsilon: float = 1e-6):
        return g.op(
            "Custom",
            x1,
            x2,
            gamma,
            input_index_i=[0, 1, 2],
            input_names_s=["x1", "x2", "gamma"],
            optional_input_names_s=[],
            output_names_s=["y", "rstd", "x"],
            type_s="AddRmsNorm",
            epsilon_f=float(epsilon),
            outputs=3,
        )


class CustomScatterUpdate(_AutogradFunction):
    """Custom scatter update operator for KV cache update."""

    @staticmethod
    def forward(
        ctx: Any,
        var: torch.Tensor,
        indices: torch.Tensor,
        updates: torch.Tensor,
        axis: int = 0,
    ):
        """Update a single cache position along the given axis."""
        del ctx
        if _TORCH_NPU is not None and var.device.type == "npu":
            out = var.clone()
            _TORCH_NPU.scatter_update_(
                out,
                indices.to(torch.int64),
                updates.to(out.dtype),
                axis=int(axis),
            )
            return out

        cache_total = int(var.size(int(axis)))
        pos = indices.reshape(-1)[0].to(torch.int64)
        e = torch.nn.functional.one_hot(pos, num_classes=cache_total).to(var.dtype)
        e_bt11 = e.view(1, 1, cache_total, 1)
        old = torch.matmul(var.transpose(2, 3), e_bt11).transpose(2, 3)
        return var + e_bt11 * (updates.to(var.dtype) - old.to(var.dtype))

    @staticmethod
    def symbolic(g: torch.Graph, var, indices, updates, axis: int = 0):
        return g.op(
            "Custom",
            var,
            indices,
            updates,
            input_index_i=[0, 1, 2],
            input_names_s=["var", "indices", "updates"],
            optional_input_names_s=[],
            output_names_s=["var"],
            type_s="Scatter",
            reduce_s="update",
            axis_i=int(axis),
        )


def _kv_cache_update(
    past: torch.Tensor,
    update: torch.Tensor,
    cache_pos: torch.Tensor,
) -> torch.Tensor:
    """Update KV cache at cache_pos and return the updated tensor."""
    if _TORCH_NPU is not None and past.device.type == "npu":
        if torch.is_tensor(cache_pos):
            indices = cache_pos.reshape(-1)[:1].to(torch.int64)
        else:
            indices = torch.tensor([int(cache_pos)], dtype=torch.int64, device=past.device)
        return CustomScatterUpdate.apply(past, indices, update, 2)

    cache_total = int(past.size(2))
    pos = cache_pos.to(torch.int64)
    e = torch.nn.functional.one_hot(pos, num_classes=cache_total).to(past.dtype)
    e_bt11 = e.view(-1, 1, cache_total, 1)
    old = torch.matmul(past.transpose(2, 3), e_bt11).transpose(2, 3)
    return past + e_bt11 * (update.to(past.dtype) - old.to(past.dtype))


class CustomSwiGlu(_AutogradFunction):
    """Custom SwiGLU activation operator for ONNX export."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int = -1):
        """Compute SwiGLU activation for the last dimension."""
        del ctx
        dim = int(dim)
        split = int(x.shape[dim]) // 2
        a, b = torch.split(x, [split, split], dim=dim)
        return torch.nn.functional.silu(a) * b

    @staticmethod
    def symbolic(g: torch.Graph, x, dim: int = -1):
        return g.op(
            "Custom",
            x,
            input_index_i=[0],
            input_names_s=["x"],
            optional_input_names_s=[],
            output_names_s=["y"],
            type_s="SwiGlu",
            dim_i=int(dim),
        )


def _rotary_mul_plain(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dim = int(x.shape[-1])
    half = dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


def _prepare_mrope_cos_sin(
    cos,
    sin,
    mrope_section,
    mrope_interleaved: bool = False,
    unsqueeze_dim: int = 1,
):
    """Prepare multimodal RoPE cos/sin tensors for talker attention."""
    if mrope_interleaved:
        dim = int(cos.shape[-1])
        modality_num = int(len(mrope_section))
        cos_half = cos[..., : dim // 2]
        sin_half = sin[..., : dim // 2]

        cos_t = cos_half[0].clone()
        sin_t = sin_half[0].clone()
        for i, n in enumerate(mrope_section[1:], 1):
            beg_idx = i
            end_idx = int(n) * modality_num
            cos_src = cos_half[i, ..., beg_idx:end_idx:modality_num]
            sin_src = sin_half[i, ..., beg_idx:end_idx:modality_num]
            cos_t[..., beg_idx:end_idx:modality_num] = cos_src
            sin_t[..., beg_idx:end_idx:modality_num] = sin_src

        cos = torch.cat([cos_t] * 2, dim=-1).unsqueeze(int(unsqueeze_dim))
        sin = torch.cat([sin_t] * 2, dim=-1).unsqueeze(int(unsqueeze_dim))
        return cos, sin

    mrope_section = mrope_section * 2
    cos_parts = [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))]
    sin_parts = [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))]
    cos = torch.cat(cos_parts, dim=-1).unsqueeze(int(unsqueeze_dim))
    sin = torch.cat(sin_parts, dim=-1).unsqueeze(int(unsqueeze_dim))
    return cos, sin


def _ensure_torchaudio_stub() -> None:
    """Provide a minimal torchaudio stub when torchaudio is unavailable."""
    try:
        _import_module("torchaudio")
        return
    except (ImportError, OSError):
        ta = types.ModuleType("torchaudio")
        compliance = types.ModuleType("torchaudio.compliance")
        kaldi = types.ModuleType("torchaudio.compliance.kaldi")
        compliance.kaldi = kaldi
        ta.compliance = compliance
        sys.modules["torchaudio"] = ta
        sys.modules["torchaudio.compliance"] = compliance
        sys.modules["torchaudio.compliance.kaldi"] = kaldi


def _export_onnx(
    model: Any,
    example_inputs: tuple,
    out_path: str,
    opset: int,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict | None = None,
    allow_custom_ops: bool = False,
) -> None:
    """Export a torch module to an ONNX file."""
    torch_mod = _require_module("torch")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    export_kwargs: dict[str, Any] = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": int(opset),
    }
    if dynamic_axes:
        export_kwargs["dynamic_axes"] = dynamic_axes
    if "dynamo" in inspect.signature(torch_mod.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    if allow_custom_ops:
        export_kwargs["operator_export_type"] = torch_mod.onnx.OperatorExportTypes.ONNX_FALLTHROUGH
    torch_mod.onnx.export(model, example_inputs, out_path, **export_kwargs)


def _load_qwen3_tts_model(model_path: str, dtype: torch.dtype):
    """Load the Qwen3-TTS model via transformers auto classes."""
    _ensure_torchaudio_stub()
    transformers = _require_module("transformers")
    auto_config = getattr(transformers, "AutoConfig")
    auto_model = getattr(transformers, "AutoModel")
    cfg_mod = _require_module("qwen_tts.core.models.configuration_qwen3_tts")
    model_mod = _require_module("qwen_tts.core.models.modeling_qwen3_tts")
    qwen3_tts_config = getattr(cfg_mod, "Qwen3TTSConfig")
    qwen3_tts_model = getattr(model_mod, "Qwen3TTSForConditionalGeneration")
    auto_config.register("qwen3_tts", qwen3_tts_config)
    auto_model.register(qwen3_tts_config, qwen3_tts_model)
    return auto_model.from_pretrained(model_path, dtype=dtype).eval()


def _collect_const_value_maps(model) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect constant values from initializers and Constant nodes."""
    _require_module("onnx")
    const_node_by_output: dict[str, Any] = {}
    const_value_by_output: dict[str, Any] = {}
    for init in model.graph.initializer:
        try:
            const_value_by_output[init.name] = onnx_numpy_helper.to_array(init)
        except (TypeError, ValueError, RuntimeError):
            pass
    for node in model.graph.node:
        if node.op_type == "Constant" and len(node.output) == 1 and node.output[0]:
            const_node_by_output[node.output[0]] = node
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    try:
                        const_value_by_output[node.output[0]] = onnx_numpy_helper.to_array(attr.t)
                    except (TypeError, ValueError, RuntimeError):
                        pass
    return const_node_by_output, const_value_by_output


def _find_cache_pos_input_name(model) -> str | None:
    for graph_input in model.graph.input:
        name = getattr(graph_input, "name", "") or ""
        if name == "cache_pos" or name.endswith("cache_pos"):
            return name
    for graph_input in model.graph.input:
        name = getattr(graph_input, "name", "") or ""
        if "cache_pos" in name:
            return name
    return None


def _ensure_actual_seq_len_for_ifa(model) -> str | None:
    """Add or reuse actual_seq_lengths input for IFA rewriting."""
    cache_pos_name = _find_cache_pos_input_name(model)
    if not cache_pos_name:
        return None
    add_out = "__ifa_actual_seq_len"
    for value_info in model.graph.value_info:
        if value_info.name == add_out:
            return add_out
    for node in model.graph.node:
        if add_out in node.output:
            return add_out
    one_name = "__ifa_actual_seq_len_one"
    one_const = onnx_helper.make_tensor(
        name=one_name,
        data_type=onnx.TensorProto.INT64,
        dims=[1],
        vals=[1],
    )
    one_node = onnx_helper.make_node(
        "Constant",
        inputs=[],
        outputs=[one_name],
        value=one_const,
        name="__ifa_actual_seq_len_one_const",
    )
    add_node = onnx_helper.make_node(
        "Add",
        inputs=[cache_pos_name, one_name],
        outputs=[add_out],
        name="__ifa_actual_seq_len_add",
    )
    model.graph.node.insert(0, add_node)
    model.graph.node.insert(0, one_node)
    return add_out


def _get_const_scalar(
    name: str,
    const_node_by_output: dict[str, Any],
    const_value_by_output: dict[str, Any],
):
    """Resolve a scalar constant for a given value name if possible."""
    if not name:
        return None
    array = const_value_by_output.get(name)
    if array is None:
        node = const_node_by_output.get(name)
        if node is None:
            return None
        for attr in node.attribute:
            if attr.name == "value" and attr.type == onnx.AttributeProto.STRING and attr.s:
                return attr.s.decode("utf-8")
        return None
    try:
        if hasattr(array, "size") and int(array.size) == 1:
            value = array.reshape(-1)[0]
            return value.item() if hasattr(value, "item") else value
    except (TypeError, ValueError, RuntimeError):
        return None
    return None


def _rewrite_if_nodes_to_squeeze(model) -> None:
    """Rewrite a specific If subgraph pattern into a Squeeze node."""
    rewritten_nodes: list[Any] = []
    if_rewritten = 0
    for node in model.graph.node:
        replacements = _maybe_rewrite_if_node_to_squeeze(node)
        if replacements is None:
            rewritten_nodes.append(node)
            continue
        rewritten_nodes.extend(replacements)
        if_rewritten += 1
    if if_rewritten and len(rewritten_nodes) == len(model.graph.node) + if_rewritten:
        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)


def _get_if_branches(node) -> tuple[Any | None, Any | None]:
    then_graph = None
    else_graph = None
    for attr in node.attribute:
        if attr.name == "then_branch":
            then_graph = attr.g
        elif attr.name == "else_branch":
            else_graph = attr.g
    return then_graph, else_graph


def _find_first_node_by_op(graph, op_type: str):
    return next((n for n in graph.node if n.op_type == op_type), None)


def _read_squeeze_axes_from_graph(then_graph, then_squeeze) -> list[int]:
    """Extract Squeeze axes from the then-branch graph if present."""
    axes_values = [1]
    if len(then_squeeze.input) <= 1 or not then_squeeze.input[1]:
        return axes_values

    axes_name_in_then = then_squeeze.input[1]
    for then_node in then_graph.node:
        if then_node.op_type != "Constant":
            continue
        if not then_node.output or then_node.output[0] != axes_name_in_then:
            continue
        for attr in then_node.attribute:
            if attr.name != "value" or not attr.HasField("t"):
                continue
            try:
                axes_array = onnx_numpy_helper.to_array(attr.t).reshape(-1).tolist()
            except (TypeError, ValueError, RuntimeError):
                return axes_values
            if axes_array:
                return [int(v) for v in axes_array]
            return axes_values
        return axes_values
    return axes_values


def _maybe_rewrite_if_node_to_squeeze(node) -> list[Any] | None:
    """Return replacement nodes if an If node matches the rewrite pattern."""
    if node.op_type != "If":
        return None
    then_graph, else_graph = _get_if_branches(node)
    if then_graph is None or else_graph is None:
        return None
    then_squeeze = _find_first_node_by_op(then_graph, "Squeeze")
    else_identity = _find_first_node_by_op(else_graph, "Identity")
    if then_squeeze is None or else_identity is None:
        return None
    if not then_squeeze.input or not else_identity.input:
        return None

    data_input = then_squeeze.input[0]
    if else_identity.input[0] != data_input:
        return None

    axes_values = _read_squeeze_axes_from_graph(then_graph, then_squeeze)
    base_name = node.name or "if_node"
    axes_const_name = f"{base_name}_axes_const_output_0"
    axes_const_node = onnx_helper.make_node(
        "Constant",
        inputs=[],
        outputs=[axes_const_name],
        name=f"{base_name}_axes_const",
        value=onnx_helper.make_tensor(
            name=f"{base_name}_axes",
            data_type=onnx.TensorProto.INT64,
            dims=[len(axes_values)],
            vals=axes_values,
        ),
    )
    squeeze_node = onnx_helper.make_node(
        "Squeeze",
        inputs=[data_input, axes_const_name],
        outputs=list(node.output),
        name=f"{base_name}_no_if",
    )
    return [axes_const_node, squeeze_node]


def _drop_unused_string_constants(model) -> None:
    """Drop unused Constant nodes that only contain string values."""
    used_inputs: set[str] = set()
    for node in model.graph.node:
        for value in node.input:
            if value:
                used_inputs.add(value)
    kept_nodes = []
    for node in model.graph.node:
        is_const = node.op_type == "Constant" and len(node.output) == 1
        is_unused = bool(node.output) and node.output[0] not in used_inputs
        if is_const and is_unused:
            has_string_value = False
            for attr in node.attribute:
                if attr.name == "value" and attr.type == onnx.AttributeProto.STRING and attr.s:
                    has_string_value = True
                    break
            if has_string_value:
                continue
        kept_nodes.append(node)
    if len(kept_nodes) != len(model.graph.node):
        del model.graph.node[:]
        model.graph.node.extend(kept_nodes)


def _topo_sort_graph_nodes(model) -> None:
    """Topologically sort graph nodes when dependencies allow."""
    ready = {value.name for value in model.graph.input}
    ready.update(init.name for init in model.graph.initializer)
    ready.update(init.name for init in model.graph.sparse_initializer)
    pending = list(model.graph.node)
    ordered = []
    while pending:
        progressed = False
        next_pending = []
        for node in pending:
            deps_ready = True
            for value in node.input:
                if value and value not in ready:
                    deps_ready = False
                    break
            if deps_ready:
                ordered.append(node)
                for value in node.output:
                    if value:
                        ready.add(value)
                progressed = True
            else:
                next_pending.append(node)
        if not progressed:
            ordered.extend(next_pending)
            break
        pending = next_pending
    if len(ordered) == len(model.graph.node):
        del model.graph.node[:]
        model.graph.node.extend(ordered)


def _save_model_external_data(model, dst_onnx: str) -> None:
    data_name = os.path.splitext(os.path.basename(dst_onnx))[0] + ".data"
    dst_dir = os.path.dirname(os.path.abspath(dst_onnx))
    os.makedirs(dst_dir, exist_ok=True)
    dst_data = os.path.join(dst_dir, data_name)
    if os.path.exists(dst_onnx):
        os.remove(dst_onnx)
    if os.path.exists(dst_data):
        os.remove(dst_data)
    onnx.save_model(
        model,
        dst_onnx,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )


def _rewrite_talker_step_to_custom_ops(src_onnx: str, dst_onnx: str) -> None:
    """Rewrite talker_step fused NPU ops to Custom nodes for Ascend."""
    model = onnx.load(src_onnx, load_external_data=True)
    const_node_by_output, const_value_by_output = _collect_const_value_maps(model)

    ifa_actual_seq_len = _ensure_actual_seq_len_for_ifa(model)

    changed = 0
    for node in model.graph.node:
        if node.domain != "npu":
            continue
        op_schema = _TALKER_STEP_CUSTOM_OP_SCHEMA.get(node.op_type)
        if op_schema is None:
            continue
        selection = _select_custom_op_inputs_and_attrs(
            node=node,
            slot_inputs=op_schema["input_slots"],
            ifa_actual_seq_len=ifa_actual_seq_len,
            const_node_by_output=const_node_by_output,
            const_value_by_output=const_value_by_output,
        )
        selected_inputs, input_names, input_index, extra_attrs = selection
        output_names = _validate_and_select_output_names(node, op_schema["output_slots"], input_names)
        _rewrite_node_as_custom(
            node=node,
            custom_type=op_schema["type"],
            selected_inputs=selected_inputs,
            input_names=input_names,
            output_names=output_names,
            input_index=input_index,
            extra_attrs=extra_attrs,
        )
        changed += 1

    if changed == 0:
        _save_model_external_data(model, dst_onnx)
        return
    _rewrite_if_nodes_to_squeeze(model)
    _drop_unused_string_constants(model)
    _topo_sort_graph_nodes(model)
    _save_model_external_data(model, dst_onnx)


_TALKER_STEP_CUSTOM_OP_SCHEMA: dict[str, dict[str, Any]] = {
    "npu_rms_norm": {
        "type": "RmsNorm",
        "input_slots": ["x", "gamma"],
        "output_slots": ["y", "rstd"],
    },
    "npu_incre_flash_attention": {
        "type": "IncreFlashAttention",
        "input_slots": [
            "query",
            "key",
            "value",
            "pse_shift",
            "atten_mask",
            "actual_seq_lengths",
            "dequant_scale1",
            "quant_scale1",
            "dequant_scale2",
            "quant_scale2",
            "quant_offset2",
            "antiquant_scale",
            "antiquant_offset",
            "block_table",
            "kv_padding_size",
            "num_heads",
            "scale_value",
            "input_layout",
            "num_key_value_heads",
            "block_size",
            "inner_precise",
        ],
        "output_slots": ["attention_out"],
    },
}


def _select_nonempty_inputs(node, slot_inputs: list[str]) -> tuple[list[str], list[str], list[int]]:
    """Select non-empty node inputs and their corresponding schema slots."""
    selected_inputs: list[str] = []
    selected_input_names: list[str] = []
    selected_input_index: list[int] = []
    for idx, name in enumerate(slot_inputs):
        value = node.input[idx] if idx < len(node.input) else ""
        if not value:
            continue
        selected_inputs.append(value)
        selected_input_names.append(name)
        selected_input_index.append(idx)
    return selected_inputs, selected_input_names, selected_input_index


def _select_custom_op_inputs_and_attrs(
    *,
    node,
    slot_inputs: list[str],
    ifa_actual_seq_len: str | None,
    const_node_by_output: dict[str, Any],
    const_value_by_output: dict[str, Any],
) -> tuple[list[str], list[str], list[int], list[Any]]:
    """Select schema-aligned inputs and attributes for Custom node rewriting."""
    if node.op_type == "npu_rms_norm":
        return _select_rms_norm_inputs_and_attrs(
            node=node,
            slot_inputs=slot_inputs,
            const_node_by_output=const_node_by_output,
            const_value_by_output=const_value_by_output,
        )
    if node.op_type == "npu_incre_flash_attention":
        return _select_ifa_inputs_and_attrs(
            node=node,
            slot_inputs=slot_inputs,
            ifa_actual_seq_len=ifa_actual_seq_len,
            const_node_by_output=const_node_by_output,
            const_value_by_output=const_value_by_output,
        )
    selected_inputs, names, indexes = _select_nonempty_inputs(node, slot_inputs)
    return selected_inputs, names, indexes, []


def _select_rms_norm_inputs_and_attrs(
    *,
    node,
    slot_inputs: list[str],
    const_node_by_output: dict[str, Any],
    const_value_by_output: dict[str, Any],
) -> tuple[list[str], list[str], list[int], list[Any]]:
    """Select RMSNorm inputs and attributes for Custom node rewriting."""
    extra_attrs: list[Any] = []
    if len(node.input) > 2:
        epsilon_value = _get_const_scalar(
            node.input[2],
            const_node_by_output,
            const_value_by_output,
        )
        if epsilon_value is not None:
            extra_attrs.append(onnx_helper.make_attribute("epsilon", float(epsilon_value)))
    selected_inputs, names, indexes = _select_nonempty_inputs(node, slot_inputs)
    return selected_inputs, names, indexes, extra_attrs


def _ensure_ifa_actual_seq_len_input(node, ifa_actual_seq_len: str | None) -> None:
    if ifa_actual_seq_len is None:
        return
    while len(node.input) <= 5:
        node.input.append("")
    if not node.input[5]:
        node.input[5] = ifa_actual_seq_len


def _select_ifa_inputs_and_attrs(
    *,
    node,
    slot_inputs: list[str],
    ifa_actual_seq_len: str | None,
    const_node_by_output: dict[str, Any],
    const_value_by_output: dict[str, Any],
) -> tuple[list[str], list[str], list[int], list[Any]]:
    """Select IFA inputs and attributes for Custom node rewriting."""
    _ensure_ifa_actual_seq_len_input(node, ifa_actual_seq_len)
    selected_inputs: list[str] = []
    selected_input_names: list[str] = []
    selected_input_index: list[int] = []
    for cann_slot_idx in range(15):
        value = node.input[cann_slot_idx] if cann_slot_idx < len(node.input) else ""
        if not value:
            continue
        selected_inputs.append(value)
        selected_input_names.append(slot_inputs[cann_slot_idx])
        selected_input_index.append(cann_slot_idx)
    extra_attrs = _read_ifa_attrs(
        node=node,
        const_node_by_output=const_node_by_output,
        const_value_by_output=const_value_by_output,
    )
    return selected_inputs, selected_input_names, selected_input_index, extra_attrs


def _read_ifa_attrs(
    *,
    node,
    const_node_by_output: dict[str, Any],
    const_value_by_output: dict[str, Any],
) -> list[Any]:
    """Read IFA attributes from constant scalar inputs."""
    extra_attrs: list[Any] = []
    attr_specs: list[tuple[str, int, Any, bool]] = [
        ("num_heads", 16, int, True),
        ("scale_value", 17, float, False),
        ("input_layout", 18, str, False),
        ("num_key_value_heads", 19, int, False),
        ("block_size", 20, int, False),
        ("inner_precise", 21, int, False),
    ]
    for attr_name, input_idx, caster, required in attr_specs:
        value = None
        if input_idx < len(node.input):
            value = _get_const_scalar(
                node.input[input_idx],
                const_node_by_output,
                const_value_by_output,
            )
        if value is None:
            if required:
                node_name = node.name or "<unnamed>"
                raise RuntimeError(f"{node_name} missing required CANN attr: {attr_name}")
            continue
        if attr_name == "block_size" and int(value) <= 0:
            value = 16
        extra_attrs.append(onnx_helper.make_attribute(attr_name, caster(value)))
    return extra_attrs


def _validate_and_select_output_names(node, slot_outputs: list[str], input_names: list[str]) -> list[str]:
    """Validate output arity and derive output_names for Custom nodes."""
    if len(node.output) > len(slot_outputs):
        raise RuntimeError(
            f"Unsupported output arity for {node.op_type}: got {len(node.output)}, "
            f"but schema only has {len(slot_outputs)} slots."
        )
    if not input_names:
        node_name = node.name or "<unnamed>"
        raise RuntimeError(f"Custom node {node_name} has empty input_names after cleanup.")
    output_names = [slot_outputs[idx] for idx, value in enumerate(node.output) if value]
    if not output_names:
        node_name = node.name or "<unnamed>"
        raise RuntimeError(f"Custom node {node_name} has empty output_names after cleanup.")
    return output_names


def _rewrite_node_as_custom(
    *,
    node,
    custom_type: str,
    selected_inputs: list[str],
    input_names: list[str],
    output_names: list[str],
    input_index: list[int],
    extra_attrs: list[Any],
) -> None:
    """Rewrite one node in-place into a MindSpore Custom node."""
    optional_input_names = list(input_names)
    node.domain = ""
    node.op_type = "Custom"
    del node.input[:]
    node.input.extend(selected_inputs)
    del node.attribute[:]
    node.attribute.extend(
        [
            onnx_helper.make_attribute("type", custom_type),
            onnx_helper.make_attribute("input_names", input_names),
            onnx_helper.make_attribute("output_names", output_names),
            onnx_helper.make_attribute("optional_input_names", optional_input_names),
            onnx_helper.make_attribute("input_index", input_index),
            onnx_helper.make_attribute("output_num", len(output_names)),
            *extra_attrs,
        ]
    )


def _rewrite_talker_prefill_kv_to_fixed_len(src_onnx: str, fixed_len: int = 512) -> None:
    """Rewrite prefill KV cache tensors to a fixed cache length."""
    fixed_len = int(fixed_len)
    model = onnx.load(src_onnx, load_external_data=True)
    graph = model.graph

    def _make_i64(name: str, values, dims):
        return onnx_helper.make_tensor(
            name=name,
            data_type=onnx.TensorProto.INT64,
            dims=dims,
            vals=values,
        )

    def _make_f32(name: str, value: float):
        return onnx_helper.make_tensor(
            name=name,
            data_type=onnx.TensorProto.FLOAT,
            dims=[],
            vals=[float(value)],
        )

    def _rename_output_tensor(old: str, new: str):
        found = False
        for node in graph.node:
            for idx, out_name in enumerate(node.output):
                if out_name == old:
                    node.output[idx] = new
                    found = True
        return found

    def _fix_one(kv_name: str):
        raw = f"{kv_name}__raw"
        if not _rename_output_tensor(kv_name, raw):
            raise RuntimeError(f"Cannot find producer output for {kv_name}")

        starts_name = f"{kv_name}__starts"
        ends_name = f"{kv_name}__ends"
        axes_name = f"{kv_name}__axes"
        graph.initializer.extend(
            [
                _make_i64(starts_name, [0], [1]),
                _make_i64(ends_name, [fixed_len], [1]),
                _make_i64(axes_name, [3], [1]),
            ]
        )

        slice_out = f"{kv_name}__slice"
        graph.node.append(
            onnx_helper.make_node(
                "Slice",
                inputs=[raw, starts_name, ends_name, axes_name],
                outputs=[slice_out],
                name=f"{kv_name}__Slice",
            )
        )

        shape_out = f"{kv_name}__shape"
        graph.node.append(
            onnx_helper.make_node(
                "Shape",
                inputs=[slice_out],
                outputs=[shape_out],
                name=f"{kv_name}__Shape",
            )
        )

        axis3_name = f"{kv_name}__axis3"
        graph.initializer.append(_make_i64(axis3_name, [3], []))
        gather_out = f"{kv_name}__seq"
        graph.node.append(
            onnx_helper.make_node(
                "Gather",
                inputs=[shape_out, axis3_name],
                outputs=[gather_out],
                name=f"{kv_name}__GatherSeq",
                axis=0,
            )
        )

        fixed_name = f"{kv_name}__fixed"
        graph.initializer.append(_make_i64(fixed_name, [fixed_len], []))
        sub_out = f"{kv_name}__padlen"
        graph.node.append(
            onnx_helper.make_node(
                "Sub",
                inputs=[fixed_name, gather_out],
                outputs=[sub_out],
                name=f"{kv_name}__Sub",
            )
        )

        unsq_out = f"{kv_name}__padlen_unsq"
        unsq_axes = f"{kv_name}__unsq_axes"
        graph.initializer.append(_make_i64(unsq_axes, [0], [1]))
        graph.node.append(
            onnx_helper.make_node(
                "Unsqueeze",
                inputs=[sub_out, unsq_axes],
                outputs=[unsq_out],
                name=f"{kv_name}__Unsqueeze",
            )
        )

        pads_begin = f"{kv_name}__pads_begin"
        pads_end_prefix = f"{kv_name}__pads_end_prefix"
        pads_end_suffix = f"{kv_name}__pads_end_suffix"
        graph.initializer.extend(
            [
                _make_i64(pads_begin, [0, 0, 0, 0, 0], [5]),
                _make_i64(pads_end_prefix, [0, 0, 0], [3]),
                _make_i64(pads_end_suffix, [0], [1]),
            ]
        )

        pads_out = f"{kv_name}__pads"
        graph.node.append(
            onnx_helper.make_node(
                "Concat",
                inputs=[pads_begin, pads_end_prefix, unsq_out, pads_end_suffix],
                outputs=[pads_out],
                name=f"{kv_name}__ConcatPads",
                axis=0,
            )
        )

        zero_f32 = f"{kv_name}__zero_f32"
        graph.initializer.append(_make_f32(zero_f32, 0.0))
        graph.node.append(
            onnx_helper.make_node(
                "Pad",
                inputs=[slice_out, pads_out, zero_f32],
                outputs=[kv_name],
                name=f"{kv_name}__Pad",
                mode="constant",
            )
        )

    _fix_one("past_k")
    _fix_one("past_v")

    for out in graph.output:
        if out.name == "past_k" and len(out.type.tensor_type.shape.dim) >= 4:
            out.type.tensor_type.shape.dim[3].dim_value = fixed_len
        if out.name == "past_v" and len(out.type.tensor_type.shape.dim) >= 4:
            out.type.tensor_type.shape.dim[3].dim_value = fixed_len

    data_name = os.path.splitext(os.path.basename(src_onnx))[0] + ".data"
    dst_dir = os.path.dirname(os.path.abspath(src_onnx))
    os.makedirs(dst_dir, exist_ok=True)
    dst_data = os.path.join(dst_dir, data_name)
    if os.path.exists(src_onnx):
        os.remove(src_onnx)
    if os.path.exists(dst_data):
        os.remove(dst_data)

    onnx.save_model(
        model,
        src_onnx,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )


def _causal_mask_2d(
    attn_2d: torch.Tensor,
    window_size: int | None,
    mask_value: float = -1e4,
) -> torch.Tensor:
    """Build a causal attention mask from a 2D padding mask."""
    b, s = attn_2d.shape
    device = attn_2d.device
    i = torch.arange(s, device=device).view(1, 1, s, 1)
    j = torch.arange(s, device=device).view(1, 1, 1, s)
    allowed = j <= i
    if window_size is not None and int(window_size) > 0:
        allowed = allowed & (j > (i - int(window_size)))
    key_ok = attn_2d.view(b, 1, 1, s).to(torch.bool)
    allowed = allowed & key_ok
    zero = torch.zeros((b, 1, s, s), device=device, dtype=torch.float32)
    neg = torch.full((b, 1, s, s), float(mask_value), device=device, dtype=torch.float32)
    return torch.where(allowed, zero, neg)


@dataclass
class TalkerKVMeta:
    """Metadata saved alongside exported talker KV ONNX models."""

    model: str
    export_seq_len: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    codec_eos_token_id: int
    opset: int
    dtype: str


def _repeat_kv(x: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    if num_key_value_groups == 1:
        return x
    return x.repeat_interleave(num_key_value_groups, dim=1)


class TalkerPrefillKVWrapper(_NNModule):
    """Talker prefill wrapper exporting logits/hidden and KV cache."""

    def __init__(
        self,
        talker,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        mask_value: float = -1e4,
        use_ascend_fused_ops: bool = False,
        use_custom_rope: bool = False,
    ):
        super().__init__()
        self.talker = talker
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.num_key_value_groups = int(num_attention_heads // num_key_value_heads)
        self.scaling = float(head_dim) ** -0.5
        self.mask_value = float(mask_value)
        self.use_ascend_fused_ops = bool(use_ascend_fused_ops)
        self.allow_custom_rope = bool(use_custom_rope)
        try:
            setattr(self.talker.config, "_attn_implementation", "eager")
            setattr(self.talker.model.config, "_attn_implementation", "eager")
        except (AttributeError, TypeError):
            pass
        self._mlp_gateup_w: list[torch.Tensor] = []
        self._mlp_gateup_b: list[torch.Tensor | None] = []
        layers = list(self.talker.model.layers)
        for li, layer in enumerate(layers):
            mlp = layer.mlp
            w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0).detach()
            self.register_buffer(f"_mlp_gateup_w_{li}", w)
            self._mlp_gateup_w.append(getattr(self, f"_mlp_gateup_w_{li}"))
            bg = mlp.gate_proj.bias
            if bg is None:
                self._mlp_gateup_b.append(None)
            else:
                b = torch.cat([bg, mlp.up_proj.bias], dim=0).detach()
                self.register_buffer(f"_mlp_gateup_b_{li}", b)
                self._mlp_gateup_b.append(getattr(self, f"_mlp_gateup_b_{li}"))

    def _attention_prefill(
        self,
        attn_mod,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask_4d: torch.Tensor,
    ):
        """Compute attention for the prefill graph and return (out, k, v)."""
        b, s, d = hidden_states.shape
        q = attn_mod.q_proj(hidden_states).view(
            b,
            s,
            self.num_attention_heads,
            self.head_dim,
        )
        q = self._rms_norm_custom(attn_mod.q_norm, q).transpose(1, 2)
        k = attn_mod.k_proj(hidden_states).view(
            b,
            s,
            self.num_key_value_heads,
            self.head_dim,
        )
        k = self._rms_norm_custom(attn_mod.k_norm, k).transpose(1, 2)
        v = attn_mod.v_proj(hidden_states).view(
            b,
            s,
            self.num_key_value_heads,
            self.head_dim,
        )
        v = v.transpose(1, 2)

        if self.allow_custom_rope:
            q = CustomRoatryMul.apply(q, cos, sin)
            k = CustomRoatryMul.apply(k, cos, sin)
        else:
            q = _rotary_mul_plain(q, cos, sin)
            k = _rotary_mul_plain(k, cos, sin)
        if self.use_ascend_fused_ops and hidden_states.device.type == "npu":
            q = q.to(hidden_states.dtype)
            k = k.to(hidden_states.dtype)

        k_for_attn = _repeat_kv(k, self.num_key_value_groups)
        v_for_attn = _repeat_kv(v, self.num_key_value_groups)

        scores = torch.matmul(q, k_for_attn.transpose(-2, -1))
        scores = scores * self.scaling
        scores = scores + attention_mask_4d
        probs = torch.softmax(scores, dim=-1).to(v_for_attn.dtype)
        out = torch.matmul(probs, v_for_attn)
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        out = attn_mod.o_proj(out)
        return out, k, v

    def _rms_norm_custom(self, norm_mod, x: torch.Tensor) -> torch.Tensor:
        if self.use_ascend_fused_ops and _TORCH_NPU is not None and x.device.type == "npu":
            y, _ = _TORCH_NPU.npu_rms_norm(
                x,
                norm_mod.weight,
                epsilon=float(norm_mod.variance_epsilon),
            )
            return y
        return norm_mod(x)

    def _mlp(
        self,
        mlp_mod,
        x: torch.Tensor,
        gateup_w: torch.Tensor | None = None,
        gateup_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MLP, optionally using a fused SwiGLU path."""
        if not self.use_ascend_fused_ops:
            return mlp_mod(x)
        act = getattr(mlp_mod, "act_fn", None)
        act_class = getattr(act, "__class__", None)
        act_class_name = getattr(act_class, "__name__", "") if act_class is not None else ""
        act_name = str(getattr(act, "__name__", "") or act_class_name or "").lower()
        if act is not None and ("silu" not in act_name and "swish" not in act_name):
            return mlp_mod(x)
        w = (
            gateup_w
            if gateup_w is not None
            else torch.cat([mlp_mod.gate_proj.weight, mlp_mod.up_proj.weight], dim=0)
        )
        bias = gateup_b
        if bias is None and gateup_w is None:
            bg = mlp_mod.gate_proj.bias
            if bg is not None:
                bias = torch.cat([bg, mlp_mod.up_proj.bias], dim=0)
        gate_up = torch.nn.functional.linear(x, w, bias)
        y = CustomSwiGlu.apply(gate_up, -1)
        return mlp_mod.down_proj(y)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        """Run talker prefill and return (logits_last, hidden_last, past_k, past_v, prompt_len)."""
        b, s, _ = inputs_embeds.shape
        device = inputs_embeds.device
        cache_position = torch.arange(s, device=device)
        position_ids = cache_position.view(1, 1, -1).expand(3, b, -1)

        position_embeddings = self.talker.model.rotary_emb(inputs_embeds, position_ids)
        cos_raw, sin_raw = position_embeddings
        attn0 = self.talker.model.layers[0].self_attn
        cos, sin = _prepare_mrope_cos_sin(
            cos_raw,
            sin_raw,
            attn0.rope_scaling["mrope_section"],
            mrope_interleaved=bool(attn0.rope_scaling["interleaved"]),
            unsqueeze_dim=1,
        )
        window = getattr(self.talker.model.config, "sliding_window", None)
        mask_value = float(torch.finfo(torch.float32).min)
        attn4d = _causal_mask_2d(attention_mask.to(torch.int64), window_size=window, mask_value=mask_value)

        layers = list(self.talker.model.layers)

        residual = inputs_embeds
        if self.use_ascend_fused_ops:
            hidden_states = self._rms_norm_custom(layers[0].input_layernorm, residual)
        else:
            hidden_states = residual
        k_layers = []
        v_layers = []
        for li, layer in enumerate(layers):
            if not self.use_ascend_fused_ops:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            attn_out, k, v = self._attention_prefill(layer.self_attn, hidden_states, cos, sin, attn4d)
            if self.use_ascend_fused_ops:
                eps = float(getattr(layer.post_attention_layernorm, "variance_epsilon", 1e-6))
                y, _, x = CustomAddRmsNorm.apply(residual, attn_out, layer.post_attention_layernorm.weight, eps)
                residual = x
                hidden_states = y
            else:
                hidden_states = residual + attn_out
                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
            mlp_out = self._mlp(layer.mlp, hidden_states, self._mlp_gateup_w[li], self._mlp_gateup_b[li])
            if self.use_ascend_fused_ops and li + 1 < len(layers):
                next_layer = layers[li + 1]
                eps_in = float(getattr(next_layer.input_layernorm, "variance_epsilon", 1e-6))
                y2, _, x2 = CustomAddRmsNorm.apply(residual, mlp_out, next_layer.input_layernorm.weight, eps_in)
                residual = x2
                hidden_states = y2
            else:
                hidden_states = residual + mlp_out
                residual = hidden_states

            k_layers.append(k)
            v_layers.append(v)

        if self.use_ascend_fused_ops:
            hidden_states = self._rms_norm_custom(self.talker.model.norm, residual)
        else:
            hidden_states = self.talker.model.norm(hidden_states)
        logits = self.talker.codec_head(hidden_states)

        prompt_len = attention_mask.to(torch.int64).sum(dim=1)
        one_i = prompt_len.new_tensor(1)
        prompt_len = torch.maximum(prompt_len, one_i)
        idx = (prompt_len - 1).view(b, 1, 1)
        logits_last = logits.gather(1, idx.expand(b, 1, logits.shape[-1]))[:, 0, :]
        hidden_last = hidden_states.gather(1, idx.expand(b, 1, hidden_states.shape[-1]))

        past_k = torch.stack(k_layers, dim=0)
        past_v = torch.stack(v_layers, dim=0)
        return logits_last, hidden_last, past_k, past_v, prompt_len


class TalkerStepKVWrapper(_NNModule):
    """Talker step wrapper exporting logits/hidden and updated KV cache."""

    def __init__(
        self,
        talker,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        cache_len: int = 512,
        mask_value: float = -1e4,
        use_ascend_fused_ops: bool = False,
        use_custom_rope: bool = False,
    ):
        super().__init__()
        self.talker = talker
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.num_key_value_groups = int(num_attention_heads // num_key_value_heads)
        self.scaling = float(head_dim) ** -0.5
        self.cache_len = int(cache_len)
        self.mask_value = float(mask_value)
        self.use_ascend_fused_ops = bool(use_ascend_fused_ops)
        self.allow_custom_rope = bool(use_custom_rope)
        try:
            setattr(self.talker.config, "_attn_implementation", "eager")
            setattr(self.talker.model.config, "_attn_implementation", "eager")
        except (AttributeError, TypeError):
            pass
        self._qkv_w: list[torch.Tensor] = []
        self._qkv_b: list[torch.Tensor | None] = []
        self._mlp_gateup_w: list[torch.Tensor] = []
        self._mlp_gateup_b: list[torch.Tensor | None] = []
        layers = list(self.talker.model.layers)
        for li, layer in enumerate(layers):
            attn = layer.self_attn
            qkv_w = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0).detach()
            self.register_buffer(f"_qkv_w_{li}", qkv_w)
            self._qkv_w.append(getattr(self, f"_qkv_w_{li}"))
            bq = attn.q_proj.bias
            if bq is None:
                self._qkv_b.append(None)
            else:
                qkv_b = torch.cat([bq, attn.k_proj.bias, attn.v_proj.bias], dim=0).detach()
                self.register_buffer(f"_qkv_b_{li}", qkv_b)
                self._qkv_b.append(getattr(self, f"_qkv_b_{li}"))
            mlp = layer.mlp
            w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0).detach()
            self.register_buffer(f"_mlp_gateup_w_{li}", w)
            self._mlp_gateup_w.append(getattr(self, f"_mlp_gateup_w_{li}"))
            bg = mlp.gate_proj.bias
            if bg is None:
                self._mlp_gateup_b.append(None)
            else:
                b = torch.cat([bg, mlp.up_proj.bias], dim=0).detach()
                self.register_buffer(f"_mlp_gateup_b_{li}", b)
                self._mlp_gateup_b.append(getattr(self, f"_mlp_gateup_b_{li}"))

    def _rms_norm(self, norm_mod, x: torch.Tensor) -> torch.Tensor:
        if self.use_ascend_fused_ops and _TORCH_NPU is not None and x.device.type == "npu":
            y, _ = _TORCH_NPU.npu_rms_norm(x, norm_mod.weight, epsilon=float(norm_mod.variance_epsilon))
            return y
        return norm_mod(x)

    def _rms_norm_custom(self, norm_mod, x: torch.Tensor) -> torch.Tensor:
        if _TORCH_NPU is not None and x.device.type == "npu":
            y, _ = _TORCH_NPU.npu_rms_norm(x, norm_mod.weight, epsilon=float(norm_mod.variance_epsilon))
            return y
        return norm_mod(x)

    def _mlp(
        self,
        mlp_mod,
        x: torch.Tensor,
        gateup_w: torch.Tensor | None = None,
        gateup_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MLP, optionally using a fused SwiGLU path."""
        if not self.use_ascend_fused_ops:
            return mlp_mod(x)
        act = getattr(mlp_mod, "act_fn", None)
        act_class = getattr(act, "__class__", None)
        act_class_name = getattr(act_class, "__name__", "") if act_class is not None else ""
        act_name = str(getattr(act, "__name__", "") or act_class_name or "").lower()
        if act is not None and ("silu" not in act_name and "swish" not in act_name):
            return mlp_mod(x)
        w = (
            gateup_w
            if gateup_w is not None
            else torch.cat([mlp_mod.gate_proj.weight, mlp_mod.up_proj.weight], dim=0)
        )
        bias = gateup_b
        if bias is None and gateup_w is None:
            bg = mlp_mod.gate_proj.bias
            if bg is not None:
                bias = torch.cat([bg, mlp_mod.up_proj.bias], dim=0)
        gate_up = torch.nn.functional.linear(x, w, bias)
        y = CustomSwiGlu.apply(gate_up, -1)
        return mlp_mod.down_proj(y)

    def _attention_step(
        self,
        attn_mod,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_pad_4d: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        cache_pos: torch.Tensor,
        qkv_w: torch.Tensor | None = None,
        qkv_b: torch.Tensor | None = None,
    ):
        """Compute attention for one step and update KV cache."""
        q_out = int(self.num_attention_heads * self.head_dim)
        kv_out = int(self.num_key_value_heads * self.head_dim)
        w = (
            qkv_w
            if qkv_w is not None
            else torch.cat(
                [attn_mod.q_proj.weight, attn_mod.k_proj.weight, attn_mod.v_proj.weight],
                dim=0,
            )
        )
        bias = qkv_b
        if bias is None and qkv_w is None:
            bq = attn_mod.q_proj.bias
            if bq is not None:
                bias = torch.cat([bq, attn_mod.k_proj.bias, attn_mod.v_proj.bias], dim=0)
        qkv = torch.nn.functional.linear(hidden_states, w, bias)
        q_raw, k_raw, v_raw = torch.split(qkv, [q_out, kv_out, kv_out], dim=-1)
        q = self._rms_norm_custom(attn_mod.q_norm, q_raw.reshape(-1, self.num_attention_heads, 1, self.head_dim))
        k = self._rms_norm_custom(attn_mod.k_norm, k_raw.reshape(-1, self.num_key_value_heads, 1, self.head_dim))
        v = v_raw.reshape(-1, self.num_key_value_heads, 1, self.head_dim)

        if self.use_ascend_fused_ops:
            q = CustomRoatryMul.apply(q, cos, sin)
            k = CustomRoatryMul.apply(k, cos, sin)
        else:
            q = _rotary_mul_plain(q, cos, sin)
            k = _rotary_mul_plain(k, cos, sin)

        k_full = _kv_cache_update(past_k, k, cache_pos)
        v_full = _kv_cache_update(past_v, v, cache_pos)

        if self.use_ascend_fused_ops:
            q_bnsd = q.contiguous()
            k_bnsd = k_full.contiguous()
            v_bnsd = v_full.contiguous()
            if key_pad_4d is not None:
                attn_mask = key_pad_4d != 0
            else:
                bsz = int(k_bnsd.shape[0])
                cache_total = int(k_bnsd.shape[2])
                attn_mask = torch.zeros(
                    (bsz, 1, 1, cache_total),
                    device=hidden_states.device,
                    dtype=torch.bool,
                )
            out = CustomIncreFlashAttention.apply(
                q_bnsd,
                k_bnsd,
                v_bnsd,
                attn_mask,
                int(self.num_attention_heads),
                float(self.scaling),
                "BNSD",
                int(self.num_key_value_heads),
            )
            if int(out.dim()) == 4:
                out = out.transpose(1, 2).reshape(-1, 1, self.num_attention_heads * self.head_dim)
            out = attn_mod.o_proj(out)
            return out, k_full, v_full

        if int(self.num_key_value_groups) > 1:
            k_full_rep = k_full.repeat_interleave(int(self.num_key_value_groups), dim=1)
            v_full_rep = v_full.repeat_interleave(int(self.num_key_value_groups), dim=1)
        else:
            k_full_rep = k_full
            v_full_rep = v_full

        scores = torch.matmul(q, k_full_rep.transpose(-2, -1))
        scores = scores * self.scaling
        if key_pad_4d is not None:
            scores = scores + key_pad_4d
        probs = torch.softmax(scores, dim=-1).to(v_full_rep.dtype)
        out = torch.matmul(probs, v_full_rep)
        out = out.transpose(1, 2).reshape(-1, 1, self.num_attention_heads * self.head_dim)
        out = attn_mod.o_proj(out)
        return out, k_full, v_full

    def forward(
        self,
        step_embed: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        position_ids_step: torch.Tensor,
        cache_len: torch.Tensor,
    ):
        """Run one talker step and return logits/hidden plus updated KV cache."""
        position_embeddings = self.talker.model.rotary_emb(step_embed, position_ids_step)
        cos_raw, sin_raw = position_embeddings
        attn0 = self.talker.model.layers[0].self_attn
        cos, sin = _prepare_mrope_cos_sin(
            cos_raw,
            sin_raw,
            attn0.rope_scaling["mrope_section"],
            mrope_interleaved=bool(attn0.rope_scaling["interleaved"]),
            unsqueeze_dim=1,
        )
        mask_value = float(torch.finfo(torch.float32).min)
        window = getattr(self.talker.model.config, "sliding_window", None)
        cache_total = int(past_k.size(3))
        cache_pos = torch.clamp(cache_len.to(torch.int64), min=0, max=cache_total - 1)
        kv_idx = torch.arange(cache_total, device=step_embed.device, dtype=torch.int64).view(1, cache_total)
        allow = kv_idx <= cache_pos.view(-1, 1)
        if window is not None and int(window) > 0:
            start = (cache_pos - int(window) + 1).view(-1, 1)
            allow = allow & (kv_idx >= start)
        key_pad_4d = torch.full(
            (allow.size(0), 1, 1, cache_total),
            mask_value,
            device=step_embed.device,
            dtype=torch.float32,
        )
        key_pad_4d = key_pad_4d.masked_fill(
            allow.view(allow.size(0), 1, 1, cache_total),
            0.0,
        ).to(step_embed.dtype)

        layers = list(self.talker.model.layers)
        residual = step_embed
        if self.use_ascend_fused_ops:
            hidden_states = self._rms_norm(layers[0].input_layernorm, residual)
        else:
            hidden_states = residual
        past_k_layers = torch.unbind(past_k, dim=0)
        past_v_layers = torch.unbind(past_v, dim=0)
        k_layers = []
        v_layers = []
        for li, layer in enumerate(layers):
            if not self.use_ascend_fused_ops:
                residual = hidden_states
                hidden_states = self._rms_norm(layer.input_layernorm, hidden_states)
            attn_out, k_new, v_new = self._attention_step(
                layer.self_attn,
                hidden_states,
                cos,
                sin,
                key_pad_4d,
                past_k_layers[li],
                past_v_layers[li],
                cache_pos=cache_pos,
                qkv_w=self._qkv_w[li],
                qkv_b=self._qkv_b[li],
            )
            if self.use_ascend_fused_ops:
                eps = float(getattr(layer.post_attention_layernorm, "variance_epsilon", 1e-6))
                y, _, x = CustomAddRmsNorm.apply(residual, attn_out, layer.post_attention_layernorm.weight, eps)
                residual = x
                hidden_states = y
            else:
                hidden_states = residual + attn_out
                residual = hidden_states
                hidden_states = self._rms_norm(layer.post_attention_layernorm, hidden_states)
            mlp_out = self._mlp(layer.mlp, hidden_states, self._mlp_gateup_w[li], self._mlp_gateup_b[li])
            if self.use_ascend_fused_ops and li + 1 < len(layers):
                next_layer = layers[li + 1]
                eps_in = float(getattr(next_layer.input_layernorm, "variance_epsilon", 1e-6))
                y2, _, x2 = CustomAddRmsNorm.apply(residual, mlp_out, next_layer.input_layernorm.weight, eps_in)
                residual = x2
                hidden_states = y2
            else:
                hidden_states = residual + mlp_out
                residual = hidden_states

            k_layers.append(k_new)
            v_layers.append(v_new)

        hidden_states = self._rms_norm(self.talker.model.norm, residual)
        logits_last = self.talker.codec_head(hidden_states)[:, 0, :]
        past_k_out = torch.stack(k_layers, dim=0)
        past_v_out = torch.stack(v_layers, dim=0)
        return logits_last, hidden_states, past_k_out, past_v_out


def export_talker_kv(
    model_path: str,
    output_dir: str,
    opset: int,
    dtype: str,
    export_seq_len: int,
    device: str = "cpu",
    ascend_fused_ops: bool = False,
    export_custom_op: bool = False,
    custom_rope: bool = False,
) -> None:
    """Export talker prefill/step ONNX models and write meta.json."""
    dt = _torch_dtype(dtype)
    os.makedirs(output_dir, exist_ok=True)
    model = _load_qwen3_tts_model(model_path, dtype=dt).to(device)
    talker = model.talker
    info = _collect_talker_kv_export_info(model, talker)

    _export_talker_prefill_kv_onnx(
        talker=talker,
        output_dir=output_dir,
        opset=int(opset),
        dt=dt,
        export_seq_len=int(export_seq_len),
        device=str(device),
        num_attention_heads=int(info["num_attention_heads"]),
        num_key_value_heads=int(info["num_key_value_heads"]),
        head_dim=int(info["head_dim"]),
        ascend_fused_ops=bool(ascend_fused_ops),
        custom_rope=bool(custom_rope),
    )
    _export_talker_step_kv_onnx(
        talker=talker,
        output_dir=output_dir,
        opset=int(opset),
        dt=dt,
        device=str(device),
        ascend_fused_ops=bool(ascend_fused_ops),
        export_custom_op=bool(export_custom_op),
        custom_rope=bool(custom_rope),
    )
    meta = TalkerKVMeta(
        model=str(model_path),
        export_seq_len=int(export_seq_len),
        hidden_size=int(info["hidden_size"]),
        num_layers=int(info["num_layers"]),
        num_attention_heads=int(info["num_attention_heads"]),
        num_key_value_heads=int(info["num_key_value_heads"]),
        head_dim=int(info["head_dim"]),
        vocab_size=int(info["vocab_size"]),
        codec_eos_token_id=int(info["eos_id"]),
        opset=int(opset),
        dtype=str(dtype),
    )
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as file:
        json.dump(asdict(meta), file, ensure_ascii=False, indent=2)


def _collect_talker_kv_export_info(model: Any, talker: Any) -> dict[str, int]:
    num_layers = int(talker.model.config.num_hidden_layers)
    hidden_size = int(talker.config.hidden_size)
    num_attention_heads = int(talker.config.num_attention_heads)
    num_key_value_heads = int(talker.config.num_key_value_heads)
    head_dim = int(getattr(talker.config, "head_dim", hidden_size // num_attention_heads))
    vocab_size = int(talker.config.vocab_size)
    eos_id = int(model.config.talker_config.codec_eos_token_id)
    return {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "eos_id": eos_id,
    }


def _export_talker_prefill_kv_onnx(
    *,
    talker: Any,
    output_dir: str,
    opset: int,
    dt: Any,
    export_seq_len: int,
    device: str,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    ascend_fused_ops: bool,
    custom_rope: bool,
) -> None:
    """Export talker_prefill.onnx and apply post-export graph rewrites."""
    hidden_size = int(talker.config.hidden_size)
    prefill_wrap = TalkerPrefillKVWrapper(
        talker=talker,
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
        head_dim=int(head_dim),
        use_ascend_fused_ops=bool(ascend_fused_ops),
        use_custom_rope=bool(custom_rope),
    ).eval()
    prefill_wrap = prefill_wrap.to(device)
    ex_inputs_embeds = torch.zeros((1, int(export_seq_len), hidden_size), dtype=dt, device=device)
    ex_attn = torch.zeros((1, int(export_seq_len)), dtype=torch.int64, device=device)
    prefill_onnx = os.path.join(output_dir, "talker_prefill.onnx")
    _export_onnx(
        prefill_wrap,
        (ex_inputs_embeds, ex_attn),
        prefill_onnx,
        opset=int(opset),
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["logits_last", "hidden_last", "past_k", "past_v", "prompt_len"],
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "prompt_len"},
            "attention_mask": {0: "batch", 1: "prompt_len"},
            "hidden_last": {0: "batch"},
            "logits_last": {0: "batch"},
            "past_k": {1: "batch"},
            "past_v": {1: "batch"},
            "prompt_len": {0: "batch"},
        },
        allow_custom_ops=bool(ascend_fused_ops),
    )
    _rewrite_talker_prefill_kv_to_fixed_len(prefill_onnx, fixed_len=512)
    _rewrite_talker_step_to_custom_ops(src_onnx=prefill_onnx, dst_onnx=prefill_onnx)


def _export_talker_step_kv_onnx(
    *,
    talker: Any,
    output_dir: str,
    opset: int,
    dt: Any,
    device: str,
    ascend_fused_ops: bool,
    export_custom_op: bool,
    custom_rope: bool,
) -> None:
    """Export talker_step.onnx and optionally rewrite Custom ops."""
    num_layers = int(talker.model.config.num_hidden_layers)
    hidden_size = int(talker.config.hidden_size)
    num_attention_heads = int(talker.config.num_attention_heads)
    num_key_value_heads = int(talker.config.num_key_value_heads)
    head_dim = int(getattr(talker.config, "head_dim", hidden_size // num_attention_heads))
    step_wrap = TalkerStepKVWrapper(
        talker=talker,
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
        head_dim=int(head_dim),
        cache_len=512,
        use_ascend_fused_ops=bool(ascend_fused_ops),
        use_custom_rope=bool(custom_rope),
    ).eval()
    step_wrap = step_wrap.to(device)
    ex_step_embed = torch.zeros((1, 1, int(hidden_size)), dtype=dt, device=device)
    ex_past_k = torch.zeros(
        (int(num_layers), 1, int(num_key_value_heads), 512, int(head_dim)),
        dtype=dt,
        device=device,
    )
    ex_past_v = torch.zeros(
        (int(num_layers), 1, int(num_key_value_heads), 512, int(head_dim)),
        dtype=dt,
        device=device,
    )
    ex_pos = torch.zeros((3, 1, 1), dtype=torch.int64, device=device)
    ex_cache_len = torch.zeros((1,), dtype=torch.int64, device=device)
    step_onnx = os.path.join(output_dir, "talker_step.onnx")
    _export_onnx(
        step_wrap,
        (ex_step_embed, ex_past_k, ex_past_v, ex_pos, ex_cache_len),
        step_onnx,
        opset=int(opset),
        input_names=["step_embed", "past_k", "past_v", "position_ids_step", "cache_len"],
        output_names=["logits_last", "hidden_last", "past_k_out", "past_v_out"],
        allow_custom_ops=bool(ascend_fused_ops or custom_rope),
    )
    _maybe_rewrite_step_custom_ops(
        output_dir=output_dir,
        step_onnx=step_onnx,
        device=device,
        ascend_fused_ops=bool(ascend_fused_ops),
        export_custom_op=bool(export_custom_op),
    )


def _maybe_rewrite_step_custom_ops(
    *,
    output_dir: str,
    step_onnx: str,
    device: str,
    ascend_fused_ops: bool,
    export_custom_op: bool,
) -> None:
    """Rewrite talker_step.onnx to Custom nodes for Ascend when enabled."""
    if not (ascend_fused_ops and (device or "").lower() == "npu" and _TORCH_NPU is not None):
        return
    if export_custom_op:
        step_npu_onnx = os.path.join(output_dir, "talker_step.npu.onnx")
        step_npu_data = os.path.join(output_dir, "talker_step.npu.data")
        step_data = os.path.join(output_dir, "talker_step.data")
        if os.path.exists(step_npu_onnx):
            os.remove(step_npu_onnx)
        if os.path.exists(step_npu_data):
            os.remove(step_npu_data)
        if os.path.exists(step_data):
            os.remove(step_data)
        if os.path.exists(step_onnx):
            os.replace(step_onnx, step_npu_onnx)
        if os.path.exists(step_data):
            os.replace(step_data, step_npu_data)
        _rewrite_talker_step_to_custom_ops(src_onnx=step_npu_onnx, dst_onnx=step_onnx)
        return
    _rewrite_talker_step_to_custom_ops(
        src_onnx=step_onnx,
        dst_onnx=os.path.join(output_dir, "talker_step.custom.onnx"),
    )


def _count_control_flow_nodes(onnx_path: str) -> dict[str, int]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    counter = Counter(n.op_type for n in model.graph.node)
    return {
        "If": int(counter.get("If", 0)),
        "Loop": int(counter.get("Loop", 0)),
        "Scan": int(counter.get("Scan", 0)),
    }


def _strip_talker_prefill_control_flow(output_dir: str) -> dict[str, Any]:
    """Strip control flow nodes from talker_prefill.onnx when possible."""
    prefill_path = os.path.join(str(output_dir), "talker_prefill.onnx")
    before = _count_control_flow_nodes(prefill_path)
    if max(before.values(), default=0) <= 0:
        return {"before": before, "after": before, "changed": False}

    conv = _require_module("convert_talker_onnx_to_mindir")
    preprocess = getattr(conv, "preprocess_prefill_onnx")
    info = preprocess(prefill_path, prefill_path)
    after = _count_control_flow_nodes(prefill_path)
    return {"before": before, "after": after, "changed": True, "preprocess": info}


def export_talker_kv_onnx(
    model_path: str,
    output_dir: str,
    opset: int = 17,
    dtype: str = "float32",
    export_seq_len: int = 512,
    device: str = "cpu",
    strip_control_flow: bool = True,
) -> None:
    """Export talker KV ONNX models suitable for ONNX Runtime."""
    export_talker_kv(
        model_path=model_path,
        output_dir=output_dir,
        opset=int(opset),
        dtype=str(dtype),
        export_seq_len=int(export_seq_len),
        device=str(device),
        ascend_fused_ops=False,
        export_custom_op=False,
    )
    if strip_control_flow:
        _strip_talker_prefill_control_flow(output_dir=str(output_dir))


class SpeechTokenizerV2DecoderOnnxWrapper(_NNModule):
    """Wrapper to export speech decoder as a standalone ONNX model."""

    def __init__(self, decoder_model: Any) -> None:
        super().__init__()
        self.decoder_model = decoder_model

    @staticmethod
    def _causal_mask(bsz: int, seq_len: int, window: int, dtype: Any, device: Any) -> Any:
        """Build causal masks for full and sliding-window attention."""
        del bsz
        mask_value = torch.finfo(torch.float32).min
        q = torch.arange(seq_len, device=device).view(1, seq_len, 1)
        kv = torch.arange(seq_len, device=device).view(1, 1, seq_len)
        allow = kv <= q
        if int(window) > 0:
            start = q - int(window) + 1
            allow = allow & (kv >= start)
        mask = torch.full((1, 1, seq_len, seq_len), mask_value, device=device, dtype=torch.float32)
        mask = mask.masked_fill(allow.view(1, 1, seq_len, seq_len), 0.0)
        return mask.to(dtype)

    def forward(self, codes: Any) -> Any:
        """Decode codec codes to waveform tensor."""
        hidden = self.decoder_model.quantizer.decode(codes)
        hidden = self.decoder_model.pre_conv(hidden).transpose(1, 2)

        _, t, _ = hidden.shape
        device = hidden.device
        mask_full = self._causal_mask(bsz=1, seq_len=int(t), window=0, dtype=hidden.dtype, device=device)
        sliding = int(getattr(self.decoder_model.pre_transformer.config, "sliding_window", 0) or 0)
        mask_sliding = self._causal_mask(bsz=1, seq_len=int(t), window=sliding, dtype=hidden.dtype, device=device)
        attn_mask = {"full_attention": mask_full, "sliding_attention": mask_sliding}

        hidden = self.decoder_model.pre_transformer(inputs_embeds=hidden, attention_mask=attn_mask).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.decoder_model.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder_model.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)


def export_speech_decoder_onnx(
    model_path: str,
    output_dir: str,
    opset: int = 17,
    dtype: str = "float32",
    device: str = "cpu",
    example_seq_len: int = 100,
) -> str:
    """Export speech decoder ONNX model."""
    torch_mod = _require_module("torch")
    qwen_tts = _require_module("qwen_tts")
    model_cls = getattr(qwen_tts, "Qwen3TTSModel")
    model = model_cls.from_pretrained(
        model_path,
        device_map=str(device),
        dtype=_torch_dtype(dtype),
    )
    speech_tokenizer = model.model.speech_tokenizer
    decoder_model = speech_tokenizer.model.decoder
    decoder_model.eval()
    try:
        setattr(decoder_model.pre_transformer.config, "_attn_implementation", "eager")
    except (AttributeError, TypeError):
        pass
    wrapper = SpeechTokenizerV2DecoderOnnxWrapper(decoder_model).eval()

    os.makedirs(output_dir, exist_ok=True)
    num_quantizers = int(decoder_model.config.num_quantizers)
    codebook_size = int(decoder_model.config.codebook_size)
    codes = torch_mod.randint(
        low=0,
        high=codebook_size,
        size=(1, num_quantizers, int(example_seq_len)),
        device=torch_mod.device("cpu"),
    )
    codes = codes.to(dtype=_torch_dtype(dtype))

    output_path = os.path.join(output_dir, "speech_decoder.onnx")
    with torch_mod.no_grad():
        _ = wrapper(codes)
        torch_mod.onnx.export(
            wrapper,
            (codes,),
            output_path,
            input_names=["codes"],
            output_names=["wav"],
            opset_version=int(opset),
            do_constant_folding=True,
            training=torch_mod.onnx.TrainingMode.EVAL,
            dynamo=False,
            dynamic_axes={
                "codes": {0: "batch_size", 2: "seq_len"},
                "wav": {0: "batch_size", 2: "output_seq_len"},
            },
        )
    return output_path


def _apply_rotary_pos_emb_plain(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE using plain tensor operations."""
    del position_ids
    cos = cos.unsqueeze(int(unsqueeze_dim))
    sin = sin.unsqueeze(int(unsqueeze_dim))
    return _rotary_mul_plain(q, cos, sin), _rotary_mul_plain(k, cos, sin)


def _apply_rotary_pos_emb_custom(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE using CustomRoatryMul for ONNX export."""
    del position_ids
    cos = cos.unsqueeze(int(unsqueeze_dim))
    sin = sin.unsqueeze(int(unsqueeze_dim))
    return CustomRoatryMul.apply(q, cos, sin), CustomRoatryMul.apply(k, cos, sin)


class GenerateProcessAndStepEmbedWrapper(_NNModule):
    """Wrapper that exports both codec ids and the next step embedding."""

    def __init__(self, code_predictor: Any) -> None:
        super().__init__()
        self.gen = GenerateProcessWrapper(code_predictor)
        self.codec_embedding = code_predictor.model.codec_embedding

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        next_id: torch.Tensor,
        last_id_hidden: torch.Tensor,
        trailing_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run generate process and compute step embedding."""
        sequences = self.gen(inputs_embeds)
        base_embed = last_id_hidden.to(torch.float32)
        pred_embeds = [
            self.codec_embedding[i](sequences[:, i]).to(torch.float32).unsqueeze(1)
            for i in range(int(sequences.shape[1]))
        ]
        codec_hiddens = torch.cat([base_embed] + pred_embeds, dim=1)
        step_embed = codec_hiddens.sum(1, keepdim=True) + trailing_step
        codec_ids = torch.cat([next_id.to(torch.int64), sequences.to(torch.int64)], dim=1)
        return codec_ids, step_embed


class GenerateProcessWrapper(_NNModule):
    """Generate-process wrapper that returns predicted codec token ids."""

    def __init__(self, code_predictor: Any) -> None:
        super().__init__()
        self.code_predictor = code_predictor
        self.projection = code_predictor.small_to_mtp_projection
        self.transformer = code_predictor.model
        self.lm_heads = code_predictor.lm_head

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Predict next codec token ids given inputs_embeds."""
        model_output = self.transformer(inputs_embeds=inputs_embeds, use_cache=True, output_hidden_states=True)
        last_hidden = model_output.last_hidden_state[:, -1, :]
        last_hidden = self.projection(last_hidden)
        logits_list = []
        for head in self.lm_heads:
            logits_list.append(head(last_hidden))
        logits = torch.stack(logits_list, dim=1)
        return torch.argmax(logits, dim=-1).to(torch.int64)


def _torch_dtype(name: str) -> torch.dtype:
    torch_mod = _require_module("torch")
    val = (name or "float32").strip().lower()
    if val in ("float16", "fp16"):
        return torch_mod.float16
    if val in ("bfloat16", "bf16"):
        return torch_mod.bfloat16
    return torch_mod.float32


def export_generate_process_onnx(
    model_path: str,
    output_dir: str,
    opset: int = 17,
    checkpoint_path: str | None = None,
    use_custom_rope: bool = False,
) -> str:
    """Export code predictor generate_process.onnx."""
    torch_mod = _require_module("torch")
    cfg_mod = _require_module("qwen_tts.core.models.configuration_qwen3_tts")
    modeling_mod = _require_module("qwen_tts.core.models.modeling_qwen3_tts")
    talker_cfg_cls = getattr(modeling_mod, "Qwen3TTSTalkerConfig")
    code_cfg_cls = getattr(cfg_mod, "Qwen3TTSTalkerCodePredictorConfig")
    code_model_cls = getattr(modeling_mod, "Qwen3TTSTalkerCodePredictorModelForConditionalGeneration")
    talker_config = talker_cfg_cls.from_pretrained(model_path)
    cfg = code_cfg_cls(
        hidden_size=talker_config.hidden_size,
        num_hidden_layers=talker_config.num_hidden_layers,
        num_attention_heads=talker_config.num_attention_heads,
        num_key_value_heads=talker_config.num_key_value_heads,
        intermediate_size=talker_config.intermediate_size,
        vocab_size=talker_config.vocab_size,
    )
    code_predictor = code_model_cls(cfg, talker_config)
    code_predictor.eval()

    if checkpoint_path is None:
        checkpoint_path = os.path.join(model_path, "model.safetensors")
    safetensors_torch = _require_module("safetensors.torch")
    load_file = getattr(safetensors_torch, "load_file")
    checkpoint = load_file(checkpoint_path)

    filtered: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not key.startswith("model.talker.code_predictor."):
            continue
        new_key = key.replace("model.talker.code_predictor.", "")
        filtered[new_key] = value
    code_predictor.load_state_dict(filtered, strict=True)
    code_predictor.eval()

    try:
        setattr(code_predictor.model.config, "_attn_implementation", "eager")
    except (AttributeError, TypeError):
        pass

    if use_custom_rope:
        modeling_mod.apply_rotary_pos_emb = _apply_rotary_pos_emb_custom
    else:
        modeling_mod.apply_rotary_pos_emb = _apply_rotary_pos_emb_plain

    wrapper = GenerateProcessAndStepEmbedWrapper(code_predictor).eval()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generate_process.onnx")

    batch_size = 1
    initial_seq_len = 2
    hidden_size = int(talker_config.hidden_size)
    inputs_embeds = torch_mod.randn(batch_size, initial_seq_len, hidden_size, dtype=torch_mod.float32)
    next_id = torch_mod.zeros((batch_size, 1), dtype=torch_mod.int64)
    last_id_hidden = torch_mod.randn(batch_size, 1, hidden_size, dtype=torch_mod.float32)
    trailing_step = torch_mod.randn(batch_size, 1, hidden_size, dtype=torch_mod.float32)

    with torch_mod.no_grad():
        _ = wrapper(inputs_embeds, next_id, last_id_hidden, trailing_step)
        torch_mod.onnx.export(
            wrapper,
            (inputs_embeds, next_id, last_id_hidden, trailing_step),
            output_path,
            input_names=["inputs_embeds", "next_id", "last_id_hidden", "trailing_step"],
            output_names=["codec_ids", "step_embed"],
            opset_version=int(opset),
            do_constant_folding=True,
            training=torch_mod.onnx.TrainingMode.EVAL,
            dynamo=False,
        )
    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for exporting Qwen3-TTS ONNX models."""
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS ONNX models in one shot.")
    parser.add_argument("--model_path", type=str, default="../Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--output_root", type=str, default=".")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--talker_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--talker_export_seq_len", type=int, default=512)
    parser.add_argument("--strip_talker_control_flow", action="store_true")
    parser.add_argument("--talker_output_dir", type=str, default="onnx_models_talker_core_kv_transpose")
    parser.add_argument("--speech_output_dir", type=str, default="onnx_models_speech_tokenizer")
    parser.add_argument("--speech_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--speech_example_seq_len", type=int, default=100)
    parser.add_argument("--code_predictor_output_dir", type=str, default="onnx_models_talker_core")
    parser.add_argument("--code_predictor_checkpoint", type=str, default="")
    parser.add_argument("--code_predictor_custom_rope", action="store_true")
    parser.add_argument("--export_talker", action="store_true")
    parser.add_argument("--export_speech", action="store_true")
    parser.add_argument("--export_code_predictor", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    selected = any([args.export_talker, args.export_speech, args.export_code_predictor])
    do_talker = args.export_talker or not selected
    do_speech = args.export_speech or not selected
    do_code = args.export_code_predictor or not selected

    if do_talker:
        out_dir = os.path.join(output_root, args.talker_output_dir)
        export_talker_kv_onnx(
            model_path=args.model_path,
            output_dir=out_dir,
            opset=int(args.opset),
            dtype=str(args.talker_dtype),
            export_seq_len=int(args.talker_export_seq_len),
            device=str(args.device),
            strip_control_flow=bool(args.strip_talker_control_flow),
        )
        print(os.path.join(out_dir, "talker_prefill.onnx"))
        print(os.path.join(out_dir, "talker_step.onnx"))

    if do_speech:
        out_dir = os.path.join(output_root, args.speech_output_dir)
        out = export_speech_decoder_onnx(
            model_path=args.model_path,
            output_dir=out_dir,
            opset=int(args.opset),
            dtype=str(args.speech_dtype),
            device=str(args.device),
            example_seq_len=int(args.speech_example_seq_len),
        )
        print(out)

    if do_code:
        out_dir = os.path.join(output_root, args.code_predictor_output_dir)
        ckpt = str(args.code_predictor_checkpoint).strip() or None
        out = export_generate_process_onnx(
            model_path=args.model_path,
            output_dir=out_dir,
            opset=int(args.opset),
            checkpoint_path=ckpt,
            use_custom_rope=bool(args.code_predictor_custom_rope),
        )
        print(out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
