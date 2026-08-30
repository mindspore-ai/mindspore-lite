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
"""ONNX graph post-processing and the NNRT executor I/O contract.

Shared by all model exporters:

* ``fuse_add_rmsnorm`` — fuse ``Add -> MsRmsNorm`` into ``MsAddRmsNorm``
* ``duplicate_shared_initializers`` — deep-copy initializers shared by
  multiple nodes so each node owns an independent copy
* ``validate_contract`` — verify the ONNX matches ``nnrt_executor.cc``
  ``ValidateModelContract`` (7 non-KV inputs in fixed order + interleaved
  past_key_i/past_val_i per layer; outputs logits + out_key_i/out_val_i).
"""

import copy
import logging
import os

import onnx
from onnx import helper

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ONNX graph post-processing
# ─────────────────────────────────────────────────────────────────────────────


def get_initializers_info(model):
    """Build a mapping from initializer names to their consuming graph nodes."""
    init_usage_map = {init.name: [] for init in model.graph.initializer}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name in init_usage_map:
                init_usage_map[input_name].append(node)

    name_2_init = {init.name: init for init in model.graph.initializer}
    return init_usage_map, name_2_init


def duplicate_shared_initializers(model):
    """Deep-copy ONNX initializers shared by multiple nodes.

    After graph rebuilds (slim / quantization) several nodes can reference the
    same weight initializer; each consuming node gets its own deep copy so a
    later in-place edit cannot affect the others.
    """
    init_usage_map, name_2_init = get_initializers_info(model)

    for init_name, nodes in init_usage_map.items():
        if len(nodes) <= 1:
            continue
        for idx, node in enumerate(nodes):
            idx_del = None
            for idx_re, i_name in enumerate(node.input):
                if i_name == init_name:
                    idx_del = idx_re
                    break
            get_init = name_2_init.get(init_name, None)
            if get_init is None or idx_del is None:
                continue

            new_init = copy.deepcopy(get_init)
            new_name = f"{init_name}_copy{idx}"
            new_init.name = new_name
            model.graph.initializer.append(new_init)
            node.input[idx_del] = new_name


def get_fusion_outputs(users):
    """Return (rms_node, add_node) candidates for an Add node's users."""
    if len(users) == 1:
        if users[0].op_type == "MsRmsNorm":
            return users[0], None
        return None, None

    if len(users) == 2:
        rms_node = None
        add_node = None
        for i, user in enumerate(users):
            if user.op_type == "MsRmsNorm":
                rms_node = user
                add_node = users[1 - i]
                break
        return rms_node, add_node
    return None, None


def fuse_add_rmsnorm(model_path, output_path):
    """Fuse ``Add -> MsRmsNorm`` into ``MsAddRmsNorm`` (NPU kernel fusion)."""
    model = onnx.load(model_path)

    node_map = {}
    for node in model.graph.node:
        for node_input in node.input:
            if node_input in node_map:
                node_map[node_input].append(node)
            else:
                node_map[node_input] = [node]

    for node in model.graph.node:
        if node.op_type != "Add":
            continue
        add_users = node_map.get(node.output[0], [])
        rms_node, add_node = get_fusion_outputs(add_users)
        if rms_node is None:
            continue

        node_name = rms_node.name.replace("rmsnorm", "addrmsnorm")
        outputs = [rms_node.output[0]] if add_node is None else [rms_node.output[0], node.output[0]]
        fused_rmsnorm = helper.make_node(
            "MsAddRmsNorm",
            name=node_name,
            inputs=[node.input[0], node.input[1], rms_node.input[1]],
            outputs=outputs,
            epsilon=rms_node.attribute[0].f,
            domain="custom",
        )

        model.graph.node.append(fused_rmsnorm)
        model.graph.node.remove(node)
        model.graph.node.remove(rms_node)

    _save_onnx(model, output_path)
    return model


def _save_onnx(model, output_path):
    if model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF:
        onnx.save(model, output_path, save_as_external_data=True, location=os.path.basename(output_path) + ".data")
    else:
        onnx.save(model, output_path)


NNRT_NON_KV_INPUTS = [
    "valid_seq_len",
    "lmhead_idx",
    "rope_cos",
    "rope_sin",
    "inputs_embeds",
    "attention_mask",
    "embedding_weight",
]


def validate_contract(model_path: str, num_layers: int, embedding_quant: bool = False):
    """Verify the exported ONNX matches the NNRT executor's I/O contract.

    Contract (see ``nnrt_executor.cc`` ``ValidateModelContract``):
      7 non-KV inputs in fixed order (embedding_weight at index 6) + interleaved
      past_key_i/past_val_i per layer; outputs logits + interleaved out_key_i/out_val_i.

    Raises:
        ValueError: on any contract mismatch.
    """
    model = onnx.load(model_path)
    inputs = [vi.name for vi in model.graph.input]
    outputs = [o.name for o in model.graph.output]

    expected_inputs = NNRT_NON_KV_INPUTS[:]
    for i in range(num_layers):
        expected_inputs.append(f"past_key_{i}")
        expected_inputs.append(f"past_val_{i}")
    expected_outputs = ["logits"]
    for i in range(num_layers):
        expected_outputs.append(f"out_key_{i}")
        expected_outputs.append(f"out_val_{i}")

    if inputs != expected_inputs:
        raise ValueError(
            f"NNRT contract mismatch: inputs {inputs} != expected {expected_inputs}"
        )
    if outputs != expected_outputs:
        raise ValueError(
            f"NNRT contract mismatch: outputs {outputs} != expected {expected_outputs}"
        )

    from onnx import TensorProto

    embedding = model.graph.input[6]
    expected_dtype = TensorProto.UINT8 if embedding_quant else TensorProto.FLOAT16
    if embedding.type.tensor_type.elem_type != expected_dtype:
        raise ValueError(
            f"embedding_weight dtype {embedding.type.tensor_type.elem_type} != expected "
            f"{expected_dtype} (embedding_quant={embedding_quant})"
        )
    logger.info("NNRT contract OK: %d inputs / %d outputs (embedding_weight %s)",
                len(inputs), len(outputs), "UINT8" if embedding_quant else "FLOAT16")
    return True
