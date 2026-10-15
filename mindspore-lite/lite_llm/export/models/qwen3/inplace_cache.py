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
"""Lower the functional Qwen3 cache graph to writable-input device operators."""

from onnx import helper


def _build_cache_aliases(graph, fused, num_layers):
    """Validate fused cache interfaces and return output-to-input aliases."""
    expected_caches = [name for layer in range(num_layers)
                       for name in (f"out_key_{layer}", f"out_val_{layer}")]
    expected_outputs = ["logits", *expected_caches]
    if len(fused) != num_layers or [value.name for value in graph.output] != expected_outputs:
        raise ValueError("Expected one fused cache update per layer and interleaved Qwen3 cache outputs")

    aliases = {}
    for layer, node in enumerate(fused):
        caches = [f"past_key_{layer}", f"past_val_{layer}"]
        outputs = expected_caches[2 * layer:2 * layer + 2]
        if len(node.input) != 8 or list(node.input[5:7]) != caches or list(node.output[1:]) != outputs:
            raise ValueError(f"Unexpected fused cache interface: {node.name}")
        aliases.update(zip(outputs, caches))
    return aliases


def _single_reader(graph, value):
    """Return the sole consumer of a graph value, or None if it is not unique."""
    readers = [node for node in graph.node if value in node.input]
    return readers[0] if len(readers) == 1 else None


def _validate_attention_consumers(graph, fused):
    """Validate the ordering and input contract of fused attention consumers."""
    for node in fused:
        query, key, value = node.output
        attention = _single_reader(graph, key)
        expected_inputs = [query, key, value]
        if (attention is None or _single_reader(graph, value) != attention
                or attention.op_type != "MsAttention" or attention.domain != "custom"
                or len(attention.input) != 6 or list(attention.input[:3]) != expected_inputs
                or attention.input[5] != node.input[7]):
            raise ValueError(f"Unexpected attention cache consumers: {node.name}")
        if _single_reader(graph, query) != attention:
            raise ValueError(f"Unexpected packed query consumers: {node.name}")


def _rewrite_cache_edges(graph, fused, aliases):
    """Replace cache outputs with writable inputs and remove external outputs."""
    for node in graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = aliases.get(name, name)
    for node in fused:
        del node.output[1:]
        attributes = [attr for attr in node.attribute if attr.name != "storage_layout"]
        del node.attribute[:]
        node.attribute.extend(attributes)
        node.attribute.append(helper.make_attribute("storage_layout", "NZ"))
    del graph.output[1:]
    values = [value for value in graph.value_info if value.name not in aliases]
    del graph.value_info[:]
    graph.value_info.extend(values)


def lower_inplace_cache(model, num_layers):
    """Connect fused attention to persistent packed NZ caches and RoPE output.

    Run after generic ONNX optimizations: ONNX cannot express writable inputs.
    MsAttention reads both caches after the fused operator produces its query.
    Its config input must use the same cache write position. Keep these edges;
    do not insert cache copies or expose cache inputs as graph outputs. The NNRT
    executor must reuse the same writable cache buffers across synchronous runs.
    """
    graph = model.graph
    fused = [node for node in graph.node if node.op_type == "MsTransRopeScatterNDUpdate"]
    aliases = _build_cache_aliases(graph, fused, num_layers)
    _validate_attention_consumers(graph, fused)
    _rewrite_cache_edges(graph, fused, aliases)
