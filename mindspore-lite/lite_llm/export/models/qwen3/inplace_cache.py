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


# Side-effect-free shape/index arithmetic that may appear inside a traced cache
# update expression and is therefore safe to prune once the device operator
# takes over the write.  ``ConstantOfShape``/``Equal``/``Where`` come from the
# constant broadcast idiom Torch emits when exporting ``index_put`` (the
# ScatterND indices are built as Where(Equal(Mul(ConstantOfShape(...)))) then
# broadcast with Expand), so omitting them rejects every real traced graph.
SHADOW_OPS = frozenset({
    "Identity", "Constant", "ConstantOfShape", "Cast", "Transpose", "Reshape",
    "Expand", "Shape", "Slice", "Concat", "Add", "Mul", "Equal", "Where",
    "Unsqueeze", "Squeeze", "Neg",
})


class _GraphIndex:
    """Index tensor edges without loading or copying initializer payloads."""

    def __init__(self, graph):
        self.nodes = list(graph.node)
        self.producers = {name: index for index, node in enumerate(self.nodes) for name in node.output}
        self.inputs = {value.name for value in graph.input}
        self.constants = {value.name for value in graph.initializer}

    def trace(self, value, anchors):
        """Collect an allowed shadow expression, distinguishing shapes from values."""
        pending = [(value, False)]
        visited, nodes, roots = set(), set(), set()
        while pending:
            name, shape_only = pending.pop()
            if not name or (name, shape_only) in visited:
                continue
            visited.add((name, shape_only))
            if name in anchors:
                if not shape_only:
                    roots.add(name)
                continue
            if name in self.constants:
                continue
            index = self.producers.get(name)
            if index is None:
                raise ValueError(f"Unexpected cache update source: {name}")
            node = self.nodes[index]
            if node.domain or node.op_type not in SHADOW_OPS:
                raise ValueError(f"Unexpected cache update operator: {node.op_type}")
            nodes.add(index)
            pending.extend((source, shape_only or node.op_type == "Shape") for source in node.input)
        return roots, nodes


def _cache_update(view, output, cache, fused, is_key):
    """Match one ScatterND copy whose value comes from this RoPE's K or V."""
    aliases = set()
    while output in view.producers:
        index = view.producers[output]
        node = view.nodes[index]
        if node.domain or node.op_type != "Identity":
            break
        aliases.add(index)
        output = node.input[0]
    index = view.producers.get(output)
    node = view.nodes[index] if index is not None else None
    if (node is None or node.domain or node.op_type != "ScatterND"
            or len(node.input) != 3 or list(node.output) != [output] or node.input[0] != cache):
        raise ValueError(f"Unexpected traced cache update for {cache}")
    anchors = set(fused.input)
    sources, payload = view.trace(node.input[2], anchors)
    expected = {fused.input[1], *fused.input[3:5]} if is_key else {fused.input[2]}
    if sources != expected:
        raise ValueError(f"Unexpected cache update values for {cache}")
    positions, indices = view.trace(node.input[1], anchors)
    if not positions <= {fused.input[7]}:
        raise ValueError(f"Unexpected cache update indices for {cache}")
    return output, aliases | payload | indices | {index}, index


def _layer_plan(view, fused, layer):
    """Validate the Q dependency, cache copies and attention inputs for a layer."""
    caches = [f"past_key_{layer}", f"past_val_{layer}"]
    if (len(fused.input) != 8 or len(fused.output) != 1
            or list(fused.input[5:7]) != caches or not set(caches) <= view.inputs):
        raise ValueError(f"Unexpected fused cache interface: {fused.name}")
    readers = [index for index, node in enumerate(view.nodes) if fused.output[0] in node.input]
    if len(readers) != 1:
        raise ValueError(f"Unexpected packed query consumers: {fused.name}")
    attention = view.nodes[readers[0]]
    if (attention.domain != "custom" or attention.op_type != "MsAttention"
            or len(attention.input) != 6 or attention.input[0] != fused.output[0]
            or attention.input[5] != fused.input[7]):
        raise ValueError(f"Unexpected attention cache consumers: {fused.name}")
    candidates, copies = set(), set()
    for offset, cache in enumerate(caches):
        output = f"out_{'key' if offset == 0 else 'val'}_{layer}"
        head, nodes, copy = _cache_update(view, output, cache, fused, offset == 0)
        source = attention.input[offset + 1]
        while source in view.producers and view.nodes[view.producers[source]].op_type == "Identity":
            source = view.nodes[view.producers[source]].input[0]
        if source != head:
            raise ValueError(f"Unexpected attention cache consumers: {fused.name}")
        candidates.update(nodes)
        copies.add(copy)
    return readers[0], candidates, copies


def _dead_shadow_nodes(view, candidates, attentions, copies, caches):
    """Keep shared shape expressions and reject externally observed cache copies."""
    live, pending = set(), []
    for index, node in enumerate(view.nodes):
        if index in candidates:
            continue
        inputs = [name for offset, name in enumerate(node.input)
                  if index not in attentions or offset not in (1, 2)]
        pending.extend(inputs)
        if any(name in caches for name in inputs) and index not in attentions:
            if node.op_type != "MsTransRopeScatterNDUpdate" or node.domain != "custom":
                raise ValueError(f"Unexpected persistent cache consumer: {node.name}")
    while pending:
        index = view.producers.get(pending.pop())
        if index not in candidates or index in live:
            continue
        live.add(index)
        pending.extend(view.nodes[index].input)
    if copies & live:
        raise ValueError("Unexpected consumers of traced cache copies")
    return candidates - live


def lower_traced_cache(graph, fused, num_layers):
    """Reconnect attention and prune only matched, unobserved shadow update chains.

    Validate the complete plan first so unsupported graphs remain unchanged.
    The fused query edge orders the device-side cache write before attention.
    Other ScatterND nodes and shared expressions are outside the removal plan.
    """
    outputs = [name for layer in range(num_layers) for name in (f"out_key_{layer}", f"out_val_{layer}")]
    if len(fused) != num_layers or [value.name for value in graph.output] != ["logits", *outputs]:
        raise ValueError("Expected interleaved Qwen3 cache outputs")
    view = _GraphIndex(graph)
    plans = [_layer_plan(view, node, layer) for layer, node in enumerate(fused)]
    attentions = {attention for attention, _, _ in plans}
    candidates = set().union(*(nodes for _, nodes, _ in plans))
    copies = set().union(*(nodes for _, _, nodes in plans))
    caches = {name for node in fused for name in node.input[5:7]}
    dead = _dead_shadow_nodes(view, candidates, attentions, copies, caches)
    for node, (attention, _, _) in zip(fused, plans):
        view.nodes[attention].input[1:3] = node.input[5:7]
        attributes = [attr for attr in node.attribute if attr.name != "storage_layout"]
        del node.attribute[:]
        node.attribute.extend(attributes)
        node.attribute.append(helper.make_attribute("storage_layout", "NZ"))
    _prune_shadow_nodes(graph, view, dead)


def _prune_shadow_nodes(graph, view, dead):
    """Remove dead update expressions and their unused tensor metadata."""
    removed = {name for index in dead for name in view.nodes[index].output}
    orphan_candidates = {name for index in dead for name in view.nodes[index].input}
    kept = [node for index, node in enumerate(view.nodes) if index not in dead]
    used = {name for node in kept for name in node.input} | view.inputs | {"logits"}
    removed.update(orphan_candidates - used)
    initializers = [value for value in graph.initializer if value.name not in removed]
    values = [value for value in graph.value_info if value.name not in removed]
    del graph.node[:]
    graph.node.extend(kept)
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    del graph.value_info[:]
    graph.value_info.extend(values)
    del graph.output[1:]


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
    if fused and all(len(node.output) == 1 for node in fused):
        lower_traced_cache(graph, fused, num_layers)
        return
    aliases = _build_cache_aliases(graph, fused, num_layers)
    _validate_attention_consumers(graph, fused)
    _rewrite_cache_edges(graph, fused, aliases)
