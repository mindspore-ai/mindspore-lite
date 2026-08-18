#!/usr/bin/env python3
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
"""
Model adapter registry for ParallelManager.

Model types are detected with a lookup table (see _MODEL_MATCH_TABLE):
entries are matched in order by class-name prefix, plus optional
extra-class/marker-attribute conditions. Adding a new model type only
requires appending an entry to _MODEL_MATCH_TABLE and _BOOST_REGISTRY.
"""
import importlib

# Wan2.2-specific attribute: some Wan2.2 pipelines have low_noise_model.
_WAN22_MARKER_ATTR = 'low_noise_model'

# Wan2.2 classes that do NOT have low_noise_model (e.g. WanTI2V, WanS2V).
_WAN22_CLASSES = frozenset({'WanTI2V', 'WanS2V'})

# Model-type match table: (key, class_prefixes, extra_classes, marker_attr).
#   class_prefixes: class name must start with one of these prefixes.
#   extra_classes:  class names that match without any marker attribute.
#   marker_attr:    instance attribute that also matches; None means the
#                   entry matches unconditionally (fallback for its prefix).
# Entries are checked in order; the first hit wins, so specific entries
# (e.g. wan2_2) must precede their generic fallback (e.g. wan2_1).
_MODEL_MATCH_TABLE = (
    ('qwen_image_edit', ('QwenImageEdit',), frozenset(), None),
    ('wan2_2', ('Wan',), _WAN22_CLASSES, _WAN22_MARKER_ATTR),
    ('wan2_1', ('Wan',), frozenset(), None),
)

_BOOST_REGISTRY = {
    'wan2_1': ('.wan2_1.boost', 'boost_wan2_1'),
    'wan2_2': ('.wan2_2.boost', 'boost_wan2_2'),
    'qwen_image_edit': ('.qwen_image_edit.boost', 'boost_qwen_image_edit'),
}


def detect_model_type(model) -> str:
    """Detect the model type by walking the match table in order."""
    cls_name = model.__class__.__name__
    for key, prefixes, extra_classes, marker_attr in _MODEL_MATCH_TABLE:
        if not cls_name.startswith(prefixes):
            continue
        if cls_name in extra_classes or marker_attr is None or hasattr(model, marker_attr):
            return key
    raise ValueError(
        f"Unsupported model type: {cls_name}. "
        f"Expected a Wan-series or Qwen-Image-Edit pipeline or model."
    )


def setup_model(model):
    """Dispatch model setup based on detected model type."""
    module_key = detect_model_type(model)
    pkg, fn = _BOOST_REGISTRY[module_key]
    mod = importlib.import_module(pkg, package=__name__)
    getattr(mod, fn)(model)
