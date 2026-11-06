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
"""HF/GGUF generation stop/suppress token policy shared by quant and export.

Pure Python (json/pathlib only). GGUF producers in this package and the export
tokenizer both resolve the policy here, so this package never imports export.
"""

import json
from pathlib import Path

# Private metadata keeps multi-EOS/suppression policy alongside GGUF's scalar EOS ID.
GGUF_GENERATION_CONFIG_KEY = "tokenizer.mslite.generation_config"


def _read_optional_json(path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_token_ids(value, field):
    """Normalise a token-id value into a sorted de-duplicated list."""
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    token_ids = []
    for token_id in values:
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"{field} must contain non-negative integer token IDs")
        if token_id not in token_ids:
            token_ids.append(token_id)
    return token_ids


def load_generation_policy(tokenizer, model_dir):
    """Resolve and validate HF or GGUF stop/suppress IDs without writing a file."""
    generation_config = getattr(tokenizer, "generation_config", None)
    if generation_config is None:
        generation_config = _read_optional_json(Path(model_dir) / "generation_config.json")
    model_config = _read_optional_json(Path(model_dir) / "config.json")

    eos_token_ids = generation_config.get("eos_token_id")
    if eos_token_ids is None:
        eos_token_ids = model_config.get("eos_token_id")
    if eos_token_ids is None:
        eos_token_ids = getattr(tokenizer, "eos_token_id", None)

    stop_token_ids = _normalize_token_ids(eos_token_ids, "eos_token_id")
    suppress_token_ids = _normalize_token_ids(generation_config.get("suppress_tokens"), "suppress_tokens")

    conflict = sorted(set(stop_token_ids) & set(suppress_token_ids))
    if conflict:
        raise ValueError(f"Stop and suppress token IDs must not overlap: {conflict}")

    return {"stop_token_ids": stop_token_ids, "suppress_token_ids": suppress_token_ids}
