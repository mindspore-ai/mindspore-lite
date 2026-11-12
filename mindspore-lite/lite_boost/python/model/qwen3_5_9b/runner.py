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
"""Service-style runner for Qwen3.5-9B: pooled prefill + captured decode.

Prefill forwards to the frozen pooled bucket pipelines; decode replays one
graph captured at startup.  Requests are serialized because the pooled cache
and the graph's static input/output buffers are shared.
"""
from __future__ import annotations

import logging

from .graph_decode import (
    init_bucket_pipelines,
    init_decode_pipeline,
    pooled_decode,
    pooled_prefill,
)

logger = logging.getLogger(__name__)


def _reset_rope_deltas(model):
    """Drop mRoPE state left by a previous request on a text-only prompt."""
    inner = getattr(model, "model", model)
    if hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None


def _default_stop_ids(model):
    """Return the checkpoint's EOS ids as a tuple (empty when unconfigured)."""
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) or config
    eos = getattr(text_config, "eos_token_id", None)
    if eos is None:
        return ()
    if isinstance(eos, int):
        return (eos,)
    return tuple(int(tok) for tok in eos)


class Qwen35GraphRunner:
    """Frozen bucketed prefill plus replay-only decode for a long-running service.

    Call ``warmup()`` once before serving: it captures every bucket's prefill
    graph and a single decode graph.  Afterwards ``prefill``/``generate`` only
    copy static inputs and replay graphs; there is no per-request capture.

    ``stop_token_ids`` defaults to the checkpoint's EOS ids; pass ``()`` to
    always generate the full budget, or an int / iterable to override.
    """

    def __init__(self, model, buckets, max_new_tokens: int = 256,
                 device: str = "npu:0", num_warmup: int = 3, stream=None,
                 stop_token_ids=None):
        self.model = model
        self.buckets = tuple(sorted(int(b) for b in buckets))
        if not self.buckets:
            raise ValueError("buckets must be a non-empty sequence")
        self.max_new_tokens = int(max_new_tokens)
        self.device = device
        self.num_warmup = num_warmup
        self.stream = stream
        if stop_token_ids is None:
            stop_token_ids = _default_stop_ids(model)
        elif isinstance(stop_token_ids, int):
            stop_token_ids = (stop_token_ids,)
        self.stop_token_ids = tuple(int(tok) for tok in stop_token_ids)
        self._warmed = False
        self.last_bucket = None
        self.last_prefill_hidden = None

    def _ensure_warmed(self):
        if not self._warmed:
            raise RuntimeError("Qwen35GraphRunner.warmup() must run before requests")

    def _bucket_for(self, real_len):
        eligible = [size for size in self.buckets if size >= real_len]
        if not eligible:
            raise ValueError(
                f"prompt length {real_len} exceeds the largest bucket "
                f"{max(self.buckets)}; overflow prefill is not supported by the "
                "service runner")
        return min(eligible)

    def warmup(self, timing: dict | None = None):
        """Capture the pooled prefill graphs and the decode graph once."""
        _reset_rope_deltas(self.model)
        init_bucket_pipelines(
            self.model, buckets=self.buckets, device=self.device,
            num_warmup=self.num_warmup, stream=self.stream, timing=timing,
            max_new_tokens=self.max_new_tokens,
        )
        init_decode_pipeline(
            self.model, device=self.device, num_warmup=self.num_warmup,
            stream=self.stream, timing=timing,
        )
        self._warmed = True
        logger.info("%s", f"[runner] warmed buckets={self.buckets} "
              f"max_new_tokens={self.max_new_tokens}")

    def prefill(self, input_ids, attention_mask=None, pixel_values=None,
                image_grid_thw=None, mm_token_type_ids=None,
                extra_token_embeds=None, timing=None):
        """Run one pooled prefill and return ``(first_token, logits)``.

        Multimodal inputs are prepared by the caller: ``pixel_values`` for
        images, or already-spliced ``input_ids`` plus ``extra_token_embeds``
        in place of the placeholder span.  The caller owns prompt expansion
        because the placeholder layout is processor semantics, not a serving
        invariant.
        """
        self._ensure_warmed()
        if pixel_values is None:
            _reset_rope_deltas(self.model)
        next_tok, logits = pooled_prefill(
            self.model, input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            extra_token_embeds=extra_token_embeds, device=self.device,
            timing=timing,
        )
        self.last_bucket = self._bucket_for(input_ids.shape[1])
        self.last_prefill_hidden = self.model._pool_outs[self.last_bucket]
        if timing is not None:
            timing["first_token_ms"] = timing.get("prefill_total", 0.0) * 1e3
        return next_tok, logits

    def decode(self, first_token, real_len, max_new_tokens=None, timing=None,
               opt_no_decode_sync=False):
        """Replay the decode graph and return at most ``max_new_tokens`` tokens.

        Generation stops early when ``self.stop_token_ids`` is hit; ``timing``
        then carries ``finish_reason``/``stop_token``/``decode_steps``.
        """
        self._ensure_warmed()
        total = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        if total > self.max_new_tokens:
            raise ValueError(
                f"decode: requested {total} tokens but the runner was initialized "
                f"for {self.max_new_tokens}")
        return pooled_decode(
            self.model, first_token, real_len, total, device=self.device,
            timing=timing, opt_no_decode_sync=opt_no_decode_sync,
            stop_token_ids=self.stop_token_ids,
        )

    def generate(self, input_ids, attention_mask=None, pixel_values=None,
                 image_grid_thw=None, mm_token_type_ids=None,
                 extra_token_embeds=None, max_new_tokens=None, timing=None,
                 opt_no_decode_sync=False):
        """Prefill once, then decode at most ``max_new_tokens`` tokens."""
        next_tok, _ = self.prefill(
            input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
            image_grid_thw=image_grid_thw, mm_token_type_ids=mm_token_type_ids,
            extra_token_embeds=extra_token_embeds, timing=timing,
        )
        return self.decode(
            next_tok, input_ids.shape[1], max_new_tokens=max_new_tokens,
            timing=timing, opt_no_decode_sync=opt_no_decode_sync,
        )
