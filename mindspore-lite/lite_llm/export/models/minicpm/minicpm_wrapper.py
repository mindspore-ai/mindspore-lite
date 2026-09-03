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
"""MiniCPM NNRT ONNX wrapper.

MiniCPM 1/2 is a LLaMA-derived decoder (verified against the upstream
``modeling_minicpm.py``: same projection / layernorm / MLP attribute names,
standard RoPE + GQA attention, per-layer BNSD KV cache), so the base
:class:`NnrtDecoderWrapper` path applies unchanged except for one numerical
difference:

* ``config.scale_emb`` (12 on MiniCPM-2B): the input embedding is scaled by
  this factor before the decoder stack.  The NNRT contract computes the
  embedding lookup CPU-side (``inputs_embeds`` graph input), so the scale is
  applied here, at the graph entry, as a single ``Mul`` — one traced node,
  no runtime change needed.  Models without ``scale_emb`` (LLaMA proper,
  MiniCPM5 which is native-llama) skip the multiplication entirely.
"""

from models._base.nnrt_decoder_wrapper import NnrtDecoderWrapper


class MiniCpmNnrtWrapper(NnrtDecoderWrapper):
    """NNRT wrapper for MiniCPM — base path + ``scale_emb`` input scaling."""

    def __init__(self, hf_model, config, op_set=None):
        super().__init__(hf_model, config, op_set)
        # float(...) guards against None / str values in exotic configs.
        self.scale_emb = float(getattr(config, "scale_emb", 1) or 1)

    def forward(self, input_ids=None, valid_seq_len=None, lmhead_idx=None,
                rope_cos=None, rope_sin=None, inputs_embeds=None,
                attention_mask=None, past_key_values=None):
        if self.scale_emb != 1.0:
            inputs_embeds = inputs_embeds * self.scale_emb
        return super().forward(
            input_ids=input_ids,
            valid_seq_len=valid_seq_len,
            lmhead_idx=lmhead_idx,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
