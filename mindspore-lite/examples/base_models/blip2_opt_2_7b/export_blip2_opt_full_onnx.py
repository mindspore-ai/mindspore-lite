"""Export Salesforce/blip2-opt-2.7b OPT as a fixed-seq full-forward ONNX (no KV cache).

Functional-first path (per "basic functionality before fusion"): the OPT decoder
is exported use_cache=False at a FIXED seq (96), with an attention_mask so the
deploy loop can pad the growing prefix each greedy step (BLIP-VQA re-feed pattern)
— slower than a fixed KV cache but avoids the fixed-shape KV-cache trace pitfalls.
Single fixed shape => no dynamicDims, light convert.

Input:  inputs_embeds[1, 96, 2560] (float32), attention_mask[1, 96] (int64), position_ids[1, 96] (int64)
Output: logits[1, 96, vocab] (float32)
"""

import argparse
import gc
import sys

import torch
import torch.nn as nn

from transformers import Blip2ForConditionalGeneration


class OptFullWrapper(nn.Module):
    """OPT full forward (no cache): inputs_embeds + attention_mask + position_ids -> logits."""

    def __init__(self, model):
        super().__init__()
        self.language_model = model.language_model

    def forward(self, inputs_embeds, attention_mask, position_ids):
        """Return logits [1, 96, vocab] for the padded prefix (mask handles padding)."""
        out = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        return out.logits


def _patch_opt_attention():
    """Replace F.scaled_dot_product_attention with the CANN Custom op (BNSD)."""
    import math

    def _as_list(items):
        return [str(x) for x in items]

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, nh, nkv, scale, layout):
            del ctx, k, v, nh, nkv, scale, layout
            return q

        @staticmethod
        def symbolic(g, q, k, v, nh, nkv, scale, layout):
            y = g.op("Custom", q, k, v, type_s="PromptFlashAttention",
                     num_heads_i=int(nh), num_key_value_heads_i=int(nkv),
                     scale_value_f=float(scale), input_layout_s=str(layout),
                     pre_tokens_i=2147483647, next_tokens_i=0, sparse_mode_i=0,
                     inner_precise_i=1, input_names_s=_as_list(["query", "key", "value"]),
                     output_names_s=_as_list(["attention_out"]))
            y.setType(q.type())
            return y

    def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
              scale=None, enable_gqa=False):
        del attn_mask, dropout_p, is_causal, enable_gqa
        nh = int(query.shape[1])
        hd = int(query.shape[-1])
        s = float(scale) if scale is not None else float(1.0 / math.sqrt(hd))
        return _Fn.apply(query, key, value, nh, nh, s, "BNSD")

    torch.nn.functional.scaled_dot_product_attention = _sdpa


def main():
    """Export the OPT full-forward ONNX (dynamic seq)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="./blip2-opt-2.7b")
    parser.add_argument("--output", default="./blip2_onnx/blip2_opt_full.onnx")
    parser.add_argument("--no-custom-op", action="store_true")
    args = parser.parse_args()

    if not args.no_custom_op:
        _patch_opt_attention()
    model = Blip2ForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.float32).eval()
    wrapper = OptFullWrapper(model).eval()
    seq = 96  # fixed max prefix length; infer pads shorter prefixes (mask=0)
    hidden = model.language_model.config.hidden_size
    inputs_embeds = torch.randn(1, seq, hidden, dtype=torch.float32)
    attention_mask = torch.ones((1, seq), dtype=torch.int64)
    position_ids = torch.arange(seq, dtype=torch.int64).unsqueeze(0)
    torch.onnx.utils.export(
        wrapper, (inputs_embeds, attention_mask, position_ids), args.output,
        input_names=["inputs_embeds", "attention_mask", "position_ids"],
        output_names=["logits"],
        opset_version=17, do_constant_folding=False,
    )
    print(f"[export] saved {args.output} (fixed seq={seq}, hidden={hidden})")
    del model, wrapper
    gc.collect()


if __name__ == "__main__":
    sys.exit(main() or 0)
