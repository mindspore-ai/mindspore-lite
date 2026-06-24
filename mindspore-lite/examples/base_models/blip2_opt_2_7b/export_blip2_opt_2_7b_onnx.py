"""Export Salesforce/blip2-opt-2.7b to four fixed-shape ONNX modules.

The model (Blip2ForConditionalGeneration) is split into four stages so that the
autoregressive OPT decoder can be exported with a static KV-cache shape:

    1. vision_model (EVA-ViT-G):   pixel_values[1,3,224,224]
                                    -> image_embeds[1,257,1408]
    2. qformer + language_projection:
            image_embeds[1,257,1408]
                                    -> query_embeds[1,32,768] (qformer)
                                    -> language_model_inputs[1,32,2560] (projection)
    3. opt prefill (use_cache=True, past=None):
            inputs_embeds[1, 32+q_len, 2560], attention_mask[1, 32+q_len],
            position_ids[1, 32+q_len]
                                    -> logits[1, 32+q_len, 50272],
                                       present_key_values[64,1,32,32+q_len,80]
    4. opt decode (use_cache=True, past given):
            inputs_embeds[1,1,2560], attention_mask[1, max_total_len],
            position_ids[1,1], past_key_values[64,1,32,max_total_len,80]
                                    -> logits[1,1,50272],
                                       present_key_values[64,1,32,max_total_len,80]

The legacy ``torch.onnx.utils.export`` path (opset 17, float32, no constant
folding) is used, mirroring the BLIP-VQA / Qwen2.5-VL conventions.
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn

try:
    from transformers import Blip2ForConditionalGeneration
except ImportError as exc:  # pragma: no cover - import guard
    print(
        "Please install transformers: pip install transformers  "
        f"(import failed: {exc})",
        file=sys.stderr,
    )
    sys.exit(1)


def _as_list_str(items):
    """Convert items to a list of string representations for ONNX attributes."""
    return [str(x) for x in items]


class _CustomPromptFlashAttention(torch.autograd.Function):
    """CANN PromptFlashAttention (full bidirectional, no mask) ONNX emitter.

    OPT self-attention is full (no causal-in-graph mask needed at export — the
    attention_mask is applied via the mask tensor). The fallback forward is a
    cheap shape-preserving stub (return query) so the trace of long sequences
    does not materialise O(seq**2) scores; the real op is the symbolic Custom node.
    """

    @staticmethod
    def forward(ctx, query, key, value, num_heads_i, num_kv_heads_i,
                scale_value_f, input_layout_s):
        """Cheap shape-preserving stub (the symbolic Custom node is the real op)."""
        del ctx, key, value, num_heads_i, num_kv_heads_i, scale_value_f, input_layout_s
        return query

    @staticmethod
    def symbolic(g, query, key, value, num_heads_i, num_kv_heads_i,
                 scale_value_f, input_layout_s):
        """Emit a Custom PromptFlashAttention node (no mask)."""
        y = g.op(
            "Custom", query, key, value,
            type_s="PromptFlashAttention",
            num_heads_i=int(num_heads_i),
            num_key_value_heads_i=int(num_kv_heads_i),
            scale_value_f=float(scale_value_f),
            input_layout_s=str(input_layout_s),
            pre_tokens_i=2147483647,
            next_tokens_i=0,
            sparse_mode_i=0,
            inner_precise_i=1,
            input_names_s=_as_list_str(["query", "key", "value"]),
            output_names_s=_as_list_str(["attention_out"]),
        )
        y.setType(query.type())
        return y


def _patch_opt_attention():
    """Replace ``F.scaled_dot_product_attention`` with the CANN Custom op.

    OPT attention calls SDPA with q/k/v in BNSD already; the replacement emits
    the Custom node directly. Applied before prefill/decode export.
    """
    import math

    def _custom_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, enable_gqa=False):
        del attn_mask, dropout_p, is_causal, enable_gqa
        num_heads = int(query.shape[1])
        head_dim = int(query.shape[-1])
        scale_val = float(scale) if scale is not None else float(1.0 / math.sqrt(head_dim))
        return _CustomPromptFlashAttention.apply(
            query, key, value, num_heads, num_heads, scale_val, "BNSD")

    torch.nn.functional.scaled_dot_product_attention = _custom_sdpa


# ---------------------------------------------------------------------------
# Stage wrappers
# ---------------------------------------------------------------------------
class VisionWrapper(nn.Module):
    """Wrap the EVA-ViT-G encoder.

    pixel_values[1,3,224,224] (float32) -> image_embeds[1,257,1408] (float32).
    """

    def __init__(self, model: Blip2ForConditionalGeneration):
        super().__init__()
        self.vision_model = model.vision_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vision_model(pixel_values=pixel_values, return_dict=True)
        return out.last_hidden_state


class QFormerWrapper(nn.Module):
    """Wrap Q-Former + language_projection.

    Inputs:
        image_embeds[1,257,1408] (float32)
    Outputs:
        query_embeds[1,32,768]   (Q-Former output, float32)
        language_model_inputs[1,32,2560] (projected to OPT dim, float32)

    The 32 learned query tokens are baked into the ONNX graph as a constant so
    the downstream MSLite infer path does not need to know about them.
    """

    def __init__(self, model: Blip2ForConditionalGeneration):
        super().__init__()
        self.query_tokens = model.query_tokens  # [1, 32, 768]
        self.qformer = model.qformer
        self.language_projection = model.language_projection  # Linear(768 -> 2560)

    def forward(self, image_embeds: torch.Tensor):
        image_atts = torch.ones(
            image_embeds.shape[:2], dtype=torch.long, device=image_embeds.device
        )
        qf_out = self.qformer(
            query_embeds=self.query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        query_embeds = qf_out.last_hidden_state  # [1, 32, 768]
        language_model_inputs = self.language_projection(query_embeds)  # [1, 32, 2560]
        return query_embeds, language_model_inputs


class OptPrefillWrapper(nn.Module):
    """Wrap OPT prefill step (past_key_values = None).

    Inputs:
        inputs_embeds[1, L, 2560]   (32 query embeds + question tokens)
        attention_mask[1, L]
        position_ids[1, L]
    Outputs:
        logits[1, L, vocab]
        present_key_values[64, 1, 32, L, 80]   (2 * num_layers, batch,
                                                 num_heads, seq, head_dim)

    The KV cache is re-stacked along dim 0 (alternating key/value per layer) so
    it can be carried as a single tensor between decode steps, matching the
    Qwen2.5-VL convention.
    """

    def __init__(self, model: Blip2ForConditionalGeneration):
        super().__init__()
        self.language_model = model.language_model  # OPTForCausalLM
        self.num_layers = model.language_model.config.num_hidden_layers
        self.num_heads = model.language_model.config.num_attention_heads

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        from transformers import DynamicCache
        out = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
            past_key_values=DynamicCache(),  # force DynamicCache (else OPT returns legacy tuple)
        )
        logits = out.logits  # [1, L, vocab]
        cache = out.past_key_values  # DynamicCache
        present = []
        for i in range(self.num_layers):
            pk = cache.key_cache[i]  # [1, 32, L, 80]
            pv = cache.value_cache[i]
            present.append(pk)
            present.append(pv)
        present_kv = torch.stack(present, dim=0)  # [64, 1, 32, L, 80]
        return logits, present_kv


class OptDecodeWrapper(nn.Module):
    """Wrap a single OPT decode step with a fixed-shape KV cache.

    Inputs:
        inputs_embeds[1, 1, 2560]
        attention_mask[1, max_total_len]
        position_ids[1, 1]
        past_key_values[64, 1, 32, max_total_len, 80]
    Outputs:
        logits[1, 1, vocab]
        present_key_values[64, 1, 32, max_total_len, 80]

    Each layer's KV cache is updated at the single current position
    (``cache_pos``) in-place via index assignment, then the rest of the cache is
    returned unchanged. The position to write is supplied as a scalar input so
    the assignment is traceable.
    """

    def __init__(self, model: Blip2ForConditionalGeneration, max_total_len: int):
        super().__init__()
        self.language_model = model.language_model
        self.num_layers = model.language_model.config.num_hidden_layers
        self.num_heads = model.language_model.config.num_attention_heads
        self.head_dim = (
            model.language_model.config.hidden_size
            // model.language_model.config.num_attention_heads
        )
        self.max_total_len = max_total_len

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: torch.Tensor,
        cache_pos: torch.Tensor,
    ):
        # cache_pos is a [1] int tensor with the current write index.
        cp = int(cache_pos[0].item())
        # Build a DynamicCache from the stacked past_key_values, sliced to [0, cp].
        from transformers import DynamicCache

        cache = DynamicCache()
        for i in range(self.num_layers):
            pk = past_key_values[2 * i][:, :, :cp, :]  # [1, 32, cp, 80]
            pv = past_key_values[2 * i + 1][:, :, :cp, :]
            cache.update(pk, pv, i)

        out = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits  # [1, 1, vocab]
        new_cache = out.past_key_values
        present = []
        for i in range(self.num_layers):
            pk_new = new_cache.key_cache[i]  # [1, 32, cp+1, 80]
            pv_new = new_cache.value_cache[i]
            # Scatter the single new column back into the fixed-shape cache.
            pk_full = past_key_values[2 * i].clone()
            pv_full = past_key_values[2 * i + 1].clone()
            pk_full[:, :, cp : cp + 1, :] = pk_new[:, :, cp : cp + 1, :]
            pv_full[:, :, cp : cp + 1, :] = pv_new[:, :, cp : cp + 1, :]
            present.append(pk_full)
            present.append(pv_full)
        present_kv = torch.stack(present, dim=0)  # [64, 1, 32, max_total_len, 80]
        return logits, present_kv


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------
def _export_module(
    module: nn.Module,
    onnx_path: Path,
    dummy_inputs: list,
    input_names: list,
    output_names: list,
    opset: int = 17,
):
    """Export a module via the legacy exporter (no constant folding)."""
    module.eval()
    torch.onnx.utils.export(
        module,
        tuple(dummy_inputs),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=False,
    )
    print(f"[export] wrote {onnx_path}")


def _clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Step 1/2: vision + qformer (vision sub-tree only)
# ---------------------------------------------------------------------------
def _export_vision_qformer(model_id: str, output_dir: Path, device: str, image_size: int):
    print("[export] Step 1/2: loading model (fp32) ...")
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    # Free the LLM (not needed for this step) to save memory.
    try:
        del model.language_model
    except AttributeError:
        pass
    _clear_cache()

    image_tokens = (image_size // 14) ** 2 + 1  # 257 for 224
    vision_hidden = model.config.vision_config.hidden_size  # 1408

    # --- Vision ---
    vision_path = output_dir / "blip2_vision.onnx"
    vision_wrapper = VisionWrapper(model).to(device)
    dummy_pixel = torch.randn(1, 3, image_size, image_size, dtype=torch.float32,
                              device=device)
    _export_module(
        vision_wrapper,
        vision_path,
        [dummy_pixel],
        input_names=["pixel_values"],
        output_names=["image_embeds"],
    )

    # --- Q-Former (+ language_projection) ---
    qformer_path = output_dir / "blip2_qformer.onnx"
    qformer_wrapper = QFormerWrapper(model).to(device)
    dummy_image_embeds = torch.randn(1, image_tokens, vision_hidden,
                                     dtype=torch.float32, device=device)
    _export_module(
        qformer_wrapper,
        qformer_path,
        [dummy_image_embeds],
        input_names=["image_embeds"],
        output_names=["query_embeds", "language_model_inputs"],
    )

    del model
    _clear_cache()


# ---------------------------------------------------------------------------
# Step 2/2: OPT prefill + decode (language sub-tree only)
# ---------------------------------------------------------------------------
def _export_opt(
    model_id: str,
    output_dir: Path,
    device: str,
    question_len: int,
    max_total_len: int,
):
    print("[export] Step 2/2: loading model (fp32) ...")
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    try:
        del model.vision_model
        del model.qformer
        del model.query_tokens
        del model.language_projection
    except AttributeError:
        pass
    _clear_cache()

    text_cfg = model.language_model.config
    hidden_size = text_cfg.hidden_size  # 2560
    num_layers = text_cfg.num_hidden_layers  # 32
    num_heads = text_cfg.num_attention_heads  # 32
    head_dim = hidden_size // num_heads  # 80
    seq_len = 32 + question_len  # 32 query embeds + question tokens

    # --- Prefill ---
    _patch_opt_attention()  # OPT attention -> CANN PromptFlashAttention Custom op
    prefill_path = output_dir / "blip2_opt_prefill.onnx"
    prefill_wrapper = OptPrefillWrapper(model).to(device)
    dummy_embeds = torch.randn(1, seq_len, hidden_size, dtype=torch.float32,
                               device=device)
    dummy_attn = torch.ones(1, seq_len, dtype=torch.long, device=device)
    dummy_pos = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    _export_module(
        prefill_wrapper,
        prefill_path,
        [dummy_embeds, dummy_attn, dummy_pos],
        input_names=["inputs_embeds", "attention_mask", "position_ids"],
        output_names=["logits", "present_key_values"],
    )

    # --- Decode ---
    decode_path = output_dir / "blip2_opt_decode.onnx"
    decode_wrapper = OptDecodeWrapper(model, max_total_len).to(device)
    dummy_step_embeds = torch.randn(1, 1, hidden_size, dtype=torch.float32,
                                    device=device)
    dummy_step_attn = torch.cat(
        [torch.ones(1, seq_len, dtype=torch.long, device=device),
         torch.zeros(1, max_total_len - seq_len, dtype=torch.long, device=device)],
        dim=1,
    )
    dummy_step_pos = torch.tensor([[seq_len]], dtype=torch.long, device=device)
    dummy_past = torch.zeros(
        2 * num_layers, 1, num_heads, max_total_len, head_dim,
        dtype=torch.float32, device=device,
    )
    dummy_cache_pos = torch.tensor([seq_len], dtype=torch.long, device=device)
    _export_module(
        decode_wrapper,
        decode_path,
        [dummy_step_embeds, dummy_step_attn, dummy_step_pos,
         dummy_past, dummy_cache_pos],
        input_names=["inputs_embeds", "attention_mask", "position_ids",
                     "past_key_values", "cache_pos"],
        output_names=["logits", "present_key_values"],
    )

    del model
    _clear_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Export Salesforce/blip2-opt-2.7b to four ONNX modules."
    )
    parser.add_argument("--model-id", default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--output-dir", default="./blip2_opt_2_7b_onnx")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--image-size", type=int, default=224,
                        help="Square vision input (default 224).")
    parser.add_argument("--question-len", type=int, default=32,
                        help="Fixed padded question length (default 32).")
    parser.add_argument("--max-total-len", type=int, default=256,
                        help="Fixed max total sequence length for the KV cache "
                             "(32 query + question + answer, default 256).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _export_vision_qformer(args.model_id, output_dir, args.device, args.image_size)
    _export_opt(args.model_id, output_dir, args.device,
                args.question_len, args.max_total_len)

    print("\n[export] DONE. Artifacts:")
    for name in ("blip2_vision.onnx", "blip2_qformer.onnx",
                 "blip2_opt_prefill.onnx", "blip2_opt_decode.onnx"):
        p = output_dir / name
        print(f"  - {p}")


if __name__ == "__main__":
    main()
