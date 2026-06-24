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

"""Export Salesforce/blip-vqa-base (BlipForQuestionAnswering) to ONNX.

BLIP VQA combines a ViT vision encoder, a BERT text encoder (cross-attending to
image embeddings) and a BERT text decoder (cross-attending to the encoded
question) that autoregressively generates the answer. Because the decoder runs
an autoregressive ``generate`` loop with a KV-cache, the model cannot be traced
into a single fixed-shape ONNX. We therefore export THREE sub-models:

  1. ``blip_vqa_vision.onnx``  - vision encoder, ``pixel_values -> image_embeds``
  2. ``blip_vqa_text_encoder.onnx`` - text encoder, question cross-attends image
  3. ``blip_vqa_text_decoder.onnx`` - text decoder single-step, ``prefix -> logits``

The decoder is exported without a KV-cache (``use_cache=False``); the inference
script re-feeds the full answer prefix each step and greedy-decodes in numpy.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.onnx import utils as onnx_utils

try:
    from transformers import BlipForQuestionAnswering
except ImportError as exc:
    print("Error: transformers package not found or version too low.")
    print("Please install: pip install -U transformers")
    raise SystemExit(1) from exc


class VisionWrapper(nn.Module):
    """Wrap BlipVisionModel to emit only ``image_embeds`` (last_hidden_state)."""

    def __init__(self, model: BlipForQuestionAnswering):
        super().__init__()
        self.vision_model = model.vision_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the vision encoder and return image embeddings (last_hidden_state)."""
        outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
        return outputs.last_hidden_state


class TextEncoderWrapper(nn.Module):
    """Wrap the BLIP text encoder (encoder mode) with image cross-attention."""

    def __init__(self, model: BlipForQuestionAnswering):
        super().__init__()
        self.text_encoder = model.text_encoder

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_embeds: torch.Tensor,
        image_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the text encoder (encoder mode) with image cross-attention."""
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state


class TextDecoderWrapper(nn.Module):
    """Wrap the BLIP text decoder for a single forward of the answer prefix.

    ``use_cache=False`` so no KV-cache tensors are present in the ONNX graph.
    The greedy decode loop is implemented in the inference script.
    """

    def __init__(self, model: BlipForQuestionAnswering):
        super().__init__()
        self.text_decoder = model.text_decoder

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the text decoder single-step on the answer prefix, returning logits."""
        outputs = self.text_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        return outputs.logits


def _export_module(
    module: nn.Module,
    onnx_path: Path,
    dummy_inputs: tuple,
    input_names: list,
    output_names: list,
    opset: int,
    dynamic_axes: dict | None = None,
):
    """Export a torch module with the legacy ONNX exporter conventions."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        onnx_utils.export(
            module,
            dummy_inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=int(opset),
            do_constant_folding=False,
            dynamic_axes=dynamic_axes,
        )


def _parse_args():
    p = argparse.ArgumentParser(
        description="Export Salesforce/blip-vqa-base to 3 ONNX sub-models."
    )
    p.add_argument(
        "--model-id",
        type=str,
        default="Salesforce/blip-vqa-base",
        help="HuggingFace model id or local directory.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./blip_vqa_onnx",
        help="Output directory for ONNX files.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Export device.",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    p.add_argument(
        "--question-len",
        type=int,
        default=20,
        help="Fixed question length (padded). Must match inference.",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Square image size (BLIP default 384).",
    )
    return p.parse_args()


def main():
    """Export BLIP VQA into vision / text-encoder / text-decoder ONNX files."""
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    image_size = int(args.image_size)
    question_len = int(args.question_len)

    print(f"Loading model {args.model_id} (float32, device={device}) ...")
    model = BlipForQuestionAnswering.from_pretrained(
        args.model_id, torch_dtype=torch.float32
    )
    model.eval().to(device)

    vision_cfg = model.config.vision_config
    text_cfg = model.config.text_config
    patch = int(getattr(vision_cfg, "patch_size", 16))
    num_image_tokens = (image_size // patch) ** 2 + 1
    vision_hidden = int(vision_cfg.hidden_size)
    text_hidden = int(text_cfg.hidden_size)
    if vision_hidden != text_hidden:
        print(
            f"Warning: vision_hidden({vision_hidden}) != text_hidden({text_hidden}); "
            "cross-attention requires them to match."
        )

    # ---- Stage 1: vision encoder ----------------------------------------
    print("Stage 1/3: exporting vision encoder ...")
    dummy_pixel = torch.randn(1, 3, image_size, image_size, device=device)
    vision_wrapper = VisionWrapper(model).to(device).eval()
    _export_module(
        vision_wrapper,
        output_dir / "blip_vqa_vision.onnx",
        (dummy_pixel,),
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        opset=args.opset,
    )

    # ---- Stage 2: text encoder (question + image cross-attention) -------
    print("Stage 2/3: exporting text encoder ...")
    dummy_input_ids = torch.randint(0, 1000, (1, question_len), device=device)
    dummy_attn = torch.ones(1, question_len, dtype=torch.long, device=device)
    dummy_image_embeds = torch.randn(1, num_image_tokens, vision_hidden, device=device)
    dummy_image_attn = torch.ones(1, num_image_tokens, dtype=torch.long, device=device)
    text_enc_wrapper = TextEncoderWrapper(model).to(device).eval()
    _export_module(
        text_enc_wrapper,
        output_dir / "blip_vqa_text_encoder.onnx",
        (
            dummy_input_ids,
            dummy_attn,
            dummy_image_embeds,
            dummy_image_attn,
        ),
        input_names=[
            "input_ids",
            "attention_mask",
            "image_embeds",
            "image_attention_mask",
        ],
        output_names=["question_embeds"],
        opset=args.opset,
    )

    # ---- Stage 3: text decoder single-step ------------------------------
    # Decoder is exported for a DYNAMIC prefix length (answer grows each greedy
    # step); dynamic_axes marks the sequence dim so the converter's
    # ge.dynamicDims (1..max) can lower it.
    print("Stage 3/3: exporting text decoder ...")
    dummy_dec_ids = torch.randint(0, 1000, (1, 5), device=device, dtype=torch.long)
    dummy_q_embeds = torch.randn(1, question_len, text_hidden, device=device)
    dummy_q_attn = torch.ones(1, question_len, dtype=torch.long, device=device)
    text_dec_wrapper = TextDecoderWrapper(model).to(device).eval()
    _export_module(
        text_dec_wrapper,
        output_dir / "blip_vqa_text_decoder.onnx",
        (dummy_dec_ids, dummy_q_embeds, dummy_q_attn),
        input_names=[
            "decoder_input_ids",
            "encoder_hidden_states",
            "encoder_attention_mask",
        ],
        output_names=["logits"],
        opset=args.opset,
        dynamic_axes={"decoder_input_ids": {1: "L"}, "logits": {1: "L"}},
    )

    print("\nExport complete. Files saved to:", output_dir)
    print(
        f"  image_size={image_size}, num_image_tokens={num_image_tokens}, "
        f"question_len={question_len}, vision_hidden={vision_hidden}, "
        f"text_hidden={text_hidden}, vocab_size={text_cfg.vocab_size}"
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
