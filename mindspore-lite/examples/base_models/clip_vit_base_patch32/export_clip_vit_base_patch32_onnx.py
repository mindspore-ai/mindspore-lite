#!/usr/bin/env python3
"""Export `openai/clip-vit-base-patch32` vision tower to a single unified ONNX model.

The exported module wraps `CLIPVisionModelWithProjection`:

  - input  : pixel_values          [batch, 3, 224, 224]   float32
  - output : image_embeds          [batch, 512]            float32   (projected)
  - output : last_hidden_state     [batch, 50, 768]        float32   (CLS + 49 patches)

The projected `image_embeds` is the canonical CLIP image embedding used for
zero-shot classification and image-text retrieval (cosine with text embeds).
Keeping the text encoder out of the ONNX makes the deployed model self-contained;
the text side can be computed once offline (see `align_*` / `infer_*` scripts).
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

try:
    from transformers import CLIPVisionModelWithProjection
except Exception:
    CLIPVisionModelWithProjection = None


class ClipVisionUnified(nn.Module):
    """Unified CLIP vision module returning (image_embeds, last_hidden_state)."""

    def __init__(self, model: CLIPVisionModelWithProjection, interpolate_pos_encoding: bool):
        super().__init__()
        self.model = model
        self.interpolate_pos_encoding = bool(interpolate_pos_encoding)

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=self.interpolate_pos_encoding,
            return_dict=True,
        )
        # image_embeds: [B, 512]; last_hidden_state: [B, 50, 768]
        return outputs.image_embeds, outputs.last_hidden_state


def _check_deps():
    if CLIPVisionModelWithProjection is None:
        print("Error: transformers not found or version is incompatible.")
        sys.exit(1)


def _export_one(
    module: nn.Module,
    onnx_path: Path,
    dummy_inputs,
    input_names,
    output_names,
    dynamic_axes,
    opset: int,
):
    """Export a PyTorch module to ONNX format (legacy exporter)."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    del dynamic_axes  # fixed-shape export (batch=1) for ascend_oriented conversion
    with torch.no_grad():
        # Legacy exporter (dynamo=False): the dynamo exporter emits an opset-18
        # graph the ACL convert pass cannot lower; legacy traces cleanly.
        torch.onnx.export(
            module,
            dummy_inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=int(opset),
            do_constant_folding=False,
            dynamo=False,
        )


def main():
    _check_deps()

    parser = argparse.ArgumentParser(
        description="Export openai/clip-vit-base-patch32 vision tower to unified ONNX."
    )
    parser.add_argument("--model-id", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--output-dir", type=str, default="./clip_onnx")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--dynamic-image-size",
        action="store_true",
        help="Export with dynamic H/W (requires interpolate_pos_encoding=True). "
        "May reduce converter compatibility.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = CLIPVisionModelWithProjection.from_pretrained(args.model_id, torch_dtype=torch.float32)
    model.eval()
    model.to(args.device)

    interpolate_pos_encoding = bool(args.dynamic_image_size)

    dummy_pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=args.device)

    if args.dynamic_image_size:
        dyn_pixel = {0: "batch", 2: "height", 3: "width"}
    else:
        dyn_pixel = {0: "batch"}

    unified = ClipVisionUnified(model, interpolate_pos_encoding=interpolate_pos_encoding)
    unified = unified.to(args.device).eval()
    _export_one(
        unified,
        output_dir / "clip_vision.onnx",
        (dummy_pixel_values,),
        input_names=["pixel_values"],
        output_names=["image_embeds", "last_hidden_state"],
        dynamic_axes={
            "pixel_values": dyn_pixel,
            "image_embeds": {0: "batch"},
            "last_hidden_state": {0: "batch"},
        },
        opset=args.opset,
    )

    print("Export complete.")
    print(f"ONNX saved to: {output_dir}")


if __name__ == "__main__":
    main()
