#!/usr/bin/env python3
"""Export `google/vit-base-patch16-224` to a single unified ONNX model."""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

try:
    from transformers import ViTForImageClassification
except Exception:
    ViTForImageClassification = None


class VitUnified(nn.Module):
    """Unified ViT model combining encoder and classifier head."""

    def __init__(self, vit_model: ViTForImageClassification, interpolate_pos_encoding: bool):
        super().__init__()
        self.model = vit_model
        self.interpolate_pos_encoding = bool(interpolate_pos_encoding)

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=self.interpolate_pos_encoding,
            return_dict=True,
        )
        return outputs.logits


def _check_deps():
    if ViTForImageClassification is None:
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
    """Export a PyTorch module to ONNX format."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy_inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=int(opset),
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )


def main():
    _check_deps()

    parser = argparse.ArgumentParser(description="Export google/vit-base-patch16-224 to unified ONNX.")
    parser.add_argument("--model-id", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--output-dir", type=str, default="./vit_onnx")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=15)
    parser.add_argument(
        "--dynamic-image-size",
        action="store_true",
        help="Export with dynamic H/W (requires interpolate_pos_encoding=True). May reduce converter compatibility.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = ViTForImageClassification.from_pretrained(args.model_id)
    model.eval()
    model.to(args.device)

    interpolate_pos_encoding = bool(args.dynamic_image_size)

    dummy_pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=args.device)

    if args.dynamic_image_size:
        dyn_pixel = {0: "batch", 2: "height", 3: "width"}
    else:
        dyn_pixel = {0: "batch"}

    unified = VitUnified(model, interpolate_pos_encoding=interpolate_pos_encoding).to(args.device).eval()
    _export_one(
        unified,
        output_dir / "vit_unified.onnx",
        (dummy_pixel_values,),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": dyn_pixel, "logits": {0: "batch"}},
        opset=args.opset,
    )

    print("Export complete.")
    print(f"ONNX saved to: {output_dir}")


if __name__ == "__main__":
    main()
