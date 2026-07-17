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

"""Export IDEA-Research/grounding-dino-base to a single ONNX model.

This script wraps :class:`transformers.GroundingDinoForObjectDetection` and
exports it to ONNX with fixed input shapes that are friendly to Ascend GE
compilation (``--optimize=ascend_oriented``).

The multi-scale deformable attention (MSDA) layer can be exported in two ways:

- non-fuse: hot-patch MSDA to use ``F.grid_sample`` so the ONNX contains only
  standard ops (GridSample). This is the compatibility path.
- fuse: emit ``MultiScaleDeformableAttnFunction`` directly in ONNX via a
  custom symbolic, so the downstream converter/backend can select the fused
  MSDA kernel when available.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForZeroShotObjectDetection

PIXEL_H = 800
PIXEL_W = 1333
MAX_TEXT_LEN = 256
OPSET_VERSION = 17


def _patch_msda_to_grid_sample():
    """Replace ``MultiScaleDeformableAttention.forward`` with grid_sample.

    The upstream forward calls a C++ CUDA kernel
    (``MultiScaleDeformableAttnFunction``) that cannot be traced by
    ``torch.onnx``.  This hot-patch replaces it with a pure-PyTorch
    implementation using ``F.grid_sample``, which ONNX exports as standard
    ``GridSample`` nodes.
    """
    from transformers.models.grounding_dino import modeling_grounding_dino as gm

    def new_forward(self, value, value_spatial_shapes,
                    value_spatial_shapes_list, level_start_index,
                    sampling_locations, attention_weights, im2col_step):
        del self, value_spatial_shapes_list, im2col_step, level_start_index
        return msda_forward_pytorch(value, value_spatial_shapes,
                                    sampling_locations, attention_weights)

    gm.MultiScaleDeformableAttention.forward = new_forward


def msda_forward_pytorch(value, value_spatial_shapes,
                         sampling_locations, attention_weights):
    """Pure-PyTorch multi-scale deformable attention using grid_sample."""
    batch_size, _, num_heads, hidden_dim = value.shape
    _, _, _, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split(
        [int(h * w) for h, w in value_spatial_shapes.tolist()], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (h, w) in enumerate(value_spatial_shapes.tolist()):
        value_l = (value_list[level_id].flatten(2).transpose(1, 2)
                   .reshape(batch_size * num_heads, hidden_dim, h, w))
        sampling_grid_l = (sampling_grids[:, :, :, level_id]
                           .transpose(1, 2).flatten(0, 1))
        sampling_value_l = F.grid_sample(
            value_l, sampling_grid_l, mode="bilinear",
            padding_mode="zeros", align_corners=False)
        sampling_value_list.append(sampling_value_l)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, sampling_locations.shape[1],
        num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2)
              * attention_weights).sum(-1)
    output = output.view(batch_size, num_heads * hidden_dim, -1)
    return output.transpose(1, 2).contiguous()


def _patch_msda_fusion():
    """Emit ``MultiScaleDeformableAttnFunction`` in ONNX via a custom symbolic."""
    from transformers.models.grounding_dino import modeling_grounding_dino as gm
    from torch.autograd import Function
    from torch.onnx.symbolic_helper import _get_tensor_dim_size

    class _MSDAFusionFn(Function):
        """Autograd shim that emits a fused ``MultiScaleDeformableAttnFunction`` op.

        The ``forward`` runs the reference PyTorch implementation so eager outputs
        stay numerically correct, while ``symbolic`` rewrites the export to a
        single custom ONNX op that downstream MSDA fusion can match against.
        """

        @staticmethod
        def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                    sampling_locations, attention_weights):
            del ctx, value_level_start_index
            return msda_forward_pytorch(value, value_spatial_shapes,
                                        sampling_locations, attention_weights)

        @staticmethod
        def symbolic(g, value, value_spatial_shapes, value_level_start_index,
                     sampling_locations, attention_weights):
            """Emit the custom MSDA op with an explicit output shape for tracing."""
            y = g.op(
                "MultiScaleDeformableAttnFunction",
                value,
                value_spatial_shapes,
                value_level_start_index,
                sampling_locations,
                attention_weights,
                input_names_s=[
                    "value",
                    "value_spatial_shapes",
                    "value_level_start_index",
                    "sampling_locations",
                    "attention_weights",
                ],
                output_names_s=["output"],
                type_s="MultiScaleDeformableAttnFunction",
            )

            value_sizes = value.type().sizes() or []
            if len(value_sizes) >= 4:
                bs = value_sizes[0]
                num_query = _get_tensor_dim_size(sampling_locations, 1) or 0
                num_heads = value_sizes[2]
                embed_dims = value_sizes[3]
                y.setType(value.type().with_sizes(
                    [bs, num_query, num_heads * embed_dims]
                ))
            return y

    def new_forward(self, value, value_spatial_shapes,
                    value_spatial_shapes_list, level_start_index,
                    sampling_locations, attention_weights, im2col_step):
        del self, value_spatial_shapes_list, im2col_step
        return _MSDAFusionFn.apply(
            value,
            value_spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )

    gm.MultiScaleDeformableAttention.forward = new_forward


class GroundingDinoOnnxWrapper(torch.nn.Module):
    """Thin wrapper exposing the detection inputs/outputs as plain tensors.

    ``text_self_attention_masks`` and ``text_position_ids`` are passed in
    explicitly to bypass ``generate_masks_with_special_tokens_and_transfer_map``
    during tracing — the upstream helper mixes ``torch.eye`` (which ONNX
    lowers to the unsupported ``EyeLike`` op) with a data-dependent loop
    over special-token positions that cannot be traced symbolically.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, pixel_mask, input_ids, token_type_ids,
                attention_mask, text_self_attention_masks, text_position_ids):
        """Run the wrapped model with pre-computed text masks injected."""
        _set_precomputed_text_masks(text_self_attention_masks,
                                    text_position_ids)
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.logits, outputs.pred_boxes


_PRECOMPUTED_TEXT_MASKS = {"masks": None, "position_ids": None}


def _set_precomputed_text_masks(masks, position_ids):
    _PRECOMPUTED_TEXT_MASKS["masks"] = masks
    _PRECOMPUTED_TEXT_MASKS["position_ids"] = position_ids


def _patch_text_mask_generator():
    """Replace the special-token mask generator with a context lookup."""
    from transformers.models.grounding_dino import modeling_grounding_dino as gm

    def _patched(unused_input_ids):
        del unused_input_ids
        return (_PRECOMPUTED_TEXT_MASKS["masks"],
                _PRECOMPUTED_TEXT_MASKS["position_ids"])

    gm.generate_masks_with_special_tokens_and_transfer_map = _patched


def export_onnx(model_dir: str, output_path: str, opset: int = OPSET_VERSION):
    """Load the HF model and export a single ONNX file at ``output_path``."""
    if os.environ.get("GROUNDING_DINO_DISABLE_MSDA_FUSION", "0") == "1":
        _patch_msda_to_grid_sample()
    else:
        _patch_msda_fusion()
    _patch_text_mask_generator()
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir)
    model.eval()
    wrapper = GroundingDinoOnnxWrapper(model)

    pixel_values = torch.zeros(1, 3, PIXEL_H, PIXEL_W, dtype=torch.float32)
    pixel_mask = torch.ones(1, PIXEL_H, PIXEL_W, dtype=torch.int64)
    input_ids = torch.zeros(1, MAX_TEXT_LEN, dtype=torch.int64)
    token_type_ids = torch.zeros(1, MAX_TEXT_LEN, dtype=torch.int64)
    attention_mask = torch.zeros(1, MAX_TEXT_LEN, dtype=torch.int64)
    text_self_attention_masks = torch.eye(MAX_TEXT_LEN, dtype=torch.bool)
    text_self_attention_masks = text_self_attention_masks.unsqueeze(0).repeat(
        1, 1, 1)
    text_position_ids = torch.arange(MAX_TEXT_LEN, dtype=torch.long).unsqueeze(0)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with torch.no_grad():
        torch.onnx.utils.export(
            wrapper,
            (pixel_values, pixel_mask, input_ids, token_type_ids,
             attention_mask, text_self_attention_masks, text_position_ids),
            output_path,
            input_names=["pixel_values", "pixel_mask", "input_ids",
                         "token_type_ids", "attention_mask",
                         "text_self_attention_masks", "text_position_ids"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes=None,
            opset_version=opset,
            do_constant_folding=True,
        )
    print(f"[export] saved ONNX to {output_path}")


def _parse_args():
    """Parse command-line arguments for ONNX export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="/data/llj/models/model_weight/grounding-dino-base",
        help="Path to the HuggingFace model directory.")
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="Directory to write the ONNX file.")
    parser.add_argument(
        "--opset",
        type=int,
        default=OPSET_VERSION,
        help="ONNX opset version.")
    parser.add_argument(
        "--name",
        default="grounding_dino_base.onnx",
        help="Output ONNX filename.")
    parser.add_argument(
        "--disable-msda-fusion",
        action="store_true",
        help="Disable MSDA fusion op emission and export the GridSample path.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    output_path = str(Path(args.output_dir) / args.name)
    if args.disable_msda_fusion:
        os.environ["GROUNDING_DINO_DISABLE_MSDA_FUSION"] = "1"
    export_onnx(args.model_dir, output_path, opset=args.opset)


if __name__ == "__main__":
    main()
