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
"""Export VGGT (Visual Geometry Grounded Transformer) model to ONNX format.

VGGT is a feed-forward 3D scene inference model that takes one or more images
and predicts camera poses, depth maps, and 3D point maps.

This script exports the full VGGT model (aggregator + camera head + depth head +
point head) as a single ONNX file. The track head is disabled since it requires
query points as an additional input.
"""

import argparse
import os
import sys
from pathlib import Path

import onnx
import torch
from torch import nn
from onnx import TensorProto, helper, numpy_helper
from vggt.layers.rope import PositionGetter
from vggt.models.vggt import VGGT
import vggt.heads.utils as _head_utils

VGGT_REPO_PATH = os.environ.get("VGGT_REPO_PATH", "/VGGT/vggt")
sys.path.insert(0, VGGT_REPO_PATH)


def _patched_position_getter_call(self, batch_size, height, width, device):
    """Patched PositionGetter.__call__ using meshgrid instead of cartesian_prod.

    torch.cartesian_prod is not supported by the ONNX exporter, so we replace it
    with torch.meshgrid + torch.stack which produce the same result.
    """
    if (height, width) not in self.position_cache:
        y_coords = torch.arange(height, device=device)
        x_coords = torch.arange(width, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        positions = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
        self.position_cache[height, width] = positions

    cached_positions = self.position_cache[height, width]
    return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


PositionGetter.__call__ = _patched_position_getter_call

# Monkey-patch torch.expm1 to exp(x) - 1 for ONNX compatibility
# expm1 is used in inverse_log_transform (point head activation) and is not
# supported by the ONNX exporter. exp(x) - 1 is mathematically equivalent.
_original_expm1 = torch.expm1
torch.expm1 = lambda x: torch.exp(x) - 1.0

# Monkey-patch make_sincos_pos_embed to use float32 instead of double
# The original uses torch.double for omega, which causes mixed-type Einsum
# errors in ONNX. float32 is sufficient for positional embeddings.

_original_make_sincos = _head_utils.make_sincos_pos_embed


def _patched_make_sincos_pos_embed(embed_dim, pos, omega_0=100):
    """Patched version using float32 to avoid ONNX mixed-type Einsum."""
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0 ** omega

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb.float()


_head_utils.make_sincos_pos_embed = _patched_make_sincos_pos_embed


class VGGTForONNX(nn.Module):
    """Wrapper module for VGGT ONNX export.

    Exposes only the images input and returns a tuple of five output tensors:
    (pose_enc, depth, depth_conf, world_points, world_points_conf).

    The track head is disabled to keep the interface simple (tracking requires
    query_points as an additional input).
    """

    def __init__(self, model: VGGT):
        """Initialize the wrapper with a VGGT model.

        Args:
            model: A VGGT model instance with camera, depth, and point heads.
        """
        super().__init__()
        self.aggregator = model.aggregator
        self.camera_head = model.camera_head
        self.depth_head = model.depth_head
        self.point_head = model.point_head

    def forward(self, images: torch.Tensor):
        """Run forward pass and return predictions as a tuple.

        Args:
            images: Input images [B, S, 3, H, W] in [0, 1] range.

        Returns:
            Tuple of (pose_enc, depth, depth_conf, world_points, world_points_conf).
        """
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        pose_enc_list = self.camera_head(aggregated_tokens_list)
        pose_enc = pose_enc_list[-1]

        depth, depth_conf = self.depth_head(
            aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
        )
        world_points, world_points_conf = self.point_head(
            aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
        )

        return pose_enc, depth, depth_conf, world_points, world_points_conf


def disable_fused_attn(model: VGGT):
    """Set fused_attn=False on all attention layers for ONNX compatibility.

    F.scaled_dot_product_attention may not be fully supported by the ONNX exporter
    or the MindSpore Lite converter. Using the manual attention implementation
    (matmul + softmax + matmul) ensures maximum compatibility.

    Also disables interpolate_antialias in the DINOv2 patch embedding, since
    aten::_upsample_bicubic2d_aa is not supported by the ONNX exporter.

    Args:
        model: The VGGT model instance.
    """
    for block in model.aggregator.frame_blocks:
        block.attn.fused_attn = False
    for block in model.aggregator.global_blocks:
        block.attn.fused_attn = False
    if model.camera_head is not None:
        for block in model.camera_head.trunk:
            if hasattr(block, "attn"):
                block.attn.fused_attn = False
    if hasattr(model.aggregator.patch_embed, "interpolate_antialias"):
        model.aggregator.patch_embed.interpolate_antialias = False


def load_model(checkpoint_path: str) -> VGGT:
    """Load VGGT model from checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.

    Returns:
        A VGGT model in eval mode with track head disabled.
    """
    model = VGGT(enable_track=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export VGGT model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/VGGT/model/vggt_1B_commercial.pt",
        help="Path to VGGT checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/vggt_1b.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=2,
        help="Number of input frames (sequence length S)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=518,
        help="Image size (height = width, must be divisible by 14)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--dynamic-frames",
        action="store_true",
        help="Export with dynamic sequence length (S) axis",
    )
    parser.add_argument(
        "--replace-gather",
        action="store_true",
        default=True,
        help="Post-process ONNX to replace scalar-index Gather with Slice+Squeeze "
        "(required for Ascend GatherV2 compatibility)",
    )
    parser.add_argument(
        "--no-replace-gather",
        action="store_false",
        dest="replace_gather",
        help="Skip Gather replacement post-processing",
    )
    return parser.parse_args()


def replace_gather_with_slice(onnx_path):
    """Replace scalar-index Gather ops with Slice+Squeeze for Ascend compatibility.

    Ascend 300I Duo GatherV2 kernel fails (Aicore trap) when gathering from data
    tensors with scalar int indices. Slice+Squeeze is mathematically equivalent
    and runs reliably on Ascend.

    Args:
        onnx_path: Path to the ONNX file (modified in-place).
    """
    m = onnx.load(str(onnx_path), load_external_data=True)
    node_map = {n.output[0]: n for n in m.graph.node if n.output}
    new_nodes = []
    replaced = 0

    for node in m.graph.node:
        if node.op_type != "Gather":
            new_nodes.append(node)
            continue

        idx_name = node.input[1] if len(node.input) > 1 else ""
        data_name = node.input[0] if node.input else ""
        const_node = node_map.get(idx_name)
        if const_node is None or const_node.op_type != "Constant":
            new_nodes.append(node)
            continue

        producer = node_map.get(data_name)
        if (producer and producer.op_type == "Shape") or "Shape" in data_name:
            new_nodes.append(node)
            continue

        idx_val = None
        for attr in const_node.attribute:
            if attr.name == "value":
                idx_val = int(numpy_helper.to_array(attr.t).item())
        if idx_val is None:
            new_nodes.append(node)
            continue

        axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i

        start_val = idx_val
        end_val = idx_val + 1 if idx_val >= 0 else 2 ** 31 - 1

        starts_name = node.name + "_starts"
        ends_name = node.name + "_ends"
        axes_name = node.name + "_axes"
        sq_axes_name = node.name + "_sq_axes"
        m.graph.initializer.extend([
            helper.make_tensor(starts_name, TensorProto.INT64, [1], [start_val]),
            helper.make_tensor(ends_name, TensorProto.INT64, [1], [end_val]),
            helper.make_tensor(axes_name, TensorProto.INT64, [1], [axis]),
            helper.make_tensor(sq_axes_name, TensorProto.INT64, [1], [axis]),
        ])

        slice_out = node.output[0] + "_slice"
        new_nodes.append(helper.make_node(
            "Slice", [data_name, starts_name, ends_name, axes_name], [slice_out],
            name=node.name + "_slice"))
        new_nodes.append(helper.make_node(
            "Squeeze", [slice_out, sq_axes_name], list(node.output),
            name=node.name + "_squeeze"))
        replaced += 1

    del m.graph.node[:]
    m.graph.node.extend(new_nodes)

    data_path = str(onnx_path) + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)
    onnx.save_model(
        m, str(onnx_path), save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(str(onnx_path)) + ".data",
        size_threshold=1024, convert_attribute=True)
    print(f"  Replaced {replaced} Gather nodes with Slice+Squeeze")


def main():
    """Main export function."""
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== VGGT ONNX Export ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output: {args.output}")
    print(f"  Num frames: {args.num_frames}")
    print(f"  Image size: {args.img_size}x{args.img_size}")
    print(f"  Opset: {args.opset}")
    print()

    model = load_model(args.checkpoint)
    disable_fused_attn(model)

    wrapper = VGGTForONNX(model)
    wrapper.eval()

    images = torch.randn(1, args.num_frames, 3, args.img_size, args.img_size, dtype=torch.float32)

    print("Running trace forward to verify model...")
    with torch.no_grad():
        outputs = wrapper(images)
    output_names = ["pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"]
    for name, out in zip(output_names, outputs):
        print(f"  {name}: {tuple(out.shape)}, dtype={out.dtype}")
    print()

    dynamic_axes = None
    if args.dynamic_frames:
        dynamic_axes = {
            "images": {1: "S"},
            "pose_enc": {1: "S"},
            "depth": {1: "S"},
            "depth_conf": {1: "S"},
            "world_points": {1: "S"},
            "world_points_conf": {1: "S"},
        }

    print("Exporting ONNX model...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            images,
            str(output_path),
            opset_version=args.opset,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_params=True,
            do_constant_folding=False,
        )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"ONNX export complete: {args.output} ({file_size_mb:.1f} MB)")

    if args.replace_gather:
        print("Post-processing: replacing scalar-index Gather ops...")
        replace_gather_with_slice(output_path)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        total_mb = file_size_mb
        data_path = output_path.with_suffix(".onnx.data")
        if data_path.exists():
            total_mb += data_path.stat().st_size / (1024 * 1024)
        print(f"Final ONNX: {args.output} ({total_mb:.1f} MB total)")


if __name__ == "__main__":
    main()
