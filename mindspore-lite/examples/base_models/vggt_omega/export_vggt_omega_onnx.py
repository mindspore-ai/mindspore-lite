# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Export VGGT-Omega to a fixed-shape fp32 ONNX for MindSpore Lite deployment.

VGGT-Omega (1B) is a feed-forward vision model that predicts camera pose and
dense depth from a short image sequence. It is not autoregressive, so a single
ONNX is exported (no prefill/decode split).

The released model wraps the forward pass in ``torch.autocast(device_type="cuda")``
which cannot run on a CUDA-less build. This script bypasses autocast by calling
the aggregator and heads directly in fp32, which is also the recommended export
dtype for MindSpore Lite conversion.

Two tracing artifacts that produce ONNX ``If`` control-flow nodes (rejected by
``converter_lite``) are removed:
  1. ``RopePositionEmbedding.forward`` is precomputed for the fixed patch grid
     and served as constants (``torch.arange`` emits range checks otherwise).
  2. ``custom_interpolate`` equal-size early return is replaced by an unconditional
     ``F.interpolate`` (the shape comparison becomes a traced ``Equal``/``If``).

These changes are math-identical for the exported fixed shape.
"""

import argparse
import os
import sys
import time

import onnx
import torch
import torch as nn
import torch.nn.functional as F
from torch import Tensor
from vggt_omega.models import VGGTOmega
from vggt_omega.models.layers.block import SelfAttentionBlock
from vggt_omega.models.layers.rope_position_encoding import RopePositionEmbedding
import vggt_omega.models.heads.dense_head as dense_head_mod
import vggt_omega.models.heads.utils as head_utils

_DEFAULT_UPSTREAM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vggt-omega")
if _DEFAULT_UPSTREAM not in sys.path:
    sys.path.insert(0, _DEFAULT_UPSTREAM)


class VGGTOmegaExport(nn.Module):
    """ONNX-friendly wrapper around VGGT-Omega without CUDA autocast."""

    def __init__(self, model: VGGTOmega) -> None:
        super().__init__()
        self.aggregator = model.aggregator
        self.camera_head = model.camera_head
        self.dense_head = model.dense_head

    def forward(self, images: Tensor):
        """Run aggregator + camera/depth heads and return core predictions.

        Args:
            images: ``[B, S, 3, H, W]`` RGB images in ``[0, 1]`` range. The
                aggregator applies ResNet mean/std normalization internally.

        Returns:
            ``(pose_enc, depth, depth_conf)`` where ``pose_enc`` is ``[B, S, 9]``,
            ``depth`` is ``[B, S, H, W, 1]`` and ``depth_conf`` is ``[B, S, H, W]``.
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)
        aggregated_tokens_list, patch_token_start = self.aggregator(images)
        pose_enc = self.camera_head(aggregated_tokens_list, patch_token_start=patch_token_start)
        depth, depth_conf = self.dense_head(
            aggregated_tokens_list, images=images, patch_token_start=patch_token_start
        )
        return pose_enc, depth, depth_conf


def _apply_export_patches() -> None:
    """Patch modules to remove ONNX control flow and list-op overhead."""
    _orig_block_forward = SelfAttentionBlock.forward

    def _block_forward(self, x_or_x_list, rope_or_rope_list=None):
        if isinstance(x_or_x_list, Tensor):
            return self._forward(x_or_x_list, rope=rope_or_rope_list)
        return _orig_block_forward(self, x_or_x_list, rope_or_rope_list)

    SelfAttentionBlock.forward = _block_forward

    def _simple_interpolate(x, size=None, scale_factor=None, mode="bilinear", align_corners=True):
        if size is None:
            size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
        return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)

    dense_head_mod.custom_interpolate = _simple_interpolate

    # ``make_sincos_pos_embed`` mixes float32 coords with a float64 ``omega`` arange,
    # which yields an ONNX Einsum bound to two dtypes (rejected by ORT/converter).
    # Use float32 throughout; the embedding is cast back to float32 anyway.
    def _sincos_pos_embed(embed_dim, pos, omega_0=100):
        half = embed_dim // 2
        omega = torch.arange(half, dtype=torch.float32, device=pos.device) / (half)
        omega = 1.0 / (omega_0 ** omega)
        out = torch.einsum("m,d->md", pos.reshape(-1), omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()

    head_utils.make_sincos_pos_embed = _sincos_pos_embed


def _install_rope_cache(model: nn.Module, patch_h: int, patch_w: int) -> None:
    """Precompute RoPE sin/cos so tracing skips arange/meshgrid/If nodes."""
    _orig_rope_forward = RopePositionEmbedding.forward

    for _, module in model.named_modules():
        if isinstance(module, RopePositionEmbedding):
            with torch.no_grad():
                sin, cos = _orig_rope_forward(module, H=patch_h, W=patch_w)
            module._rope_sin = sin.detach().contiguous()
            module._rope_cos = cos.detach().contiguous()

    def _rope_forward(self, **_kwargs):
        """Return precomputed RoPE; H/W are ignored (fixed shape export)."""
        return self._rope_sin, self._rope_cos

    RopePositionEmbedding.forward = _rope_forward


def load_model(checkpoint: str) -> VGGTOmega:
    """Load the released VGGT-Omega checkpoint in fp32 eval mode."""
    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model


def export_onnx(
    checkpoint: str,
    output_dir: str,
    num_frames: int,
    img_h: int,
    img_w: int,
    opset: int,
) -> str:
    """Export the VGGT-Omega ONNX with a fixed input shape."""
    patch_size = 16
    if img_h % patch_size or img_w % patch_size:
        raise ValueError(f"Image size must be a multiple of {patch_size}, got {img_h}x{img_w}")

    os.makedirs(output_dir, exist_ok=True)
    model = load_model(checkpoint)

    patch_h, patch_w = img_h // patch_size, img_w // patch_size
    _install_rope_cache(model, patch_h, patch_w)
    _apply_export_patches()

    wrapper = VGGTOmegaExport(model).eval()
    dummy = torch.rand(1, num_frames, 3, img_h, img_w, dtype=torch.float32)

    t0 = time.time()
    with torch.inference_mode():
        with torch.no_grad():
            torch.onnx.utils.export(
                wrapper,
                dummy,
                os.path.join(output_dir, "vggt_omega.onnx"),
                opset_version=opset,
                input_names=["images"],
                output_names=["pose_enc", "depth", "depth_conf"],
                do_constant_folding=True,
            )
    print(f"[export] ONNX written in {time.time() - t0:.1f}s")

    onnx_path = os.path.join(output_dir, "vggt_omega.onnx")
    _consolidate_external_data(onnx_path)
    _report_onnx(onnx_path)
    return onnx_path


def _consolidate_external_data(onnx_path: str) -> None:
    """Fold the many per-initializer external data files into one ``.onnx.data``.

    ``torch.onnx`` emits one external file per large initializer, which clutters
    the output directory and breaks when copied without the siblings. Reloading
    the model (with external data) and re-saving with ``all_tensors_to_one_file``
    yields a clean ``vggt_omega.onnx`` + ``vggt_omega.onnx.data`` pair.
    """
    output_dir = os.path.dirname(os.path.abspath(onnx_path))
    model = onnx.load(onnx_path, load_external_data=True)
    data_name = "vggt_omega.onnx.data"
    for fname in os.listdir(output_dir):
        full = os.path.join(output_dir, fname)
        if not os.path.isfile(full):
            continue
        if fname in (os.path.basename(onnx_path), data_name):
            continue
        os.remove(full)
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=True,
    )
    print(f"[export] consolidated external data -> {data_name}")


def _report_onnx(onnx_path: str) -> None:
    """Print ONNX I/O, opset, size and check for unsupported If nodes."""
    model = onnx.load(onnx_path)
    from collections import Counter

    ops = Counter(n.op_type for n in model.graph.node)
    num_if = ops.get("If", 0)
    size_mb = os.path.getsize(onnx_path) / 1e6

    opset = model.opset_import[0].version
    print(f"[export] nodes={len(model.graph.node)} size={size_mb:.1f}MB opset={opset}")
    print(f"[export] If nodes={num_if} (must be 0 for converter_lite)")
    print("[export] inputs:")
    for inp in model.graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {dims}")
    print("[export] outputs:")
    for out in model.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {dims}")
    top = dict(sorted(ops.items(), key=lambda x: -x[1])[:12])
    print(f"[export] top ops: {top}")

    if num_if != 0:
        raise RuntimeError(f"ONNX still contains {num_if} If nodes; conversion would fail.")


def main() -> None:
    """Parse arguments and run the ONNX export."""
    parser = argparse.ArgumentParser(description="Export VGGT-Omega to ONNX")
    parser.add_argument("--checkpoint", default="/VGGT-omega/model/vggt_omega_1b_512.pt")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--img-h", type=int, default=512)
    parser.add_argument("--img-w", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    print(f"[export] shape: 1 x {args.num_frames} x 3 x {args.img_h} x {args.img_w}")
    export_onnx(
        args.checkpoint,
        args.output_dir,
        args.num_frames,
        args.img_h,
        args.img_w,
        args.opset,
    )


if __name__ == "__main__":
    main()
