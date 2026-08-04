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
"""Export SAM2 (sam2.1-hiera-base-plus) image predictor to two ONNX models.

The SAM2 image segmentation pipeline is split into two ONNX graphs so that
each sub-graph stays small enough for Ascend GE compilation and avoids the
video-path memory attention (which relies on complex-number RoPE ops that
ONNX does not support):

  1. sam2_encoder -- Hiera trunk + FPN neck + conv_s0/conv_s1 projections
     (image -> image_embed + high-res feature maps)
  2. sam2_decoder -- SAM prompt encoder + mask decoder (image features +
     point prompts -> mask logits + IoU predictions)

Usage:
    python export_sam2_onnx.py --ckpt /path/to/sam2.1_hiera_base_plus.pt \
        --config configs/sam2.1/sam2.1_hiera_b+.yaml --output-dir ./onnx
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch import nn

IMAGE_SIZE = 1024
OPSET_VERSION = 17


def _patch_sdpa():
    """Replace F.scaled_dot_product_attention with manual matmul attention.

    PyTorch's ONNX export of SDPA emits ``If`` control-flow nodes (choosing
    between efficient / fallback attention paths) that the MindSpore Lite
    Ascend converter cannot handle. Manual matmul + softmax + matmul is
    numerically equivalent and exports to a clean, branch-free graph.
    """
    import sam2.modeling.backbones.hieradet as hd
    import sam2.modeling.sam.transformer as tr

    def msa_forward(self, x):
        batch, h, w, _ = x.shape
        qkv = self.qkv(x).reshape(batch, h * w, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, 2)
        if self.q_pool:
            q = hd.do_pool(q.reshape(batch, h, w, -1), self.q_pool)
            h, w = q.shape[1:3]
            q = q.reshape(batch, h * w, self.num_heads, -1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, h, w, -1)
        return self.proj(out)

    def attn_forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        b, n, c = q.shape
        q = q.reshape(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)
        b, n, c = k.shape
        k = k.reshape(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)
        b, n, c = v.shape
        v = v.reshape(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)
        scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        b, n_heads, n_tokens, c_per_head = out.shape
        out = out.transpose(1, 2).reshape(b, n_tokens, n_heads * c_per_head)
        return self.out_proj(out)

    hd.MultiScaleAttention.forward = msa_forward
    tr.Attention.forward = attn_forward

    # Hiera._get_pos_embed uses window_embed.tile([shape-derived dims]) whose
    # ONNX export emits If guards on the repeat factors. Precompute the pos
    # embed for the fixed 1024 input (patch grid 256x256) so it becomes a
    # constant in the exported graph.
    def _pos_embed_const(self, hw):
        if not hasattr(self, "cached_pos_embed"):
            h, w = int(hw[0]), int(hw[1])
            window_embed = self.pos_embed_window
            pos_embed = F.interpolate(
                self.pos_embed, size=(h, w), mode="bicubic"
            )
            ws = window_embed.shape[-1]
            window_tiled = window_embed.repeat(1, 1, h // ws, w // ws)
            pos_embed = (pos_embed + window_tiled).permute(0, 2, 3, 1)
            self.cached_pos_embed = pos_embed
        return self.cached_pos_embed

    hd.Hiera._get_pos_embed = _pos_embed_const  # pylint: disable=protected-access


_patch_sdpa()


class EncoderWrapper(nn.Module):
    """Wraps the SAM2 Hiera backbone + FPN neck for ONNX export.

    Input:  image [1, 3, 1024, 1024] float32
    Output: image_embed   [1, 256, 64, 64]   (fpn level-2 feat + no_mem_embed)
            high_res_s0   [1, 32, 256, 256]  (conv_s0 on fpn level-0)
            high_res_s1   [1, 64, 128, 128]  (conv_s1 on fpn level-1)
    """

    def __init__(self, sam_model):
        super().__init__()
        self.image_encoder = sam_model.image_encoder
        self.conv_s0 = sam_model.sam_mask_decoder.conv_s0
        self.conv_s1 = sam_model.sam_mask_decoder.conv_s1
        self.no_mem_embed = sam_model.no_mem_embed

    def forward(self, image):
        """Run the vision backbone and project high-resolution feature maps."""
        backbone_out = self.image_encoder(image)
        fpn = backbone_out["backbone_fpn"]
        # Project level-0 and level-1 features for the SAM mask decoder.
        high_res_s0 = self.conv_s0(fpn[0])
        high_res_s1 = self.conv_s1(fpn[1])
        # Add the no-memory embedding to the top-level (level-2) feature map,
        # matching SAM2ImagePredictor.set_image when directly_add_no_mem_embed.
        image_embed = fpn[2] + self.no_mem_embed.view(1, -1, 1, 1)
        return image_embed, high_res_s0, high_res_s1


class DecoderWrapper(nn.Module):
    """Wraps the SAM prompt encoder + mask decoder for ONNX export.

    Inputs:
        image_embed    [1, 256, 64, 64]
        high_res_s0    [1, 32, 256, 256]
        high_res_s1    [1, 64, 128, 128]
        point_coords   [B, P, 2]  absolute pixel coords in the 1024x1024 frame
        point_labels   [B, P]     1=foreground, 0=background, -1=padding

    Outputs:
        low_res_masks  [B, 3, 256, 256]  mask logits (1/4 stride, multimask)
        iou_predictions [B, 3]           estimated IoU of each mask
    """

    def __init__(self, sam_model):
        super().__init__()
        self.sam_prompt_encoder = sam_model.sam_prompt_encoder
        self.sam_mask_decoder = sam_model.sam_mask_decoder

    def forward(self, image_embed, high_res_s0, high_res_s1,
                point_coords, point_labels):
        """Run the prompt encoder and mask decoder for point prompts."""
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        high_res_features = [high_res_s0, high_res_s1]
        low_res_masks, iou_predictions, _, _ = self.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        return low_res_masks, iou_predictions


def build_model(ckpt_path, config_file, device="cpu"):
    """Build the SAM2Base model from a checkpoint and config."""
    from sam2.build_sam import build_sam2

    model = build_sam2(
        config_file=config_file,
        ckpt_path=ckpt_path,
        device=device,
        mode="eval",
        apply_postprocessing=True,
    )
    model.eval()
    return model


def _export_module(wrapper, dummy_inputs, output_path, input_names,
                   output_names, opset_version=OPSET_VERSION):
    """Trace and export a single wrapper module to ONNX."""
    wrapper.eval()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with torch.no_grad():
        outputs = wrapper(*dummy_inputs)
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    print(f"  Output shapes: {[tuple(o.shape) for o in outputs]}")

    print(f"  Exporting to {output_path} ...")
    with torch.no_grad():
        torch.onnx.utils.export(
            wrapper,
            dummy_inputs,
            output_path,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,
            do_constant_folding=True,
        )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    data_path = output_path + ".data"
    if os.path.exists(data_path):
        file_size += os.path.getsize(data_path) / (1024 * 1024)
    print(f"  Done ({file_size:.1f} MB)")


def export_encoder(model, output_dir):
    """Export the Hiera + FPN vision backbone to ONNX."""
    print("\n=== Exporting sam2_encoder ===")
    wrapper = EncoderWrapper(model)
    dummy_image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    output_path = os.path.join(output_dir, "sam2_encoder.onnx")
    _export_module(
        wrapper, (dummy_image,), output_path,
        input_names=["image"],
        output_names=["image_embed", "high_res_s0", "high_res_s1"],
    )


def export_decoder(model, output_dir):
    """Export the SAM prompt encoder + mask decoder to ONNX."""
    print("\n=== Exporting sam2_decoder ===")
    wrapper = DecoderWrapper(model)

    dummy_image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    with torch.no_grad():
        enc = EncoderWrapper(model)
        image_embed, high_res_s0, high_res_s1 = enc(dummy_image)

    point_coords = torch.tensor([[[512.0, 512.0]]], dtype=torch.float32)
    point_labels = torch.tensor([[1]], dtype=torch.int32)

    dummy_inputs = (image_embed, high_res_s0, high_res_s1,
                    point_coords, point_labels)
    output_path = os.path.join(output_dir, "sam2_decoder.onnx")
    _export_module(
        wrapper, dummy_inputs, output_path,
        input_names=["image_embed", "high_res_s0", "high_res_s1",
                     "point_coords", "point_labels"],
        output_names=["low_res_masks", "iou_predictions"],
    )


def main():
    """Parse arguments and export both ONNX models."""
    parser = argparse.ArgumentParser(description="Export SAM2 to two ONNX models")
    parser.add_argument(
        "--ckpt", type=str,
        default="/path/to/sam2.1_hiera_base_plus.pt",
        help="Path to SAM2 checkpoint",
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM2 hydra config name",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./onnx",
        help="Output directory for ONNX files",
    )
    parser.add_argument("--opset", type=int, default=OPSET_VERSION)
    args = parser.parse_args()

    print("Building SAM2 model...")
    model = build_model(args.ckpt, args.config)

    export_encoder(model, args.output_dir)
    export_decoder(model, args.output_dir)

    print("\n=== All ONNX exports complete ===")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".onnx"):
            path = os.path.join(args.output_dir, f)
            size = os.path.getsize(path) / (1024 * 1024)
            data_path = path + ".data"
            if os.path.exists(data_path):
                size += os.path.getsize(data_path) / (1024 * 1024)
            print(f"  {f}: {size:.1f} MB")


if __name__ == "__main__":
    main()
