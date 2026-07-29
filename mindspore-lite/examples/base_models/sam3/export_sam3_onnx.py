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
"""Export SAM3 image model to three ONNX models.

Splits the SAM3 image segmentation pipeline into three ONNX graphs:
  1. sam3_image_encoder     — ViT backbone + FPN neck (image → features)
  2. sam3_language_encoder  — CLIP text encoder (tokens → text features)
  3. sam3_decoder           — DETR encoder/decoder + segmentation head

This modular approach keeps each graph small enough for Ascend GE
compilation and avoids cross-module ONNX operator incompatibilities.

Usage:
    python export_sam3_onnx.py --checkpoint /path/to/sam3.1_multiplex.pt \
        --output-dir ./onnx
"""

import argparse
import os

import torch
from torch import nn

IMAGE_SIZE = 1008
CONTEXT_LENGTH = 32
OPSET_VERSION = 17


class ImageEncoderWrapper(nn.Module):
    """Wraps the SAM3 ViT + FPN vision backbone for ONNX export.

    Input:  image [1, 3, 1008, 1008] float32
    Output: backbone_fpn_0 [1, 256, 288, 288]
            backbone_fpn_1 [1, 256, 144, 144]
            backbone_fpn_2 [1, 256, 72, 72]
            vision_pos_enc_0 [1, 256, 288, 288]
            vision_pos_enc_1 [1, 256, 144, 144]
            vision_pos_enc_2 [1, 256, 72, 72]
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone

    def forward(self, image):
        """Run vision backbone forward pass."""
        out = self.backbone.forward_image(image)
        fpn = out["backbone_fpn"]
        pos = out["vision_pos_enc"]
        return fpn[0], fpn[1], fpn[2], pos[0], pos[1], pos[2]


class LanguageEncoderWrapper(nn.Module):
    """Wraps the SAM3 CLIP text encoder for ONNX export.

    Input:  text_tokens [1, 32] int64
    Output: language_features [32, 1, 256] float32
            language_mask [1, 32] bool
    """

    def __init__(self, model):
        super().__init__()
        self.text_encoder = model.backbone.language_backbone

    def forward(self, text_tokens):
        """Run CLIP text encoder from pre-tokenized input.

        Returns text features (seq-first) and inverted attention mask.
        """
        encoder = self.text_encoder.encoder
        text_attention_mask = (text_tokens != 0).bool()
        _, text_memory = encoder(text_tokens)
        text_attention_mask_inv = text_attention_mask.ne(1)
        text_memory = text_memory.transpose(0, 1)
        text_memory_resized = self.text_encoder.resizer(text_memory)
        return text_memory_resized, text_attention_mask_inv


class DecoderWrapper(nn.Module):
    """Wraps the SAM3 DETR encoder/decoder + segmentation head.

    Inputs:
        backbone_fpn_0 [1, 256, 288, 288]
        backbone_fpn_1 [1, 256, 144, 144]
        backbone_fpn_2 [1, 256, 72, 72]
        vision_pos_enc_0 [1, 256, 288, 288]
        vision_pos_enc_1 [1, 256, 144, 144]
        vision_pos_enc_2 [1, 256, 72, 72]
        language_features [32, 1, 256]
        language_mask [1, 32]

    Outputs:
        pred_logits [1, 200, 1]
        pred_boxes  [1, 200, 4]
        pred_masks  [1, 200, 288, 288]
        presence_logit [1, 1]
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.geometry_encoder = model.geometry_encoder
        self.transformer = model.transformer
        self.segmentation_head = model.segmentation_head
        self.dot_prod_scoring = model.dot_prod_scoring
        self.num_feature_levels = model.num_feature_levels

    def _encode_prompt(self, text_features, text_mask):
        """Encode text and geometry prompts into a combined prompt tensor."""
        txt_feats = text_features
        txt_masks = text_mask
        geo_feats = torch.zeros((0, txt_feats.shape[1], txt_feats.shape[2]),
                                dtype=torch.float32)
        geo_masks = torch.zeros((txt_masks.shape[0], 0), dtype=torch.bool)
        prompt = torch.cat([txt_feats, geo_feats], dim=0)
        prompt_mask = torch.cat([txt_masks, geo_masks], dim=1)
        return prompt, prompt_mask

    def _run_encoder(self, img_feats, img_pos_embeds, prompt, prompt_mask, vis_feat_sizes):
        """Run the transformer encoder to fuse image and text features."""
        prompt_pos_embed = torch.zeros_like(prompt)
        memory = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=None,
        )
        return memory

    def _run_decoder(self, memory, pos_embed, src_mask, prompt, prompt_mask, encoder_out):
        """Run the transformer decoder with 200 object queries."""
        decoder = self.transformer.decoder
        bs = memory.shape[1]
        query_embed = decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        hs, reference_boxes, dec_presence_out, _ = decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=src_mask,
            pos=pos_embed,
            reference_boxes=None,
            level_start_index=encoder_out["level_start_index"],
            spatial_shapes=encoder_out["spatial_shapes"],
            valid_ratios=encoder_out["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )
        hs = hs.transpose(1, 2)
        reference_boxes = reference_boxes.transpose(1, 2)
        if dec_presence_out is not None:
            dec_presence_out = dec_presence_out.transpose(1, 2)
        return hs, reference_boxes, dec_presence_out

    def _update_scores_and_boxes(self, hs, reference_boxes, prompt, prompt_mask):
        """Compute detection scores and bounding boxes from decoder hidden states."""
        from sam3.model.model_misc import inverse_sigmoid

        num_o2o = hs.size(2)
        outputs_class = self.dot_prod_scoring(hs, prompt, prompt_mask)
        box_head = self.transformer.decoder.bbox_embed
        anchor_box_offsets = box_head(hs)
        reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid()
        return outputs_class, outputs_coord

    def _run_segmentation(self, backbone_fpn, encoder_hidden_states, prompt, prompt_mask, hs):
        """Run the segmentation head to produce mask predictions."""
        from sam3.model.act_ckpt_utils import activation_ckpt_wrapper

        obj_queries = hs
        seg_outputs = activation_ckpt_wrapper(self.segmentation_head)(
            backbone_feats=backbone_fpn,
            obj_queries=obj_queries,
            image_ids=torch.tensor([0], dtype=torch.long),
            encoder_hidden_states=encoder_hidden_states,
            act_ckpt_enable=False,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        return seg_outputs

    def forward(self, fpn0, fpn1, fpn2, pos0, pos1, pos2,
                language_features, language_mask):
        """Run the full decoder pipeline."""
        backbone_fpn = [fpn0, fpn1, fpn2]
        vision_pos_enc = [pos0, pos1, pos2]

        enc_fpn = backbone_fpn[-self.num_feature_levels:]
        enc_pos = vision_pos_enc[-self.num_feature_levels:]
        vis_feat_sizes = [x.shape[-2:] for x in enc_pos]

        img_ids = torch.tensor([0], dtype=torch.long)
        img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in enc_fpn]
        img_pos_embeds = [x[img_ids].flatten(2).permute(2, 0, 1) for x in enc_pos]

        prompt, prompt_mask = self._encode_prompt(
            language_features, language_mask, img_feats, img_pos_embeds, vis_feat_sizes
        )

        memory_out = self._run_encoder(
            img_feats, img_pos_embeds, prompt, prompt_mask, vis_feat_sizes
        )

        encoder_hidden_states = memory_out["memory"]
        encoder_out = {
            "level_start_index": memory_out["level_start_index"],
            "spatial_shapes": memory_out["spatial_shapes"],
            "valid_ratios": memory_out["valid_ratios"],
        }

        hs, reference_boxes, dec_presence_out = self._run_decoder(
            encoder_hidden_states, memory_out["pos_embed"],
            memory_out["padding_mask"], prompt, prompt_mask, encoder_out
        )

        outputs_class, outputs_coord = self._update_scores_and_boxes(
            hs, reference_boxes, prompt, prompt_mask, dec_presence_out
        )

        seg_outputs = self._run_segmentation(
            backbone_fpn, encoder_hidden_states, prompt, prompt_mask, hs
        )

        pred_logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]
        pred_masks = seg_outputs["pred_masks"]
        presence_logit = dec_presence_out[-1] if dec_presence_out is not None else \
            torch.zeros((1, 1), dtype=torch.float32)

        return pred_logits, pred_boxes, pred_masks, presence_logit


def build_model(checkpoint_path):
    """Build the SAM3 image model from a checkpoint."""
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        device="cpu",
        eval_mode=True,
    )
    return model


def _export_module(wrapper, dummy_inputs, output_path, input_names, output_names,
                   opset_version=OPSET_VERSION):
    """Export a single wrapper module to ONNX.

    Args:
        wrapper: nn.Module wrapper.
        dummy_inputs: Tuple of dummy input tensors.
        output_path: Path to save the ONNX file.
        input_names: List of input names.
        output_names: List of output names.
        opset_version: ONNX opset version.
    """
    wrapper.eval()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with torch.no_grad():
        outputs = wrapper(*dummy_inputs)
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    print(f"  Output shapes: {[o.shape for o in outputs]}")

    print(f"  Exporting to {output_path} ...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_inputs, strict=False)
        torch.onnx.utils.export(
            traced,
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
        data_size = os.path.getsize(data_path) / (1024 * 1024)
        file_size += data_size
    print(f"  Done ({file_size:.1f} MB)")


def export_image_encoder(model, output_dir):
    """Export the ViT + FPN vision backbone to ONNX."""
    print("\n=== Exporting sam3_image_encoder ===")
    trunk = model.backbone.vision_backbone.trunk
    feat_size = IMAGE_SIZE // trunk.patch_embed.proj.kernel_size[0]
    trunk.precompute_pos_embed(feat_size, feat_size)
    wrapper = ImageEncoderWrapper(model)
    dummy_image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    output_path = os.path.join(output_dir, "sam3_image_encoder.onnx")
    _export_module(
        wrapper, (dummy_image,), output_path,
        input_names=["image"],
        output_names=["backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
                       "vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2"],
    )


def export_language_encoder(model, output_dir):
    """Export the CLIP text encoder to ONNX."""
    print("\n=== Exporting sam3_language_encoder ===")
    wrapper = LanguageEncoderWrapper(model)
    dummy_text = torch.zeros(1, CONTEXT_LENGTH, dtype=torch.long)
    dummy_text[0, 0] = 49406
    dummy_text[0, 1] = 320
    dummy_text[0, 2] = 49407
    output_path = os.path.join(output_dir, "sam3_language_encoder.onnx")
    _export_module(
        wrapper, (dummy_text,), output_path,
        input_names=["text_tokens"],
        output_names=["language_features", "language_mask"],
    )


def export_decoder(model, output_dir):
    """Export the DETR decoder + segmentation head to ONNX."""
    print("\n=== Exporting sam3_decoder ===")
    wrapper = DecoderWrapper(model)

    dummy_image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    with torch.no_grad():
        backbone_out = model.backbone.forward_image(dummy_image)
    fpn = backbone_out["backbone_fpn"]
    pos = backbone_out["vision_pos_enc"]

    dummy_lang = torch.randn(32, 1, 256, dtype=torch.float32)
    dummy_mask = torch.zeros(1, 32, dtype=torch.bool)
    dummy_mask[0, :3] = True

    dummy_inputs = (fpn[0], fpn[1], fpn[2], pos[0], pos[1], pos[2],
                    dummy_lang, dummy_mask)
    output_path = os.path.join(output_dir, "sam3_decoder.onnx")
    _export_module(
        wrapper, dummy_inputs, output_path,
        input_names=["backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
                       "vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2",
                       "language_features", "language_mask"],
        output_names=["pred_logits", "pred_boxes", "pred_masks", "presence_logit"],
    )


def main():
    """Parse arguments and export all three ONNX models."""
    parser = argparse.ArgumentParser(description="Export SAM3 to three ONNX models")
    parser.add_argument(
        "--checkpoint", type=str,
        default="/home/BYD/SAM3/weight/sam3.1_multiplex.pt",
        help="Path to SAM3 checkpoint",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./onnx",
        help="Output directory for ONNX files",
    )
    parser.add_argument("--opset", type=int, default=OPSET_VERSION)
    args = parser.parse_args()

    print("Building SAM3 model...")
    model = build_model(args.checkpoint)

    export_image_encoder(model, args.output_dir)
    export_language_encoder(model, args.output_dir)
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
