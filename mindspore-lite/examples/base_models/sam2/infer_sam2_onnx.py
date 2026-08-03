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
"""SAM2 ONNX Runtime inference and precision alignment script.

Loads the exported `sam2_encoder.onnx` and `sam2_decoder.onnx` models, runs
the full image segmentation pipeline on a sample image with a point prompt,
and (optionally) compares the output against the original PyTorch SAM2 model
to verify cosine similarity > 0.99.

Usage:
    python infer_sam2_onnx.py \
        --encoder ./onnx/sam2_encoder.onnx \
        --decoder ./onnx/sam2_decoder.onnx \
        --image ./truck.jpg --point 500 375
"""

import argparse
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_path):
    """Load an image and resize/normalize it for SAM2.

    Returns (input_tensor [1,3,1024,1024] float32, orig_hw (h, w)).
    """
    img = Image.open(image_path).convert("RGB")
    orig_hw = (img.size[1], img.size[0])
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[None, ...]  # 1xCxHxW
    return np.ascontiguousarray(arr, dtype=np.float32), orig_hw


def transform_point(point_xy, orig_hw):
    """Map a point from original image pixels to the 1024x1024 frame."""
    x, y = float(point_xy[0]), float(point_xy[1])
    h, w = orig_hw
    coords = np.array([[[x / w * IMAGE_SIZE, y / h * IMAGE_SIZE]]],
                      dtype=np.float32)
    labels = np.array([[1]], dtype=np.int32)
    return coords, labels


def build_session(onnx_path, providers=None):
    """Create an ONNX Runtime inference session."""
    if providers is None:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    return sess


def run_encoder(session, image):
    """Run the encoder ONNX model."""
    feeds = {session.get_inputs()[0].name: image}
    outs = session.run(None, feeds)
    names = [o.name for o in session.get_outputs()]
    return dict(zip(names, outs))


def run_decoder(session, image_embed, high_res_s0, high_res_s1,
                point_coords, point_labels):
    """Run the decoder ONNX model."""
    inputs = {
        "image_embed": image_embed,
        "high_res_s0": high_res_s0,
        "high_res_s1": high_res_s1,
        "point_coords": point_coords,
        "point_labels": point_labels,
    }
    outs = session.run(None, inputs)
    names = [o.name for o in session.get_outputs()]
    return dict(zip(names, outs))


def postprocess_masks(low_res_masks, iou_predictions, orig_hw):
    """Pick the best mask, upsample to original size, and threshold."""
    best_idx = int(np.argmax(iou_predictions[0]))
    best_logit = low_res_masks[0, best_idx][None, None]  # 1x1x256x256
    best_logit = np.clip(best_logit, -32.0, 32.0)
    logit_img = best_logit[0, 0].astype(np.float32)
    logit_pil = Image.fromarray(logit_img, mode="F")
    logit_pil = logit_pil.resize((orig_hw[1], orig_hw[0]), Image.BILINEAR)
    mask = np.asarray(logit_pil, dtype=np.float32) > 0.0
    return mask, float(iou_predictions[0, best_idx])


def cos_sim(a, b):
    """Cosine similarity between two flattened arrays."""
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


def torch_reference(ckpt, config, image_path, point_xy):
    """Run the original PyTorch SAM2ImagePredictor for precision comparison."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(config_file=config, ckpt_path=ckpt, device="cpu",
                       mode="eval", apply_postprocessing=True)
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(np.array(Image.open(image_path).convert("RGB")))
    _, ious, low_res = predictor.predict(
        point_coords=np.array([point_xy]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    return low_res, ious


def save_mask_overlay(image_path, mask, out_path):
    """Save a simple mask overlay image (green highlight on the object)."""
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    green = np.zeros_like(arr)
    green[..., 1] = 255
    overlay = np.where(mask[..., None], (arr // 2 + green // 2), arr)
    Image.fromarray(overlay.astype(np.uint8)).save(out_path)


def main():
    """Run ONNX inference and (optionally) precision alignment."""
    parser = argparse.ArgumentParser(description="SAM2 ONNX Runtime inference")
    parser.add_argument("--encoder", required=True, help="Encoder ONNX path")
    parser.add_argument("--decoder", required=True, help="Decoder ONNX path")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--point", nargs=2, type=float, default=[500, 375],
                        metavar=("X", "Y"), help="Foreground point (pixels)")
    parser.add_argument("--ckpt", default=None, help="PyTorch ckpt for check")
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--output", default=None, help="Path to save mask overlay")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    image, orig_hw = preprocess(args.image)
    point_coords, point_labels = transform_point(args.point, orig_hw)

    enc = build_session(args.encoder)
    dec = build_session(args.decoder)

    t0 = time.perf_counter()
    enc_out = run_encoder(enc, image)
    t1 = time.perf_counter()
    dec_out = run_decoder(
        dec, enc_out["image_embed"], enc_out["high_res_s0"],
        enc_out["high_res_s1"], point_coords, point_labels,
    )
    t2 = time.perf_counter()

    low_res_masks = dec_out["low_res_masks"]
    iou_predictions = dec_out["iou_predictions"]
    mask, best_iou = postprocess_masks(low_res_masks, iou_predictions, orig_hw)

    print("=== SAM2 ONNX Inference ===")
    print(f"  image: {args.image} (orig {orig_hw[1]}x{orig_hw[0]})")
    print(f"  point: ({args.point[0]}, {args.point[1]})")
    print(f"  encoder_out shapes: image_embed={enc_out['image_embed'].shape}, "
          f"s0={enc_out['high_res_s0'].shape}, s1={enc_out['high_res_s1'].shape}")
    print(f"  low_res_masks: {low_res_masks.shape}, iou_predictions: {iou_predictions.shape}")
    print(f"  ious: {np.round(iou_predictions[0], 4).tolist()}")
    print(f"  best_mask_iou: {best_iou:.4f}, foreground_pixels: {int(mask.sum())}")
    print(f"  latency_ms: encoder={1000*(t1-t0):.1f}, decoder={1000*(t2-t1):.1f}, "
          f"total={1000*(t2-t0):.1f}")

    if args.output:
        save_mask_overlay(args.image, mask, args.output)
        print(f"  saved overlay: {args.output}")

    if args.ckpt:
        print("\n=== Precision alignment (ONNX vs PyTorch) ===")
        ref_low, ref_ious = torch_reference(
            args.ckpt, args.config, args.image, args.point,
        )
        sim = cos_sim(low_res_masks, ref_low)
        iou_sim = cos_sim(iou_predictions, ref_ious)
        max_abs = float(np.max(np.abs(low_res_masks - ref_low)))
        print(f"  low_res_masks cos_sim: {sim:.6f}")
        print(f"  iou_predictions cos_sim: {iou_sim:.6f}")
        print(f"  max_abs_error (masks): {max_abs:.6f}")
        print(f"  torch ious: {np.round(ref_ious, 4).tolist()}")
        if sim > 0.99:
            print("  [PASS] cosine similarity > 0.99")
        else:
            print("  [FAIL] cosine similarity <= 0.99")

    if args.runs > 1:
        print(f"\n=== Performance ({args.runs} runs) ===")
        enc_lat, dec_lat = [], []
        for _ in range(max(3, args.runs // 5)):
            run_encoder(enc, image)
        for _ in range(args.runs):
            t0 = time.perf_counter()
            eo = run_encoder(enc, image)
            t1 = time.perf_counter()
            run_decoder(dec, eo["image_embed"], eo["high_res_s0"],
                        eo["high_res_s1"], point_coords, point_labels)
            t2 = time.perf_counter()
            enc_lat.append(1000 * (t1 - t0))
            dec_lat.append(1000 * (t2 - t1))
        enc_lat = np.array(enc_lat)
        dec_lat = np.array(dec_lat)
        print(f"  encoder ms: mean={enc_lat.mean():.1f}, "
              f"p50={np.percentile(enc_lat, 50):.1f}")
        print(f"  decoder ms: mean={dec_lat.mean():.1f}, "
              f"p50={np.percentile(dec_lat, 50):.1f}")
        total = enc_lat + dec_lat
        print(f"  total ms:   mean={total.mean():.1f}, "
              f"p50={np.percentile(total, 50):.1f}")


if __name__ == "__main__":
    main()
