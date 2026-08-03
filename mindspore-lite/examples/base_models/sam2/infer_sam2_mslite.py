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
"""SAM2 MindSpore Lite (MindIR) inference script for Ascend.

Loads the converted `sam2_encoder.mindir` and `sam2_decoder.mindir` models,
runs the full image segmentation pipeline on a sample image with a point
prompt, and reports the predicted mask together with end-to-end performance.

This script does NOT depend on torch; all preprocessing and postprocessing
are implemented with numpy and PIL only.

Usage:
    python infer_sam2_mslite.py \
        --encoder ./mindir/sam2_encoder.mindir \
        --decoder ./mindir/sam2_decoder.mindir \
        --image ./truck.jpg --point 500 375 \
        --device ascend
"""

import argparse
import time

import numpy as np
from PIL import Image

try:
    import mindspore_lite as mslite
except ImportError as exc:
    raise ImportError("mindspore_lite is required for this script") from exc

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


def build_model(model_path, device, device_id):
    """Build a MindSpore Lite Model from a MindIR file."""
    context = mslite.Context()
    context.target = [device]
    if device == "ascend":
        context.ascend.device_id = int(device_id)
    model = mslite.Model()
    model.build_from_file(model_path, mslite.ModelType.MINDIR, context)
    return model


def _np_dtype_to_mslite(np_dt):
    """Map numpy dtype to mslite DataType."""
    dt = np.dtype(np_dt)
    m = {
        np.dtype(np.float32): mslite.DataType.FLOAT32,
        np.dtype(np.float16): mslite.DataType.FLOAT16,
        np.dtype(np.int32): mslite.DataType.INT32,
        np.dtype(np.int64): mslite.DataType.INT64,
        np.dtype(np.bool_): mslite.DataType.BOOL,
    }
    return m.get(dt, mslite.DataType.FLOAT32)


def _mslite_to_np_dtype(ms_dt):
    """Map mslite DataType to numpy dtype."""
    m = {
        mslite.DataType.FLOAT32: np.float32,
        mslite.DataType.FLOAT16: np.float16,
        mslite.DataType.INT32: np.int32,
        mslite.DataType.INT64: np.int64,
        mslite.DataType.BOOL: np.bool_,
    }
    return m.get(ms_dt, np.float32)


def make_device_tensor(np_arr, device_id):
    """Create a pre-allocated Ascend device Tensor from numpy data.

    The tensor lives on the device and can be reused across predict() calls,
    eliminating repeated Host→Device copies for inputs that don't change
    between iterations (e.g. the image, point prompts).
    """
    device_str = f"ascend:{int(device_id)}"
    t = mslite.Tensor(
        shape=list(np_arr.shape),
        dtype=_np_dtype_to_mslite(np_arr.dtype),
        device=device_str,
    )
    t.set_data_from_numpy(np.ascontiguousarray(np_arr))
    return t


def describe_io(model, tag):
    """Print the input/output name, dtype and shape of a model."""
    inputs = model.get_inputs()
    print(f"  {tag} inputs:")
    for t in inputs:
        print(f"    {t.name} dtype={t.dtype} shape={t.shape}")
    outputs = model.get_outputs()
    print(f"  {tag} outputs:")
    for t in outputs:
        print(f"    {t.name} dtype={t.dtype} shape={t.shape}")
    return inputs, outputs


def _to_tensor(np_array, ref_input):
    """Convert a numpy array to an MSTensor matching the ref input dtype."""
    dtype_map = {
        mslite.DataType.FLOAT32: np.float32, mslite.DataType.FLOAT16: np.float16,
        mslite.DataType.INT32: np.int32, mslite.DataType.INT64: np.int64,
        mslite.DataType.BOOL: np.bool_,
    }
    target = dtype_map.get(ref_input.dtype, np.float32)
    return mslite.Tensor(np.ascontiguousarray(np_array.astype(target)))


def run_encoder(model, image, zero_copy=False):
    """Run the encoder MindIR model.

    If zero_copy=True, returns raw MSTensor outputs (stay on device, no D2H).
    Otherwise returns numpy arrays (D2H copy occurs).
    """
    outputs = model.predict([_to_tensor(image, model.get_inputs()[0])])
    out_names = [o.name for o in model.get_outputs()]
    if zero_copy:
        return dict(zip(out_names, outputs))
    return dict(zip(out_names, [o.get_data_to_numpy() for o in outputs]))


def run_decoder(model, feed):
    """Run the decoder MindIR model.

    Feed dict may contain MSTensor objects (device tensors) which are passed
    directly to predict() without Host→Device copy.
    """
    inputs = model.get_inputs()
    tensors = []
    for inp in inputs:
        if inp.name not in feed:
            raise ValueError(f"missing input '{inp.name}'")
        val = feed[inp.name]
        if hasattr(val, "get_data_to_numpy"):
            tensors.append(val)
        else:
            tensors.append(_to_tensor(val, inp))
    outputs = model.predict(tensors)
    out_names = [o.name for o in model.get_outputs()]
    return dict(zip(out_names, [o.get_data_to_numpy() for o in outputs]))


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


def load_onnx_reference(encoder_onnx, decoder_onnx, image_path, point_xy):
    """Run the ONNX models to produce reference outputs for alignment."""
    import onnxruntime as ort

    image, orig_hw = preprocess(image_path)
    point_coords, point_labels = transform_point(point_xy, orig_hw)
    enc = ort.InferenceSession(encoder_onnx, providers=["CPUExecutionProvider"])
    enc_out = enc.run(None, {enc.get_inputs()[0].name: image})
    enc_names = [o.name for o in enc.get_outputs()]
    enc_out = dict(zip(enc_names, enc_out))
    dec = ort.InferenceSession(decoder_onnx, providers=["CPUExecutionProvider"])
    dec_in = {
        "image_embed": enc_out["image_embed"],
        "high_res_s0": enc_out["high_res_s0"],
        "high_res_s1": enc_out["high_res_s1"],
        "point_coords": point_coords,
        "point_labels": point_labels,
    }
    dec_out = dec.run(None, dec_in)
    dec_names = [o.name for o in dec.get_outputs()]
    return dict(zip(dec_names, dec_out))


def save_mask_overlay(image_path, mask, out_path):
    """Save a simple mask overlay image (green highlight on the object)."""
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    green = np.zeros_like(arr)
    green[..., 1] = 255
    overlay = np.where(mask[..., None], (arr // 2 + green // 2), arr)
    Image.fromarray(overlay.astype(np.uint8)).save(out_path)


def main():
    """Run MindSpore Lite inference and (optionally) precision alignment."""
    parser = argparse.ArgumentParser(description="SAM2 MindSpore Lite inference")
    parser.add_argument("--encoder", required=True, help="Encoder MindIR path")
    parser.add_argument("--decoder", required=True, help="Decoder MindIR path")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--point", nargs=2, type=float, default=[500, 375],
                        metavar=("X", "Y"), help="Foreground point (pixels)")
    parser.add_argument("--device", default="ascend", choices=["cpu", "ascend"])
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--encoder-onnx", default=None)
    parser.add_argument("--decoder-onnx", default=None)
    parser.add_argument("--output", default=None, help="Path to save mask overlay")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--zero-copy", action="store_true", default=True,
                        help="Use zero-copy: pass device tensors between encoder/decoder")
    parser.add_argument("--no-zero-copy", dest="zero_copy", action="store_false",
                        help="Disable zero-copy (force Host round-trip)")
    args = parser.parse_args()

    image, orig_hw = preprocess(args.image)
    point_coords, point_labels = transform_point(args.point, orig_hw)

    print("=== Building MindSpore Lite models ===")
    enc = build_model(args.encoder, args.device, args.device_id)
    dec = build_model(args.decoder, args.device, args.device_id)
    describe_io(enc, "encoder")
    describe_io(dec, "decoder")
    print(f"  zero_copy={args.zero_copy}")

    # Pre-allocate device tensors for fixed inputs (eliminates repeated H2D)
    # Auto-adapt dtype to each model's declared input dtype
    use_prealloc = args.zero_copy and args.device == "ascend"
    if use_prealloc:
        enc_in0 = enc.get_inputs()[0]
        img_tensor = make_device_tensor(
            image.astype(_mslite_to_np_dtype(enc_in0.dtype)), args.device_id)
        dec_ins = {inp.name: inp for inp in dec.get_inputs()}
        pc_name = "point_coords_fp16" if "point_coords_fp16" in dec_ins else "point_coords"
        pc_tensor = make_device_tensor(
            point_coords.astype(_mslite_to_np_dtype(dec_ins[pc_name].dtype)),
            args.device_id)
        pl_tensor = make_device_tensor(point_labels, args.device_id)

    def _enc_predict():
        if use_prealloc:
            outputs = enc.predict([img_tensor])
        else:
            enc_in0 = enc.get_inputs()[0]
            img = image.astype(_mslite_to_np_dtype(enc_in0.dtype))
            outputs = enc.predict([_to_tensor(img, enc_in0)])
        out_names = [o.name for o in enc.get_outputs()]
        if args.zero_copy:
            return dict(zip(out_names, outputs))
        return dict(zip(out_names, [o.get_data_to_numpy() for o in outputs]))

    def _dec_predict(eo):
        dec_ins = {inp.name: inp for inp in dec.get_inputs()}
        feed = {}
        for base in ["image_embed", "high_res_s0", "high_res_s1"]:
            enc_key = base + "_fp16" if base + "_fp16" in eo else base
            dec_key = base + "_fp16" if base + "_fp16" in dec_ins else base
            feed[dec_key] = eo[enc_key]
        if use_prealloc:
            feed[pc_name] = pc_tensor
            feed["point_labels"] = pl_tensor
        else:
            feed[pc_name] = point_coords.astype(
                _mslite_to_np_dtype(dec_ins[pc_name].dtype))
            feed["point_labels"] = point_labels
        return run_decoder(dec, feed)

    t0 = time.perf_counter()
    enc_out = _enc_predict()
    t1 = time.perf_counter()
    dec_out = _dec_predict(enc_out)
    t2 = time.perf_counter()

    # Auto-detect output keys (may have _fp16 suffix)
    masks_key = "low_res_masks_fp16" if "low_res_masks_fp16" in dec_out else "low_res_masks"
    iou_key = "iou_predictions_fp16" if "iou_predictions_fp16" in dec_out else "iou_predictions"
    low_res_masks = dec_out[masks_key].astype(np.float32)
    iou_predictions = dec_out[iou_key].astype(np.float32)
    mask, best_iou = postprocess_masks(low_res_masks, iou_predictions, orig_hw)

    print("\n=== SAM2 MindSpore Lite Inference ===")
    print(f"  image: {args.image} (orig {orig_hw[1]}x{orig_hw[0]})")
    print(f"  point: ({args.point[0]}, {args.point[1]})")
    print(f"  low_res_masks: {low_res_masks.shape}, iou_predictions: {iou_predictions.shape}")
    print(f"  ious: {np.round(iou_predictions[0], 4).tolist()}")
    print(f"  best_mask_iou: {best_iou:.4f}, foreground_pixels: {int(mask.sum())}")
    print(f"  latency_ms: encoder={1000*(t1-t0):.1f}, decoder={1000*(t2-t1):.1f}, "
          f"total={1000*(t2-t0):.1f}")

    if args.output:
        save_mask_overlay(args.image, mask, args.output)
        print(f"  saved overlay: {args.output}")

    if args.encoder_onnx and args.decoder_onnx:
        print("\n=== Precision alignment (MindIR vs ONNX) ===")
        ref = load_onnx_reference(args.encoder_onnx, args.decoder_onnx,
                                  args.image, args.point)
        sim = cos_sim(low_res_masks, ref["low_res_masks"])
        iou_sim = cos_sim(iou_predictions, ref["iou_predictions"])
        max_abs = float(np.max(np.abs(low_res_masks - ref["low_res_masks"])))
        print(f"  low_res_masks cos_sim: {sim:.6f}")
        print(f"  iou_predictions cos_sim: {iou_sim:.6f}")
        print(f"  max_abs_error (masks): {max_abs:.6f}")
        print(f"  onnx ious: {np.round(ref['iou_predictions'][0], 4).tolist()}")
        if sim > 0.99:
            print("  [PASS] cosine similarity > 0.99")
        else:
            print("  [FAIL] cosine similarity <= 0.99")

    if args.runs > 1:
        print(f"\n=== Performance ({args.runs} runs, device={args.device}) ===")
        for _ in range(max(3, args.runs // 5)):
            _enc_predict()
        enc_lat, dec_lat = [], []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            eo = _enc_predict()
            t1 = time.perf_counter()
            _dec_predict(eo)
            t2 = time.perf_counter()
            enc_lat.append(1000 * (t1 - t0))
            dec_lat.append(1000 * (t2 - t1))
        enc_lat = np.array(enc_lat)
        dec_lat = np.array(dec_lat)
        total = enc_lat + dec_lat
        print(f"  encoder ms: mean={enc_lat.mean():.1f}, "
              f"p50={np.percentile(enc_lat, 50):.1f}")
        print(f"  decoder ms: mean={dec_lat.mean():.1f}, "
              f"p50={np.percentile(dec_lat, 50):.1f}")
        print(f"  total ms:   mean={total.mean():.1f}, "
              f"p50={np.percentile(total, 50):.1f}")


if __name__ == "__main__":
    main()
