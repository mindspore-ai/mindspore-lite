"""MindSpore Lite inference for the ConvNeXt-UperNet segmentation model.

Loads the converted MindIR, preprocesses an image (numpy/PIL only, no torch),
runs Ascend inference, and produces a colour segmentation mask with timing.
Optionally aligns the output against an ONNX Runtime reference.
"""

import argparse
import os
import time

import numpy as np
from PIL import Image

import mindspore_lite as msl
import onnxruntime as ort

INPUT_SIZE = 512
INPUT_NAME = "input"
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair", "car", "water",
    "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island", "computer",
    "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel",
    "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television receiver", "airplane", "dirt track", "apparel",
    "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet",
    "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy",
    "washer", "plaything", "swimming pool", "stool", "barrel", "basket",
    "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
    "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle",
    "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce",
    "vase", "traffic light", "tray", "ashcan", "fan", "pier", "crt screen",
    "plate", "monitor", "bulletin board", "shower", "radiator", "glass",
    "clock", "flag"]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ConvNeXt-UperNet MindSpore Lite inference")
    parser.add_argument("--mindir", default="./outputs/upernet_convnext_tiny_mindir.mindir",
                        help="path to the MindIR model")
    parser.add_argument("--input", default="/tmp/opencode/test_ade20k.jpg",
                        help="path to the input image")
    parser.add_argument("--output", default="./outputs/seg_mslite.png",
                        help="path to save the colour segmentation mask")
    parser.add_argument("--device", default="ascend", help="target device")
    parser.add_argument("--device-id", type=int, default=0, help="device id")
    parser.add_argument("--onnx", default="./outputs/upernet_convnext_tiny.onnx",
                        help="ONNX model for optional precision alignment")
    parser.add_argument("--align", action="store_true",
                        help="compare mslite output with onnx reference")
    return parser.parse_args()


def preprocess(image_path):
    """Load image, resize, normalise, return CHW float32 numpy array."""
    img = Image.open(image_path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, 0)


def build_palette(num_classes):
    """Generate a deterministic colour palette for visualisation."""
    palette = []
    for i in range(num_classes):
        r, g, b = 0, 0, 0
        cid = i
        for j in range(8):
            r |= ((cid >> 0) & 1) << (7 - j)
            g |= ((cid >> 1) & 1) << (7 - j)
            b |= ((cid >> 2) & 1) << (7 - j)
            cid >>= 3
        palette.append([r, g, b])
    return np.array(palette, dtype=np.uint8)


def build_model(mindir_path, device, device_id):
    """Build the MindSpore Lite model from a MindIR file."""
    ctx = msl.Context()
    ctx.target = [device]
    if device == "ascend":
        ctx.ascend.device_id = device_id
    model = msl.Model()
    model.build_from_file(mindir_path, msl.ModelType.MINDIR, ctx)
    return model


def run_inference(model, inp_array):
    """Feed a numpy input to the model and return the output numpy array."""
    inputs = model.get_inputs()
    for tensor in inputs:
        if tensor.name == INPUT_NAME or len(inputs) == 1:
            tensor.set_data_from_numpy(inp_array)
            break
    outputs = model.predict(inputs)
    return outputs[0].get_data_to_numpy()


def align_with_onnx(onnx_path, inp_array, mslite_out):
    """Compare MindSpore Lite output against an ONNX Runtime reference."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {INPUT_NAME: inp_array})[0]
    a = mslite_out.flatten()
    b = onnx_out.flatten()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    max_err = float(np.max(np.abs(mslite_out - onnx_out)))
    arg_match = float(np.mean(
        mslite_out.argmax(1) == onnx_out.argmax(1)))
    print(f"[align] cosine={cos:.6f}  max_abs_err={max_err:.6e}  "
          f"argmax_match_ratio={arg_match:.6f}")
    if cos > 0.99:
        print("[align] PASS: mslite vs onnx cosine > 0.99")
    else:
        print("[align] WARN: cosine below 0.99 threshold")


def main():
    """Entry point: build model, preprocess, infer, save mask, report timing."""
    args = parse_args()
    model = build_model(args.mindir, args.device, args.device_id)

    t0 = time.time()
    inp = preprocess(args.input)
    t1 = time.time()
    logits = run_inference(model, inp)
    t2 = time.time()

    seg = np.argmax(logits[0], axis=0).astype(np.uint8)
    palette = build_palette(len(ADE20K_CLASSES))
    mask = palette[seg]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    Image.fromarray(mask).save(args.output)

    pre_ms = (t1 - t0) * 1000
    infer_ms = (t2 - t1) * 1000
    total_ms = (t2 - t0) * 1000
    unique, counts = np.unique(seg, return_counts=True)
    top = sorted(zip(unique.tolist(), counts.tolist()), key=lambda kv: -kv[1])[:5]
    label_summary = ", ".join(
        f"{ADE20K_CLASSES[c]}({n})" for c, n in top if c < len(ADE20K_CLASSES))
    print(f"[mslite-infer] preprocess={pre_ms:.1f}ms  inference={infer_ms:.1f}ms  "
          f"total={total_ms:.1f}ms")
    print(f"[mslite-infer] output shape={logits.shape}  logits range="
          f"[{logits.min():.3f}, {logits.max():.3f}]")
    print(f"[mslite-infer] top classes: {label_summary}")
    print(f"[mslite-infer] segmentation mask saved to {args.output}")

    if args.align:
        align_with_onnx(args.onnx, inp, logits)


if __name__ == "__main__":
    main()
