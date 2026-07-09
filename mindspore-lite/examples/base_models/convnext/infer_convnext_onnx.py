"""ONNX Runtime inference for the ConvNeXt-UperNet segmentation model.

Loads an image, preprocesses it (ADE20K mean/std, resize to 512x512), runs the
ONNX model, and produces a colour segmentation mask plus timing data.
"""

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

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
    parser = argparse.ArgumentParser(description="ConvNeXt-UperNet ONNX inference")
    parser.add_argument("--onnx", default="./outputs/upernet_convnext_tiny.onnx",
                        help="path to the ONNX model")
    parser.add_argument("--input", default="./test_ade20k.jpg",
                        help="path to the input image")
    parser.add_argument("--output", default="./outputs/seg_onnx.png",
                        help="path to save the colour segmentation mask")
    parser.add_argument("--provider", default="CPUExecutionProvider",
                        help="ONNX Runtime execution provider")
    return parser.parse_args()


def preprocess(image_path):
    """Load image, resize, normalise, and return CHW float32 array."""
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


def colorize(seg, palette):
    """Map a per-pixel class index array to an RGB colour image."""
    return palette[seg]


def main():
    """Entry point: preprocess, run ONNX inference, save mask, report timing."""
    args = parse_args()
    sess = ort.InferenceSession(args.onnx, providers=[args.provider])

    t0 = time.time()
    inp = preprocess(args.input)
    t1 = time.time()
    logits = sess.run(None, {INPUT_NAME: inp})[0]
    t2 = time.time()

    seg = np.argmax(logits[0], axis=0).astype(np.uint8)
    palette = build_palette(len(ADE20K_CLASSES))
    mask = colorize(seg, palette)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    Image.fromarray(mask).save(args.output)

    pre_ms = (t1 - t0) * 1000
    infer_ms = (t2 - t1) * 1000
    total_ms = (t2 - t0) * 1000
    unique, counts = np.unique(seg, return_counts=True)
    top = sorted(zip(unique.tolist(), counts.tolist()),
                 key=lambda kv: -kv[1])[:5]
    label_summary = ", ".join(
        f"{ADE20K_CLASSES[c]}({n})" for c, n in top if c < len(ADE20K_CLASSES))
    print(f"[onnx-infer] preprocess={pre_ms:.1f}ms  inference={infer_ms:.1f}ms  "
          f"total={total_ms:.1f}ms")
    print(f"[onnx-infer] output shape={logits.shape}  logits range="
          f"[{logits.min():.3f}, {logits.max():.3f}]")
    print(f"[onnx-infer] top classes: {label_summary}")
    print(f"[onnx-infer] segmentation mask saved to {args.output}")


if __name__ == "__main__":
    main()
