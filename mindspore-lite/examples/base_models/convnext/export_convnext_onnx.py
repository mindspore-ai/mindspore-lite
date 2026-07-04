"""Export the ConvNeXt-UperNet segmentation model to ONNX.

Loads the mmseg checkpoint into a pure-PyTorch model, exports a single ONNX
graph with fixed input shape (1, 3, 512, 512), then reports the ONNX I/O and
operator statistics.
"""

import argparse
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from convnext_model import UPerNetConvNeXt, load_pretrained

INPUT_SIZE = 512
OPSET = 17
ONNX_NAME = "upernet_convnext_tiny.onnx"
INPUT_NAME = "input"
OUTPUT_NAME = "seg_logits"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export ConvNeXt-UperNet to ONNX")
    parser.add_argument("--weight", default="upernet_convnext_tiny_1k_512x512.pth",
                        help="path to the .pth checkpoint")
    parser.add_argument("--output-dir", default="./outputs", help="ONNX output directory")
    parser.add_argument("--opset", type=int, default=OPSET, help="ONNX opset version")
    return parser.parse_args()


def build_model(weight_path):
    """Build the model and load pretrained weights."""
    model = UPerNetConvNeXt()
    num_loaded, skipped = load_pretrained(model, weight_path)
    print(f"[export] loaded {num_loaded} weight tensors, skipped {len(skipped)} (auxiliary_head)")
    model.eval()
    return model


def export_onnx(model, output_dir, opset):
    """Export the model to a fixed-shape ONNX file."""
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, ONNX_NAME)
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            opset_version=opset,
            do_constant_folding=True,
        )
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"[export] saved {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path


def summarize_onnx(onnx_path):
    """Print ONNX graph I/O and operator statistics."""
    model = onnx.load(onnx_path)
    opset = next((o.version for o in model.opset_import), "?")
    print(f"[onnx] opset={opset}  nodes={len(model.graph.node)}  "
          f"size={os.path.getsize(onnx_path) / 1024 / 1024:.1f}MB")
    for inp in model.graph.input:
        dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"[onnx] input  {inp.name}: {dims}")
    for out in model.graph.output:
        dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"[onnx] output {out.name}: {dims}")
    ops = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    for op_type, count in sorted(ops.items()):
        print(f"[onnx] op  {op_type:24s} x{count}")


def verify_onnx(model, onnx_path):
    """Cross-check ONNX Runtime output against the PyTorch forward pass."""
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {INPUT_NAME: dummy.numpy()})[0]
    cos = _cosine(torch_out.flatten(), onnx_out.flatten())
    max_err = np.max(np.abs(torch_out - onnx_out))
    print(f"[verify] cosine={cos:.6f}  max_abs_err={max_err:.6e}  "
          f"shape={onnx_out.shape}")
    assert cos > 0.99, f"cosine similarity {cos} below 0.99 threshold"
    print("[verify] PASS: torch vs onnx cosine > 0.99")


def _cosine(a, b):
    """Cosine similarity between two flattened numpy arrays."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    """Entry point: build model, export ONNX, verify."""
    args = parse_args()
    model = build_model(args.weight)
    t0 = time.time()
    onnx_path = export_onnx(model, args.output_dir, args.opset)
    t1 = time.time()
    print(f"[export] export time: {t1 - t0:.1f}s")
    summarize_onnx(onnx_path)
    verify_onnx(model, onnx_path)


if __name__ == "__main__":
    main()
