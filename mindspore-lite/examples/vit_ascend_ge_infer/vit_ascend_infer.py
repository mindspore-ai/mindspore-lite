# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
MindSpore Lite ViT Model Inference Example on Atlas 800I A2.

This module demonstrates high-performance inference deployment for Vision
Transformer (ViT) models using the Ascend GE (Graph Engine) backend.
"""

import argparse
import sys
import time
from typing import Tuple

import mindspore_lite as mslite
import numpy as np


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
         argparse.Namespace containing parsed arguments with model path,
         batch size, and device ID.
    """
    parser = argparse.ArgumentParser(
        description="ViT Model Inference on Ascend using MindSpore Lite"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the .mindir model file'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for inference (default: 256)'
    )
    parser.add_argument(
        '--device_id',
        type=int,
        default=0,
        help='Ascend NPU device ID (default: 0)'
    )
    return parser.parse_args()


def build_context(device_id: int) -> mslite.Context:
    """
    Build MindSpore Lite context for Ascend device.

    Args:
        device_id: Ascend NPU device ID

    Returns:
        mslite.Context configured with GE backend and FP16 precision.
    """
    context = mslite.Context()
    context.target = ["Ascend"]
    context.ascend.device_id = device_id
    context.ascend.provider = "ge"
    context.ascend.precision_mode = "enforce_fp16"
    return context


def load_model(model_path: str, context: mslite.Context) -> mslite.Model:
    """
    Load and build MindSpore Lite model.

    Args:
        model_path: Path to the .mindir model file.
        context: MindSpore Lite context.

    Returns:
        mslite.Model built and ready for inference.

    Raises:
        SystemExit: If model building fails.
    """
    print(f"[INFO] Loading and compiling model: {model_path}")
    print("[INFO] Note: Graph compilation may take 1-3 minutes...")
    model = mslite.Model()
    try:
        model.build_from_file(model_path, mslite.ModelType.MINDIR, context)
        print("[INFO] Model built successfully")
        return model
    except RuntimeError as e:
        print(f"[ERROR] Model build failed: {e}")
        sys.exit(1)


def prepare_input_data(
    model: mslite.Model,
    batch_size: int
) -> Tuple[mslite.Tensor, np.ndarray]:
    """
    Prepare input data for inference.

    Args:
        model (mslite.Model): MindSpore Lite model.
        batch_size (int): Batch size for inference.

    Returns:
        Tuple containing the input tensor and numpy array.
    """
    print(
        f"[INFO] Preparing input data with shape "
        f"(Batch={batch_size}, C=3, H=224, W=224)"
    )
    input_shape = (batch_size, 3, 224, 224)
    input_data = np.random.randn(*input_shape).astype(np.float32)

    input_tensor = model.get_inputs()[0]
    input_tensor.set_data_from_numpy(np.ascontiguousarray(input_data))

    return input_tensor, input_data


def warmup_model(model: mslite.Model, input_tensor: mslite.Tensor) -> None:
    """
    Warm up the model with several inference iterations.

    Args:
        model: MindSpore Lite model.
        input_tensor: Input tensor for inference.
    """
    print("[INFO] Warming up NPU (3 iterations)...")
    for _ in range(3):
        model.predict([input_tensor])


def benchmark_model(
    model: mslite.Model,
    input_tensor: mslite.Tensor,
    batch_size: int,
    num_iterations: int = 50
) -> Tuple[float, float]:
    """
    Benchmark model performance.

    Args:
        model: MindSpore Lite model.
        input_tensor: Input tensor for inference.
        batch_size: Batch size for inference.
        num_iterations: Number of iterations for benchmarking (default: 50).

    Returns:
        Tuple containing average latency (ms) and throughput (FPS).
    """
    print(f"[INFO] Benchmarking for {num_iterations} iterations...")
    start_time = time.time()
    for _ in range(num_iterations):
        model.predict([input_tensor])
    end_time = time.time()

    avg_latency = ((end_time - start_time) / num_iterations) * 1000
    throughput = batch_size / ((end_time - start_time) / num_iterations)

    return avg_latency, throughput


def print_performance_summary(
    batch_size: int,
    avg_latency: float,
    throughput: float
) -> None:
    """
    Print performance summary.

    Args:
        batch_size: Batch size used for inference.
        avg_latency: Average latency in milliseconds.
        throughput: Throughput in frames per second.
    """
    print("=" * 60)
    print("Performance Summary (Ascend GE Engine)")
    print("=" * 60)
    print(f"Batch Size       : {batch_size}")
    print(f"Average Latency  : {avg_latency:.2f} ms")
    print(f"Throughput       : {throughput:.2f} FPS")
    print("=" * 60)


def main() -> None:
    """Main function for ViT model inference on Ascend."""
    args = parse_args()

    print(f"[INFO] Initializing Ascend context on device {args.device_id}...")
    context = build_context(args.device_id)

    model = load_model(args.model_path, context)

    input_tensor, _ = prepare_input_data(model, args.batch_size)

    warmup_model(model, input_tensor)

    avg_latency, throughput = benchmark_model(
        model, input_tensor, args.batch_size
    )

    print_performance_summary(args.batch_size, avg_latency, throughput)


if __name__ == '__main__':
    main()
