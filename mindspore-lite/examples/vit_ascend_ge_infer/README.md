# ViT (Vision Transformer) High-Performance Deployment on Atlas 800I A2

<!--
Copyright 2026 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## Overview

This tutorial demonstrates how to deploy and optimize open-source Vision Transformer (ViT) models on Atlas 800I A2 hardware using MindSpore Lite. With proper engine scheduling and environment configuration, this solution achieves **~227ms** end-to-end inference latency at `BatchSize=256` (well within the 410ms target) while maintaining high accuracy with cosine similarity **> 0.999**.

## 1. Hardware and Software Requirements

### Hardware

- Ascend Atlas 800I A2

### Operating System
- openEuler or Ubuntu Linux

### Software Dependencies
- `Python` >= 3.11.4
- `CANN` >= 8.2.RC1
- `MindSpore Lite` >= 2.7.0
- PyTorch and timm (for model export)

## 2. Deployment Strategy and Best Practices

When deploying ViT models with special operator arrangements (such as `Conv2D` followed by `Flatten/Reshape`), converter frontend optimizations may cause `EZ9999` dimension mismatch errors due to imperfect layout splitting.

### Recommended Approach

1. **Clean Graph Conversion**: During the `converter_lite` stage, do NOT enable any hardware-specific frontend optimizations or add `--device=Ascend`. This generates a clean, cross-platform `.mindir` model that preserves the graph topology.

2. **GE Backend Compilation**: At Python runtime, set `provider="ge"` in the Context to invoke the Ascend Graph Engine (GE) compiler for online compilation. GE automatically applies low-level optimizations such as NHWC layout transformation.

3. **Maximum Performance**: Set `precision_mode="enforce_fp16"` to leverage the full parallel computing power of the CUBE matrix units.

4. **Resource Isolation**: On multi-core physical machines, limit CPU threads to prevent OpenBLAS memory overflow (see commands below).

## 3. Quick Start Guide

### Step 1: Export ONNX Model

Use the provided script to export a standard NCHW static batch model:

```bash
python export_vit_onnx.py --batch_size 256 --output vit_base_b256.onnx
```

**注意**: The export script enables `do_constant_folding=True` during ONNX export, which pre-computes and folds constant expressions in the graph, reducing model size and eliminating redundant computation at inference time.

### Step 2: Convert to MindIR Format

Use MindSpore Lite converter to convert ONNX to universal MindIR format.

**Important**: Do NOT add any optimization or device parameters at this stage.

```bash
converter_lite --fmk=ONNX --modelFile=vit_base_b256.onnx --outputFile=vit_base_b256
```

### Step 3: Run High-Performance Inference

Use the provided `vit_ascend_infer.py` script to execute benchmarking.

**注意**: On large-core servers, set thread environment variables to prevent CPU operator library crashes:

```bash
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
```

Run the inference script:

```bash
python vit_ascend_infer.py --model_path vit_base_b256.mindir --batch_size 256 --device_id 0
```

## 4. Performance and Accuracy Baseline

### Performance Metrics
- **Average Latency**: ~227 ms
- **Throughput**: ~1130 FPS
- Well within the target of 410 ms

### Accuracy Metrics (compared to PyTorch CPU FP32)
- **Max Absolute Error**: ~0.075
- **Cosine Similarity**: 0.99996 (extremely high consistency)

### Ascend Profiling Data

Profiling data collected on Atlas 800I A2 using `msprof` (CANN >= 8.2.RC1), BatchSize=256, 53 iterations.

#### Host-Side API Timing

| API Level | API Name      | Avg (ms) | Min (ms) | Max (ms) | Count |
|-----------|---------------|----------|----------|----------|-------|
| model     | ModelExecute  | 135.30   | 134.97   | 139.15   | 53    |
| model     | InputCopy     | 67.60    | 28.92    | 98.76    | 53    |
| model     | OutputCopy    | 1.40     | 0.37     | 4.20     | 53    |
| model     | ModelLoad     | 192.50   | —        | —        | 1     |
| acl       | RunGraphAsync | 0.11     | 0.07     | 0.44     | 53    |

> **注意**: `InputCopy` accounts for a significant portion of end-to-end latency due to host-to-device data transfer at BatchSize=256. `ModelExecute` represents pure NPU compute time (~135ms).

#### GE Automatic Operator Fusion

The GE backend automatically performs the following operator fusions at compile time (no manual modification required):

| Fused Op Name              | Original Ops Fused                              |
|----------------------------|-------------------------------------------------|
| `node_MatMul_xx_batch_matmul` | MatMul + AddBias (all attention/FFN layers)  |
| `node_gelu/Gelu`           | Add + Div + Erf + Mul + Gelu (FFN activation)   |
| `node_conv2d_post`         | Conv2D post-processing                          |
| `node_permute / Transpose` | Layout permutation ops                          |

#### How to Collect Profiling Data

```bash
msprof --application="python vit_ascend_infer.py --model_path vit_base_b256.mindir --batch_size 256 --device_id 0" --output=./profiling_output
```

Output is saved to `/root/PROF_xxx/mindstudio_profiler_output/`. If you need further guidance, please submit an issue or discussion in the [MindSpore Lite Community](https://gitcode.com/mindspore/mindspore-lite/issues).

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
