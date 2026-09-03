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
"""MindIR fixed-shape inference with seed-fixed random inputs.

Verifies the MindIR model runs correctly on Ascend NPU. Performance and
accuracy are covered by benchmark_bevdet_mslite.py.

Run from examples/base_models/bevdet/:
    python infer_bevdet_mslite.py \
        --model output/bevdet_r50.mindir \
        --device-id 0
"""
import argparse
import sys

import numpy as np
import mindspore_lite as mslite


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description='BEVDet MindIR fixed-shape inference with random inputs')
    p.add_argument("--model", required=True, help="MindIR model file")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for reproducible inputs")
    p.add_argument("--depth-size", type=int, default=498432,
                   help="upper bound for ranks_depth randint "
                        "(B·N·D·H_feat·W_feat for BEVDet-R50)")
    p.add_argument("--feat-size", type=int, default=4224,
                   help="upper bound for ranks_feat randint "
                        "(B·N·H_feat·W_feat for BEVDet-R50)")
    p.add_argument("--bev-size", type=int, default=16384,
                   help="upper bound for ranks_bev randint "
                        "(bev_z·bev_h·bev_w for BEVDet-R50)")
    return p.parse_args()


def main():
    """Run MindIR inference with seed-fixed random inputs."""
    args = parse_args()
    print("=== BEVDet MindIR Fixed-Shape Inference (random inputs) ===")
    print(f"  Model:    {args.model}")
    print(f"  Seed:     {args.seed}")
    print(f"  Device:   ascend:{args.device_id}")

    rng = np.random.default_rng(args.seed)

    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = args.device_id

    ms_model = mslite.Model()
    ms_model.build_from_file(args.model, mslite.ModelType.MINDIR, context)

    ms_inputs = ms_model.get_inputs()
    print("\n  MindIR inputs:")
    for inp in ms_inputs:
        print(f"    {inp.name}: shape={list(inp.shape)}, dtype={inp.dtype}")

    # Verify fixed shape (no -1 dimension allowed)
    for inp in ms_inputs:
        if any(d == -1 for d in inp.shape):
            print(f"ERROR: '{inp.name}' has dynamic shape {list(inp.shape)}; "
                  f"this script only supports fixed-shape models.")
            sys.exit(1)

    upper_bounds = {
        "ranks_depth": args.depth_size,
        "ranks_feat": args.feat_size,
        "ranks_bev": args.bev_size,
    }

    for inp in ms_inputs:
        if inp.name == "img":
            arr = rng.standard_normal(inp.shape).astype(np.float32)
        elif inp.name in upper_bounds:
            arr = rng.integers(0, upper_bounds[inp.name],
                               inp.shape, dtype=np.int32)
        else:
            print(f"ERROR: unknown input '{inp.name}'")
            sys.exit(1)
        inp.set_data_from_numpy(arr)

    ms_outputs = ms_model.predict(ms_inputs)

    print("\n  Output shapes:")
    for i, out in enumerate(ms_outputs):
        print(f"    [{i}] shape={list(out.shape)}, dtype={out.dtype}")


if __name__ == "__main__":
    main()
