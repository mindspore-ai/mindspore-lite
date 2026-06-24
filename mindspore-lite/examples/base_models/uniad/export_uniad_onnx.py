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
"""Export UniAD (unified end-to-end autonomous driving) to ONNX.

UniAD jointly runs BEVFormer encoder + multi-task heads (detection, tracking,
motion, occupancy, planning). This is a multi-component model with several
stage-2 blockers (Temporal Deformable Attention, track/occ modules). This
scaffold exposes the multi-view input and the planning trajectory as the
representative output; stage-2 verification will block on the deformable op and
multi-module glue — record and shelve per plan.
"""

import argparse
import importlib
from pathlib import Path

import torch
from torch import nn


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="Export UniAD to ONNX.")
    parser.add_argument("--model-module", type=str, default="model")
    parser.add_argument("--model-class", type=str, default="UniAD")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="uniad_onnx/uniad.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--ncams", type=int, default=6)
    parser.add_argument("--img-h", type=int, default=320)
    parser.add_argument("--img-w", type=int, default=800)
    parser.add_argument("--pred-len", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


class UniADWrapper(nn.Module):
    """UniAD wrapper: multi-view images -> planning trajectory (+ task outputs)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, imgs):
        """imgs [1,N,3,H,W] -> (planning_traj [1,pred,3], det_cls [1,Q,cls])."""
        out = self.model(imgs)
        if isinstance(out, dict):
            traj = out.get("planning", out.get("ego_traj", list(out.values())[0]))
            det = out.get("det_cls", list(out.values())[-1])
            return traj, det
        if isinstance(out, (tuple, list)):
            return out[0], out[-1]
        return out, out


def main():
    """main entry."""
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    module = importlib.import_module(args.model_module)
    ModelCls = getattr(module, args.model_class)
    try:
        model = ModelCls(ncams=args.ncams, pred_len=args.pred_len)
    except TypeError:
        model = ModelCls()
    state = torch.load(args.checkpoint, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=False)
    model.to(args.device).eval()

    wrapper = UniADWrapper(model).to(args.device).eval()
    dummy = torch.randn(1, args.ncams, 3, args.img_h, args.img_w, device=args.device,
                        dtype=torch.float32)
    print(f"Exporting UniAD ONNX, input shape={tuple(dummy.shape)}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, args.output, opset_version=args.opset,
            input_names=["imgs"],
            output_names=["planning_traj", "det_cls"],
            dynamic_axes={"imgs": {0: "batch"},
                          "planning_traj": {0: "batch"},
                          "det_cls": {0: "batch"}},
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
        )
    print(f"Successfully exported to: {args.output}")


if __name__ == "__main__":
    main()
