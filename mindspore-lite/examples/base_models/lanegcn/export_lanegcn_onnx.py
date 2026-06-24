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
"""Export LaneGCN (lane-graph convolution trajectory forecasting) to ONNX.

Note: LaneGCN uses graph convolutions over lane nodes. Dense-matmul equivalents
are typically export-friendly; scattered gather/scatter ops may need stage-2
op handling (see README FAQ).
"""

import argparse
import importlib
from pathlib import Path

import torch
from torch import nn


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="Export LaneGCN to ONNX.")
    parser.add_argument("--model-module", type=str, default="model")
    parser.add_argument("--model-class", type=str, default="LaneGCN")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="lanegcn_onnx/lanegcn.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--obs-len", type=int, default=20)
    parser.add_argument("--actor-feat", type=int, default=6)
    parser.add_argument("--lane-num", type=int, default=500)
    parser.add_argument("--lane-feat", type=int, default=4)
    parser.add_argument("--pred-len", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


class LaneGCNWrapper(nn.Module):
    """LaneGCN wrapper: actor history + lane nodes -> future trajectory."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, actor_hist, lane_nodes):
        """actor_hist [1,T,F], lane_nodes [1,M,F] -> traj [1,pred,2]."""
        out = self.model(actor_hist, lane_nodes)
        if isinstance(out, (tuple, list)):
            return out[0]
        if isinstance(out, dict):
            return out.get("traj", list(out.values())[0])
        return out


def main():
    """main entry."""
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    module = importlib.import_module(args.model_module)
    ModelCls = getattr(module, args.model_class)
    try:
        model = ModelCls(obs_len=args.obs_len, pred_len=args.pred_len)
    except TypeError:
        model = ModelCls()
    state = torch.load(args.checkpoint, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=False)
    model.to(args.device).eval()

    wrapper = LaneGCNWrapper(model).to(args.device).eval()
    actor = torch.randn(1, args.obs_len, args.actor_feat, device=args.device)
    lanes = torch.randn(1, args.lane_num, args.lane_feat, device=args.device)
    print(f"Exporting LaneGCN ONNX: actor_hist={tuple(actor.shape)}, lane_nodes={tuple(lanes.shape)}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (actor, lanes), args.output, opset_version=args.opset,
            input_names=["actor_hist", "lane_nodes"],
            output_names=["trajectory"],
            dynamic_axes={"actor_hist": {0: "batch"},
                          "lane_nodes": {0: "batch"},
                          "trajectory": {0: "batch"}},
        )
    print(f"Successfully exported to: {args.output}")


if __name__ == "__main__":
    main()
