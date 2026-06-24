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
"""Export VectorNet (trajectory forecasting) to ONNX.

VectorNet encodes agent history and map polylines with a hierarchical graph and
predicts the target agent's future trajectory. Inputs are vectorized (no image).
"""

import argparse
import importlib
from pathlib import Path

import torch
from torch import nn


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="Export VectorNet to ONNX.")
    parser.add_argument("--model-module", type=str, default="model")
    parser.add_argument("--model-class", type=str, default="VectorNet")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="vectornet_onnx/vectornet.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--obs-len", type=int, default=20)
    parser.add_argument("--agent-feat", type=int, default=4)
    parser.add_argument("--map-poly-num", type=int, default=100)
    parser.add_argument("--map-feat", type=int, default=9)
    parser.add_argument("--pred-len", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


class VectorNetWrapper(nn.Module):
    """VectorNet wrapper: agent history + map polylines -> future trajectory."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, agent_hist, map_polyline):
        """agent_hist [1,obs,feat], map_polyline [1,M,feat] -> traj [1,pred,2]."""
        return self.model(agent_hist, map_polyline)


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
    state = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=False)
    model.to(args.device).eval()

    wrapper = VectorNetWrapper(model).to(args.device).eval()
    agent_hist = torch.randn(1, args.obs_len, args.agent_feat, device=args.device)
    map_polyline = torch.randn(1, args.map_poly_num, args.map_feat, device=args.device)
    print(f"Exporting VectorNet ONNX: agent_hist={tuple(agent_hist.shape)}, "
          f"map_polyline={tuple(map_polyline.shape)}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (agent_hist, map_polyline), args.output, opset_version=args.opset,
            input_names=["agent_hist", "map_polyline"],
            output_names=["trajectory"],
            dynamic_axes={"agent_hist": {0: "batch"},
                          "map_polyline": {0: "batch"},
                          "trajectory": {0: "batch"}},
        )
    print(f"Successfully exported to: {args.output}")


if __name__ == "__main__":
    main()
