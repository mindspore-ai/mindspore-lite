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
"""Export HiVT (hierarchical vector transformer forecasting) to ONNX."""

import argparse
import importlib
from pathlib import Path

import torch
from torch import nn


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="Export HiVT to ONNX.")
    parser.add_argument("--model-module", type=str, default="model")
    parser.add_argument("--model-class", type=str, default="HiVT")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="hivt_onnx/hivt.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--obs-len", type=int, default=20)
    parser.add_argument("--agent-feat", type=int, default=7)
    parser.add_argument("--map-num", type=int, default=200)
    parser.add_argument("--map-feat", type=int, default=7)
    parser.add_argument("--pred-len", type=int, default=30)
    parser.add_argument("--num-modes", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


class HiVTWrapper(nn.Module):
    """HiVT wrapper: actor history + map lanes -> multi-modal trajectory."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, actor_history, map_lanes):
        """actor_history [1,N,T,F], map_lanes [1,M,F] -> traj [1,K,pred,2]."""
        out = self.model(actor_history, map_lanes)
        if isinstance(out, (tuple, list)):
            loc = out[0]
        elif isinstance(out, dict):
            loc = out.get("loc", list(out.values())[0])
        else:
            loc = out
        return loc


def main():
    """main entry."""
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    module = importlib.import_module(args.model_module)
    ModelCls = getattr(module, args.model_class)
    try:
        model = ModelCls(obs_len=args.obs_len, pred_len=args.pred_len, n_modes=args.num_modes)
    except TypeError:
        model = ModelCls()
    state = torch.load(args.checkpoint, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=False)
    model.to(args.device).eval()

    wrapper = HiVTWrapper(model).to(args.device).eval()
    actor = torch.randn(1, args.num_agents, args.obs_len, args.agent_feat, device=args.device)
    mp = torch.randn(1, args.map_num, args.map_feat, device=args.device)
    print(f"Exporting HiVT ONNX: actor_history={tuple(actor.shape)}, map_lanes={tuple(mp.shape)}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (actor, mp), args.output, opset_version=args.opset,
            input_names=["actor_history", "map_lanes"],
            output_names=["trajectory"],
            dynamic_axes={"actor_history": {0: "batch"},
                          "map_lanes": {0: "batch"},
                          "trajectory": {0: "batch"}},
        )
    print(f"Successfully exported to: {args.output}")


if __name__ == "__main__":
    main()
