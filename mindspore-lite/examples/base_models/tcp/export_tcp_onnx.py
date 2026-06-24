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
"""Export TCP (trajectory-guided control prediction, end-to-end driving) to ONNX."""

import argparse
import importlib
from pathlib import Path

import torch
from torch import nn


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="Export TCP to ONNX.")
    parser.add_argument("--model-module", type=str, default="model")
    parser.add_argument("--model-class", type=str, default="TCP")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="tcp_onnx/tcp.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--img-h", type=int, default=256)
    parser.add_argument("--img-w", type=int, default=512)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


class TCPWrapper(nn.Module):
    """TCP wrapper: front image + speed + command -> control + trajectory."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, front_img, speed, command):
        """front_img [1,3,H,W], speed [1,1], command one-hot [1,4] -> (control, traj)."""
        out = self.model(front_img, speed, command)
        if isinstance(out, dict):
            control = out.get("control", list(out.values())[0])
            traj = out.get("trajectory", out.get("waypoints", list(out.values())[-1]))
            return control, traj
        if isinstance(out, (tuple, list)):
            return out[0], out[-1]
        return out, out


def main():
    """main entry."""
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    module = importlib.import_module(args.model_module)
    ModelCls = getattr(module, args.model_class)
    model = ModelCls()
    state = torch.load(args.checkpoint, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=False)
    model.to(args.device).eval()

    wrapper = TCPWrapper(model).to(args.device).eval()
    front = torch.randn(1, 3, args.img_h, args.img_w, device=args.device)
    speed = torch.randn(1, 1, device=args.device)
    command = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=args.device)  # one-hot, e.g. follow
    print(f"Exporting TCP ONNX: front_img={tuple(front.shape)}, speed={tuple(speed.shape)}, "
          f"command={tuple(command.shape)}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (front, speed, command), args.output, opset_version=args.opset,
            input_names=["front_img", "speed", "command"],
            output_names=["control", "trajectory"],
            dynamic_axes={"front_img": {0: "batch"},
                          "speed": {0: "batch"},
                          "command": {0: "batch"},
                          "control": {0: "batch"},
                          "trajectory": {0: "batch"}},
        )
    print(f"Successfully exported to: {args.output}")


if __name__ == "__main__":
    main()
