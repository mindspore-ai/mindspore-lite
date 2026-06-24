#!/usr/bin/env python3
"""Export RIFE-Lite (轻量版中间光流插帧) to ONNX.

与 RIFE 同源(hzwer/RIFE 的 lite 变体/轻量 checkpoint),封装 ``forward(img0,img1)->mid``。
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

_RIFE_TIMESTEP = 0.5


class RifeLiteInterp(nn.Module):
    """RIFE-Lite 插帧封装:两帧 -> 中点中间帧。"""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.model = net

    def forward(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        out = self.model(img0, img1, _RIFE_TIMESTEP)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


def _load_model(rife_dir, ckpt, model_file, device):
    rife_dir = str(Path(rife_dir).resolve())
    if rife_dir not in sys.path:
        sys.path.insert(0, rife_dir)
    pkg, cls = model_file.rsplit(".", 1)
    try:
        mod = __import__(pkg, fromlist=[cls])  # noqa: WPS433
        model_cls = getattr(mod, cls)
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"无法从 --rife-dir={rife_dir} 导入 {model_file}: " + repr(exc)) from exc
    net = model_cls()
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state, strict=False)
    net.eval().to(device)
    return net


def _export_one(module, onnx_path, dummy_inputs, input_names, output_names, opset):
    """Export to ONNX (legacy exporter, fixed shapes)."""
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module, dummy_inputs, str(onnx_path),
            input_names=input_names, output_names=output_names,
            opset_version=int(opset), do_constant_folding=False, dynamo=False,
        )
    print(f"[export] saved {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Export RIFE-Lite to ONNX.")
    parser.add_argument("--repo-dir", type=str, default="./RIFE")
    parser.add_argument("--model-file", type=str, default="model.RIFE_HDv3")
    parser.add_argument("--ckpt", type=str, default="./RIFE_Lite.pth")
    parser.add_argument("--output-dir", type=str, default="./rife_lite_onnx")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()
    if args.height % 32 or args.width % 32:
        raise SystemExit("height/width 须可被 32 整除")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    net = _load_model(args.rife_dir, args.ckpt, args.model_file, args.device)
    wrapper = RifeLiteInterp(net).to(args.device).eval()
    d0 = torch.randn(1, 3, args.height, args.width, dtype=torch.float32, device=args.device)
    d1 = torch.randn(1, 3, args.height, args.width, dtype=torch.float32, device=args.device)
    _export_one(wrapper, output_dir / "rife_lite.onnx", (d0, d1),
                input_names=["img0", "img1"], output_names=["mid_frame"], opset=args.opset)
    print("Export complete.")


if __name__ == "__main__":
    main()
