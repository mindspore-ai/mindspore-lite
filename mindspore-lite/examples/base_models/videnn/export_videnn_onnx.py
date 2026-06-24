#!/usr/bin/env python3
"""Export ViDeNN (Video Denoising CNN) to ONNX.

ViDeNN 用时序+空域双子网对视频去噪。本脚本封装上游模型的完整 forward 为
``forward(seq) -> denoised``,seq 为 [B, 2, 3, H, W](相邻两帧),输出去噪帧。
注意: 上游 forward 签名需在 Phase 2 用实际源码核对(部分版本需噪声 sigma 输入)。
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


class VidennWrapper(nn.Module):
    """ViDeNN 封装:2 帧序列 -> 去噪帧。"""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.model = net

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out = self.model(seq)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


def _load_model(repo_dir, ckpt, device):
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        from model import ViDeNN  # noqa: WPS433
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            f"无法从 --repo-dir={repo_dir} 导入 ViDeNN,请确认已 clone "
            "clausmichele/ViDeNN。原始错误: " + repr(exc)) from exc
    net = ViDeNN()
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state, strict=False)
    net.eval().to(device)
    return net


def _export_one(module, onnx_path, dummy_inputs, input_names, output_names, opset):
    """Export a PyTorch module to ONNX (legacy exporter, fixed shapes)."""
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
    parser = argparse.ArgumentParser(description="Export ViDeNN to ONNX.")
    parser.add_argument("--repo-dir", type=str, default="./videnn_src")
    parser.add_argument("--ckpt", type=str, default="./videnn.pth")
    parser.add_argument("--output-dir", type=str, default="./videnn_onnx")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.height % 16 or args.width % 16:
        raise SystemExit("height/width 须可被 16 整除")

    net = _load_model(args.repo_dir, args.ckpt, args.device)
    wrapper = VidennWrapper(net).to(args.device).eval()
    seq = torch.randn(1, 2, 3, args.height, args.width, dtype=torch.float32, device=args.device)
    _export_one(
        wrapper, output_dir / "videnn.onnx",
        (seq,),
        input_names=["seq"], output_names=["denoised"],
        opset=args.opset,
    )
    print("Export complete.")


if __name__ == "__main__":
    main()
