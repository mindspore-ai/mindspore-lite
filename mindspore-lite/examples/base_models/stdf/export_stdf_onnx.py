#!/usr/bin/env python3
"""Export STDF (Spatio-Temporal Deblocking Filter) to ONNX.

STDF 用于视频压缩伪影去除: 输入低质量帧序列(默认 7 帧), 经时空对齐+融合输出增强帧。
封装为 ``forward(seq) -> enhanced``,seq 为 [B, 7, 3, H, W]。
注意: 上游 forward 签名/帧数需在 Phase 2 用实际源码核对。
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


class StdfWrapper(nn.Module):
    """STDF 封装:帧序列 -> 增强帧。"""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.model = net

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out = self.model(seq)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


def _load_model(repo_dir, ckpt, num_frames, device):
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        from model import STDF  # noqa: WPS433 (上游路径,Phase 2 核对)
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            f"无法从 --repo-dir={repo_dir} 导入 STDF,请确认已 clone "
            "RyanXingHL/STDF。原始错误: " + repr(exc)) from exc
    net = STDF()
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
    parser = argparse.ArgumentParser(description="Export STDF to ONNX.")
    parser.add_argument("--repo-dir", type=str, default="./stdf_src")
    parser.add_argument("--ckpt", type=str, default="./stdf.pth")
    parser.add_argument("--output-dir", type=str, default="./stdf_onnx")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.height % 16 or args.width % 16:
        raise SystemExit("height/width 须可被 16 整除")

    net = _load_model(args.repo_dir, args.ckpt, args.num_frames, args.device)
    wrapper = StdfWrapper(net).to(args.device).eval()
    seq = torch.randn(1, args.num_frames, 3, args.height, args.width,
                      dtype=torch.float32, device=args.device)

    _export_one(
        wrapper, output_dir / "stdf.onnx", (seq,),
        input_names=["seq"], output_names=["enhanced"], opset=args.opset,
    )
    print("Export complete.")


if __name__ == "__main__":
    main()
