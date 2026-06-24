#!/usr/bin/env python3
"""Export FastDVDnet (Fast and Accurate Video Denoising) to ONNX.

FastDVDnet 用 5 帧含噪序列 + 噪声 sigma 去除中心帧噪声。
上游 m-tassano/fastdvdnet 的 ``FastDVDnet.forward(x, noise_sigma)``:
x 为 [B, 5, C, H, W],noise_sigma 为 [B,1];输出去噪中心帧 [B, C, H, W]。
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


class FastDvdNetWrapper(nn.Module):
    """FastDVDnet 封装:5 帧序列 + 噪声 sigma -> 去噪中心帧。"""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.model = net

    def forward(self, seq: torch.Tensor, noise_sigma: torch.Tensor) -> torch.Tensor:
        # 上游 forward(x, noise_map): x 为 flattened [B,N*C,H,W], noise_map [B,1,H,W]。
        # seq 为 [B,N,3,H,W], sigma 为 [B,1]; 内部展开以匹配上游接口。
        b, n, c, h, w = seq.shape
        x = seq.reshape(b, n * c, h, w)
        noise_map = noise_sigma.expand(1, 1, h, w)
        out = self.model(x, noise_map)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out  # [B,3,H,W]


def _load_model(repo_dir, ckpt, num_input_frames, device):
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        from models import FastDVDnet  # noqa: WPS433 (上游 models.py)
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            f"无法从 --repo-dir={repo_dir} 导入 FastDVDnet,请确认已 clone "
            "m-tassano/fastdvdnet。原始错误: " + repr(exc)) from exc
    net = FastDVDnet(num_input_frames=num_input_frames)
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
    parser = argparse.ArgumentParser(description="Export FastDVDnet to ONNX.")
    parser.add_argument("--repo-dir", type=str, default="./fastdvdnet_src",
                        help="上游 m-tassano/fastdvdnet 源码目录(含 fastdvdnet.py)")
    parser.add_argument("--ckpt", type=str, default="./fastdvdnet.pth")
    parser.add_argument("--output-dir", type=str, default="./fastdvdnet_onnx")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=5, help="输入帧数(默认 5)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.height % 16 or args.width % 16:
        raise SystemExit("height/width 须可被 16 整除(UNet 下采样)")

    net = _load_model(args.repo_dir, args.ckpt, args.num_frames, args.device)
    wrapper = FastDvdNetWrapper(net).to(args.device).eval()
    seq = torch.randn(1, args.num_frames, 3, args.height, args.width, dtype=torch.float32, device=args.device)
    sigma = torch.tensor([[5.0]], dtype=torch.float32, device=args.device)
    _export_one(
        wrapper, output_dir / "fastdvdnet.onnx",
        (seq, sigma),
        input_names=["seq", "noise_sigma"], output_names=["denoised"],
        opset=args.opset,
    )
    print("Export complete.")


if __name__ == "__main__":
    main()
