#!/usr/bin/env python3
"""Export Deep-Video-Deblurring (视频去模糊) to ONNX。

输入多帧模糊序列, 输出清晰中心帧。封装 ``forward(seq) -> sharp_frame``,
seq 为 [B, N, 3, H, W](N=7)。上游 shuochsu/Deep-Video-Deblurring。
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


class DeepVideoDeblurringNet(nn.Module):
    """Deep-Video-Deblurring 封装:多帧模糊 -> 清晰中心帧。"""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.model = net

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out = self.model(seq)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


def _load_model(repo_dir, ckpt, model_file, device):
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    pkg, cls = model_file.rsplit(".", 1)
    try:
        mod = __import__(pkg, fromlist=[cls])  # noqa: WPS433
        model_cls = getattr(mod, cls)
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"无法从 --repo-dir={repo_dir} 导入 {model_file}: " + repr(exc)) from exc
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
    parser = argparse.ArgumentParser(description="Export Deep-Video-Deblurring to ONNX.")
    parser.add_argument("--repo-dir", type=str, default="./deep_video_deblurring_src")
    parser.add_argument("--model-file", type=str, default="model.DeepVideoDeblurring")
    parser.add_argument("--ckpt", type=str, default="./deep_video_deblurring.pth")
    parser.add_argument("--output-dir", type=str, default="./deep_video_deblurring_onnx")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--num-frames", type=int, default=7)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()
    if args.height % 16 or args.width % 16:
        raise SystemExit("height/width 须可被 16 整除")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    net = _load_model(args.repo_dir, args.ckpt, args.model_file, args.device)
    wrapper = DeepVideoDeblurringNet(net).to(args.device).eval()
    seq = torch.randn(1, args.num_frames, 3, args.height, args.width,
                      dtype=torch.float32, device=args.device)
    _export_one(wrapper, output_dir / "deep_video_deblurring.onnx", (seq,),
                input_names=["seq"], output_names=["sharp_frame"], opset=args.opset)
    print("Export complete.")


if __name__ == "__main__":
    main()
