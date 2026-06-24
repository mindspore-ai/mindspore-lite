#!/usr/bin/env python3
"""Export CAIN (Channel Attention Is All You Need for Video Frame Interpolation) to ONNX.

CAIN 是 CVPR2020 的纯 CNN 视频插帧模型,输入两帧 [img0, img1],输出中间帧。
本脚本从上游仓库 myungsub/CAIN 导入模型类 ``CAIN``,加载预训练权重,封装为
``forward(img0, img1) -> mid_frame`` 的单 ONNX(用于 ONNXRuntime 验证与 MindIR 转换)。
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


class CainInterp(nn.Module):
    """CAIN 插帧封装:两帧输入 -> 单中间帧输出。

    上游 ``CAIN.forward`` 通常接收沿通道拼接的 [B,6,H,W] 张量并返回中间帧
    (部分版本返回 tuple/dict,此处统一取首张量)。
    """

    def __init__(self, cain: nn.Module):
        super().__init__()
        self.model = cain

    def forward(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        out = self.model(img0, img1)  # CAIN.forward(x1, x2) -> (out, feats)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("img", out.get("frame", next(iter(out.values()))))
        return out  # [B,3,H,W]


def _load_model(cain_dir: str, ckpt: str, depth: int, device: str) -> nn.Module:
    """Load upstream CAIN model and pretrained checkpoint."""
    cain_dir = str(Path(cain_dir).resolve())
    if cain_dir not in sys.path:
        sys.path.insert(0, cain_dir)
    try:
        from model.cain import CAIN  # noqa: WPS433 (上游 model/cain.py)
    except Exception as exc:  # pragma: no cover - 依赖上游源码
        raise SystemExit(
            f"无法从 --cain-dir={cain_dir} 导入 CAIN,请确认已 clone myungsub/CAIN 并"
            "提供正确路径(目录需含 model.py)。原始错误: " + repr(exc)) from exc

    net = CAIN(depth=depth)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)} "
              "(strict=False; 若数量偏大请检查 depth 与权重是否匹配)")
    net.eval().to(device)
    return net


def _export_one(module, onnx_path, dummy_inputs, input_names, output_names, opset):
    """Export a PyTorch module to ONNX (legacy exporter, fixed shapes).

    必须用 dynamo=False(legacy), 否则 torch 2.9 默认 dynamo 导出器产生的图
    converter_lite ACL 无法降级(NULL ptr)。固定 shape 配合 ascend_oriented。
    """
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
    parser = argparse.ArgumentParser(description="Export CAIN to a single ONNX.")
    parser.add_argument("--cain-dir", type=str, default="./CAIN",
                        help="上游 myungsub/CAIN 源码目录(含 model.py)")
    parser.add_argument("--ckpt", type=str, default="./pretrained_CAIN.pth",
                        help="预训练权重路径(如 pretrained_CAIN.pth)")
    parser.add_argument("--output-dir", type=str, default="./cain_onnx")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--height", type=int, default=256,
                        help="固定输入高(须可被 2^(depth+1) 整除)")
    parser.add_argument("--width", type=int, default=256,
                        help="固定输入宽(须可被 2^(depth+1) 整除)")
    parser.add_argument("--depth", type=int, default=3, help="CAIN depth(须与权重一致)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    net = _load_model(args.cain_dir, args.ckpt, args.depth, args.device)
    wrapper = CainInterp(net).to(args.device).eval()

    h, w = int(args.height), int(args.width)
    div = 2 ** (int(args.depth) + 1)
    if h % div != 0 or w % div != 0:
        raise SystemExit(f"height/width 须可被 {div} 整除(当前 {h}x{w})")

    dummy0 = torch.randn(1, 3, h, w, dtype=torch.float32, device=args.device)
    dummy1 = torch.randn(1, 3, h, w, dtype=torch.float32, device=args.device)
    _export_one(
        wrapper, output_dir / "cain.onnx",
        (dummy0, dummy1),
        input_names=["img0", "img1"], output_names=["mid_frame"],
        opset=args.opset,
    )
    print("Export complete.")


if __name__ == "__main__":
    main()
