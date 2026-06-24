#!/usr/bin/env python3
"""Export EDVR-M (Video Restoration, DCNv1 + 时空注意力, M=中等规模) to ONNX。

EDVR 用 deformable conv 对齐 + 时空注意力做视频超分/修复。
封装 ``forward(lr_seq) -> sr_frame``,lr_seq 为 [B, N, 3, H, W],输出 4x 超分中心帧。

⚠️ 重要: deformable conv(DCNv1)目前 converter_lite 不原生支持,Phase 2 需自定义
AscendC 算子或导出侧等价改写,否则转换会受阻。EDVR-L 同源仅换规模。
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


class EdvrMSR(nn.Module):
    """EDVR-M 超分封装(含 DCNv1 对齐)。"""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.model = net

    def forward(self, lr_seq: torch.Tensor) -> torch.Tensor:
        out = self.model(lr_seq)
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
    parser = argparse.ArgumentParser(description="Export EDVR-M to ONNX.")
    parser.add_argument("--repo-dir", type=str, default="./mmagic_src")
    parser.add_argument("--model-file", type=str, default="mmagic.models.edvr_net.EDVRNet")
    parser.add_argument("--ckpt", type=str, default="./edvr_m_x4.pth")
    parser.add_argument("--output-dir", type=str, default="./edvr_m_onnx")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--num-frames", type=int, default=7)
    parser.add_argument("--lr-height", type=int, default=64)
    parser.add_argument("--lr-width", type=int, default=64)
    args = parser.parse_args()
    if args.lr_height % 4 or args.lr_width % 4:
        raise SystemExit("lr-height/lr-width 须可被 4 整除")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    net = _load_model(args.repo_dir, args.ckpt, args.model_file, args.device)
    wrapper = EdvrMSR(net).to(args.device).eval()
    lr = torch.randn(1, args.num_frames, 3, args.lr_height, args.lr_width,
                     dtype=torch.float32, device=args.device)
    _export_one(wrapper, output_dir / "edvr_m.onnx", (lr,),
                input_names=["lr_seq"], output_names=["sr_frame"], opset=args.opset)
    print("Export complete. (注意: DCNv1 需 Phase 2 自定义算子才能转 MindIR)")


if __name__ == "__main__":
    main()
