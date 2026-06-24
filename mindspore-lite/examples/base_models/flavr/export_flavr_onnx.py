#!/usr/bin/env python3
"""Export FLAVR (Flow-Agnostic VFI, 3D 卷积) to ONNX。

FLAVR 用 3D 卷积(UNet_3D_3D)处理 4 帧序列插帧(无光流)。上游 forward 接收帧
*列表*并 torch.stack, 本 wrapper 改为接收 tensor seq [B,4,3,H,W] 并内联复刻 forward
(joinType=concat), n_outputs=1 输出单中间帧。
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


class FlavrInterp(nn.Module):
    """FLAVR(UNet_3D_3D) 插帧封装:4 帧 seq -> 1 中间帧(内联复刻 forward)。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        m = self.model
        images = seq.permute(0, 2, 1, 3, 4)  # [B,N,C,H,W] -> [B,C,N,H,W](encoder 期望)
        mean_ = images.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)
        images = images - mean_
        x_0, x_1, x_2, x_3, x_4 = m.encoder(images)
        dx_3 = m.lrelu(m.decoder[0](x_4))
        dx_3 = torch.cat([dx_3, x_3], dim=1)
        dx_2 = m.lrelu(m.decoder[1](dx_3))
        dx_2 = torch.cat([dx_2, x_2], dim=1)
        dx_1 = m.lrelu(m.decoder[2](dx_2))
        dx_1 = torch.cat([dx_1, x_1], dim=1)
        dx_0 = m.lrelu(m.decoder[3](dx_1))
        dx_0 = torch.cat([dx_0, x_0], dim=1)
        dx_out = m.lrelu(m.decoder[4](dx_0))
        dx_out = torch.cat(torch.unbind(dx_out, 2), 1)  # 时间维并入通道 [B,64*T,H,W]
        out = m.lrelu(m.feature_fuse(dx_out))
        out = m.outconv(out)
        out = out[:, 0:3] + mean_.squeeze(2)  # 中间帧(n_outputs=1)
        return out


def _load_model(repo_dir, ckpt, model_file, block, n_inputs, n_outputs, device):
    """Build UNet_3D_3D with FLAVR args and load checkpoint."""
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    pkg, cls = model_file.rsplit(".", 1)
    try:
        mod = __import__(pkg, fromlist=[cls])  # noqa: WPS433
        model_cls = getattr(mod, cls)
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"无法从 --repo-dir={repo_dir} 导入 {model_file}: " + repr(exc)) from exc
    net = model_cls(block, n_inputs=n_inputs, n_outputs=n_outputs,
                    joinType="concat", upmode="transpose")
    if ckpt and Path(ckpt).exists():
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
    parser = argparse.ArgumentParser(description="Export FLAVR to ONNX.")
    parser.add_argument("--repo-dir", type=str, default="./flavr_src")
    parser.add_argument("--model-file", type=str, default="model.FLAVR_arch.UNet_3D_3D")
    parser.add_argument("--block", type=str, default="unet_18", help="resnet_3D encoder block 名")
    parser.add_argument("--ckpt", type=str, default="./flavr.pth")
    parser.add_argument("--n-inputs", type=int, default=4)
    parser.add_argument("--n-outputs", type=int, default=1, help="插值输出帧数(1=单中间帧)")
    parser.add_argument("--output-dir", type=str, default="./flavr_onnx")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--height", type=int, default=256, help="须被 16 整除(Conv3D 下采样)")
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()
    if args.height % 16 or args.width % 16:
        raise SystemExit("height/width 须可被 16 整除")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    net = _load_model(args.repo_dir, args.ckpt, args.model_file, args.block,
                      args.n_inputs, args.n_outputs, args.device)
    wrapper = FlavrInterp(net).to(args.device).eval()
    seq = torch.randn(1, args.n_inputs, 3, args.height, args.width,
                      dtype=torch.float32, device=args.device)
    _export_one(wrapper, output_dir / "flavr.onnx", (seq,),
                input_names=["seq"], output_names=["mid_frame"], opset=args.opset)
    print("Export complete.")


if __name__ == "__main__":
    main()
