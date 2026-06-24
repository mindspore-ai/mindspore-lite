#!/usr/bin/env python3
"""CAIN MindSpore Lite 推理脚本: 输入两帧, 输出中间帧(MindIR)。

严格遵守 SKILL 规范: 不引入 torch 依赖、不在代码内设置 precision_mode(由 config.ini 控制)。
内置内存预算守护: 建模型前检查 RAM/HBM, 任一已用 >80% 则告警退出, 避免逼近内存上限。
"""

import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image

try:
    import mindspore_lite as mslite  # type: ignore
except Exception:
    mslite = None


# ---------------------------------------------------------------------------
# 内存预算守护(<80% 硬约束)
# ---------------------------------------------------------------------------
def _ram_usage():
    """Return (total_mb, used_mb, used_pct) of system RAM from /proc/meminfo."""
    info = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                key, _, val = line.partition(":")
                info[key.strip()] = int(val.strip().split()[0])  # kB
    except Exception:
        return None, None, None
    total = info.get("MemTotal", 0)
    avail = info.get("MemAvailable", 0)
    if total <= 0:
        return None, None, None
    used = total - avail
    return total // 1024, used // 1024, round(used / total * 100.0, 1)


def _npu_hbm_for_device(device_id):
    """Best-effort parse of npu-smi Memory-Usage(MB) for the given device."""
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return None, None
    rows = [ln for ln in out.splitlines() if re.search(r"\d{4}:\d{2}:\d{2}\.\d", ln)]
    idx = int(device_id)
    if idx >= len(rows):
        return None, None
    m = re.search(r"(\d+)\s*/\s*(\d+)", rows[idx])
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))  # used_mb, total_mb


def _check_memory_budget(device_id, limit_pct=80):
    """Print RAM/HBM report; return True if usage within budget."""
    ram_total, ram_used, ram_pct = _ram_usage()
    print("[memory-budget] RAM:", end=" ")
    if ram_total is not None:
        print(f"used {ram_used}/{ram_total} MB ({ram_pct}%)")
    else:
        print("unavailable")
    hbm_used, hbm_total = (None, None)
    if str(device_id).lower() == "ascend" or True:
        hbm_used, hbm_total = _npu_hbm_for_device(_resolve_dev_id(device_id))
    if hbm_total:
        hbm_pct = round(hbm_used / hbm_total * 100.0, 1)
        print(f"[memory-budget] NPU{device_id} HBM: used {hbm_used}/{hbm_total} MB ({hbm_pct}%)")
    else:
        hbm_pct = None
        print(f"[memory-budget] NPU{device_id} HBM: unavailable(npu-smi 解析失败)")
    over = []
    if ram_pct is not None and ram_pct > limit_pct:
        over.append(f"RAM {ram_pct}% > {limit_pct}%")
    if hbm_pct is not None and hbm_pct > limit_pct:
        over.append(f"HBM {hbm_pct}% > {limit_pct}%")
    if over:
        print("[memory-budget] !!! 超出 80% 预算: " + "; ".join(over)
              + "。请先释放内存或降低帧数/分辨率后再运行。")
        return False
    return True


def _resolve_dev_id(device_id):
    """Coerce device_id to int for npu-smi lookup."""
    try:
        return int(device_id)
    except (TypeError, ValueError):
        return 0


def _read_proc_status_mb(key):
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(key + ":"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) // 1024
    except Exception:
        return None
    return None


def _proc_peak():
    return {"rss_mb": _read_proc_status_mb("VmRSS"), "hwm_mb": _read_proc_status_mb("VmHWM")}


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
def _load_image(path, height, width):
    """Load image, resize to (width,height), return float32 [1,3,H,W] in [0,1]."""
    img = Image.open(path).convert("RGB").resize((int(width), int(height)), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...]


class CainMsliteInferencer:
    """CAIN MindSpore Lite 推理封装。"""

    def __init__(self, mindir_path, device="ascend", device_id=0):
        if mslite is None:
            raise RuntimeError("mindspore_lite 未安装")
        self.context = mslite.Context()
        self.context.target = [device]
        if device == "ascend":
            self.context.ascend.device_id = int(device_id)
        self.model = mslite.Model()
        self.model.build_from_file(mindir_path, mslite.ModelType.MINDIR, self.context)

    def forward(self, img0, img1):
        out = self.model.predict([mslite.Tensor(img0), mslite.Tensor(img1)])
        return out[0].get_data_to_numpy()


def _parse_args():
    p = argparse.ArgumentParser(description="CAIN MindSpore Lite 推理(MindIR)")
    p.add_argument("--mindir", type=str, required=True, help="cain.mindir 路径")
    p.add_argument("--img0", type=str, required=True)
    p.add_argument("--img1", type=str, required=True)
    p.add_argument("--output", type=str, default="./cain_mid_mslite.png")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--device", type=str, default="ascend", choices=["cpu", "ascend"])
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=10)
    return p.parse_args()


def main():
    args = _parse_args()
    for key in ("img0", "img1", "mindir"):
        if not os.path.exists(getattr(args, key)):
            raise FileNotFoundError(getattr(args, key))

    if not _check_memory_budget(args.device_id, limit_pct=80):
        sys.exit(2)

    infer = CainMsliteInferencer(args.mindir, device=args.device, device_id=args.device_id)
    x0 = _load_image(args.img0, args.height, args.width)
    x1 = _load_image(args.img1, args.height, args.width)

    for _ in range(int(args.warmup)):
        _ = infer.forward(x0, x1)
    lat = []
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        mid = infer.forward(x0, x1)
        lat.append((time.perf_counter() - t0) * 1000.0)
    peak = _proc_peak()

    mid = np.clip(mid[0], 0.0, 1.0)
    Image.fromarray((np.transpose(mid, (1, 2, 0)) * 255.0 + 0.5).astype(np.uint8)).save(args.output)

    lat_np = np.array(lat, dtype=np.float32)
    print(f"[mslite] saved mid frame -> {args.output}")
    print("Perf:")
    print(f"  input: {args.height}x{args.width}, warmup={args.warmup} runs={args.runs}")
    print(f"  latency_ms_mean: {float(lat_np.mean()):.3f}")
    print(f"  latency_ms_p50:  {float(np.percentile(lat_np, 50)):.3f}")
    print(f"  latency_ms_p90:  {float(np.percentile(lat_np, 90)):.3f}")
    print(f"  proc_rss_mb: {peak['rss_mb']} (hwm={peak['hwm_mb']})")


if __name__ == "__main__":
    main()
