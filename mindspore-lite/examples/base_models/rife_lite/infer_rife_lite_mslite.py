#!/usr/bin/env python3
"""RIFE_LITE MindSpore Lite 推理脚本(MindIR)。

严格遵守 SKILL 规范: 不引入 torch 依赖、不在代码内设置 precision_mode。
内置内存预算守护: 建模型前检查 RAM/HBM, 任一已用 >80% 则告警退出。
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


# 内存预算守护(<80% 硬约束) -------------------------------------------------
def _ram_usage():
    info = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                key, _, val = line.partition(":")
                info[key.strip()] = int(val.strip().split()[0])
    except Exception:
        return None, None, None
    total = info.get("MemTotal", 0)
    avail = info.get("MemAvailable", 0)
    if total <= 0:
        return None, None, None
    used = total - avail
    return total // 1024, used // 1024, round(used / total * 100.0, 1)


def _npu_hbm_for_device(device_id):
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None, None
    rows = [ln for ln in out.splitlines() if re.search(r"\d{4}:\d{2}:\d{2}\.\d", ln)]
    if int(device_id) >= len(rows):
        return None, None
    m = re.search(r"(\d+)\s*/\s*(\d+)", rows[int(device_id)])
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _check_memory_budget(device_id, limit_pct=80):
    total, used, pct = _ram_usage()
    print("[memory-budget] RAM:", end=" ")
    print(f"used {used}/{total} MB ({pct}%)" if total is not None else "unavailable")
    hu, ht = _npu_hbm_for_device(int(device_id))
    hpct = round(hu / ht * 100.0, 1) if ht else None
    if ht:
        print(f"[memory-budget] NPU{device_id} HBM: used {hu}/{ht} MB ({hpct}%)")
    else:
        print(f"[memory-budget] NPU{device_id} HBM: unavailable")
    over = []
    if pct is not None and pct > limit_pct:
        over.append(f"RAM {pct}%")
    if hpct is not None and hpct > limit_pct:
        over.append(f"HBM {hpct}%")
    if over:
        print("[memory-budget] !!! 超出 80% 预算: " + "; ".join(over) + "。请释放内存或降分辨率。")
        return False
    return True


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


# 推理 ---------------------------------------------------------------------
def _load_image(path, height, width):
    img = Image.open(path).convert("RGB").resize((int(width), int(height)), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...]


class RifeLiteMsliteInferencer:
    """RIFE_LITE MindSpore Lite 推理封装。"""

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
    p = argparse.ArgumentParser(description="RIFE_LITE MindSpore Lite 推理(MindIR)")
    p.add_argument("--mindir", type=str, required=True)
    p.add_argument("--img0", type=str, required=True)
    p.add_argument("--img1", type=str, required=True)
    p.add_argument("--output", type=str, default="./rife_lite_mid_mslite.png")
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

    infer = RifeLiteMsliteInferencer(args.mindir, device=args.device, device_id=args.device_id)
    x0 = _load_image(args.img0, args.height, args.width)
    x1 = _load_image(args.img1, args.height, args.width)

    for _ in range(int(args.warmup)):
        _ = infer.forward(x0, x1)
    lat = []
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        out = infer.forward(x0, x1)
        lat.append((time.perf_counter() - t0) * 1000.0)
    peak = _proc_peak()

    out = np.clip(out[0], 0.0, 1.0)
    Image.fromarray((np.transpose(out, (1, 2, 0)) * 255.0 + 0.5).astype(np.uint8)).save(args.output)
    lat_np = np.array(lat, dtype=np.float32)
    print(f"[mslite] saved -> {args.output}")
    print(f"  latency_ms_mean: {float(lat_np.mean()):.3f} p50: {float(np.percentile(lat_np, 50)):.3f}")
    print(f"  proc_rss_mb: {peak['rss_mb']} (hwm={peak['hwm_mb']})")


if __name__ == "__main__":
    main()
