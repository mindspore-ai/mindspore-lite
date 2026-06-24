#!/usr/bin/env python3
"""GR-1 policy MindSpore Lite inference (MindIR): video -> action chunk."""

import argparse
import os
import time

import numpy as np

try:
    import mindspore_lite as mslite  # type: ignore
except Exception:
    mslite = None


def _read_proc_status_kb(key):
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(key + ":"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:
        return None
    return None


def _memory_snapshot():
    rss = _read_proc_status_kb("VmRSS")
    hwm = _read_proc_status_kb("VmHWM")
    return {"vmrss_gb": round(rss / 1048576, 3) if rss else None,
            "vmhwm_gb": round(hwm / 1048576, 3) if hwm else None}


def _build_model(model_path, device="cpu", device_id=0):
    if mslite is None:
        raise RuntimeError("mindspore_lite not installed.")
    if device not in ("cpu", "ascend"):
        raise ValueError("device must be cpu or ascend")
    context = mslite.Context()
    context.target = [device]
    if device == "ascend":
        context.ascend.device_id = int(device_id)
    model = mslite.Model()
    model.build_from_file(model_path, mslite.ModelType.MINDIR, context)
    return model, model.get_inputs()


_MS_DTYPE = {
    "FLOAT32": np.float32, "FLOAT16": np.float16, "FLOAT64": np.float64,
    "INT32": np.int32, "INT64": np.int64, "INT16": np.int16, "INT8": np.int8,
    "UINT8": np.uint8, "UINT32": np.uint32, "UINT64": np.uint64, "BOOL": np.bool_,
}


def _run_model(model, inputs, feed):
    name_to_tensor = {t.name: t for t in inputs}
    if set(feed.keys()) != set(name_to_tensor.keys()):
        raise RuntimeError(
            f"Input name mismatch. Model expects {sorted(name_to_tensor.keys())}, "
            f"got {sorted(feed.keys())}.")
    ordered = []
    for t in inputs:
        arr = np.ascontiguousarray(feed[t.name])
        target = _MS_DTYPE.get(getattr(t.dtype, "name", str(t.dtype)), np.float32)
        if arr.dtype != target:
            arr = arr.astype(target)
        ordered.append(mslite.Tensor(arr))
    outputs = model.predict(ordered)
    return [o.get_data_to_numpy() for o in outputs]


def _load_video(path, num_frames, img_size):
    from PIL import Image
    frames = []
    for fn in sorted(os.listdir(path))[:num_frames]:
        img = Image.open(os.path.join(path, fn)).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
        frames.append(np.transpose(arr, (2, 0, 1)))
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.stack(frames, axis=0)[None, :].astype(np.float32)


def _parse_args():
    p = argparse.ArgumentParser(description="GR-1 policy MindSpore Lite inference (MindIR)")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--video-dir", type=str, default="")
    p.add_argument("--num-frames", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="ascend", choices=["cpu", "ascend"])
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=20)
    return p.parse_args()


def main():
    args = _parse_args()
    if mslite is None:
        raise RuntimeError("mindspore_lite not installed.")
    model, inputs = _build_model(args.model, device=args.device, device_id=args.device_id)

    rng = np.random.default_rng(args.seed)
    video = (_load_video(args.video_dir, args.num_frames, args.img_size)
             if args.video_dir and os.path.isdir(args.video_dir)
             else rng.standard_normal((1, args.num_frames, 3, args.img_size, args.img_size)).astype(np.float32))

    feed = {"video": video}
    mem_before = _memory_snapshot()
    for _ in range(int(args.warmup)):
        _run_model(model, inputs, feed)
    lat = []
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        outs = _run_model(model, inputs, feed)
        lat.append((time.perf_counter() - t0) * 1000.0)
    mem_after = _memory_snapshot()

    action = outs[0]
    print("Output:")
    print(f"  action shape={action.shape} dtype={action.dtype}")
    print(f"  action[0,0]={np.array2string(action[0, 0], precision=4)}")
    print(f"  action_abs_max={float(np.abs(action).max()):.6f}")

    lat_np = np.asarray(lat, dtype=np.float32)
    print("Perf:")
    print(f"  video: {video.shape}  device: {args.device}")
    print(f"  warmup: {args.warmup}  runs: {args.runs}")
    print(f"  latency_ms_mean: {float(lat_np.mean()):.4f}")
    print(f"  latency_ms_p50:  {float(np.percentile(lat_np, 50)):.4f}")
    print(f"  latency_ms_p90:  {float(np.percentile(lat_np, 90)):.4f}")
    print(f"  mem_before: {mem_before}")
    print(f"  mem_after:  {mem_after}")


if __name__ == "__main__":
    main()
