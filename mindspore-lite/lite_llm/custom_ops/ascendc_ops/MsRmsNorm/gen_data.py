#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Generate deterministic FP16 inputs and the torch_custom-backed RMSNorm golden.

Golden 唯一真标杆来源：``torch_custom.ms_rms_norm.MsRmsNorm.apply``（eager 参考
实现）。本脚本只负责确定性输入采样与落盘，数学计算全部委托给 torch_custom，
避免多套参考实现各自漂移。
"""

import json
from pathlib import Path
import sys

import numpy as np

# 仓库根 = ascendc_ops/MsRmsNorm/ 上两级（torch_custom 在仓库根下）
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def rms_norm_reference(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """Golden：委托给 torch_custom 的 MsRmsNorm eager 参考实现（唯一真源）。"""
    import torch

    from torch_custom.ms_rms_norm import MsRmsNorm

    x_t = torch.from_numpy(np.ascontiguousarray(x))
    w_t = torch.from_numpy(np.ascontiguousarray(w))
    return MsRmsNorm.apply(x_t, w_t, eps).numpy()


def generate(config_path: Path, output_dir: Path) -> None:
    """Generate deterministic FP16 inputs and the golden output from config."""
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    x_shape = tuple(config["inputs"][0]["shape"])
    w_shape = tuple(config["inputs"][1]["shape"])
    eps = float(config["attrs"]["eps"])
    if w_shape != (x_shape[-1],):
        raise ValueError(f"w shape {w_shape} must equal the last x dimension {(x_shape[-1],)}")

    rng = np.random.default_rng(20260805)
    x = rng.uniform(-2.0, 2.0, x_shape).astype(np.float16)
    w = rng.uniform(0.5, 1.5, w_shape).astype(np.float16)
    y = rms_norm_reference(x, w, eps)

    output_dir.mkdir(parents=True, exist_ok=True)
    x.tofile(output_dir / config["inputs"][0]["data_file"])
    w.tofile(output_dir / config["inputs"][1]["data_file"])
    y.tofile(output_dir / config["outputs"][0]["data_file"])
    print(
        f"generated x={x.shape} w={w.shape} eps={eps:g} "
        f"output={output_dir} elements={y.size}"
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} CONFIG_JSON OUTPUT_DIR")
    generate(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
