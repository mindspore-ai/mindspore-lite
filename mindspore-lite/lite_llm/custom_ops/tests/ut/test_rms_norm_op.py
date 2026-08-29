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
"""Host reference and ONNX -> OMC -> device tests for MsRmsNorm."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch
from torch.onnx import OperatorExportTypes


REPO_ROOT = Path(__file__).resolve().parents[2]

# pylint: disable=wrong-import-position  # test helpers resolve via sys.path
from base_test import TestCaseBasic, build_omg_environment
from binrunner_test import BinRunnerTestCaseBasic
from torch_custom.ms_rms_norm import MsRmsNorm


DeviceTestBase = (
    BinRunnerTestCaseBasic
    if os.environ.get("DEVICE_TRANSPORT", "hdc").strip().lower() == "binapp"
    else TestCaseBasic
)

# 精度契约：组合式容差 + 千分之一超标率门限
RTOL = 1.0e-3
MAX_FAIL_RATIO = 1.0e-3


def get_name(x_shape: list[int], w_shape: list[int]) -> str:
    return "_".join(map(str, x_shape + w_shape))


def rms_norm_reference(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """Golden：唯一真标杆来源 = torch_custom eager 参考实现（MsRmsNorm.apply）。"""
    x_t = torch.from_numpy(np.ascontiguousarray(x))
    w_t = torch.from_numpy(np.ascontiguousarray(w))
    return MsRmsNorm.apply(x_t, w_t, eps).numpy()


def test_reference_matches_independent_numpy() -> None:
    """test_reference_matches_independent_numpy: helper."""
    rng = np.random.default_rng(20260814)
    x = rng.uniform(-2.0, 2.0, (3, 32)).astype(np.float16)
    w = rng.uniform(0.5, 1.5, (32,)).astype(np.float16)
    actual = rms_norm_reference(x, w, 1.0e-5)
    x_f32 = x.astype(np.float32)
    expected = (
        x_f32
        / np.sqrt(np.mean(x_f32 * x_f32, axis=-1, keepdims=True) + 1.0e-5)
        * w.astype(np.float32)
    ).astype(np.float16)
    np.testing.assert_array_equal(actual, expected)


def test_reference_supports_optional_gamma() -> None:
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float16).reshape(2, 32)
    actual = MsRmsNorm.apply(torch.from_numpy(x), None, 1.0e-6).numpy()
    x_f32 = x.astype(np.float32)
    expected = (
        x_f32
        / np.sqrt(np.mean(x_f32 * x_f32, axis=-1, keepdims=True) + 1.0e-6)
    ).astype(np.float16)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("hidden", [896, 2048, 2560, 4096, 8192])
def test_reference_accepts_model_hidden_sizes(hidden: int) -> None:
    x = torch.ones((1, hidden), dtype=torch.float16)
    w = torch.ones((hidden,), dtype=torch.float16)
    assert MsRmsNorm.apply(x, w, 1.0e-5).shape == x.shape


def test_reference_rejects_bad_contract() -> None:
    """test_reference_rejects_bad_contract: helper."""
    with pytest.raises(ValueError, match="multiple of 16"):
        MsRmsNorm.apply(
            torch.ones((2, 31), dtype=torch.float16),
            torch.ones((31,), dtype=torch.float16),
            1.0e-6,
        )
    with pytest.raises(ValueError, match="w must be"):
        MsRmsNorm.apply(
            torch.ones((2, 32), dtype=torch.float16),
            torch.ones((16,), dtype=torch.float16),
            1.0e-6,
        )
    with pytest.raises(TypeError, match="x must be float16"):
        MsRmsNorm.apply(torch.ones((2, 32)), None, 1.0e-6)
    with pytest.raises(ValueError, match="epsilon"):
        MsRmsNorm.apply(torch.ones((2, 32), dtype=torch.float16), None, -1.0)
    with pytest.raises(ValueError, match="<= 8192"):
        MsRmsNorm.apply(torch.ones((1, 8208), dtype=torch.float16), None, 1.0e-6)


def find_omg() -> str:
    """定位 OMG 可执行文件；找不到抛 EnvironmentError。"""
    ddk_path = os.environ.get("DDK_PATH", "")
    default_omg = Path(ddk_path) / "tools/tools_omg/master/omg" if ddk_path else None
    omg = os.environ.get("OMG_PATH")
    if not omg and default_omg and default_omg.is_file():
        omg = str(default_omg)
    if not omg:
        omg = shutil.which("omg")
    if not omg:
        raise EnvironmentError(
            "Cannot find OMG; set OMG_PATH or activate the DDK environment"
        )
    return omg


def omg_environment(platform: str) -> dict:
    return build_omg_environment(platform)


def export_ms_rms_norm_onnx(
    path: Path, x_shape: list[int], w_shape: list[int], eps: float, dtype=torch.float16
) -> None:
    """导出单节点 custom::MsRmsNorm ONNX（opset 18 / ONNX_FALLTHROUGH）。"""
    model = MsRmsNormModel(eps).eval()
    dummy_x = torch.zeros(x_shape, dtype=dtype)
    dummy_w = torch.ones(w_shape, dtype=dtype)
    torch.onnx.export(
        model,
        (dummy_x, dummy_w),
        path,
        opset_version=18,
        input_names=["x", "w"],
        output_names=["output"],
        do_constant_folding=True,
        operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
        custom_opsets={"custom": 1},
        dynamo=False,
    )


class MsRmsNormModel(torch.nn.Module):
    """Minimal PyTorch model containing one exportable MsRmsNorm call."""

    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return MsRmsNorm.apply(x, w, self.epsilon)


@pytest.mark.device
class TestRmsNorm(DeviceTestBase):
    """TestRmsNorm: internal helper type."""
    configs = [
        # Qwen2.5-0.5B: hidden_size=896, Decode S=1 and Chunked Prefill
        # S in {64, 128}; rms_norm_eps=1e-6.
        # Platform,     x_shape,       w_shape, eps,      atol
        ["kirin9020", [1, 1, 896], [896], 1.0e-6, 1.0e-2],
        ["kirin9020", [1, 64, 896], [896], 1.0e-6, 1.0e-2],
        ["kirin9020", [1, 128, 896], [896], 1.0e-6, 1.0e-2],
        # MiniMind-3 (Qwen3 dense): hidden_size=768, head_dim=96, kv_heads=4.
        # rank-3 input_layernorm: Decode S=1 / Chunked Prefill S=64.
        ["kirin9020", [1, 1, 768], [768], 1.0e-6, 1.0e-2],
        ["kirin9020", [1, 64, 768], [768], 1.0e-6, 1.0e-2],
        # rank-4 per-head Q/K RMSNorm (q_norm/k_norm): [1, S, heads, head_dim].
        ["kirin9020", [1, 1, 8, 96], [96], 1.0e-6, 1.0e-2],
        ["kirin9020", [1, 64, 8, 96], [96], 1.0e-6, 1.0e-2],
        # Existing Qwen2.5-3B hidden_size=2048 coverage; keep the explicit
        # epsilon=1e-5 override used by the original device cases.
        ["kirin9020", [128, 2048], [2048], 1.0e-5, 1.0e-2],
        ["kirin9020", [1, 2048], [2048], 1.0e-5, 1.0e-2],
    ]

    @pytest.mark.parametrize("platform,x_shape,w_shape,eps,atol", configs)
    def test_case(self, platform, x_shape, w_shape, eps, atol, ext_platform):
        target_platform = ext_platform or platform
        suffix = get_name(x_shape, w_shape)
        inputs = self.gen_data(x_shape, w_shape, eps, suffix)
        omc = self.gen_omc(x_shape, w_shape, eps, suffix, target_platform)
        self.upload(omc, inputs)
        self.exec(omc, inputs)
        self.validate(x_shape, w_shape, eps, atol, suffix)
        self.display_perf(suffix)

    def gen_data(
        self,
        x_shape: list[int],
        w_shape: list[int],
        eps: float,  # pylint: disable=unused-argument
        suffix: str,
    ) -> list[str]:
        """Gen data."""
        if w_shape != [x_shape[-1]]:
            raise ValueError(f"w_shape {w_shape} must equal [{x_shape[-1]}]")

        rng = np.random.default_rng(20260806)
        x = rng.uniform(-2.0, 2.0, x_shape).astype(np.float16)
        w = rng.uniform(0.5, 1.5, w_shape).astype(np.float16)
        x_name = f"x_{suffix}.bin"
        w_name = f"w_{suffix}.bin"
        x.tofile(self.local_path(x_name))
        w.tofile(self.local_path(w_name))
        return [x_name, w_name]

    def gen_omc(
        self,
        x_shape: list[int],
        w_shape: list[int],
        eps: float,  # pylint: disable=unused-argument
        suffix: str,
        platform: str,
    ) -> str:
        """Gen omc."""
        onnx_path = self.local_path(f"rms_norm_{suffix}.onnx")
        export_ms_rms_norm_onnx(onnx_path, x_shape, w_shape, eps)
        omc_stem = self.local_path(f"rms_norm_{suffix}")
        omc_path = omc_stem.with_suffix(".omc")

        omg = find_omg()
        command = [
            omg,
            f"--model={onnx_path}",
            "--framework=5",
            "--target=omc",
            f"--output={omc_stem}",
            f"--platform={platform}",
        ]
        print("$ " + " ".join(map(str, command)), flush=True)
        subprocess.run(command, check=True, env=omg_environment(platform))
        if not omc_path.is_file():
            raise FileNotFoundError(f"OMG did not generate {omc_path}")
        return omc_path.name

    def exec(self, omc: str, inputs: list[str]) -> None:
        """exec: helper."""
        remote_omc = self.remote_path(omc)
        if self.transport == "binapp":
            if len(inputs) != 2:
                raise ValueError(f"RmsNorm BinApp runner expects two inputs, got {inputs}")
            remote_x, remote_w = (self.remote_path(item) for item in inputs)
            remote_y = self.remote_path("y.bin")
            # model_run_tool 需要 --model/--input/--output 风格参数（非位置参数）
            command_line = " ".join(
                [
                    self.remote_model_run_tools,
                    "--model",
                    remote_omc,
                    "--input",
                    remote_x,
                    "--input",
                    remote_w,
                    "--output",
                    remote_y,
                ]
            )
            self.run_binrunner(
                "run",
                command_line,
                "--timeout",
                os.environ.get("BINAPP_RUN_TIMEOUT", "60"),
            )
            return

        remote_inputs = ",".join(self.remote_path(item) for item in inputs)
        model_input_flag = os.environ.get("MODEL_INPUT_FLAG", "--input").rstrip("=")
        perf_arguments = ["--enable_item=1"] if self.test_perf else []
        self.run_hdc(
            "shell",
            "env",
            f"LD_LIBRARY_PATH={self.model_run_ld_library_path}",
            self.remote_model_run_tools,
            "--model",
            remote_omc,
            f"{model_input_flag}={remote_inputs}",
            f"--output_dir={self.remote_dir}/",
            *perf_arguments,
        )
        self.run_hdc(
            "shell",
            "mv",
            f"{self.remote_dir}/output_0",
            f"{self.remote_dir}/y.bin",
        )

    def validate(
        self,
        x_shape: list[int],
        w_shape: list[int],
        eps: float,  # pylint: disable=unused-argument
        atol: float,
        suffix: str,
        rtol: float = RTOL,
        max_fail_ratio: float = MAX_FAIL_RATIO,
    ) -> None:
        """组合式容差校验 + 三件套输出（max_abs_diff / 超标数 / 超标率）。"""
        local_y = self.local_path(f"y_{suffix}.bin")
        self.download("y.bin", local_y)

        x = np.fromfile(self.local_path(f"x_{suffix}.bin"), dtype=np.float16).reshape(x_shape)
        w = np.fromfile(self.local_path(f"w_{suffix}.bin"), dtype=np.float16).reshape(w_shape)
        expect = rms_norm_reference(x, w, eps)

        y_raw = np.fromfile(local_y, dtype=np.float16)
        actual_size = int(np.prod(x_shape))
        if y_raw.size >= 2 * actual_size:
            actual = y_raw[:actual_size].reshape(x_shape)
        elif y_raw.size == actual_size:
            actual = y_raw.reshape(x_shape)
        else:
            raise AssertionError(
                f"Unexpected output size: got {y_raw.size}, "
                f"expected at least {actual_size} FP16 values"
            )

        actual_f32 = actual.astype(np.float32)
        expect_f32 = expect.astype(np.float32)
        diff = np.abs(actual_f32 - expect_f32)
        tolerance = atol + rtol * np.abs(expect_f32)
        failed = int(np.count_nonzero(diff > tolerance))
        fail_ratio = failed / actual_size
        max_abs_diff = float(diff.max(initial=0.0))
        print(
            f"RmsNorm suffix={suffix}: elements={actual_size}, "
            f"max_abs_diff={max_abs_diff:.8g}, failed={failed}, "
            f"fail_ratio={fail_ratio:.6g}, threshold={max_fail_ratio:g}, "
            f"tol=(atol={atol:g}, rtol={rtol:g})",
            flush=True,
        )
        assert fail_ratio < max_fail_ratio, (
            f"fail_ratio {fail_ratio:.6g} >= {max_fail_ratio:g} "
            f"(failed={failed}/{actual_size}, max_abs_diff={max_abs_diff:.8g})"
        )


# ---------------------------------------------------------------------------
# 校验负例（代表性格局 B）：构造非法 ONNX -> OMG 转换预期失败。
# 需要 OMG（DDK 环境），无 OMG 时 skip。不需要真机。
# ---------------------------------------------------------------------------

def _omg_available() -> bool:
    try:
        find_omg()
        return True
    except EnvironmentError:
        return False


def _write_raw_onnx(
    path: Path,
    x_shape: list[int],
    w_shape: list[int] | None,
    x_dtype: int,
) -> None:
    """Build a raw custom node so invalid Host contracts bypass eager checks."""
    import onnx
    from onnx import TensorProto, helper

    inputs = [helper.make_tensor_value_info("x", x_dtype, x_shape)]
    node_inputs = ["x"]
    if w_shape is not None:
        inputs.append(
            helper.make_tensor_value_info("w", TensorProto.FLOAT16, w_shape)
        )
        node_inputs.append("w")
    output = helper.make_tensor_value_info("output", x_dtype, x_shape)
    node = helper.make_node(
        "MsRmsNorm", node_inputs, ["output"], domain="custom", epsilon=1.0e-5
    )
    graph = helper.make_graph([node], "invalid_ms_rms_norm", inputs, [output])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom", 1)],
    )
    onnx.save(model, path)


@pytest.mark.skipif(not _omg_available(), reason="OMG not available (DDK environment required)")
def test_invalid_rank_rejected(tmp_path):
    """标量 x 必须被 Host shape 推断拒绝。"""
    from onnx import TensorProto

    onnx_path = tmp_path / "rms_norm_invalid_rank.onnx"
    _write_raw_onnx(onnx_path, [], [1], TensorProto.FLOAT16)
    omg = find_omg()
    command = [
        omg,
        f"--model={onnx_path}",
        "--framework=5",
        "--target=omc",
        f"--output={tmp_path / 'rms_norm_invalid_rank'}",
        "--platform=kirin9020",
    ]
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(command, check=True, env=omg_environment("kirin9020"))


@pytest.mark.skipif(not _omg_available(), reason="OMG not available (DDK environment required)")
def test_invalid_dtype_rejected(tmp_path):
    """fp32 x 必须被 host 校验拦截（InferDataType GRAPH_FAILED）。"""
    from onnx import TensorProto

    onnx_path = tmp_path / "rms_norm_invalid_dtype.onnx"
    _write_raw_onnx(onnx_path, [1, 2048], [2048], TensorProto.FLOAT)
    omg = find_omg()
    command = [
        omg,
        f"--model={onnx_path}",
        "--framework=5",
        "--target=omc",
        f"--output={tmp_path / 'rms_norm_invalid_dtype'}",
        "--platform=kirin9020",
    ]
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(command, check=True, env=omg_environment("kirin9020"))
