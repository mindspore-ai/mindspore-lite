# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Test load from buffer
"""

from typing import Tuple, List, Dict
from dataclasses import dataclass, replace
from pathlib import Path
import subprocess
import pytest
import mindspore_lite as mslite
import numpy as np
from utils import expect_error


@dataclass
class ModelArgs:
    model_path: str
    weight_path: str = None
    model_type: str = mslite.ModelType.MINDIR
    device: str = "ascend"
    use_ge: bool = False
    config: dict = None
    device_id: int = None
    inputs: Tuple[np.ndarray] = None


def _prepare_build_context(context, model_build_args: ModelArgs):
    context.target = [model_build_args.device]
    if model_build_args.device == "ascend":
        if model_build_args.use_ge:
            context.ascend.provider = "ge"
        if model_build_args.device_id is not None:
            context.ascend.device_id = model_build_args.device_id


def build_model_from_file(model_build_args: ModelArgs):
    """
    build model
    """
    context = mslite.Context()
    model = mslite.Model()

    _prepare_build_context(context, model_build_args)

    model.build_from_file(
        model_build_args.model_path,
        model_build_args.model_type,
        context,
        config_dict=model_build_args.config,
    )
    return model


def build_model_from_buffer(model_build_args: ModelArgs):
    """
    build model
    """
    context = mslite.Context()
    model = mslite.Model()

    _prepare_build_context(context, model_build_args)

    with open(model_build_args.model_path, "rb") as f:
        model_bytes = f.read()

    weight_bytes = None
    if model_build_args.weight_path is not None:
        with open(model_build_args.weight_path, "rb") as f:
            weight_bytes = f.read()

    model.build_from_buffer(
        model_bytes,
        weight_bytes,
        model_build_args.model_type,
        context,
        config_dict=model_build_args.config,
    )
    return model


def _fill_model_build_args(obj, output_dir, device_id):
    """
    prepare model build args
    """
    model_path = obj.model_path.format(output_dir=output_dir)
    weight_path = obj.weight_path.format(output_dir=output_dir) if obj.weight_path is not None else None
    return replace(obj, model_path=model_path, weight_path=weight_path, device_id=device_id)


@pytest.fixture(scope="module", autouse=True)
def module_setup_and_teardown_fixture(so_path, mindir_dir, output_dir):
    """Convert bert_model.onnx if needed (skip when already present)."""
    acl_output = Path(output_dir) / "bert_model.onnx.mindir"
    cpu_output = Path(output_dir) / "bert_model.onnx.cpu.mindir"
    if acl_output.exists() and cpu_output.exists():
        empty_mindir = Path(output_dir) / "emptyfile"
        empty_mindir.unlink(missing_ok=True)
        empty_mindir.touch()
        yield
        return

    fmk = "ONNX"
    model_path = Path(mindir_dir) / "bert_model.onnx"
    acl_output_path = Path(output_dir) / "bert_model.onnx"
    acl_optimize = "ascend_oriented"
    acl_cmd = [
        Path(so_path) / "tools/converter/converter/converter_lite",
        f"--optimize={acl_optimize}",
        f"--modelFile={model_path}",
        f"--outputFile={acl_output_path}",
        f"--fmk={fmk}",
    ]
    subprocess.run(acl_cmd, check=True)

    cpu_output_path = Path(output_dir) / "bert_model.onnx.cpu"
    cpu_optimize = "general"
    cpu_cmd = [
        Path(so_path) / "tools/converter/converter/converter_lite",
        f"--optimize={cpu_optimize}",
        f"--modelFile={model_path}",
        f"--outputFile={cpu_output_path}",
        f"--fmk={fmk}",
    ]
    subprocess.run(cpu_cmd, check=True)
    empty_mindir = Path(output_dir) / "emptyfile"
    empty_mindir.unlink(missing_ok=True)
    empty_mindir.touch()
    yield


@pytest.mark.parametrize(
    "args",
    (
        ModelArgs(
            "{output_dir}/sd1.5_unet.onnx_graph.mindir",
            "{output_dir}/sd1.5_unet.onnx_variables/data_0",
            inputs=(
                np.ones((2, 4, 64, 64)).astype(np.float32),
                np.ones((1,)).astype(np.float32),
                np.ones((2, 77, 768)).astype(np.float32),
            ),
        ),
        ModelArgs(
            "{output_dir}/bert_model.onnx.mindir",
            None,
            inputs=(
                np.ones((1, 128)).astype(np.int32),
                np.ones((1, 128)).astype(np.int32),
                np.ones((1, 128)).astype(np.int32),
            ),
        ),
        ModelArgs(
            "{output_dir}/bert_model.onnx.mindir",
            "{output_dir}/sd1.5_unet.onnx_variables/data_0",
            inputs=(
                np.ones((1, 128)).astype(np.int32),
                np.ones((1, 128)).astype(np.int32),
                np.ones((1, 128)).astype(np.int32),
            ),
        ),
    ),
)
def test_build_from_buffer_correct(args: ModelArgs, output_dir: str, device_id: List[int]):
    """
    test model build form buffer
    """
    model_build_args = _fill_model_build_args(args, output_dir, device_id[0])

    assert model_build_args.inputs is not None

    model_from_file = build_model_from_file(model_build_args)
    model_from_buffer = build_model_from_buffer(model_build_args)

    model_input = [mslite.Tensor(tensor=i, device=f"ascend:{device_id[0]}") for i in model_build_args.inputs]

    output_file = model_from_file.predict(model_input)
    output_buffer = model_from_buffer.predict(model_input)

    for of, ob in zip(output_file, output_buffer):
        np.testing.assert_allclose(of.get_data_to_numpy(), ob.get_data_to_numpy())


@pytest.mark.parametrize(
    "args,error_type,msg",
    (
        (
            ModelArgs(
                "{output_dir}/sd1.5_unet.onnx_graph.mindir",
                None,
            ),
            RuntimeError,
            "build_from_buffer failed! Error is func_graph is nullptr, failed to load MindIR model!",
        ),
        (
            ModelArgs(
                "{output_dir}/sd1.5_unet.onnx_graph.mindir",
                "{output_dir}/emptyfile",
            ),
            RuntimeError,
            "build_from_buffer failed! Error is func_graph is nullptr, failed to load MindIR model!",
        ),
        (
            ModelArgs(
                "{output_dir}/emptyfile",
                None,
            ),
            RuntimeError,
            "build_from_buffer failed, model_bytes is empty.",
        ),
    ),
)
def test_build_from_buffer_lack_weight(
    args: ModelArgs, error_type: Exception, msg: str, output_dir: str, device_id: List[int]
):
    """
    test model build form buffer. lack weight
    """
    model_build_args = _fill_model_build_args(args, output_dir, device_id[0])

    with expect_error(error_type) as exec_info:
        build_model_from_buffer(model_build_args)
    assert msg in str(exec_info.value)


@pytest.mark.parametrize(
    "build_args,error_type,msg",
    (
        # model_bytes
        ({"model_bytes": None}, TypeError, "model_bytes must be bytes"),
        ({"model_bytes": bytes()}, RuntimeError, "build_from_buffer failed, model_bytes is empty."),
        # weight_bytes
        ({"model_bytes": b"0", "weight_bytes": str()}, TypeError, "weight_bytes must be bytes"),
        # model_type
        (
            {"model_bytes": b"0"},
            TypeError,
            "model_type must be ModelType",
        ),
        (
            {"model_bytes": b"0", "model_type": mslite.ModelType.MINDIR_LITE},
            RuntimeError,
            "build_from_buffer failed, model_type should be MINDIR",
        ),
        # context
        (
            {"model_bytes": b"0", "model_type": mslite.ModelType.MINDIR, "context": str()},
            TypeError,
            "context must be Context",
        ),
        # config_path
        (
            {"model_bytes": b"0", "model_type": mslite.ModelType.MINDIR, "config_path": 1},
            TypeError,
            "config_path must be str",
        ),
        # config_dict
        (
            {"model_bytes": b"0", "model_type": mslite.ModelType.MINDIR, "config_dict": 1},
            TypeError,
            "config_dict must be dict",
        ),
    ),
)
def test_build_from_buffer_arg_type(
    build_args: Dict, error_type: Exception, msg: str, output_dir: str, device_id: List[int]
):
    """
    test model build form buffer. check args type.
    """
    args = ModelArgs(str())
    model_build_args = _fill_model_build_args(args, output_dir, device_id[0])

    with expect_error(error_type) as exec_info:
        context = mslite.Context()
        model = mslite.Model()

        _prepare_build_context(context, model_build_args)

        model.build_from_buffer(**build_args)

    assert msg in str(exec_info.value)
