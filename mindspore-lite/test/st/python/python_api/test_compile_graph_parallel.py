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
Test lite python API.
Compile graph parallel config test.
"""
from itertools import product, chain
from dataclasses import dataclass, replace
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import pytest
import mindspore_lite as mslite
from utils import ScopeTimeRecord

PARALLEL_CONFIG_DICT_ON = {
    "common_context": {"compile_graph_parallel": "on"},
}

PARALLEL_CONFIG_DICT_OFF = {
    "common_context": {"compile_graph_parallel": "off"},
}


@dataclass
class ModelBuildArgs:
    path: str
    model_type: str
    device: str
    use_ge: bool = False
    config: dict = None
    device_id: int = None


def build_model(model_build_args: ModelBuildArgs):
    """
    build model
    """
    try:
        context = mslite.Context()
        model = mslite.Model()
        context.target = [model_build_args.device]
        if model_build_args.device == "ascend":
            if model_build_args.use_ge:
                context.ascend.provider = "ge"
            if model_build_args.device_id is not None:
                context.ascend.device_id = model_build_args.device_id
        model.build_from_file(
            str(model_build_args.path),
            model_build_args.model_type,
            context,
            config_dict=model_build_args.config,
        )
    except (RuntimeError, TypeError) as e:
        print(e)
        return False
    return True


def build_in_loop(loop, *args, **kwargs):
    for _ in range(loop):
        assert build_model(*args, **kwargs)
    return True


@pytest.fixture(scope="module", autouse=True)
def module_setup_and_teardown_fixture(so_path, mindir_dir, output_dir):
    """Convert bert_model.onnx if needed (skip when already present)."""
    acl_output = Path(output_dir) / "bert_model.onnx.mindir"
    cpu_output = Path(output_dir) / "bert_model.onnx.cpu.mindir"
    if acl_output.exists() and cpu_output.exists():
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
    yield


@pytest.mark.parametrize(
    "model_build_args",
    (
        ModelBuildArgs("{output_dir}/bert_model.onnx.mindir", mslite.ModelType.MINDIR, "ascend", False),
        ModelBuildArgs("{output_dir}/bert_model.onnx.cpu.mindir", mslite.ModelType.MINDIR, "cpu", False),
    ),
)
@pytest.mark.backend("mslite_large_model_inference_arm_ascend910B")
def test_compile_graph_parallel_speedup(
    model_build_args, output_dir, device_id, parallel=2, thread_pool_timeout=300, time_threshold=0.8
):
    """
    test speedup
    """
    model_build_args = replace(
        model_build_args, path=model_build_args.path.format(output_dir=output_dir), device_id=device_id[0]
    )
    pool = ThreadPoolExecutor(max_workers=parallel)
    with ScopeTimeRecord() as record_disable_parallel:
        assert build_in_loop(parallel, replace(model_build_args, config=PARALLEL_CONFIG_DICT_OFF))

    with ScopeTimeRecord() as record_enable_parallel:
        tasks = [
            pool.submit(build_model, replace(model_build_args, config=PARALLEL_CONFIG_DICT_ON)) for i in range(parallel)
        ]
        assert all(task.result() for task in as_completed(tasks, timeout=thread_pool_timeout))

    print(f"build time disable compile_graph_parallel: {record_disable_parallel.duration} ms")
    print(f"build time enable compile_graph_parallel: {record_enable_parallel.duration} ms")
    assert record_enable_parallel.duration < record_disable_parallel.duration * time_threshold


def product_case(test_case):
    """
    Generate permutation parallel config for models.
    For n models, 2^n configurations will be generated.
    For example, 2 models, configurations will be [off, off], [off, on], [on, off], [on, on]
    """
    model_num = len(test_case)
    product_args = [[PARALLEL_CONFIG_DICT_OFF, PARALLEL_CONFIG_DICT_ON]] * model_num

    return [
        [replace(model, config=deepcopy(c)) for model, c in zip(test_case, config)] for config in product(*product_args)
    ]


def prepare_different_provider_args():
    """
    Generate cases.
    """
    args = (
        (
            ModelBuildArgs("{output_dir}/sd1.5_unet.onnx_graph.mindir", mslite.ModelType.MINDIR, "ascend", False),
            ModelBuildArgs("{output_dir}/bert_model.onnx.cpu.mindir", mslite.ModelType.MINDIR, "cpu", False),
        ),
        (
            ModelBuildArgs("{output_dir}/bert_model.onnx.mindir", mslite.ModelType.MINDIR, "ascend", False),
            ModelBuildArgs("{output_dir}/bert_model.onnx.cpu.mindir", mslite.ModelType.MINDIR, "cpu", False),
        ),
    )

    return tuple(chain.from_iterable(map(product_case, args)))


DIFFERENT_PROVIDER_MODELS_BUILD_ARGS = prepare_different_provider_args()


@pytest.mark.parametrize("model_build_list", DIFFERENT_PROVIDER_MODELS_BUILD_ARGS)
@pytest.mark.backend("mslite_large_model_inference_arm_ascend910B")
def test_compile_graph_parallel_different_provider(
    model_build_list, output_dir, device_id, thread_pool_timeout=300, loop=3
):
    """
    test different provider
    """
    model_build_list = tuple(
        replace(i, path=i.path.format(output_dir=output_dir), device_id=device_id[0]) for i in model_build_list
    )

    parallel = len(model_build_list)
    pool = ThreadPoolExecutor(max_workers=parallel)

    with ScopeTimeRecord() as record:
        tasks = [pool.submit(build_in_loop, loop, model) for model in model_build_list]
        assert all(task.result() for task in as_completed(tasks, timeout=thread_pool_timeout))
    print(f"build time with args: {model_build_list} {record.duration} ms")
