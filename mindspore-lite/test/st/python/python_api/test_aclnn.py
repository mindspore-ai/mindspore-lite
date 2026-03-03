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
"""
Test load from buffer
"""

import itertools
import os
import tempfile
import pytest
import mindspore_lite as mslite


def create_temp_file(content, suffix=".ini"):
    """
    create temp config file
    """
    fd, path = tempfile.mkstemp(suffix=suffix, text=True)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


QKV_NAME_TEMPLATE = "/blocks.{}/self_attn/{}_proj/MatMul"
ACLNN_NODES = ",".join(
    QKV_NAME_TEMPLATE.format(layer, part) for layer, part in itertools.product(range(0, 28), ["q", "k", "v"])
)

STATIC_SHAPE_CONFIG_STR = f"""
[acl_build_options]
input_format="ND"
input_shape="input_ids:128,1,1024;attention_mask:1,1,128,256;position_ids:1,128;past_key_values:28,2,1,128,8,128"
[ascend_context]
aclnn_nodes={ACLNN_NODES}
"""

DYN_SHAPE_CONFIG_STR = f"""
[acl_build_options]
input_format="ND"
input_shape="input_ids:-1,1,1024;attention_mask:1,1,-1,-1;position_ids:1,-1;past_key_values:28,2,1,-1,8,128"
ge.dynamicDims="128,128,256,128,128;256,256,512,256,256"
[ascend_context]
aclnn_nodes={ACLNN_NODES}
"""

CONFIGS = {
    "static": STATIC_SHAPE_CONFIG_STR,
    "dynamic": DYN_SHAPE_CONFIG_STR,
}


@pytest.mark.parametrize(
    "config",
    ("static", "dynamic"),
)
@pytest.mark.backend("mslite_large_model_inference_arm_ascend910B")
def test_aclnn_convert(mindir_dir, output_dir, config):
    """
    module setup
    convert qwen3 model
    """
    config_str = CONFIGS[config]
    config_file = create_temp_file(config_str)
    try:
        converter = mslite.Converter()
        converter.save_type = mslite.ModelType.MINDIR
        converter.optimize = "ascend_oriented"
        model_name = "qwen3_0.6B_fp32.onnx"
        converter.convert(
            mslite.FmkType.ONNX,
            f"{mindir_dir}/{model_name}",
            f"{output_dir}/{model_name}",
            config_file=config_file,
        )
    finally:
        os.remove(config_file)
