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
Test for MindSpore Lite Model
"""

import mindspore_lite as mslite
import numpy as np

# ----------- config file -----------
# [acl_build_options]
# input_format="ND"
# input_shape="sample:2,4,-1,-1;timestep:1;encoder_hidden_states:2,77,768"
# ge.dynamicDims="64,64;96,96"

MODEL_FILE = "./sd1.5_unet.onnx_graph.mindir"
DEVICE_ID = 0
OUTPUT_SHAPE_RANGE_1 = (2, 4, 64, 64)
OUTPUT_SHAPE_RANGE_2 = (2, 4, 96, 96)

INPUT_1_SHAPE_RANGE_1 = (2, 4, 64, 64)
INPUT_2_SHAPE_RANGE_1 = (1,)
INPUT_3_SHAPE_RANGE_1 = (2, 77, 768)
INPUT_1_SHAPE_RANGE_2 = (2, 4, 96, 96)
INPUT_2_SHAPE_RANGE_2 = (1,)
INPUT_3_SHAPE_RANGE_2 = (2, 77, 768)


def create_model():
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    model = mslite.Model()
    model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context)
    return model


def create_tensor(shape, dtype):
    data = np.random.random(shape).astype(dtype)
    device_tensor = mslite.Tensor(data, device="ascend:" + str(DEVICE_ID))
    host_tensor = mslite.Tensor(data)
    return host_tensor, device_tensor


def compare_result(data1, data2):
    err = np.sum(np.abs(data1 - data2))
    assert err < 0.00001


def test_compare_device_input():
    """
    The all inputs use copy free, the other inputs use copy free, and the output does not use copy free
    """
    # create model
    model = create_model()
    model.resize(model.get_inputs(), [INPUT_1_SHAPE_RANGE_1, INPUT_2_SHAPE_RANGE_1, INPUT_3_SHAPE_RANGE_1])

    # create tensor for image_size = 64
    input_1_host, input_1_device = create_tensor(INPUT_1_SHAPE_RANGE_1, np.float32)
    input_2_host, input_2_device = create_tensor(INPUT_2_SHAPE_RANGE_1, np.float32)
    input_3_host, input_3_device = create_tensor(INPUT_3_SHAPE_RANGE_1, np.float32)

    # predict for shape of 64
    output_host_1 = model.predict([input_1_host, input_2_host, input_3_host])
    output_host_2 = model.predict([input_1_device, input_2_device, input_3_device])
    assert output_host_1[0].device == "None:-1"
    assert output_host_2[0].device == "None:-1"
    compare_result(output_host_1[0].get_data_to_numpy(), output_host_2[0].get_data_to_numpy())


def test_compare_part_device_input_last():
    """
    The last input does not use copy free, the other inputs use copy free, and the output does not use copy free
    """
    # create model
    model = create_model()
    model.resize(model.get_inputs(), [INPUT_1_SHAPE_RANGE_1, INPUT_2_SHAPE_RANGE_1, INPUT_3_SHAPE_RANGE_1])

    # create tensor for image_size = 64
    input_1_host, input_1_device = create_tensor(INPUT_1_SHAPE_RANGE_1, np.float32)
    input_2_host, input_2_device = create_tensor(INPUT_2_SHAPE_RANGE_1, np.float32)
    input_3_host, input_3_device = create_tensor(INPUT_3_SHAPE_RANGE_1, np.float32)

    # predict for shape of 64
    output_host_1 = model.predict([input_1_host, input_2_host, input_3_host])
    output_host_2 = model.predict([input_1_host, input_2_host, input_3_device])
    assert output_host_1[0].device == "None:-1"
    assert output_host_2[0].device == "None:-1"
    compare_result(output_host_1[0].get_data_to_numpy(), output_host_2[0].get_data_to_numpy())


def test_compare_part_device_input_first():
    """
    The first input does not use copy free, the other inputs use copy free, and the output does not use copy free
    """
    # create model
    model = create_model()
    model.resize(model.get_inputs(), [INPUT_1_SHAPE_RANGE_1, INPUT_2_SHAPE_RANGE_1, INPUT_3_SHAPE_RANGE_1])

    # create tensor for image_size = 64
    input_1_host, input_1_device = create_tensor(INPUT_1_SHAPE_RANGE_1, np.float32)
    input_2_host, input_2_device = create_tensor(INPUT_2_SHAPE_RANGE_1, np.float32)
    input_3_host, input_3_device = create_tensor(INPUT_3_SHAPE_RANGE_1, np.float32)

    # predict for shape of 64
    output_host_1 = model.predict([input_1_host, input_2_host, input_3_host])
    output_host_2 = model.predict([input_1_device, input_2_device, input_3_host])
    assert output_host_1[0].device == "None:-1"
    assert output_host_2[0].device == "None:-1"
    compare_result(output_host_1[0].get_data_to_numpy(), output_host_2[0].get_data_to_numpy())


def test_compare_part_device_input_middle():
    """
    The middle input does not use copy free, the other inputs use copy free, and the output does not use copy free
    """
    # create model
    model = create_model()
    model.resize(model.get_inputs(), [INPUT_1_SHAPE_RANGE_1, INPUT_2_SHAPE_RANGE_1, INPUT_3_SHAPE_RANGE_1])

    # create tensor for image_size = 64
    input_1_host, input_1_device = create_tensor(INPUT_1_SHAPE_RANGE_1, np.float32)
    input_2_host, input_2_device = create_tensor(INPUT_2_SHAPE_RANGE_1, np.float32)
    input_3_host, input_3_device = create_tensor(INPUT_3_SHAPE_RANGE_1, np.float32)

    # predict for shape of 64
    output_host_1 = model.predict([input_1_host, input_2_host, input_3_host])
    output_host_2 = model.predict([input_1_device, input_2_host, input_3_device])
    assert output_host_1[0].device == "None:-1"
    assert output_host_2[0].device == "None:-1"
    compare_result(output_host_1[0].get_data_to_numpy(), output_host_2[0].get_data_to_numpy())


def test_compare_part_device_input_last_and_device_output():
    """
    The first input does not use copy free, the other inputs use copy free, the output uses copy free,
    and the shape of the model is transformed
    """
    # create model
    model = create_model()
    model.resize(model.get_inputs(), [INPUT_1_SHAPE_RANGE_1, INPUT_2_SHAPE_RANGE_1, INPUT_3_SHAPE_RANGE_1])

    # create tensor for image_size = 64
    input_1_host, input_1_device = create_tensor(INPUT_1_SHAPE_RANGE_1, np.float32)
    input_2_host, input_2_device = create_tensor(INPUT_2_SHAPE_RANGE_1, np.float32)
    input_3_host, input_3_device = create_tensor(INPUT_3_SHAPE_RANGE_1, np.float32)

    output_host, output_device = create_tensor(OUTPUT_SHAPE_RANGE_1, np.float32)

    # predict for shape of 64
    output_host_1 = model.predict([input_1_host, input_2_host, input_3_host], [output_host])
    output_host_2 = model.predict([input_1_host, input_2_host, input_3_device], [output_device])
    assert output_host_1[0].device == "None:-1"
    assert output_host_2[0].device == "ascend:" + str(DEVICE_ID)
    compare_result(output_host_1[0].get_data_to_numpy(), output_host_2[0].get_data_to_numpy())

    # create tensor for image_size = 64
    input_1_host, input_1_device = create_tensor(INPUT_1_SHAPE_RANGE_2, np.float32)
    input_2_host, input_2_device = create_tensor(INPUT_2_SHAPE_RANGE_2, np.float32)
    input_3_host, input_3_device = create_tensor(INPUT_3_SHAPE_RANGE_2, np.float32)

    output_host, output_device = create_tensor(OUTPUT_SHAPE_RANGE_2, np.float32)

    # predict for shape of 64
    model.resize(model.get_inputs(), [INPUT_1_SHAPE_RANGE_2, INPUT_2_SHAPE_RANGE_2, INPUT_3_SHAPE_RANGE_2])
    output_host_1 = model.predict([input_1_host, input_2_host, input_3_host], [output_host])
    output_host_2 = model.predict([input_1_host, input_2_host, input_3_device], [output_device])
    assert output_host_1[0].device == "None:-1"
    assert output_host_2[0].device == "ascend:" + str(DEVICE_ID)
    compare_result(output_host_1[0].get_data_to_numpy(), output_host_2[0].get_data_to_numpy())


def test_compare_part_device_input_first_end_and_device_output():
    """
    The last input does not use copy free, the other inputs use copy free,
    the output uses copy free, and the shape of the model is transformed
    """
    # create model
    model = create_model()
    model.resize(model.get_inputs(), [INPUT_1_SHAPE_RANGE_1, INPUT_2_SHAPE_RANGE_1, INPUT_3_SHAPE_RANGE_1])

    # create tensor for image_size = 64
    input_1_host, input_1_device = create_tensor(INPUT_1_SHAPE_RANGE_1, np.float32)
    input_2_host, input_2_device = create_tensor(INPUT_2_SHAPE_RANGE_1, np.float32)
    input_3_host, input_3_device = create_tensor(INPUT_3_SHAPE_RANGE_1, np.float32)

    output_host, output_device = create_tensor(OUTPUT_SHAPE_RANGE_1, np.float32)

    # predict for shape of 64
    output_host_1 = model.predict([input_1_host, input_2_host, input_3_host], [output_host])
    output_host_2 = model.predict([input_1_device, input_2_device, input_3_host], [output_device])
    assert output_host_1[0].device == "None:-1"
    assert output_host_2[0].device == "ascend:" + str(DEVICE_ID)
    compare_result(output_host_1[0].get_data_to_numpy(), output_host_2[0].get_data_to_numpy())

    # create tensor for image_size = 96
    input_1_host, input_1_device = create_tensor(INPUT_1_SHAPE_RANGE_2, np.float32)
    input_2_host, input_2_device = create_tensor(INPUT_2_SHAPE_RANGE_2, np.float32)
    input_3_host, input_3_device = create_tensor(INPUT_3_SHAPE_RANGE_2, np.float32)

    output_host, output_device = create_tensor(OUTPUT_SHAPE_RANGE_2, np.float32)

    # predict for shape of 96
    model.resize(model.get_inputs(), [INPUT_1_SHAPE_RANGE_2, INPUT_2_SHAPE_RANGE_2, INPUT_3_SHAPE_RANGE_2])
    output_host_1 = model.predict([input_1_host, input_2_host, input_3_host], [output_host])
    output_host_2 = model.predict([input_1_device, input_2_device, input_3_host], [output_device])
    assert output_host_1[0].device == "None:-1"
    assert output_host_2[0].device == "ascend:" + str(DEVICE_ID)
    compare_result(output_host_1[0].get_data_to_numpy(), output_host_2[0].get_data_to_numpy())


def test_compare_part_device_input_middle_end_and_device_output():
    """
    Do not use copy free input in the middle, use copy free input for other inputs, use copy free output,
    and transform the shape of the model
    """
    # create model
    model = create_model()
    model.resize(model.get_inputs(), [INPUT_1_SHAPE_RANGE_1, INPUT_2_SHAPE_RANGE_1, INPUT_3_SHAPE_RANGE_1])

    # create tensor for image_size = 64
    input_1_host, input_1_device = create_tensor(INPUT_1_SHAPE_RANGE_1, np.float32)
    input_2_host, input_2_device = create_tensor(INPUT_2_SHAPE_RANGE_1, np.float32)
    input_3_host, input_3_device = create_tensor(INPUT_3_SHAPE_RANGE_1, np.float32)

    output_host, output_device = create_tensor(OUTPUT_SHAPE_RANGE_1, np.float32)

    # predict for shape of 64
    output_host_1 = model.predict([input_1_host, input_2_host, input_3_host], [output_host])
    output_host_2 = model.predict([input_1_device, input_2_host, input_3_device], [output_device])
    assert output_host_1[0].device == "None:-1"
    assert output_host_2[0].device == "ascend:" + str(DEVICE_ID)
    compare_result(output_host_1[0].get_data_to_numpy(), output_host_2[0].get_data_to_numpy())

    # create tensor for image_size = 96
    input_1_host, input_1_device = create_tensor(INPUT_1_SHAPE_RANGE_2, np.float32)
    input_2_host, input_2_device = create_tensor(INPUT_2_SHAPE_RANGE_2, np.float32)
    input_3_host, input_3_device = create_tensor(INPUT_3_SHAPE_RANGE_2, np.float32)

    output_host, output_device = create_tensor(OUTPUT_SHAPE_RANGE_2, np.float32)

    # predict for shape of 96
    model.resize(model.get_inputs(), [INPUT_1_SHAPE_RANGE_2, INPUT_2_SHAPE_RANGE_2, INPUT_3_SHAPE_RANGE_2])
    output_host_1 = model.predict([input_1_host, input_2_host, input_3_host], [output_host])
    output_host_2 = model.predict([input_1_device, input_2_host, input_3_device], [output_device])
    assert output_host_1[0].device == "None:-1"
    assert output_host_2[0].device == "ascend:" + str(DEVICE_ID)
    compare_result(output_host_1[0].get_data_to_numpy(), output_host_2[0].get_data_to_numpy())
