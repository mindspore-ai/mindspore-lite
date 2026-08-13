/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <memory>
#include "include/api/model_parallel_runner.h"
#include "common/common_test.h"
#include "src/common/file_utils.h"

namespace mindspore {
namespace {
const char in_data_path[] = "./mobilenetv2.ms.bin";
const char model_path[] = "./mobilenetv2.ms";
const size_t kInputDataSize = 1 * 224 * 224 * 3 * sizeof(float);
const size_t kOutputDataSize = 1 * 1001 * sizeof(float);

const char mindir_model_path[] = "./matmul_ops_for_ut.static.onnx.mindir";  // static model
const char invalid_model_path[] = "./test_invalid_model.mindir";            // exist but no data

const std::vector<int64_t> kShape0{10, 11, 12};
const std::vector<int64_t> kShape1{12, 13};
const std::vector<int64_t> kShape2{10, 11, 13};
constexpr int kInputNum2 = 2;
constexpr int kOutputNum1 = 1;
constexpr int kDeviceId10 = 10;

std::shared_ptr<Context> CreateContext() {
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr.";
    return nullptr;
  }
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::AscendDeviceInfo>();
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return nullptr;
  }
  device_list.push_back(device_info);
  return context;
}

void SetInputTensorData(std::vector<MSTensor> *inputs) {
  ASSERT_EQ(inputs->size(), 1);
  auto &input = inputs->front();
  auto data_size = input.DataSize();
  ASSERT_EQ(data_size, kInputDataSize);
  size_t size;
  auto bin_buf = lite::ReadFile(in_data_path, &size);
  ASSERT_NE(bin_buf, nullptr);
  ASSERT_EQ(size, kInputDataSize);
  input.SetData(bin_buf);
  return;
}
}  // namespace

class ModelParallelRunnerTest : public mindspore::CommonTest {
 public:
  ModelParallelRunnerTest() {}
};

TEST_F(ModelParallelRunnerTest, InitWithoutRunnerConfig) {
  ModelParallelRunner runner;
  auto status = runner.Init(model_path);
  ASSERT_EQ(status, kSuccess);
}

TEST_F(ModelParallelRunnerTest, RunnerConfigWithWorkNum) {
  auto config = std::make_shared<RunnerConfig>();
  ASSERT_NE(nullptr, config);

  config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(model_path, config);
  ASSERT_EQ(status, kSuccess);
}

TEST_F(ModelParallelRunnerTest, RunnerConfigWithContext) {
  auto config = std::make_shared<RunnerConfig>();
  ASSERT_NE(nullptr, config);

  auto context = std::make_shared<Context>();
  ASSERT_NE(nullptr, context);
  context->SetThreadNum(1);
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  ASSERT_NE(nullptr, device_info);
  device_list.push_back(device_info);
  ASSERT_EQ(device_list.size(), 1);

  config->SetContext(context);
  config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(model_path, config);
  ASSERT_EQ(status, kSuccess);
}

TEST_F(ModelParallelRunnerTest, RunnerGetInput) {
  auto config = std::make_shared<RunnerConfig>();
  ASSERT_NE(nullptr, config);

  auto context = std::make_shared<Context>();
  ASSERT_NE(nullptr, context);
  context->SetThreadNum(1);
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  ASSERT_NE(nullptr, device_info);
  device_list.push_back(device_info);
  ASSERT_EQ(device_list.size(), 1);

  config->SetContext(context);
  config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(model_path, config);
  ASSERT_EQ(status, kSuccess);
  auto inputs = runner.GetInputs();
  ASSERT_EQ(inputs.size(), 1);
}

TEST_F(ModelParallelRunnerTest, RunnerGetOutput) {
  auto config = std::make_shared<RunnerConfig>();
  ASSERT_NE(nullptr, config);

  auto context = std::make_shared<Context>();
  ASSERT_NE(nullptr, context);
  context->SetThreadNum(1);
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  ASSERT_NE(nullptr, device_info);
  device_list.push_back(device_info);
  ASSERT_EQ(device_list.size(), 1);

  config->SetContext(context);
  config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(model_path, config);
  ASSERT_EQ(status, kSuccess);
  auto outputs = runner.GetOutputs();
  ASSERT_EQ(outputs.size(), 1);
}

TEST_F(ModelParallelRunnerTest, PredictWithoutInput) {
  auto config = std::make_shared<RunnerConfig>();
  ASSERT_NE(nullptr, config);

  auto context = std::make_shared<Context>();
  ASSERT_NE(nullptr, context);
  context->SetThreadNum(2);
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  ASSERT_NE(nullptr, device_info);
  device_list.push_back(device_info);
  ASSERT_EQ(device_list.size(), 1);

  config->SetContext(context);
  config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(model_path, config);
  ASSERT_EQ(status, kSuccess);
  std::vector<MSTensor> inputs;
  std::vector<MSTensor> outputs;
  status = runner.Predict(inputs, &outputs);
  ASSERT_NE(status, kSuccess);
  status = runner.Predict(inputs, &outputs);
  ASSERT_NE(status, kSuccess);
}

TEST_F(ModelParallelRunnerTest, RunnerPredict) {
  auto config = std::make_shared<RunnerConfig>();
  ASSERT_NE(nullptr, config);

  auto context = std::make_shared<Context>();
  ASSERT_NE(nullptr, context);
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  ASSERT_NE(nullptr, device_info);
  device_list.push_back(device_info);
  ASSERT_EQ(device_list.size(), 1);

  config->SetContext(context);
  config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(model_path, config);
  ASSERT_EQ(status, kSuccess);

  auto inputs = runner.GetInputs();
  SetInputTensorData(&inputs);
  std::vector<MSTensor> outputs;
  status = runner.Predict(inputs, &outputs);
  ASSERT_EQ(status, kSuccess);
  // free user data
  for (auto &tensor : inputs) {
    char *data = static_cast<char *>(tensor.MutableData());
    delete[] data;
    tensor.SetData(nullptr);
  }

  inputs = runner.GetInputs();
  SetInputTensorData(&inputs);
  outputs.clear();
  status = runner.Predict(inputs, &outputs);
  ASSERT_EQ(status, kSuccess);
  // free user data
  for (auto &tensor : inputs) {
    char *data = static_cast<char *>(tensor.MutableData());
    delete[] data;
    tensor.SetData(nullptr);
  }
}

// repeat build
TEST_F(ModelParallelRunnerTest, TestInitAPIWithRepeatBuild) {
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto runner_config = std::make_shared<RunnerConfig>();
  ASSERT_NE(runner_config, nullptr);
  runner_config->SetContext(context);
  runner_config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(mindir_model_path, runner_config);
  ASSERT_EQ(status, kSuccess);
  status = runner.Init(mindir_model_path, runner_config);
  ASSERT_EQ(status, kSuccess);
}

// worker num is -1
TEST_F(ModelParallelRunnerTest, TestInitAPIWithInvalidWorkerNum) {
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto runner_config = std::make_shared<RunnerConfig>();
  ASSERT_NE(runner_config, nullptr);
  runner_config->SetContext(context);
  runner_config->SetWorkersNum(-1);
  ModelParallelRunner runner;
  auto status = runner.Init(mindir_model_path, runner_config);
  ASSERT_EQ(status, kLiteError);
  auto status_string = status.ToString() == "InitModelPoolConfig failed.";
  ASSERT_TRUE(status_string);
}

// invalid device ids
TEST_F(ModelParallelRunnerTest, TestInitAPIWithInvalidDeviceIds) {
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto runner_config = std::make_shared<RunnerConfig>();
  ASSERT_NE(runner_config, nullptr);
  runner_config->SetContext(context);
  runner_config->SetWorkersNum(3);
  const std::vector<uint32_t> devices_id_list = {1, 2, 10};  // device id 10 is invalid
  runner_config->SetDeviceIds(devices_id_list);
  ModelParallelRunner runner;
  auto status = runner.Init(mindir_model_path, runner_config);
  ASSERT_EQ(status, kLiteError);
  auto status_string = status.ToString() == "Create worker init failed.";
  ASSERT_TRUE(status_string);
}

// model path is ""
TEST_F(ModelParallelRunnerTest, TestInitAPIWithModelPathIsEmpty) {
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto runner_config = std::make_shared<RunnerConfig>();
  ASSERT_NE(runner_config, nullptr);
  runner_config->SetContext(context);
  runner_config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init("", runner_config);
  ASSERT_EQ(status, kLiteFileError);
  auto status_string = status.ToString() == "Read model failed, please check your model file.";
  ASSERT_TRUE(status_string);
}

// predict success
TEST_F(ModelParallelRunnerTest, TestRunnerPredictAPISuccess) {
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto runner_config = std::make_shared<RunnerConfig>();
  ASSERT_NE(runner_config, nullptr);
  runner_config->SetContext(context);
  runner_config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(mindir_model_path, runner_config);
  ASSERT_EQ(status, kSuccess);
  auto inputs = runner.GetInputs();
  ASSERT_EQ(inputs.size(), kInputNum2);
  for (auto &input : inputs) {
    input.MutableData();
  }
  std::vector<MSTensor> outputs;
  status = runner.Predict(inputs, &outputs);
  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(status, kSuccess);
}

// call predict Without init
TEST_F(ModelParallelRunnerTest, TestRunnerPredictAPIWithoutInit) {
  ModelParallelRunner runner;
  std::vector<MSTensor> inputs;
  std::vector<MSTensor> outputs;
  auto status = runner.Predict(inputs, &outputs);
  ASSERT_EQ(status, kLiteUninitializedObj);
  auto status_string =
    status.ToString() == "model_parallel_runner_impl_ is nullptr, ModelParallelRunner is not initialized.";
  ASSERT_TRUE(status_string);
}

// Input is Empty
TEST_F(ModelParallelRunnerTest, TestRunnerPredictAPIWithInputIsEmpty) {
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto runner_config = std::make_shared<RunnerConfig>();
  ASSERT_NE(runner_config, nullptr);
  runner_config->SetContext(context);
  runner_config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(mindir_model_path, runner_config);
  ASSERT_EQ(status, kSuccess);
  std::vector<MSTensor> inputs = {};
  std::vector<MSTensor> outputs;
  status = runner.Predict(inputs, &outputs);
  ASSERT_EQ(status, kLiteInputParamInvalid);
  auto status_string = status.ToString() == "inputs is invalid.";
  ASSERT_TRUE(status_string);
}

// inputs size is invalid
TEST_F(ModelParallelRunnerTest, TestRunnerPredictAPIWithInvalidInputsSize) {
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto runner_config = std::make_shared<RunnerConfig>();
  ASSERT_NE(runner_config, nullptr);
  runner_config->SetContext(context);
  runner_config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(mindir_model_path, runner_config);
  ASSERT_EQ(status, kSuccess);
  auto inputs = runner.GetInputs();
  ASSERT_EQ(inputs.size(), kInputNum2);
  for (auto &input : inputs) {
    input.MutableData();
  }
  inputs.push_back(MSTensor(nullptr));
  std::vector<MSTensor> outputs;
  status = runner.Predict(inputs, &outputs);
  ASSERT_EQ(status, kLiteInputParamInvalid);
  auto status_string = status.ToString() == "Model inputs size != the given inputs size.";
  ASSERT_TRUE(status_string);
}

// invalid input dtype
TEST_F(ModelParallelRunnerTest, TestRunnerPredictAPIWithInvalidInputDtype) {
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto runner_config = std::make_shared<RunnerConfig>();
  ASSERT_NE(runner_config, nullptr);
  runner_config->SetContext(context);
  runner_config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(mindir_model_path, runner_config);
  ASSERT_EQ(status, kSuccess);
  auto inputs = runner.GetInputs();
  ASSERT_EQ(inputs.size(), kInputNum2);
  for (auto &input : inputs) {
    input.MutableData();
  }
  inputs[0].SetDataType(mindspore::DataType::kNumberTypeBool);
  std::vector<MSTensor> outputs;
  status = runner.Predict(inputs, &outputs);
  ASSERT_EQ(status, kLiteInputParamInvalid);
  auto status_string = status.ToString() == "Input data type is wrong.";
  ASSERT_TRUE(status_string);
}

// inputs shape value = -1
TEST_F(ModelParallelRunnerTest, TestRunnerPredictAPIWithIllegalInputShapeValue) {
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto runner_config = std::make_shared<RunnerConfig>();
  ASSERT_NE(runner_config, nullptr);
  runner_config->SetContext(context);
  runner_config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(mindir_model_path, runner_config);
  ASSERT_EQ(status, kSuccess);
  auto inputs = runner.GetInputs();
  ASSERT_EQ(inputs.size(), kInputNum2);
  for (auto &input : inputs) {
    input.MutableData();
  }
  inputs[0].SetShape({10, 11, -1});
  std::vector<MSTensor> outputs;
  status = runner.Predict(inputs, &outputs);
  ASSERT_EQ(status, kLiteInputParamInvalid);
  auto status_string = status.ToString() == "Invalid shape!";
  ASSERT_TRUE(status_string);
}

// inputs shape value = 1000000
TEST_F(ModelParallelRunnerTest, TestRunnerPredictAPIWithInvalidInputShapeValue) {
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto runner_config = std::make_shared<RunnerConfig>();
  ASSERT_NE(runner_config, nullptr);
  runner_config->SetContext(context);
  runner_config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(mindir_model_path, runner_config);
  ASSERT_EQ(status, kSuccess);
  auto inputs = runner.GetInputs();
  ASSERT_EQ(inputs.size(), kInputNum2);
  for (auto &input : inputs) {
    input.MutableData();
  }
  inputs[0].SetShape({10, 11, 1000000});
  std::vector<MSTensor> outputs;
  status = runner.Predict(inputs, &outputs);
  ASSERT_EQ(status, kLiteInputParamInvalid);
  auto status_string = status.ToString() == "Not support dynamic input.";
  ASSERT_TRUE(status_string);
}

// output Is nullptr
TEST_F(ModelParallelRunnerTest, TestRunnerPredictAPIWithOutputIsNullptr) {
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto runner_config = std::make_shared<RunnerConfig>();
  ASSERT_NE(runner_config, nullptr);
  runner_config->SetContext(context);
  runner_config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(mindir_model_path, runner_config);
  ASSERT_EQ(status, kSuccess);
  auto inputs = runner.GetInputs();
  ASSERT_EQ(inputs.size(), kInputNum2);
  for (auto &input : inputs) {
    input.MutableData();
  }
  status = runner.Predict(inputs, nullptr);
  ASSERT_EQ(status, kLiteOutputParamInvalid);
  auto status_string = status.ToString() == "Outputs is nullptr.";
  ASSERT_TRUE(status_string);
}

// invalid output size
TEST_F(ModelParallelRunnerTest, TestRunnerPredictAPIWithInvalidOutputsSize) {
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto runner_config = std::make_shared<RunnerConfig>();
  ASSERT_NE(runner_config, nullptr);
  runner_config->SetContext(context);
  runner_config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(mindir_model_path, runner_config);
  ASSERT_EQ(status, kSuccess);
  auto inputs = runner.GetInputs();
  ASSERT_EQ(inputs.size(), kInputNum2);
  for (auto &input : inputs) {
    input.MutableData();
  }
  std::vector<MSTensor> outputs;
  std::vector<void *> outputs_data;
  auto model_outputs = runner.GetOutputs();
  for (auto &output : model_outputs) {
    void *data = malloc(output.DataSize());
    outputs_data.push_back(data);
    auto tensor =
      mindspore::MSTensor::CreateTensor(output.Name(), output.DataType(), output.Shape(), data, output.DataSize());
    outputs.push_back(*tensor);
    delete tensor;
  }
  outputs.push_back(MSTensor(nullptr));
  status = runner.Predict(inputs, &outputs);
  ASSERT_EQ(status, kLiteOutputParamInvalid);
  auto status_string = status.ToString() == "outputs size wrong.";
  ASSERT_TRUE(status_string);
  for (size_t i = 0; i < outputs_data.size(); i++) {
    void *data = outputs_data[i];
    free(data);
    data = nullptr;
  }
}

// call GetInputs Without init, return {}
TEST_F(ModelParallelRunnerTest, TestGetInputsAPIWithoutInit) {
  ModelParallelRunner runner;
  auto model_inputs = runner.GetInputs();
  ASSERT_EQ(model_inputs.empty(), true);
}

// TestGetInputsAPI
TEST_F(ModelParallelRunnerTest, TestRunnerGetInputsAPI) {
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto runner_config = std::make_shared<RunnerConfig>();
  ASSERT_NE(runner_config, nullptr);
  runner_config->SetContext(context);
  runner_config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(mindir_model_path, runner_config);
  ASSERT_EQ(status, kSuccess);
  auto model_inputs = runner.GetInputs();
  ASSERT_EQ(model_inputs.size(), kInputNum2);
  ASSERT_EQ(model_inputs[0].Shape(), kShape0);
  ASSERT_EQ(model_inputs[1].Shape(), kShape1);
  ASSERT_EQ(model_inputs[0].DataType(), mindspore::DataType::kNumberTypeFloat16);
  ASSERT_EQ(model_inputs[1].DataType(), mindspore::DataType::kNumberTypeFloat16);
}

// call GetOutputs Without init, return {}
TEST_F(ModelParallelRunnerTest, TestGetOutputsAPIWithoutInit) {
  ModelParallelRunner runner;
  auto model_outputs = runner.GetOutputs();
  ASSERT_EQ(model_outputs.empty(), true);
}

// TestGetOutputsAPI
TEST_F(ModelParallelRunnerTest, TestRunnerGetOutputsAPI) {
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto runner_config = std::make_shared<RunnerConfig>();
  ASSERT_NE(runner_config, nullptr);
  runner_config->SetContext(context);
  runner_config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(mindir_model_path, runner_config);
  ASSERT_EQ(status, kSuccess);
  auto model_outputs = runner.GetOutputs();
  ASSERT_EQ(model_outputs.size(), kOutputNum1);
  ASSERT_EQ(model_outputs[0].Shape(), kShape2);
  ASSERT_EQ(model_outputs[0].DataType(), mindspore::DataType::kNumberTypeFloat16);
}
}  // namespace mindspore
