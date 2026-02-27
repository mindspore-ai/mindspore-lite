/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#include "include/api/model.h"
#include "include/api/context.h"
#include "gtest/gtest.h"
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "include/api/types.h"

namespace mindspore {
namespace {
const char model_path[] = "./matmul_ops_for_ut.static.mindir";           // static model
const char invalid_model_path[] = "./test_invalid_model.mindir";         // exist but no data
const char model_dynamic_path[] = "./matmul_ops_for_ut.dynamic.mindir";  // dynamic model
const char config_file_path[] = "./config.ini";
const char invalid_config_file_path[] = "./invalid_config.ini";

const std::vector<int64_t> kShape0{10, 11, 12};
const std::vector<int64_t> kShape1{12, 13};
const std::vector<int64_t> kShape2{10, 11, 13};
const std::vector<int64_t> kShape3{100, 11, 12};
const std::vector<int64_t> kShape4{};
constexpr int kInputNum1 = 1;
constexpr int kInputNum2 = 2;
constexpr int kOutputNum1 = 1;
constexpr int kDeviceId10 = 10;

std::shared_ptr<Context> CreateContext() {
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    return nullptr;
  }
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::AscendDeviceInfo>();
  if (device_info == nullptr) {
    return nullptr;
  }
  device_list.push_back(device_info);
  return context;
}

}  // namespace

class ModelTest : public mindspore::CommonTest {
 public:
  ModelTest() {}
};

TEST_F(ModelTest, TestModelBuildAndPredictSuccess) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  auto inputs = model->GetInputs();
  for (auto &input : inputs) {
    input.MutableData();
  }
  std::vector<MSTensor> outputs;
  status = model->Predict(inputs, &outputs);
  ASSERT_EQ(status, kSuccess);
}

// for model build
TEST_F(ModelTest, TestBuildAPIWithoutModelName) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build("", mindspore::kMindIR, context);
  ASSERT_EQ(status, kLiteFileError);
  auto status_string = status.ToString() == "Model path is empty!";
  ASSERT_TRUE(status_string);
}

TEST_F(ModelTest, TestBuildAPIWithoutModelFile) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build("not_find_model.mindir", mindspore::kMindIR, context);
  ASSERT_EQ(status, kLiteFileError);
  auto status_string = status.ToString() == "Failed to read buffer from model file!";
  ASSERT_TRUE(status_string);
}

TEST_F(ModelTest, TestModelBuildWithNullContext) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, nullptr);
  ASSERT_EQ(status, kLiteNullptr);
  auto status_string = status.ToString() == "context is nullptr!";
  ASSERT_TRUE(status_string);
}

TEST_F(ModelTest, TestModelBuildWithThreadNumInvalid) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  context->SetThreadNum(-1);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kLiteParamInvalid);
  auto status_string = status.ToString() == "Invalid thread num!";
  ASSERT_TRUE(status_string);
}

TEST_F(ModelTest, TestBuildAPICheckIsAscend) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = std::make_shared<mindspore::Context>();
  ASSERT_NE(context, nullptr);
  auto &device_list = context->MutableDeviceInfo();
  std::shared_ptr<DeviceInfoContext> null_device = nullptr;
  device_list.push_back(null_device);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kLiteNullptr);
  auto status_string = status.ToString() == "device_info is nullptr!";
  ASSERT_TRUE(status_string);
}

// for model predict
TEST_F(ModelTest, TestModelPredictWithInvalidDtype) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<void *> inputs_data;
  auto model_inputs = model->GetInputs();
  for (auto &input : model_inputs) {
    void *data = malloc(input.DataSize());
    inputs_data.push_back(data);
    auto tensor =
      mindspore::MSTensor::CreateTensor(input.Name(), input.DataType(), input.Shape(), data, input.DataSize());
    tensor->SetDataType(mindspore::DataType::kNumberTypeBool);
    inputs.push_back(*tensor);
    delete tensor;
  }
  std::vector<MSTensor> outputs;
  status = model->Predict(inputs, &outputs);
  ASSERT_EQ(status, kLiteInputParamInvalid);
  auto status_string = status.ToString() == "Input data type is wrong.";
  ASSERT_TRUE(status_string);
  for (size_t i = 0; i < inputs_data.size(); i++) {
    void *data = inputs_data[i];
    free(data);
    data = nullptr;
  }
}

TEST_F(ModelTest, TestModelPredictWithInvalidShape) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<void *> inputs_data;
  auto model_inputs = model->GetInputs();
  for (auto &input : model_inputs) {
    void *data = malloc(input.DataSize());
    inputs_data.push_back(data);
    auto tensor =
      mindspore::MSTensor::CreateTensor(input.Name(), input.DataType(), input.Shape(), data, input.DataSize());
    tensor->SetShape({10, 11, 12});
    inputs.push_back(*tensor);
    delete tensor;
  }
  std::vector<MSTensor> outputs;
  status = model->Predict(inputs, &outputs);
  ASSERT_EQ(status, kLiteInputParamInvalid);
  auto status_string = status.ToString() == "Not support dynamic input.";
  ASSERT_TRUE(status_string);

  for (size_t i = 0; i < inputs_data.size(); i++) {
    void *data = inputs_data[i];
    free(data);
    data = nullptr;
  }
}

TEST_F(ModelTest, TestModelPredictWithInvalidInputSize) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<void *> inputs_data;
  auto model_inputs = model->GetInputs();
  for (auto &input : model_inputs) {
    void *data = malloc(input.DataSize());
    inputs_data.push_back(data);
    auto tensor =
      mindspore::MSTensor::CreateTensor(input.Name(), input.DataType(), input.Shape(), data, input.DataSize());
    inputs.push_back(*tensor);
    delete tensor;
  }
  inputs.push_back(MSTensor(nullptr));
  std::vector<MSTensor> outputs;
  status = model->Predict(inputs, &outputs);
  ASSERT_EQ(status, kLiteInputParamInvalid);
  auto status_string = status.ToString() == "The given input size != graph input size.";
  ASSERT_TRUE(status_string);
  for (size_t i = 0; i < inputs_data.size(); i++) {
    void *data = inputs_data[i];
    free(data);
    data = nullptr;
  }
}

// LoadConfig success
TEST_F(ModelTest, TestLoadConfigAPISuccess) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto status = model->LoadConfig(config_file_path);
  ASSERT_EQ(status, kSuccess);
}

// AfterBuildToCallLoadConfig
TEST_F(ModelTest, TestAfterBuildToCallLoadConfig) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  status = model->LoadConfig(config_file_path);
  ASSERT_EQ(status, kLiteError);
  auto status_string = status.ToString() == "Model has been called build, please call LoadConfig before build.";
  ASSERT_TRUE(status_string);
}

// Configfile Is Empty
TEST_F(ModelTest, TestLoadConfigAPIWithConfigfileIsEmpty) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto status = model->LoadConfig("");
  ASSERT_EQ(status, kLiteFileError);
  auto status_string = status.ToString() == "GetAllSectionInfoFromConfigFile failed, please check your config file.";
  ASSERT_TRUE(status_string);
}

// TEST UpdateConfig API success
TEST_F(ModelTest, TestUpdateConfigAPISuccess) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->LoadConfig(config_file_path);
  ASSERT_EQ(status, kSuccess);
  status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  std::string section = "[acl_build_options]";
  std::pair<std::string, std::string> config("input_format", "ND");
  status = model->UpdateConfig(section, config);
  ASSERT_EQ(status, kSuccess);
}

// The number of sections in the configuration file is too large.
TEST_F(ModelTest, TestUpdateConfigAPIWithInvalidSectionNum) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->LoadConfig(invalid_config_file_path);
  ASSERT_EQ(status, kSuccess);
  status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  std::string section = "[acl_build_options]";
  std::pair<std::string, std::string> config("input_format", "ND");
  status = model->UpdateConfig(section, config);
  ASSERT_EQ(status, kLiteParamInvalid);
  auto status_string = status.ToString() == "The config has too many sections!";
  ASSERT_TRUE(status_string);
}

// Build Success
TEST_F(ModelTest, TestBuildAPISuccess) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
}

// invalid_model
TEST_F(ModelTest, TestBuildAPIWithInvalidModel) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(invalid_model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kLiteFileError);
  auto status_string = status.ToString() == "Failed to read buffer from model file!";
  ASSERT_TRUE(status_string);
}

// repeat rebuild
TEST_F(ModelTest, TestBuildAPIWithRepeatBuild) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kLiteModelRebuild);
  auto status_string = status.ToString() == "Model has been called build!";
  ASSERT_TRUE(status_string);
}

// the model type of kONNX
TEST_F(ModelTest, TestBuildAPIWithModelTypeIskONNX) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kONNX, context);
  ASSERT_EQ(status, kLiteNullptr);
  auto status_string = status.ToString() == "func_graph is nullptr, failed to load MindIR model!";
  ASSERT_TRUE(status_string);
}

// The model type of kDataFlow
TEST_F(ModelTest, TestBuildAPIWithModelTypeIskDataFlow) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, ModelType::kDataFlow, context);
  ASSERT_EQ(status, kLiteNullptr);
  auto status_string = status.ToString() == "func_graph is nullptr, failed to load MindIR model!";
  ASSERT_TRUE(status_string);
}

// The model type of kMindIR_Lite
TEST_F(ModelTest, TestBuildAPIWithModelTypeIskMindIRLite) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, ModelType::kMindIR_Lite, context);
  ASSERT_EQ(status, kLiteAclInitFailed);
  auto status_string = status.ToString() == "Call aclmdlLoadFromMem failed.";
  ASSERT_TRUE(status_string);
}

// context is nullptr
TEST_F(ModelTest, TestBuildAPIWithContextIsNullptr) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, nullptr);
  ASSERT_EQ(status, kLiteNullptr);
  auto status_string = status.ToString() == "context is nullptr!";
  ASSERT_TRUE(status_string);
}

// predict success
TEST_F(ModelTest, TestModelPredictAPISuccess) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  auto inputs = model->GetInputs();
  ASSERT_EQ(inputs.size(), kInputNum2);
  for (auto &input : inputs) {
    input.MutableData();
  }
  std::vector<MSTensor> outputs;
  status = model->Predict(inputs, &outputs);
  ASSERT_EQ(outputs.size(), kOutputNum1);
  ASSERT_EQ(status, kSuccess);
}

// invalid device id
TEST_F(ModelTest, TestBuildAPIWithInvalidDeviceId) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = std::make_shared<mindspore::Context>();
  ASSERT_NE(context, nullptr);
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::AscendDeviceInfo>();
  ASSERT_NE(context, nullptr);
  device_info->SetDeviceID(kDeviceId10);
  device_list.push_back(device_info);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kLiteParamInvalid);
  auto status_string = status.ToString() == "Acl open device failed, device_id is invalid.";
  ASSERT_TRUE(status_string);
}

// call predict Without build
TEST_F(ModelTest, TestWithCallPredictWithoutBuild) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<void *> inputs_data;
  auto model_inputs = model->GetInputs();
  for (auto &input : model_inputs) {
    void *data = malloc(input.DataSize());
    inputs_data.push_back(data);
    auto tensor =
      mindspore::MSTensor::CreateTensor(input.Name(), input.DataType(), input.Shape(), data, input.DataSize());
    inputs.push_back(*tensor);
    delete tensor;
  }
  std::vector<MSTensor> outputs;
  auto status = model->Predict(inputs, &outputs);
  ASSERT_EQ(status, kLiteUninitializedObj);
  auto status_string = status.ToString() == "Model has not been called Build, or Model Build has failed";
  ASSERT_TRUE(status_string);
  for (size_t i = 0; i < inputs_data.size(); i++) {
    void *data = inputs_data[i];
    free(data);
    data = nullptr;
  }
}

// user give input shape value = -1
TEST_F(ModelTest, TestModelPredictAPIWithInputShape) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<void *> inputs_data;
  auto model_inputs = model->GetInputs();
  for (auto &input : model_inputs) {
    void *data = malloc(input.DataSize());
    inputs_data.push_back(data);
    auto tensor =
      mindspore::MSTensor::CreateTensor(input.Name(), input.DataType(), input.Shape(), data, input.DataSize());
    inputs.push_back(*tensor);
    delete tensor;
  }
  inputs[0].SetShape({10, 11, -1});
  std::vector<MSTensor> outputs;
  status = model->Predict(inputs, &outputs);
  ASSERT_EQ(status, kLiteInputParamInvalid);
  auto status_string = status.ToString() == "shape is wrong!";
  ASSERT_TRUE(status_string);
  for (size_t i = 0; i < inputs_data.size(); i++) {
    void *data = inputs_data[i];
    free(data);
    data = nullptr;
  }
}

// invalid input format, but predict success
TEST_F(ModelTest, TestModelPredictAPIWithInvalidInputFormat) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<void *> inputs_data;
  auto model_inputs = model->GetInputs();
  for (auto &input : model_inputs) {
    void *data = malloc(input.DataSize());
    inputs_data.push_back(data);
    auto tensor = mindspore::MSTensor::CreateTensor(input.Name(), input.DataType(), input.Shape(), data,
                                                    input.DataSize(), "ascend", 0);
    inputs.push_back(*tensor);
    delete tensor;
  }
  inputs[0].SetFormat(mindspore::Format::NC4HW4);
  inputs[1].SetFormat(mindspore::Format::FRACTAL_NZ);
  std::vector<MSTensor> outputs;
  status = model->Predict(inputs, &outputs);
  ASSERT_EQ(status, kSuccess);
  for (size_t i = 0; i < inputs_data.size(); i++) {
    void *data = inputs_data[i];
    free(data);
    data = nullptr;
  }
}

// Input Is Empty
TEST_F(ModelTest, TestModelPredictAPIWithInvalidInputIsEmpty) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  std::vector<mindspore::MSTensor> inputs = {};
  std::vector<mindspore::MSTensor> outputs;
  status = model->Predict(inputs, &outputs);
  ASSERT_EQ(status, kLiteInputParamInvalid);
  auto status_string = status.ToString() == "user input tensor is empty!";
  ASSERT_TRUE(status_string);
}

// output Is nullptr
TEST_F(ModelTest, TestModelPredictAPIWithInvalidOutputIsNullptr) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<void *> inputs_data;
  auto model_inputs = model->GetInputs();
  for (auto &input : model_inputs) {
    void *data = malloc(input.DataSize());
    inputs_data.push_back(data);
    auto tensor =
      mindspore::MSTensor::CreateTensor(input.Name(), input.DataType(), input.Shape(), data, input.DataSize());
    inputs.push_back(*tensor);
    delete tensor;
  }
  status = model->Predict(inputs, nullptr);
  ASSERT_EQ(status, kLiteOutputParamInvalid);
  auto status_string = status.ToString() == "outputs pointer is nullptr";
  ASSERT_TRUE(status_string);
  for (size_t i = 0; i < inputs_data.size(); i++) {
    void *data = inputs_data[i];
    free(data);
    data = nullptr;
  }
}

// InvalidOutputSize
TEST_F(ModelTest, TestModelPredictAPIWithInvalidOutputSize) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<void *> inputs_data;
  auto model_inputs = model->GetInputs();
  for (auto &input : model_inputs) {
    void *data = malloc(input.DataSize());
    inputs_data.push_back(data);
    auto tensor =
      mindspore::MSTensor::CreateTensor(input.Name(), input.DataType(), input.Shape(), data, input.DataSize());
    inputs.push_back(*tensor);
    delete tensor;
  }
  std::vector<MSTensor> outputs;
  std::vector<void *> outputs_data;
  auto model_outputs = model->GetOutputs();
  for (auto &output : model_outputs) {
    void *data = malloc(output.DataSize());
    outputs_data.push_back(data);
    auto tensor =
      mindspore::MSTensor::CreateTensor(output.Name(), output.DataType(), output.Shape(), data, output.DataSize());
    outputs.push_back(*tensor);
    delete tensor;
  }
  outputs.push_back(MSTensor(nullptr));
  status = model->Predict(inputs, &outputs);
  ASSERT_EQ(status, kLiteOutputParamInvalid);
  auto status_string = status.ToString() == "outputs size wrong.";
  ASSERT_TRUE(status_string);
  for (size_t i = 0; i < inputs_data.size(); i++) {
    void *data = inputs_data[i];
    free(data);
    data = nullptr;
  }
  for (size_t i = 0; i < outputs_data.size(); i++) {
    void *data = outputs_data[i];
    free(data);
    data = nullptr;
  }
}

// resize success. Note: model is dynamic
TEST_F(ModelTest, TestResizeAPISuccess) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_dynamic_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  std::vector<std::vector<int64_t>> dims;
  auto model_inputs = model->GetInputs();
  dims.push_back(kShape3);
  dims.push_back(kShape1);
  status = model->Resize(model_inputs, dims);
  ASSERT_EQ(status, kSuccess);
}

// call resize Without build
TEST_F(ModelTest, TestWithCallResizeWithoutBuild) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<void *> inputs_data;
  std::vector<std::vector<int64_t>> dims;
  auto model_inputs = model->GetInputs();
  for (auto &input : model_inputs) {
    void *data = malloc(input.DataSize());
    inputs_data.push_back(data);
    auto tensor =
      mindspore::MSTensor::CreateTensor(input.Name(), input.DataType(), input.Shape(), data, input.DataSize());
    inputs.push_back(*tensor);
    dims.push_back(input.Shape());
    delete tensor;
  }
  auto status = model->Resize(inputs, dims);
  ASSERT_EQ(status, kLiteUninitializedObj);
  auto status_string = status.ToString() == "Model has not been called Build, or Model Build has failed!";
  ASSERT_TRUE(status_string);
  for (size_t i = 0; i < inputs_data.size(); i++) {
    void *data = inputs_data[i];
    free(data);
    data = nullptr;
  }
}

// input is empty for Resize API
TEST_F(ModelTest, TestResizeAPIWithInputIsEmpty) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  std::vector<mindspore::MSTensor> inputs = {};
  std::vector<std::vector<int64_t>> dims;
  auto model_inputs = model->GetInputs();
  for (auto &input : model_inputs) {
    dims.push_back(input.Shape());
  }
  status = model->Resize(inputs, dims);
  ASSERT_EQ(status, kLiteInputParamInvalid);
  auto status_string = status.ToString() == "Inputs is empty!";
  ASSERT_TRUE(status_string);
}

// dims is empty for Resize API
TEST_F(ModelTest, TestResizeAPIWithDimsIsEmpty) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<void *> inputs_data;
  std::vector<std::vector<int64_t>> dims = {};
  auto model_inputs = model->GetInputs();
  for (auto &input : model_inputs) {
    void *data = malloc(input.DataSize());
    inputs_data.push_back(data);
    auto tensor =
      mindspore::MSTensor::CreateTensor(input.Name(), input.DataType(), input.Shape(), data, input.DataSize());
    inputs.push_back(*tensor);
    delete tensor;
  }
  status = model->Resize(inputs, dims);
  ASSERT_EQ(status, kLiteInputParamInvalid);
  auto status_string = status.ToString() == "dims is empty!";
  ASSERT_TRUE(status_string);
  for (size_t i = 0; i < inputs_data.size(); i++) {
    void *data = inputs_data[i];
    free(data);
    data = nullptr;
  }
}

// Illegal dims value for Resize API
TEST_F(ModelTest, TestResizeAPIWithIllegalDimsValue) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<void *> inputs_data;
  std::vector<std::vector<int64_t>> dims;
  auto model_inputs = model->GetInputs();
  for (auto &input : model_inputs) {
    void *data = malloc(input.DataSize());
    inputs_data.push_back(data);
    auto tensor =
      mindspore::MSTensor::CreateTensor(input.Name(), input.DataType(), input.Shape(), data, input.DataSize());
    inputs.push_back(*tensor);
    delete tensor;
  }
  dims.push_back({10, 11, -1});   // -1 is illegal
  dims.push_back({INT_MAX, 13});  // INT_MAX is illegal
  status = model->Resize(inputs, dims);
  ASSERT_EQ(status, kLiteInputParamInvalid);
  auto status_string = status.ToString() == "Invalid shape!";
  ASSERT_TRUE(status_string);
  for (size_t i = 0; i < inputs_data.size(); i++) {
    void *data = inputs_data[i];
    free(data);
    data = nullptr;
  }
}

// invalid dims value for Resize API
TEST_F(ModelTest, TestResizeAPIWithInvalidDimsValue) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<void *> inputs_data;
  std::vector<std::vector<int64_t>> dims;
  auto model_inputs = model->GetInputs();
  for (auto &input : model_inputs) {
    void *data = malloc(input.DataSize());
    inputs_data.push_back(data);
    auto tensor =
      mindspore::MSTensor::CreateTensor(input.Name(), input.DataType(), input.Shape(), data, input.DataSize());
    inputs.push_back(*tensor);
    delete tensor;
  }
  dims.push_back({10000, 11, 12});  // 10000 is invalid
  dims.push_back({12, 13});
  status = model->Resize(inputs, dims);
  ASSERT_EQ(status, kLiteInputParamInvalid);
  auto status_string = status.ToString() == "Not support dynamic input.";
  ASSERT_TRUE(status_string);
  for (size_t i = 0; i < inputs_data.size(); i++) {
    void *data = inputs_data[i];
    free(data);
    data = nullptr;
  }
}

// dims size is invalid for Resize API
TEST_F(ModelTest, TestResizeAPIWithInvalidDimsSize) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<void *> inputs_data;
  std::vector<std::vector<int64_t>> dims;
  auto model_inputs = model->GetInputs();
  for (auto &input : model_inputs) {
    void *data = malloc(input.DataSize());
    inputs_data.push_back(data);
    auto tensor =
      mindspore::MSTensor::CreateTensor(input.Name(), input.DataType(), input.Shape(), data, input.DataSize());
    inputs.push_back(*tensor);
    delete tensor;
  }
  dims.push_back({10, 11, 12});
  status = model->Resize(inputs, dims);
  ASSERT_EQ(status, kLiteInputParamInvalid);
  auto status_string = status.ToString() == "The size of inputs does not match the size of dims!";
  ASSERT_TRUE(status_string);
  for (size_t i = 0; i < inputs_data.size(); i++) {
    void *data = inputs_data[i];
    free(data);
    data = nullptr;
  }
}

// inputs size is invalid for Resize API
TEST_F(ModelTest, TestResizeAPIWithInvalidInputSize) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<void *> inputs_data;
  std::vector<std::vector<int64_t>> dims;
  auto model_inputs = model->GetInputs();
  for (auto &input : model_inputs) {
    void *data = malloc(input.DataSize());
    inputs_data.push_back(data);
    auto tensor =
      mindspore::MSTensor::CreateTensor(input.Name(), input.DataType(), input.Shape(), data, input.DataSize());
    inputs.push_back(*tensor);
    dims.push_back(input.Shape());
    delete tensor;
  }
  inputs.push_back(MSTensor(nullptr));
  dims.push_back({});
  status = model->Resize(inputs, dims);
  ASSERT_EQ(status, kLiteInputParamInvalid);
  auto status_string = status.ToString() == "The given input size is inconsistent with the input size of the model.";
  ASSERT_TRUE(status_string);
  for (size_t i = 0; i < inputs_data.size(); i++) {
    void *data = inputs_data[i];
    free(data);
    data = nullptr;
  }
}

// call GetInputs before Build, return {}
TEST_F(ModelTest, TestGetInputsAPIWithoutBuild) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto model_inputs = model->GetInputs();
  ASSERT_EQ(model_inputs.empty(), true);
}

// TestGetInputsAPI
TEST_F(ModelTest, TestGetInputsAPI) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  auto model_inputs = model->GetInputs();
  ASSERT_EQ(model_inputs.size(), kInputNum2);
  ASSERT_EQ(model_inputs[0].Shape(), kShape0);
  ASSERT_EQ(model_inputs[1].Shape(), kShape1);
  ASSERT_EQ(model_inputs[0].DataType(), mindspore::DataType::kNumberTypeFloat16);
  ASSERT_EQ(model_inputs[1].DataType(), mindspore::DataType::kNumberTypeFloat16);
}

// Test GetInputByTensorName API Success
TEST_F(ModelTest, TestGetInputByTensorName) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  auto model_inputs = model->GetInputs();
  auto input_tensor0 = model->GetInputByTensorName("input1");
  ASSERT_EQ(input_tensor0.Shape(), kShape0);
  auto input_tensor1 = model->GetInputByTensorName("input2");
  ASSERT_EQ(input_tensor1.Shape(), kShape1);
  auto input_tensor2 = model->GetInputByTensorName("error_name");
  ASSERT_EQ(input_tensor2.Shape(), kShape4);
}

// call GetInputByTensorName before build, return MSTensor(nullptr)
TEST_F(ModelTest, TestGetInputByTensorNameWithoutBuild) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto input_tensor = model->GetInputByTensorName("input1");
  ASSERT_EQ(input_tensor.Shape(), kShape4);
}

// call GetOutputs before Build, return {}
TEST_F(ModelTest, TestGetOutputsAPIWithoutBuild) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto model_outputs = model->GetOutputs();
  ASSERT_EQ(model_outputs.empty(), true);
}

// GetOutputs API
TEST_F(ModelTest, TestGetOutputsAPI) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  auto model_outputs = model->GetOutputs();
  ASSERT_EQ(model_outputs.size(), kOutputNum1);
  ASSERT_EQ(model_outputs[0].Shape(), kShape2);
  ASSERT_EQ(model_outputs[0].DataType(), mindspore::DataType::kNumberTypeFloat16);
}

// Test GetOutputByTensorName API Success
TEST_F(ModelTest, TestGetOutputByTensorName) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  auto output_tensor0 = model->GetOutputByTensorName("output");
  ASSERT_EQ(output_tensor0.Shape(), kShape2);
  auto output_tensor1 = model->GetOutputByTensorName("error_name");
  ASSERT_EQ(output_tensor1.Shape(), kShape4);
}

// call GetOutputByTensorName before build, return MSTensor(nullptr)
TEST_F(ModelTest, TestGetOutputByTensorNameWithoutBuild) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto output_tensor = model->GetOutputByTensorName("output");
  ASSERT_EQ(output_tensor.Shape(), kShape4);
}

// call finalize without build
TEST_F(ModelTest, TestWithCallFinalizeWithoutBuild) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Finalize();
  ASSERT_EQ(status, kLiteUninitializedObj);
  auto status_string = status.ToString() == "session_ is nullptr, please build model first!";
  ASSERT_TRUE(status_string);
}

// call finalize success
TEST_F(ModelTest, TestFinalizeAPISuccess) {
  auto model = std::make_shared<mindspore::Model>();
  ASSERT_NE(model, nullptr);
  auto context = CreateContext();
  ASSERT_NE(context, nullptr);
  auto status = model->Build(model_path, mindspore::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
  status = model->Finalize();
  ASSERT_EQ(status, kSuccess);
}
}  // namespace mindspore
