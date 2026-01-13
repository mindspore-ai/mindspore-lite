/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include <unordered_map>
#include "litert/cxx_api/tensor/tensor_impl.h"
#include "include/api/data_type.h"
#include "src/litert/delegate/npu/offline_model_kernel.h"

namespace {
constexpr int32_t kHiaiFrequencyType = 3;  // HIGH
constexpr int32_t kHiaiDeviceType = 0;     // NPU
const char kHiaiModelName[] = "Third_Party_Model";
}  // namespace

namespace mindspore {
OfflineModelKernel::~OfflineModelKernel() {
  model_manager_client_ = nullptr;
  model_builder_ = nullptr;
  for (auto t : offline_model_inputs_tensors_) {
    t.reset();
  }

  for (auto t : offline_model_outputs_tensors_) {
    t.reset();
  }
}

int OfflineModelKernel::Prepare() {
  model_manager_client_ = std::make_shared<hiai::AiModelMngerClient>();
  if (model_manager_client_ == nullptr) {
    MS_LOG(ERROR) << "Alloc AiModelMngerClient failed.";
    return lite::RET_ERROR;
  }
  model_builder_ = std::make_shared<hiai::AiModelBuilder>(model_manager_client_);
  if (model_builder_ == nullptr) {
    MS_LOG(ERROR) << "Alloc AiModelBuilder failed.";
    return lite::RET_ERROR;
  }
  auto client_ret = model_manager_client_->Init(nullptr);  // sync mode
  if (client_ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Init modelBuilder failed.";
    return lite::RET_ERROR;
  }
  // Build Model
  int build_ret = BuildHiaiModel(offline_model_buffer_, offline_model_size_);
  if (build_ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Build offline model buffer failed.";
    return lite::RET_ERROR;
  }
  MS_LOG(INFO) << "OfflineModelKernel build offline model buffer success.";
  return lite::RET_OK;
}

int OfflineModelKernel::BuildHiaiModel(uint8_t *model_data, size_t model_data_length) {
  MS_LOG(INFO) << "OfflineModelKernel Build Function start.";
  MS_CHECK_TRUE_RET(model_data_length != 0, kLiteError);
  if (model_data == nullptr) {
    MS_LOG(ERROR) << "Current model_data is invalid, please check model file.";
    return lite::RET_ERROR;
  }
  void *offline_model_data = model_data;
  if (kHiaiFrequencyType == -1 || kHiaiDeviceType == -1) {
    MS_LOG(ERROR) << "Create model description failed. Current kHiaiFrequencyType is :" << kHiaiFrequencyType
                  << ", kHiaiDeviceType:" << kHiaiDeviceType << ".";
    return lite::RET_ERROR;
  }
  std::vector<std::shared_ptr<hiai::AiModelDescription>> model_descs;
  std::unordered_map<std::shared_ptr<hiai::AiModelBuilder>, hiai::MemBuffer *> builder_buffer_map;
  MS_LOG(INFO) << "Create model description: version [" << model_manager_client_->GetVersion()
               << "], kHiaiFrequencyType is " << kHiaiFrequencyType << ", kHiaiDeviceType is " << kHiaiDeviceType
               << ".";
  std::shared_ptr<hiai::AiModelDescription> model_desc =
    std::make_shared<hiai::AiModelDescription>(kHiaiModelName, kHiaiFrequencyType, 0, 1, kHiaiDeviceType);
  if (model_desc == nullptr) {
    MS_LOG(ERROR) << "Alloc AiModelDescription failed.";
    return lite::RET_ERROR;
  }
  model_descs.push_back(model_desc);
  auto model_buffer = model_builder_->InputMemBufferCreate(offline_model_data, model_data_length);
  if (model_buffer == nullptr) {
    MS_LOG(ERROR) << "Hiai Model Builder input memory buffer create failed, model data size:" << model_data_length;
    return lite::RET_ERROR;
  }
  builder_buffer_map.insert({model_builder_, model_buffer});
  model_desc->SetModelBuffer(model_buffer->GetMemBufferData(), model_buffer->GetMemBufferSize());
  MS_LOG(INFO) << "Hiai Model Builder set offline model buffer success.";

  if (!model_descs.empty()) {
    auto load_ret = model_manager_client_->Load(model_descs);
    if (load_ret != hiai::AI_SUCCESS) {
      for (auto it : builder_buffer_map) {
        it.first->MemBufferDestroy(it.second);
      }
      builder_buffer_map.clear();
      MS_LOG(ERROR) << "Hiai Client load offline model failed and clear offline model buffer.";
      return lite::RET_ERROR;
    }
    MS_LOG(INFO) << "Hiai Client load offline model success.";
    model_descs.clear();
  }
  // Init OfflineModel IO tensor
  if (InitHiaiIOTensors() != lite::RET_OK) {
    MS_LOG(ERROR) << "OfflineModelKernel InitHiaiIOTensors failed.";
    return lite::RET_ERROR;
  }
  MS_LOG(INFO) << "OfflineModelKernel init offline model tensor and load model success.";
  return lite::RET_OK;
}

int OfflineModelKernel::InitHiaiIOTensors() {
  std::vector<hiai::TensorDimension> input_dimension;
  std::vector<hiai::TensorDimension> output_dimension;
  if (model_manager_client_ == nullptr) {
    MS_LOG(ERROR) << "Hiai Client is nullptr.";
    return lite::RET_ERROR;
  }
  auto get_io_dim_ret = model_manager_client_->GetModelIOTensorDim(kHiaiModelName, input_dimension, output_dimension);
  if (get_io_dim_ret != hiai::AI_SUCCESS) {
    MS_LOG(ERROR) << "Get offline model input and output tensor dims failed." << get_io_dim_ret;
    return lite::RET_ERROR;
  }
  MS_LOG(DEBUG) << "Input NCHW :" << input_dimension[0].GetNumber() << " " << input_dimension[0].GetChannel() << " "
                << input_dimension[0].GetHeight() << " " << input_dimension[0].GetWidth();
  MS_LOG(DEBUG) << "Output NCHW :" << output_dimension[0].GetNumber() << " " << output_dimension[0].GetChannel() << " "
                << output_dimension[0].GetHeight() << " " << output_dimension[0].GetWidth();

  MS_LOG(DEBUG) << "Init input ai_tensors.";
  auto in_tensor_ret = InitHiaiTensorWithMSTensor(input_dimension, inputs_, offline_model_inputs_tensors_);
  if (in_tensor_ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Update offline model input tensor vector failed. " << in_tensor_ret;
    return lite::RET_ERROR;
  }
  MS_LOG(DEBUG) << "Init output ai_tensors.";
  auto out_tensor_ret = InitHiaiTensorWithMSTensor(output_dimension, outputs_, offline_model_outputs_tensors_);
  if (out_tensor_ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Update offline model output tensor vector failed. " << out_tensor_ret;
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

int OfflineModelKernel::InitHiaiTensorWithMSTensor(
  const std::vector<hiai::TensorDimension> &dimension, const std::vector<MSTensor> &ms_tensors,
  std::vector<std::shared_ptr<hiai::AiTensor>> &offline_model_tensors) {
  if (dimension.empty()) {
    MS_LOG(ERROR) << "Offline model tensor dimension is empty.";
    return lite::RET_ERROR;
  }
  MS_LOG(DEBUG) << " dimension size:" << dimension.size();
  for (int i = 0; i < dimension.size(); i++) {
    std::shared_ptr<hiai::AiTensor> ai_tensor = std::make_shared<hiai::AiTensor>();
    if (ai_tensor == nullptr) {
      MS_LOG(ERROR) << "Alloc AiTensor failed.";
      return lite::RET_ERROR;
    }
    if (ai_tensor->Init(&dimension[i], lite::MSDataTypeToHIAIDataType(ms_tensors[i].DataType())) != hiai::AI_SUCCESS) {
      MS_LOG(ERROR) << "AiTensor init failed.";
      return lite::RET_ERROR;
    }
    offline_model_tensors.push_back(ai_tensor);
  }
  return lite::RET_OK;
}

int OfflineModelKernel::Execute() {
  // Get MS INPUT Tensors
  MS_LOG(INFO) << "Before OfflineModelKernel execute, MSTensorData need to be copy to AiTensorData. Inputs_ size: "
               << inputs_.size() << ", outputs_ size: " << outputs_.size();
  auto ms_hiai_ret = CopyMSTensorsDataToHiaiTensorsData();
  if (ms_hiai_ret != lite::RET_OK) {
    MS_LOG(ERROR) << "CopyMSTensorsDataToHiaiTensorsData failed.";
    return lite::RET_ERROR;
  }
  auto execute_ret = ExecuteHiaiModel();
  if (execute_ret != hiai::AI_SUCCESS) {
    MS_LOG(ERROR) << "OfflineModelKernel ExecuteHiaiModel failed.";
    return lite::RET_ERROR;
  }
  MS_LOG(INFO) << "OfflineModelKernel ExecuteHiaiModel success.";
  auto hiai_ms_ret = CopyHiaiTensorsDataToMSTensorsData();
  if (hiai_ms_ret != lite::RET_OK) {
    MS_LOG(ERROR) << "ConvertAiTensorToMSTensor failed.";
    return lite::RET_ERROR;
  }
  MS_LOG(INFO) << "OfflineModelKernel ExecuteHiaiModel done, and CopyHiaiTensorsDataToMSTensorsData success.";
  return lite::RET_OK;
}

int OfflineModelKernel::ExecuteHiaiModel() {
  hiai::AiContext context;
  std::string key = "model_name";
  std::string value = kHiaiModelName;
  context.AddPara(key, value);
  int32_t stamp;
  if (model_manager_client_ == nullptr) {
    MS_LOG(ERROR) << "Hiai client is nullptr.";
    return lite::RET_ERROR;
  }
  int ret =
    model_manager_client_->Process(context, offline_model_inputs_tensors_, offline_model_outputs_tensors_, 3000, stamp);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "OfflineModelKernel Predict failed by Hiai client using Process function.";
    return lite::RET_ERROR;
  }
  MS_LOG(INFO) << "OfflineModelKernel Predict model success, ret:" << ret << " stamp:" << stamp;
  return lite::RET_OK;
}

int OfflineModelKernel::CopyMSTensorsDataToHiaiTensorsData() {
  MS_LOG(INFO) << "ConvertMSTensorToAiTensor ms_input tensor num:" << inputs_.size()
               << " ai_input tensor num:" << offline_model_inputs_tensors_.size();
  if (offline_model_inputs_tensors_.size() != inputs_.size()) {
    MS_LOG(ERROR) << "ms_input and ai_input have different size. ms_input tensor num:" << inputs_.size()
                  << " ai_input tensor num:" << offline_model_outputs_tensors_.size();
    return lite::RET_ERROR;
  }
  for (size_t i = 0; i < offline_model_inputs_tensors_.size(); i++) {
    if (offline_model_inputs_tensors_.at(i)->GetSize() != inputs_.at(i).DataSize()) {
      MS_LOG(ERROR) << "ms_input and ai_input have different dataSize. ms_input tensor dataSize "
                    << inputs_.at(i).DataSize()
                    << " ai_input tensor num:" << offline_model_inputs_tensors_.at(i)->GetSize();
      return lite::RET_ERROR;
    }
    auto src_buffer = inputs_.at(i).MutableData();
    if (src_buffer == nullptr) {
      MS_LOG(ERROR) << "For " << kHiaiModelName << ", the ms_input at [" << i
                    << "], tensor name:" << inputs_.at(i).Name() << " buffer is null.";
      return lite::RET_ERROR;
    }
    auto dest_buffer = offline_model_inputs_tensors_.at(i)->GetBuffer();
    if (dest_buffer == nullptr) {
      MS_LOG(ERROR) << "For " << kHiaiModelName << ", the ai_input at [" << i << "], buffer is null.";
      return lite::RET_ERROR;
    }
    std::memcpy(dest_buffer, src_buffer, inputs_.at(i).DataSize());
  }
  MS_LOG(INFO) << "ConvertMSTensorToAiTensor success.";
  return lite::RET_OK;
}

int OfflineModelKernel::CopyHiaiTensorsDataToMSTensorsData() {
  MS_LOG(INFO) << "ConvertAiTensorToMSTensor ms_output tensor num:" << outputs_.size()
               << " ai_output tensor num:" << offline_model_outputs_tensors_.size();
  if (offline_model_outputs_tensors_.size() != outputs_.size()) {
    MS_LOG(ERROR) << "ms_output and ai_output have different size. ms_output tensor num:" << outputs_.size()
                  << " ai_output tensor num:" << offline_model_outputs_tensors_.size();
    return lite::RET_ERROR;
  }
  for (size_t i = 0; i < offline_model_outputs_tensors_.size(); i++) {
    if (offline_model_outputs_tensors_.at(i)->GetSize() != outputs_.at(i).DataSize()) {
      MS_LOG(ERROR) << "ms_output and ai_output have different dataSize. ms_output tensor dataSize "
                    << outputs_.at(i).DataSize()
                    << " ai_output tensor num:" << offline_model_outputs_tensors_.at(i)->GetSize();
      return lite::RET_ERROR;
    }
    auto src_buffer = offline_model_outputs_tensors_.at(i)->GetBuffer();
    if (src_buffer == nullptr) {
      MS_LOG(ERROR) << "For " << kHiaiModelName << ", the ai_output at [" << i << "], buffer is null.";
      return lite::RET_ERROR;
    }
    auto dest_buffer = outputs_.at(i).MutableData();
    if (dest_buffer == nullptr) {
      MS_LOG(ERROR) << "For " << kHiaiModelName << ", the ms_output at [" << i
                    << "], tensor name:" << outputs_.at(i).Name() << " buffer is null.";
      return lite::RET_ERROR;
    }
    std::memcpy(dest_buffer, src_buffer, offline_model_outputs_tensors_.at(i)->GetSize());
  }
  MS_LOG(INFO) << "ConvertAiTensorToMSTensor success.";
  return lite::RET_OK;
}
}  // namespace mindspore
