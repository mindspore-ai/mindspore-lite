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
#ifndef LITE_OFFLINE_MODEL_KERNEL_H
#define LITE_OFFLINE_MODEL_KERNEL_H
#include <vector>
#include <utility>
#include <string>
#include <memory>
#include "include/api/kernel.h"
#include "src/common/log_adapter.h"
#include "src/litert/inner_context.h"
#include "include/errorcode.h"
#include "include/HiAiModelManagerService.h"
#include "src/litert/delegate/delegate_utils.h"

namespace mindspore {
class OfflineModelKernel : public kernel::Kernel {
  /**
   * We decide to make the whole model into one kernel.
   * */
 public:
  OfflineModelKernel(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs,
                     uint8_t *offline_model_buffer, size_t offline_model_size)
      : kernel::Kernel(inputs, outputs, nullptr, nullptr),
        offline_model_buffer_(offline_model_buffer),
        offline_model_size_(offline_model_size) {}
  int Prepare() override;
  int ReSize() override { return kSuccess; }
  int Execute() override;
  ~OfflineModelKernel();

 private:
  int BuildHiaiModel(uint8_t *modelData, size_t modelDataLength);
  int InitHiaiIOTensors();
  int InitHiaiTensorWithMSTensor(const std::vector<hiai::TensorDimension> &input_dimension,
                                 const std::vector<MSTensor> &ms_tensors,
                                 std::vector<std::shared_ptr<hiai::AiTensor>> &offline_model_tensors);

  int ExecuteHiaiModel();
  int CopyMSTensorsDataToHiaiTensorsData();
  int CopyHiaiTensorsDataToMSTensorsData();

  std::shared_ptr<hiai::AiModelMngerClient> model_manager_client_ = nullptr;
  std::shared_ptr<hiai::AiModelBuilder> model_builder_ = nullptr;
  uint8_t *offline_model_buffer_;
  size_t offline_model_size_;
  std::vector<std::shared_ptr<hiai::AiTensor>> offline_model_inputs_tensors_;
  std::vector<std::shared_ptr<hiai::AiTensor>> offline_model_outputs_tensors_;
};
}  // namespace mindspore
#endif  // LITE_OFFLINE_MODEL_KERNEL_H
