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

#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_DSP_CUSTOM_KERNEL_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_DSP_CUSTOM_KERNEL_H_

#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <memory>
#include "src/litert/cxx_api/tensor/tensor_impl.h"
#include "src/tensor.h"
#include "include/registry/register_kernel_interface.h"
#include "include/registry/register_kernel.h"
#include "include/errorcode.h"
#include "src/litert/kernel/dsp/dsp_runtime.h"
#include "src/common/log_adapter.h"

using mindspore::kernel::Kernel;
namespace mindspore::lite {

inline void ConvertMSTensorsToLiteTensors(const std::vector<MSTensor> &src_tensors,
                                          std::vector<mindspore::lite::Tensor *> *dst_tensors);

class CustomKernel : public Kernel {
 public:
  CustomKernel(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs,
               const schema::Primitive *primitive, const mindspore::Context *ctx)
      : Kernel(inputs, outputs, primitive, ctx) {
    dsp_runtime_ = dsp_runtime_wrap_.GetInstance();
  }
  ~CustomKernel() override = default;
  // Prepare will be called during graph compilation
  int Prepare() override { return kSuccess; }
  // Execute is called to compute.
  int Execute() override;
  // called before Run
  virtual int PreProcess();
  virtual int Run() { return kSuccess; }
  virtual int CheckSpecs(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs) { return kSuccess; }
  int ReSize() override { return kSuccess; }
  void SetKernelArg(const std::vector<uint64_t> &kernel_args = {}) { kernel_args_ = kernel_args; }
  void SetMemType(lite::dsp::MemType mem_type) { out_mem_type_ = mem_type; }
  lite::dsp::MemType GetMemType() { return out_mem_type_; }
  void SetCoreMask(int core_mask) { core_mask_ = core_mask; }
  void SetKernelName(const std::string &kernel_name) { kernel_name_ = kernel_name; }
  std::vector<uint8_t> GetAttrByKey(std::string attr_key) { return attrs_[attr_key]; }
  void ParseAttrData();
  int CheckOutputs(const std::vector<mindspore::MSTensor> &outputs);
  void ConvertTensors();

 protected:
  std::vector<lite::Tensor *> in_tensors_;
  std::vector<lite::Tensor *> out_tensors_;
  std::vector<uint64_t> kernel_args_;
  lite::dsp::MemType out_mem_type_{lite::dsp::MemType::DDR};
  int core_mask_{0xf};
  std::string kernel_name_;
  lite::dsp::DSPRuntime *dsp_runtime_;

 private:
  lite::dsp::DSPRuntimeInnerWrapper dsp_runtime_wrap_;
  std::map<std::string, std::vector<uint8_t>> attrs_;
};

template <class T>
std::shared_ptr<T> CustomKernelCreator(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs,
                                       const schema::Primitive *primitive, const mindspore::Context *ctx) {
  auto kernel = std::make_shared<T>(inputs, outputs, primitive, ctx);
  auto type = kernel->GetAttr("type");
  auto ret = kernel->CheckSpecs(inputs, outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Check " << type << " specification failed!";
    return nullptr;
  }
  return kernel;
}
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_DSP_CUSTOM_KERNEL_H_
