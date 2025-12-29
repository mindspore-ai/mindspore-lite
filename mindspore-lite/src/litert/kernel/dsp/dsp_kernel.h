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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_DSP_KERNEL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_DSP_KERNEL_H_

#include <vector>
#include <set>
#include <map>
#include <memory>
#include <string>
#include <cfloat>
#include "src/litert/lite_kernel.h"
#include "src/executor/kernel_exec.h"
#include "include/errorcode.h"
#include "src/litert/kernel/dsp/dsp_runtime.h"
#include "src/litert/kernel/dsp/dsp_allocator.h"
#include "src/litert/tensor_category.h"
#include "nnacl_c/resize_parameter.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
constexpr int INPUT_TENSOR_SIZE_1 = 1;
constexpr int INPUT_TENSOR_SIZE_2 = 2;
constexpr int INPUT_TENSOR_SIZE_3 = 3;
constexpr int INPUT_TENSOR_SIZE_4 = 4;
constexpr int INPUT_TENSOR_SIZE_5 = 5;
constexpr int INPUT_TENSOR_SIZE_6 = 6;
constexpr int INPUT_TENSOR_SIZE_16 = 16;
constexpr int OUTPUT_TENSOR_SIZE_1 = 1;
constexpr int OUTPUT_TENSOR_SIZE_2 = 2;
constexpr int OUTPUT_TENSOR_SIZE_3 = 3;
constexpr int OUTPUT_TENSOR_SIZE_4 = 4;

class DSPKernel : public LiteKernel {
 public:
  DSPKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
            const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    dsp_runtime_ = dsp_runtime_wrapper_.GetInstance();
  }
  ~DSPKernel() override = default;

  int Prepare() override { return RET_OK; }
  int PreProcess() override;
  int ReSize() override;
  int Run() override { return RET_ERROR; }

  bool MallocDataDone();
  virtual int CheckSpecs() { return RET_OK; }
  lite::dsp::MemType GetMemType() { return out_mem_type_; }
  void SetMemType(lite::dsp::MemType mem_type) { out_mem_type_ = mem_type; }
  void SetKernelArg(const std::vector<uint64_t> &kernel_args = {}) { kernel_args_ = kernel_args; }
  int InferShape() override;
  void SetCoreMask(int core_mask) { core_mask_ = core_mask; }
  void SetKernelName(const std::string &kernel_name) { kernel_name_ = kernel_name; }

 protected:
  lite::dsp::DSPRuntime *dsp_runtime_;
  std::vector<uint64_t> kernel_args_;
  lite::dsp::MemType out_mem_type_{lite::dsp::MemType::DDR};
  int core_mask_{0xf};
  std::string kernel_name_;

 private:
  lite::dsp::DSPRuntimeInnerWrapper dsp_runtime_wrapper_;
};

template <class T>
kernel::LiteKernel *DSPKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                     const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                     const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  auto *kernel = new (std::nothrow) T(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(WARNING) << "kernel " << opParameter->name_ << "is nullptr.";
    return nullptr;
  }
  auto shape = outputs.front()->shape();
  if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
    MS_LOG(WARNING) << "kernel " << opParameter->name_ << "don't infer shape yet!";
    return kernel;
  }
  if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
    MS_LOG(WARNING) << "kernel " << opParameter->name_ << "don't support output shape has zero.";
    delete kernel;
    return nullptr;
  }
  auto ret = kernel->CheckSpecs();
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(WARNING) << "Check " << opParameter->name_ << " specification failed!";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

inline const std::string GetDataTypePrefix(int data_type) {
  if (data_type == kNumberTypeFloat16) return "hp";
  if (data_type == kNumberTypeFloat32) return "fp";
  if (data_type == kNumberTypeFloat64) return "dp";
  if (data_type == kNumberTypeInt8) return "i8";
  if (data_type == kNumberTypeInt16) return "i16";
  if (data_type == kNumberTypeInt32) return "i32";
  if (data_type == kNumberTypeComplex64) return "c64";
  if (data_type == kNumberTypeComplex128) return "c128";
  return "";
}

inline char GetMemTypeSuffix(lite::dsp::MemType mem_type) {
  switch (mem_type) {
    case lite::dsp::MemType::DDR:
    case lite::dsp::MemType::SMC:
      return 's';
    case lite::dsp::MemType::L2:
      return 'p';
    default:
      return 's';
  }
}

inline std::string GenerateKernelName(int data_type, lite::dsp::MemType mem_type, const std::string op_name) {
  const std::string prefix = GetDataTypePrefix(data_type);
  char suffix = GetMemTypeSuffix(mem_type);
  auto result = prefix + "_" + op_name + "_" + suffix;
  return result;
}
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_DSP_KERNEL_H_
