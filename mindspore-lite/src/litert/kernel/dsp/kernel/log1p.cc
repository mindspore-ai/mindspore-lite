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

#include "src/litert/kernel/dsp/kernel/log1p.h"
#include <climits>
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Log1p;

namespace mindspore::kernel {

constexpr int kInIdx = 0;
constexpr int kOutIdx = 0;

// log1p_func is keyed by the INPUT dtype. Integer inputs produce float output
// (i8/i16/i32_log1p_* write float), while floating inputs keep their dtype.
// NOTE: LiteRT's Log1p infer (CommonInferShape) sets output dtype == input dtype,
// so a real graph never produces the int-in/fp32-out combination these integer
// kernels require. The integer REG_KERNEL entries below are therefore registered
// but effectively unreachable in real models (CheckSpecs rejects and falls back
// to CPU); they are kept for dsplib parity. The fp32/fp16 paths are the usable ones.
static TypeId ExpectedOutDtype(TypeId in_dtype) {
  switch (in_dtype) {
    case kNumberTypeInt8:
    case kNumberTypeInt16:
    case kNumberTypeInt32:
      return kNumberTypeFloat32;
    default:
      return in_dtype;
  }
}

static bool IsSupportedLog1pDtype(TypeId dtype) {
  if (dtype == kNumberTypeFloat32 || dtype == kNumberTypeInt16 || dtype == kNumberTypeInt32) {
    return true;
  }
#ifdef SUPPORT_FT04
  if (dtype == kNumberTypeFloat16) {
    return true;
  }
#endif
#ifdef SUPPORT_FT78
  if (dtype == kNumberTypeInt8 || dtype == kNumberTypeFloat64) {
    return true;
  }
#endif
  return false;
}

int Log1pDSPKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "log1p kernel input size invalid: in=" << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "log1p kernel output size invalid: out=" << out_tensors_.size();
    return RET_ERROR;
  }
  auto *input = in_tensors_[kInIdx];
  auto *output = out_tensors_[kOutIdx];
  if (!IsSupportedLog1pDtype(input->data_type())) {
    MS_LOG(ERROR) << "log1p unsupported input dtype: " << static_cast<int>(input->data_type());
    return RET_ERROR;
  }
  if (output->data_type() != ExpectedOutDtype(input->data_type())) {
    MS_LOG(ERROR) << "log1p output dtype mismatch: in=" << static_cast<int>(input->data_type())
                  << " out=" << static_cast<int>(output->data_type());
    return RET_ERROR;
  }
  const auto elems = output->ElementsNum();
  if (elems <= 0 || elems > INT_MAX) {
    MS_LOG(ERROR) << "log1p invalid length: " << elems;
    return RET_ERROR;
  }
  return RET_OK;
}

int Log1pDSPKernel::Prepare() { return RET_OK; }

int Log1pDSPKernel::Run() {
  auto *input = in_tensors_[kInIdx];
  auto *output = out_tensors_[kOutIdx];

  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "Log1pDSPKernel allocator is null.";
    return RET_ERROR;
  }
  if (input->data() == nullptr || output->data() == nullptr) {
    MS_LOG(ERROR) << "Log1pDSPKernel input/output data is null.";
    return RET_ERROR;
  }
  uint64_t in_ptr = allocator->GetDeviceMemPtr(input->data());
  uint64_t out_ptr = allocator->GetDeviceMemPtr(output->data());
  if (in_ptr == 0 || out_ptr == 0) {
    MS_LOG(ERROR) << "Log1pDSPKernel device ptr is null.";
    return RET_ERROR;
  }

  const auto elems = output->ElementsNum();
  const int dtype = static_cast<int>(input->data_type());
  const int mtype = static_cast<int>(GetMemType());
  int ret = log1p_func(in_ptr, out_ptr, static_cast<uint64_t>(elems), core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Log1pDSPKernel run failed, dtype=" << dtype;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_Log1p, DSPKernelCreator<Log1pDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_Log1p, DSPKernelCreator<Log1pDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_Log1p, DSPKernelCreator<Log1pDSPKernel>)

#ifdef SUPPORT_FT04
// FT04 only: fp16.
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_Log1p, DSPKernelCreator<Log1pDSPKernel>)
#endif

#ifdef SUPPORT_FT78
// FT78 only: int8 and fp64. No FT78 on-device test in this change.
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_Log1p, DSPKernelCreator<Log1pDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_Log1p, DSPKernelCreator<Log1pDSPKernel>)
#endif
}  // namespace mindspore::kernel
