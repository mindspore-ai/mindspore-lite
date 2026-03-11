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

#include "src/litert/kernel/dsp/kernel/matmul_fusion.h"
#include <algorithm>
#include <string>
#include "src/litert/kernel_registry.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel/cpu/nnacl_c/matmul_parameter.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_MatMulFusion;

namespace mindspore::kernel {

int MatMulFusionDSPKernel::Prepare() { return RET_OK; }

int MatMulFusionDSPKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_2 && in_tensors_.size() != INPUT_TENSOR_SIZE_3) {
    MS_LOG(WARNING) << "MatMulFusion expects 2 or 3 inputs, got " << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "MatMulFusion expects 1 output, got " << out_tensors_.size();
    return RET_ERROR;
  }
  int M = 0, N = 0, K = 0;
  if (GetMNK(&M, &N, &K) != RET_OK) {
    MS_LOG(WARNING) << "MatMulFusion shape inference failed.";
    return RET_ERROR;
  }
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_3) {
    auto bias_shape = in_tensors_[2]->shape();
    if (bias_shape.size() != 2 || bias_shape[0] != M || bias_shape[1] != N) {
      MS_LOG(WARNING) << "Bias shape mismatch MxN: got " << bias_shape;
      return RET_ERROR;
    }
  }
  auto out_shape = out_tensors_[0]->shape();
  if (out_shape.size() != 2 || out_shape[0] != M || out_shape[1] != N) {
    MS_LOG(WARNING) << "Output shape mismatch expected (" << M << "," << N << ")";
    return RET_ERROR;
  }
  return RET_OK;
}

int MatMulFusionDSPKernel::GetMNK(int *M, int *N, int *K) const {
  if (M == nullptr || N == nullptr || K == nullptr) return RET_ERROR;
  const auto &a_shape = in_tensors_[0]->shape();
  const auto &b_shape = in_tensors_[1]->shape();
  if (a_shape.size() != 2 || b_shape.size() != 2) {
    MS_LOG(WARNING) << "A/B must be rank-2";
    return RET_ERROR;
  }
  int aM = a_shape[0];
  int aK = a_shape[1];
  int bK = b_shape[0];
  int bN = b_shape[1];
  if (aK != bK) {
    MS_LOG(WARNING) << "Inner dimension mismatch: " << aK << " vs " << bK;
    return RET_ERROR;
  }
  *M = aM;
  *K = aK;
  *N = bN;
  return RET_OK;
}

int MatMulFusionDSPKernel::GetActTypeCode(int *code) const {
  if (code == nullptr) return RET_ERROR;
  int act = 0;
  auto *param = reinterpret_cast<MatMulParameter *>(op_parameter_);
  if (param != nullptr) {
    switch (param->act_type_) {
      case ActType_Relu:
        act = 1;
        break;
      case ActType_Relu6:
        act = 2;
        break;
      default:
        act = 0;
        break;
    }
  }
  *code = act;
  return RET_OK;
}

int MatMulFusionDSPKernel::Run() {
  int M = 0, N = 0, K = 0;
  if (GetMNK(&M, &N, &K) != RET_OK) {
    MS_LOG(ERROR) << "MatMulFusion GetMNK failed";
    return RET_ERROR;
  }
  int act_code = 0;
  (void)GetActTypeCode(&act_code);

  auto allocator = dsp_runtime_->GetAllocator();
  uint64_t a_ptr = allocator->GetDeviceMemPtr(in_tensors_[0]->data());
  uint64_t b_ptr = allocator->GetDeviceMemPtr(in_tensors_[1]->data());
  uint64_t out_ptr = allocator->GetDeviceMemPtr(out_tensors_[0]->data());
  uint64_t bias_ptr = 0;
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_3) {
    bias_ptr = allocator->GetDeviceMemPtr(in_tensors_[2]->data());
  }

  auto data_type = in_tensors_[0]->data_type();
  auto mem_type = GetMemType();
  int dtype = static_cast<int>(data_type);
  int mtype = static_cast<int>(mem_type);

  int ret = matmul_fusion_func(a_ptr, b_ptr, out_ptr, bias_ptr, M, N, K, act_code, core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << this->name() << " Run failed!";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_MatMulFusion, DSPKernelCreator<MatMulFusionDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_MatMulFusion, DSPKernelCreator<MatMulFusionDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_MatMulFusion, DSPKernelCreator<MatMulFusionDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex64, PrimitiveType_MatMulFusion, DSPKernelCreator<MatMulFusionDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_MatMulFusion, DSPKernelCreator<MatMulFusionDSPKernel>)
#endif

#ifdef SUPPORT_FT78
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_MatMulFusion, DSPKernelCreator<MatMulFusionDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_MatMulFusion, DSPKernelCreator<MatMulFusionDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex128, PrimitiveType_MatMulFusion, DSPKernelCreator<MatMulFusionDSPKernel>)
#endif
}  // namespace mindspore::kernel
