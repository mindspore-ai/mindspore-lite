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

#include "src/litert/kernel/dsp/kernel/nll_loss_grad.h"
#include <cstdint>
#include <vector>
#include "src/litert/kernel_registry.h"
#include "schema/inner/model_generated.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_NLLLossGrad;

namespace mindspore::kernel {

namespace {
constexpr size_t kLogitsIdx = 0;
constexpr size_t kLossGradIdx = 1;
constexpr size_t kLabelsIdx = 2;
constexpr size_t kWeightIdx = 3;
constexpr size_t kTotalWeightIdx = 4;
constexpr size_t kDxIdx = 0;
constexpr size_t kInputCount = 5;
constexpr size_t kMatrixRank = 2;
constexpr size_t kVectorRank = 1;
constexpr size_t kScalarRank = 0;
constexpr size_t kDimN = 0;
constexpr size_t kDimC = 1;
// SDK NLLLossGrad codes (nll_loss_grad.rst): 0=Sum, 1=Mean, 2=None.
// Lite ReductionType is Sum=0/Mean=1/None=2, so the numeric values happen to
// match -- unlike NLLLoss forward (SDK 0=None/1=Sum/2=Mean). Do not "fix" this
// by reusing the forward mapping.
constexpr int kSdkGradSum = 0;
constexpr int kSdkGradMean = 1;
constexpr int kSdkGradNone = 2;

// Validate input/output dtypes and shapes against the logits [N,C] layout.
bool CheckTensorLayout(const std::vector<lite::Tensor *> &in, const std::vector<lite::Tensor *> &out, TypeId dtype) {
  if (dtype != kNumberTypeFloat32 && dtype != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "NLLLossGrad unsupported dtype: " << static_cast<int>(dtype);
    return false;
  }
  if (in[kLabelsIdx]->data_type() != kNumberTypeInt32) {
    MS_LOG(ERROR) << "NLLLossGrad labels must be int32.";
    return false;
  }
  for (size_t i : {kLossGradIdx, kWeightIdx, kTotalWeightIdx}) {
    if (in[i]->data_type() != dtype) {
      MS_LOG(ERROR) << "NLLLossGrad input dtype mismatch with logits at index " << i;
      return false;
    }
  }
  if (out[kDxIdx]->data_type() != dtype) {
    MS_LOG(ERROR) << "NLLLossGrad output dtype mismatch with logits.";
    return false;
  }
  if (in[kLogitsIdx]->shape().size() != kMatrixRank || in[kLabelsIdx]->shape().size() != kVectorRank ||
      in[kWeightIdx]->shape().size() != kVectorRank || in[kTotalWeightIdx]->shape().size() != kScalarRank ||
      out[kDxIdx]->shape().size() != kMatrixRank) {
    MS_LOG(ERROR) << "NLLLossGrad expects logits/dx 2D, labels/weight 1D, total_weight scalar.";
    return false;
  }
  const int batch = in[kLogitsIdx]->shape()[kDimN];
  const int class_num = in[kLogitsIdx]->shape()[kDimC];
  if (batch < 1 || class_num < 1 || in[kLabelsIdx]->ElementsNum() != batch ||
      in[kWeightIdx]->ElementsNum() != class_num ||
      out[kDxIdx]->ElementsNum() != static_cast<int64_t>(batch) * class_num) {
    MS_LOG(ERROR) << "NLLLossGrad invalid shapes/sizes.";
    return false;
  }
  return true;
}
}  // namespace

int NLLLossGradDSPKernel::CheckSpecs() {
  if (in_tensors_.size() != kInputCount || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "NLLLossGrad unexpected io sizes, in: " << in_tensors_.size() << ", out: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (!CheckTensorLayout(in_tensors_, out_tensors_, in_tensors_[kLogitsIdx]->data_type())) {
    return RET_ERROR;
  }
  auto *param = reinterpret_cast<const NLLLossParameter *>(op_parameter_);
  if (param == nullptr) {
    MS_LOG(ERROR) << "NLLLossGrad param is null.";
    return RET_ERROR;
  }
  const int sdk_reduction = static_cast<int>(param->reduction_type_);
  if (sdk_reduction != kSdkGradSum && sdk_reduction != kSdkGradMean && sdk_reduction != kSdkGradNone) {
    MS_LOG(ERROR) << "NLLLossGrad unsupported reduction: " << sdk_reduction;
    return RET_ERROR;
  }
  // loss_grad is [N] for None and a scalar for Sum/Mean. Validate both rank and
  // element count: the SDK reads loss_grad[i] for i in [0,batch) under None,
  // while the framework infer only checks rank.
  const bool is_none = sdk_reduction == kSdkGradNone;
  const size_t expected_loss_grad_rank = is_none ? kVectorRank : kScalarRank;
  const int64_t expected_loss_grad_elems = is_none ? in_tensors_[kLogitsIdx]->shape()[kDimN] : 1;
  if (in_tensors_[kLossGradIdx]->shape().size() != expected_loss_grad_rank ||
      in_tensors_[kLossGradIdx]->ElementsNum() != expected_loss_grad_elems) {
    MS_LOG(ERROR) << "NLLLossGrad loss_grad shape inconsistent with reduction " << sdk_reduction;
    return RET_ERROR;
  }
  return RET_OK;
}

int NLLLossGradDSPKernel::Prepare() {
  param_ = reinterpret_cast<NLLLossParameter *>(op_parameter_);
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "NLLLossGrad param is null.";
    return RET_ERROR;
  }
  const auto &shape = in_tensors_[kLogitsIdx]->shape();
  batch_ = shape[kDimN];
  class_num_ = shape[kDimC];
  return RET_OK;
}

int NLLLossGradDSPKernel::ValidateLabels() {
  auto *data = in_tensors_[kLabelsIdx]->data();
  if (data == nullptr) {
    MS_LOG(ERROR) << "NLLLossGrad labels data is null.";
    return RET_ERROR;
  }
  const auto *labels = reinterpret_cast<const int32_t *>(data);
  for (int i = 0; i < batch_; ++i) {
    if (labels[i] < 0 || labels[i] >= class_num_) {
      MS_LOG(ERROR) << "NLLLossGrad label " << labels[i] << " out of range [0, " << class_num_ << ").";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int NLLLossGradDSPKernel::Run() {
  if (param_ == nullptr || batch_ < 1 || class_num_ < 1) {
    MS_LOG(ERROR) << "NLLLossGrad not prepared.";
    return RET_ERROR;
  }
  if (ValidateLabels() != RET_OK) {
    return RET_ERROR;
  }
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "NLLLossGrad allocator is null.";
    return RET_ERROR;
  }
  uint64_t logits_dev = allocator->GetDeviceMemPtr(in_tensors_[kLogitsIdx]->data());
  uint64_t loss_grad_dev = allocator->GetDeviceMemPtr(in_tensors_[kLossGradIdx]->data());
  uint64_t labels_dev = allocator->GetDeviceMemPtr(in_tensors_[kLabelsIdx]->data());
  uint64_t weight_dev = allocator->GetDeviceMemPtr(in_tensors_[kWeightIdx]->data());
  uint64_t total_weight_dev = allocator->GetDeviceMemPtr(in_tensors_[kTotalWeightIdx]->data());
  uint64_t dx_dev = allocator->GetDeviceMemPtr(out_tensors_[kDxIdx]->data());
  if (logits_dev == 0 || loss_grad_dev == 0 || labels_dev == 0 || weight_dev == 0 || total_weight_dev == 0 ||
      dx_dev == 0) {
    MS_LOG(ERROR) << "NLLLossGrad device ptr is null.";
    return RET_ERROR;
  }
  const int dtype = static_cast<int>(in_tensors_[kLogitsIdx]->data_type());
  const int mtype = static_cast<int>(GetMemType());
  int ret = nll_loss_grad_func(logits_dev, loss_grad_dev, labels_dev, weight_dev, total_weight_dev, dx_dev, batch_,
                               class_num_, static_cast<int>(param_->reduction_type_), core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << this->name() << " Run failed!";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_NLLLossGrad, DSPKernelCreator<NLLLossGradDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_NLLLossGrad, DSPKernelCreator<NLLLossGradDSPKernel>)
#endif
}  // namespace mindspore::kernel
