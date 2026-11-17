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

#include "src/litert/kernel/dsp/kernel/nll_loss.h"
#include <cstdint>
#include "src/litert/kernel_registry.h"
#include "schema/inner/model_generated.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_NLLLoss;

namespace mindspore::kernel {

namespace {
constexpr size_t kLogitsIdx = 0;
constexpr size_t kLabelsIdx = 1;
constexpr size_t kWeightIdx = 2;
constexpr size_t kLossIdx = 0;
constexpr size_t kTotalWeightIdx = 1;
constexpr size_t kLogitsRank = 2;
constexpr size_t kVectorRank = 1;
constexpr size_t kDimsBatch = 0;
constexpr size_t kDimsClass = 1;
// SDK reduction codes (nll_loss.rst): 0=None, 1=Sum, 2=Mean.
constexpr int kSdkReductionNone = 0;
constexpr int kSdkReductionSum = 1;
constexpr int kSdkReductionMean = 2;
constexpr int kSdkReductionMin = 0;
constexpr int kSdkReductionMax = 2;

// Validate the three inputs' dtype/rank/sizes against the logits [N,C] shape.
bool CheckInputLayout(const std::vector<lite::Tensor *> &in, TypeId dtype) {
  if (dtype != kNumberTypeFloat32 && dtype != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "NLLLoss unsupported logits dtype: " << static_cast<int>(dtype);
    return false;
  }
  if (in[kLabelsIdx]->data_type() != kNumberTypeInt32) {
    MS_LOG(ERROR) << "NLLLoss labels must be int32.";
    return false;
  }
  if (in[kWeightIdx]->data_type() != dtype) {
    MS_LOG(ERROR) << "NLLLoss weight dtype mismatch with logits.";
    return false;
  }
  if (in[kLogitsIdx]->shape().size() != kLogitsRank || in[kLabelsIdx]->shape().size() != kVectorRank ||
      in[kWeightIdx]->shape().size() != kVectorRank) {
    MS_LOG(ERROR) << "NLLLoss expects logits 2D and labels/weight 1D.";
    return false;
  }
  const int batch = in[kLogitsIdx]->shape()[kDimsBatch];
  const int class_num = in[kLogitsIdx]->shape()[kDimsClass];
  if (batch < 1 || class_num < 1 || in[kLabelsIdx]->ElementsNum() != batch ||
      in[kWeightIdx]->ElementsNum() != class_num) {
    MS_LOG(ERROR) << "NLLLoss invalid logits shape or labels/weight size.";
    return false;
  }
  return true;
}
}  // namespace

int NLLLossDSPKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_3 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_2) {
    MS_LOG(ERROR) << "NLLLoss unexpected io sizes, in: " << in_tensors_.size() << ", out: " << out_tensors_.size();
    return RET_ERROR;
  }
  const auto dtype = in_tensors_[kLogitsIdx]->data_type();
  if (!CheckInputLayout(in_tensors_, dtype)) {
    return RET_ERROR;
  }
  if (out_tensors_[kLossIdx]->data_type() != dtype || out_tensors_[kTotalWeightIdx]->data_type() != dtype) {
    MS_LOG(ERROR) << "NLLLoss output dtype mismatch with logits.";
    return RET_ERROR;
  }
  const int batch = in_tensors_[kLogitsIdx]->shape()[kDimsBatch];
  // CheckSpecs runs before Prepare (which sets param_), so read op_parameter_ directly.
  auto *param = reinterpret_cast<const NLLLossParameter *>(op_parameter_);
  if (param == nullptr) {
    MS_LOG(ERROR) << "NLLLoss param is null.";
    return RET_ERROR;
  }
  const int sdk_reduction = GetSdkReduction(param->reduction_type_);
  if (sdk_reduction < kSdkReductionMin || sdk_reduction > kSdkReductionMax) {
    MS_LOG(ERROR) << "NLLLoss unsupported reduction: " << static_cast<int>(param->reduction_type_);
    return RET_ERROR;
  }
  // The SDK writes N per-sample losses for None and a single value for Sum/Mean.
  // Reject a mismatched output shape to avoid a device-side out-of-bounds write.
  const int64_t expected_loss_elems = sdk_reduction == kSdkReductionNone ? batch : 1;
  if (out_tensors_[kLossIdx]->ElementsNum() != expected_loss_elems ||
      out_tensors_[kTotalWeightIdx]->ElementsNum() != 1) {
    MS_LOG(ERROR) << "NLLLoss output shape inconsistent with reduction " << sdk_reduction;
    return RET_ERROR;
  }
  return RET_OK;
}

int NLLLossDSPKernel::Prepare() {
  param_ = reinterpret_cast<NLLLossParameter *>(op_parameter_);
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "NLLLoss param is null.";
    return RET_ERROR;
  }
  const auto &shape = in_tensors_[kLogitsIdx]->shape();
  batch_ = shape[kDimsBatch];
  class_num_ = shape[kDimsClass];
  return RET_OK;
}

int NLLLossDSPKernel::GetSdkReduction(ReductionType reduction) {
  switch (reduction) {
    case Reduction_Sum:
      return kSdkReductionSum;
    case Reduction_Mean:
      return kSdkReductionMean;
    case Reduction_None:
      return kSdkReductionNone;
    default:
      return -1;
  }
}

int NLLLossDSPKernel::ValidateLabels() {
  auto *data = in_tensors_[kLabelsIdx]->data();
  if (data == nullptr) {
    MS_LOG(ERROR) << "NLLLoss labels data is null.";
    return RET_ERROR;
  }
  const auto *labels = reinterpret_cast<const int32_t *>(data);
  for (int i = 0; i < batch_; ++i) {
    if (labels[i] < 0 || labels[i] >= class_num_) {
      MS_LOG(ERROR) << "NLLLoss label " << labels[i] << " out of range [0, " << class_num_ << ").";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int NLLLossDSPKernel::Run() {
  if (param_ == nullptr || batch_ < 1 || class_num_ < 1) {
    MS_LOG(ERROR) << "NLLLoss not prepared.";
    return RET_ERROR;
  }
  if (ValidateLabels() != RET_OK) {
    return RET_ERROR;
  }
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "NLLLoss allocator is null.";
    return RET_ERROR;
  }
  uint64_t logits_dev = allocator->GetDeviceMemPtr(in_tensors_[kLogitsIdx]->data());
  uint64_t labels_dev = allocator->GetDeviceMemPtr(in_tensors_[kLabelsIdx]->data());
  uint64_t weight_dev = allocator->GetDeviceMemPtr(in_tensors_[kWeightIdx]->data());
  uint64_t loss_dev = allocator->GetDeviceMemPtr(out_tensors_[kLossIdx]->data());
  uint64_t total_weight_dev = allocator->GetDeviceMemPtr(out_tensors_[kTotalWeightIdx]->data());
  if (logits_dev == 0 || labels_dev == 0 || weight_dev == 0 || loss_dev == 0 || total_weight_dev == 0) {
    MS_LOG(ERROR) << "NLLLoss device ptr is null.";
    return RET_ERROR;
  }
  const int dtype = static_cast<int>(in_tensors_[kLogitsIdx]->data_type());
  const int mtype = static_cast<int>(GetMemType());
  int ret = nll_loss_func(logits_dev, labels_dev, weight_dev, loss_dev, total_weight_dev, batch_, class_num_,
                          GetSdkReduction(param_->reduction_type_), core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << this->name() << " Run failed!";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_NLLLoss, DSPKernelCreator<NLLLossDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_NLLLoss, DSPKernelCreator<NLLLossDSPKernel>)
#endif
}  // namespace mindspore::kernel
