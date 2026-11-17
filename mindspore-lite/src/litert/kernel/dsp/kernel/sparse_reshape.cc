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

#include "src/litert/kernel/dsp/kernel/sparse_reshape.h"
#include <climits>
#include <cstdint>
#include <vector>
#include "src/litert/kernel_registry.h"
#include "schema/inner/model_generated.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SparseReshape;

namespace mindspore::kernel {

namespace {
constexpr size_t kIndicesIdx = 0;
constexpr size_t kInShapeIdx = 1;
constexpr size_t kOutShapeIdx = 2;
constexpr size_t kOutIndicesIdx = 0;
constexpr size_t kOutOutShapeIdx = 1;
// A requested output dimension of -1 is inferred from the input dense size.
constexpr int32_t kUnknownDim = -1;
// Sentinel for "the -1 dimension has not been seen yet" in ResolveOutShape.
constexpr int kNoUnknownDim = -1;
constexpr int kMinRank = 1;
constexpr int kIndicesRank = 2;

// Multiply into an int64 accumulator, refusing to wrap on overflow.
bool MulOverflow(int64_t x, int64_t y, int64_t *result) { return __builtin_mul_overflow(x, y, result); }

// Product of a shape array into an int64 accumulator. Rejects negative dims and
// overflow. `allow_unknown` permits a single -1 placeholder (SparseReshape newshape),
// reported via *unknown_dim (index, or -1 when absent); the product excludes it.
bool ShapeProduct(const int32_t *shape, int rank, bool allow_unknown, int64_t *product, int *unknown_dim) {
  *product = 1;
  if (unknown_dim != nullptr) {
    *unknown_dim = kNoUnknownDim;
  }
  for (int i = 0; i < rank; ++i) {
    const int32_t dim = shape[i];
    if (allow_unknown && dim == kUnknownDim) {
      if (unknown_dim == nullptr || *unknown_dim != kNoUnknownDim) {
        MS_LOG(ERROR) << "SparseReshape allows at most one -1 dimension.";
        return false;
      }
      *unknown_dim = i;
      continue;
    }
    if (dim < 0) {
      MS_LOG(ERROR) << "SparseReshape shape rank-" << i << " must be non-negative, got " << dim;
      return false;
    }
    if (MulOverflow(*product, static_cast<int64_t>(dim), product)) {
      MS_LOG(ERROR) << "SparseReshape shape product overflows.";
      return false;
    }
  }
  return true;
}
}  // namespace

int SparseReshapeDSPKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_3 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_2) {
    MS_LOG(ERROR) << "SparseReshape unexpected io sizes, in: " << in_tensors_.size()
                  << ", out: " << out_tensors_.size();
    return RET_ERROR;
  }
  auto *indices = in_tensors_[kIndicesIdx];
  auto *in_shape = in_tensors_[kInShapeIdx];
  auto *in_out_shape = in_tensors_[kOutShapeIdx];
  auto *out_indices = out_tensors_[kOutIndicesIdx];
  auto *out_out_shape = out_tensors_[kOutOutShapeIdx];
  // All five tensors are int32 index/shape tensors; verify uniformly.
  lite::Tensor *all_tensors[] = {indices, in_shape, in_out_shape, out_indices, out_out_shape};
  for (lite::Tensor *t : all_tensors) {
    if (t->data_type() != kNumberTypeInt32) {
      MS_LOG(ERROR) << "SparseReshape requires int32 index/shape tensors, got " << static_cast<int>(t->data_type());
      return RET_ERROR;
    }
  }
  if (indices->shape().size() != kIndicesRank) {
    MS_LOG(ERROR) << "SparseReshape indices must be rank 2, got " << indices->shape().size();
    return RET_ERROR;
  }
  const int64_t input_rank = in_shape->ElementsNum();
  const int64_t output_rank = in_out_shape->ElementsNum();
  if (input_rank < kMinRank || output_rank < kMinRank) {
    MS_LOG(ERROR) << "SparseReshape shape tensors must be non-empty.";
    return RET_ERROR;
  }
  const int64_t n = indices->shape()[0];
  if (indices->shape()[1] != input_rank) {
    MS_LOG(ERROR) << "SparseReshape indices inner size " << indices->shape()[1] << " != input rank " << input_rank;
    return RET_ERROR;
  }
  // Keep the empty-indices rejection identical to Run(); the DSP kernel requires N > 0.
  if (n <= 0) {
    MS_LOG(ERROR) << "SparseReshape expects non-empty indices, got N=" << n;
    return RET_ERROR;
  }
  if (out_out_shape->ElementsNum() != output_rank || indices->ElementsNum() != n * input_rank ||
      out_indices->ElementsNum() != n * output_rank) {
    MS_LOG(ERROR) << "SparseReshape tensor element counts inconsistent with ranks, N=" << n
                  << ", in_rank=" << input_rank << ", out_rank=" << output_rank;
    return RET_ERROR;
  }
  return RET_OK;
}

int SparseReshapeDSPKernel::Prepare() { return RET_OK; }

int SparseReshapeDSPKernel::ResolveOutShape(const int32_t *inshape, const int32_t *outshape, int input_rank,
                                            int output_rank, int32_t *resolved) const {
  if (inshape == nullptr || outshape == nullptr || resolved == nullptr) {
    return RET_ERROR;
  }
  int64_t dense_size = 1;
  int64_t out_num = 1;
  if (!ShapeProduct(inshape, input_rank, false, &dense_size, nullptr)) {
    return RET_ERROR;
  }
  int unknown = kNoUnknownDim;
  if (!ShapeProduct(outshape, output_rank, true, &out_num, &unknown)) {
    return RET_ERROR;
  }
  for (int d = 0; d < output_rank; ++d) {
    resolved[d] = outshape[d];
  }
  if (unknown == kNoUnknownDim) {
    if (out_num != dense_size) {
      MS_LOG(ERROR) << "SparseReshape output dense size " << out_num << " != input dense size " << dense_size;
      return RET_ERROR;
    }
    return RET_OK;
  }
  if (out_num == 0) {
    MS_LOG(ERROR) << "SparseReshape cannot infer -1 with a zero-sized newshape dimension.";
    return RET_ERROR;
  }
  const int64_t missing = dense_size / out_num;
  int64_t reconstructed = 0;
  if (MulOverflow(out_num, missing, &reconstructed) || reconstructed != dense_size) {
    MS_LOG(ERROR) << "SparseReshape requested shape is not compatible with input dense size " << dense_size;
    return RET_ERROR;
  }
  // The inferred dimension is written back as int32; reject values that would truncate.
  if (missing <= 0 || missing > INT32_MAX) {
    MS_LOG(ERROR) << "SparseReshape inferred dimension out of int32 range: " << missing;
    return RET_ERROR;
  }
  resolved[unknown] = static_cast<int32_t>(missing);
  return RET_OK;
}

int SparseReshapeDSPKernel::ResolveAndWriteOutShape(int input_rank, int output_rank, int64_t *n) {
  auto *indices = in_tensors_[kIndicesIdx];
  auto *in_shape_tensor = in_tensors_[kInShapeIdx];
  auto *in_out_shape_tensor = in_tensors_[kOutShapeIdx];
  auto *out_out_shape = out_tensors_[kOutOutShapeIdx];
  if (indices->data() == nullptr || in_shape_tensor->data() == nullptr || in_out_shape_tensor->data() == nullptr ||
      out_out_shape->data() == nullptr) {
    MS_LOG(ERROR) << "SparseReshape tensor data is null.";
    return RET_ERROR;
  }
  if (indices->shape().size() != kIndicesRank) {
    MS_LOG(ERROR) << "SparseReshape indices must be rank 2, got " << indices->shape().size();
    return RET_ERROR;
  }
  *n = indices->shape()[0];
  if (*n <= 0) {
    MS_LOG(ERROR) << "SparseReshape expects non-empty indices, got N=" << *n;
    return RET_ERROR;
  }
  const auto *inshape = reinterpret_cast<const int32_t *>(in_shape_tensor->data());
  const auto *outshape = reinterpret_cast<const int32_t *>(in_out_shape_tensor->data());
  std::vector<int32_t> resolved(output_rank, 0);
  if (ResolveOutShape(inshape, outshape, input_rank, output_rank, resolved.data()) != RET_OK) {
    return RET_ERROR;
  }
  auto *out_out_shape_data = reinterpret_cast<int32_t *>(out_out_shape->data());
  for (int d = 0; d < output_rank; ++d) {
    out_out_shape_data[d] = resolved[d];
  }
  return RET_OK;
}

int SparseReshapeDSPKernel::Run() {
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "SparseReshape allocator is null.";
    return RET_ERROR;
  }
  auto *indices = in_tensors_[kIndicesIdx];
  auto *in_shape_tensor = in_tensors_[kInShapeIdx];
  auto *out_indices = out_tensors_[kOutIndicesIdx];
  auto *out_out_shape = out_tensors_[kOutOutShapeIdx];
  if (out_indices->data() == nullptr) {
    MS_LOG(ERROR) << "SparseReshape out indices data is null.";
    return RET_ERROR;
  }
  const int input_rank = static_cast<int>(in_shape_tensor->ElementsNum());
  const int output_rank = static_cast<int>(out_out_shape->ElementsNum());
  int64_t n = 0;
  if (ResolveAndWriteOutShape(input_rank, output_rank, &n) != RET_OK) {
    return RET_ERROR;
  }

  uint64_t indices_dev = allocator->GetDeviceMemPtr(indices->data());
  uint64_t out_indices_dev = allocator->GetDeviceMemPtr(out_indices->data());
  uint64_t in_shape_dev = allocator->GetDeviceMemPtr(in_shape_tensor->data());
  uint64_t out_shape_dev = allocator->GetDeviceMemPtr(out_out_shape->data());
  if (indices_dev == 0 || out_indices_dev == 0 || in_shape_dev == 0 || out_shape_dev == 0) {
    MS_LOG(ERROR) << "SparseReshape device ptr is null.";
    return RET_ERROR;
  }

  const int dtype = static_cast<int>(indices->data_type());
  const int mtype = static_cast<int>(GetMemType());
  // The resolved output shape serves as both the target shape and the out-shape output.
  int ret = sparse_reshape_func(indices_dev, out_indices_dev, in_shape_dev, out_shape_dev, out_shape_dev,
                                static_cast<uint64_t>(input_rank), static_cast<uint64_t>(output_rank),
                                static_cast<uint64_t>(n), core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << this->name() << " Run failed!";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_SparseReshape, DSPKernelCreator<SparseReshapeDSPKernel>)
}  // namespace mindspore::kernel
