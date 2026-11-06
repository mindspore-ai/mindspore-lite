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
 * ============================================================================
 */
// Shape rules (same convention as the built-in BatchMatMulV3: right-aligned numpy broadcasting).
#ifndef REF_SHAPE_RULES_H_
#define REF_SHAPE_RULES_H_

#include <cstddef>
#include <cstdint>

#include "exe_graph/runtime/shape.h"  // gert::Shape (keep this header self-contained)

namespace inner_batch_matmul_ref {
// A matrix operand contributes two trailing dims ((M, K) / (K, N)); everything to their left broadcasts.
constexpr size_t kMatDims = 2;
constexpr size_t kMinRank = kMatDims;
constexpr size_t kMaxRank = 8;
constexpr uint32_t IBMM_MAX_BATCH_DIM = 6;

struct MatMulDims {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  size_t x1BatchRank = 0;
  size_t x2BatchRank = 0;
  size_t outBatchRank = 0;
  int64_t outBatch[IBMM_MAX_BATCH_DIM] = {0};
};

// Copies the static dims of an operand; the array reference pins the capacity so callers cannot
// pass a shorter buffer than the rank bound they validated against.
template <size_t N>
inline bool GetStaticDims(const gert::Shape &shape, size_t maxRank, int64_t (&dims)[N], size_t &rank) {
  rank = shape.GetDimNum();
  if (rank < kMatDims || rank > maxRank) {
    return false;
  }
  for (size_t i = 0; i < rank; ++i) {
    dims[i] = shape.GetDim(i);
    if (dims[i] < 0) {
      return false;
    }
  }
  return true;
}

// The op does not implement transposed multiply, so both operands are always read as [.., M, K] and
// [.., K, N]; a transpose request is rejected by the caller before the shapes are parsed.
inline bool GetMatMulDims(const int64_t *x1Dims, size_t x1Rank, const int64_t *x2Dims, size_t x2Rank,
                          MatMulDims &dims) {
  const int64_t m = x1Dims[x1Rank - kMatDims];
  const int64_t kA = x1Dims[x1Rank - 1];
  const int64_t kB = x2Dims[x2Rank - kMatDims];
  const int64_t n = x2Dims[x2Rank - 1];
  if (kA != kB || m <= 0 || n <= 0 || kA <= 0) {
    return false;
  }
  dims.m = m;
  dims.n = n;
  dims.k = kA;
  dims.x1BatchRank = x1Rank - kMatDims;
  dims.x2BatchRank = x2Rank - kMatDims;
  return true;
}

inline bool BroadcastBatchDims(const int64_t *x1BatchDims, size_t x1BatchRank, const int64_t *x2BatchDims,
                               size_t x2BatchRank, MatMulDims &dims) {
  const size_t outRank = x1BatchRank > x2BatchRank ? x1BatchRank : x2BatchRank;
  if (outRank > IBMM_MAX_BATCH_DIM) {
    return false;
  }
  for (size_t i = 0; i < outRank; ++i) {
    const size_t fromRight = outRank - 1 - i;
    const int64_t d1 = fromRight < x1BatchRank ? x1BatchDims[x1BatchRank - 1 - fromRight] : 1;
    const int64_t d2 = fromRight < x2BatchRank ? x2BatchDims[x2BatchRank - 1 - fromRight] : 1;
    if (d1 <= 0 || d2 <= 0) {
      return false;
    }
    if (d1 != d2 && d1 != 1 && d2 != 1) {
      return false;
    }
    dims.outBatch[outRank - 1 - fromRight] = d1 > d2 ? d1 : d2;
  }
  dims.outBatchRank = outRank;
  return true;
}

// Expand the batch dims of an input to 6 dims, right aligned (pad with 1; a broadcast dim is 1).
inline bool AlignBatchDims(const int64_t *inDims, size_t inBatchRank, const int64_t (&outBatch)[IBMM_MAX_BATCH_DIM],
                           size_t outBatchRank, int64_t (&aligned)[IBMM_MAX_BATCH_DIM]) {
  for (size_t i = 0; i < IBMM_MAX_BATCH_DIM; ++i) {
    aligned[i] = 1;
  }
  if (outBatchRank > IBMM_MAX_BATCH_DIM) {
    return false;
  }
  for (size_t fromRight = 0; fromRight < outBatchRank; ++fromRight) {
    const size_t pos = IBMM_MAX_BATCH_DIM - 1 - fromRight;
    const int64_t inDim = fromRight < inBatchRank ? inDims[inBatchRank - 1 - fromRight] : 1;
    const int64_t outDim = outBatch[outBatchRank - 1 - fromRight];
    if (inDim != outDim && inDim != 1) {
      return false;
    }
    aligned[pos] = inDim;
  }
  return true;
}
}  // namespace inner_batch_matmul_ref

#endif  // REF_SHAPE_RULES_H_
