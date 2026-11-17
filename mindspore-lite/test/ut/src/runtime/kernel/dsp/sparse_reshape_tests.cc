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

#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>
#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel/dsp/dsp_subgraph.h"
#include "src/litert/kernel_registry.h"

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

namespace {
constexpr int kUnknownDim = -1;
constexpr int kRejectN = 4;
constexpr int32_t kIndexSeedMultiplier = 7;
constexpr int32_t kIndexSeedIncrement = 3;

// Reference: flat index from the input shape, then decomposed by the resolved output shape.
void RefSparseReshape(const int32_t *in_indices, const std::vector<int32_t> &in_shape,
                      const std::vector<int32_t> &out_shape, int n, int32_t *out_indices) {
  const int input_rank = static_cast<int>(in_shape.size());
  const int output_rank = static_cast<int>(out_shape.size());
  std::vector<int64_t> in_stride(input_rank, 1);
  for (int d = input_rank - 2; d >= 0; --d) {
    in_stride[d] = in_stride[d + 1] * in_shape[d + 1];
  }
  std::vector<int64_t> out_stride(output_rank, 1);
  for (int d = output_rank - 2; d >= 0; --d) {
    out_stride[d] = out_stride[d + 1] * out_shape[d + 1];
  }
  for (int i = 0; i < n; ++i) {
    int64_t flat = 0;
    for (int j = 0; j < input_rank; ++j) {
      flat += in_indices[i * input_rank + j] * in_stride[j];
    }
    for (int j = 0; j < output_rank; ++j) {
      out_indices[i * output_rank + j] = static_cast<int32_t>(flat / out_stride[j]);
      flat %= out_stride[j];
    }
  }
}

// Deterministic in-bounds coordinates derived from a rolling unsigned counter
// (unsigned wrap keeps the modulo result non-negative, unlike signed overflow).
void FillIndices(int32_t *indices, int n, const std::vector<int32_t> &in_shape) {
  const int input_rank = static_cast<int>(in_shape.size());
  uint32_t seed = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < input_rank; ++j) {
      seed = seed * static_cast<uint32_t>(kIndexSeedMultiplier) + static_cast<uint32_t>(kIndexSeedIncrement);
      // in_shape dimensions are positive in every test case; guard against divide-by-zero.
      const uint32_t dim = static_cast<uint32_t>(in_shape[j] > 0 ? in_shape[j] : 1);
      indices[i * input_rank + j] = static_cast<int32_t>(seed % dim);
    }
  }
}

// Ownership transfers to the kernel (~LiteKernel frees op_parameter_ with free()),
// so the parameter must come from malloc rather than new.
OpParameter *CreateParam() {
  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (param == nullptr) {
    return nullptr;
  }
  std::memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_SparseReshape);
  return param;
}

std::vector<int32_t> ResolveOutShape(const std::vector<int32_t> &in_shape, const std::vector<int32_t> &new_shape) {
  int64_t dense =
    std::accumulate(in_shape.begin(), in_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  int64_t dividend = 1;
  int unknown = -1;
  std::vector<int32_t> resolved(new_shape.begin(), new_shape.end());
  for (size_t d = 0; d < new_shape.size(); ++d) {
    if (new_shape[d] == kUnknownDim) {
      unknown = static_cast<int>(d);
    } else {
      dividend *= new_shape[d];
    }
  }
  if (unknown != -1) {
    resolved[unknown] = static_cast<int32_t>(dense / dividend);
  }
  return resolved;
}
}  // namespace

class TestDSP_SparseReshape : public DSPCommonTest {
 public:
  void RunCase(const std::vector<int32_t> &in_shape, const std::vector<int32_t> &new_shape, int n) {
    InitDSPRuntime();
    const int input_rank = static_cast<int>(in_shape.size());
    const int output_rank = static_cast<int>(new_shape.size());
    const std::vector<int32_t> out_shape = ResolveOutShape(in_shape, new_shape);

    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;

    auto *t_indices =
      new lite::Tensor(kNumberTypeInt32, {n, input_rank}, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, t_indices->MallocData(allocator_));
    ASSERT_NE(t_indices->MutableData(), nullptr);
    inputs.push_back(t_indices);
    auto *indices_data = reinterpret_cast<int32_t *>(t_indices->MutableData());
    FillIndices(indices_data, n, in_shape);

    auto *t_in_shape = new lite::Tensor(kNumberTypeInt32, {input_rank}, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, t_in_shape->MallocData(allocator_));
    ASSERT_NE(t_in_shape->MutableData(), nullptr);
    std::memcpy(t_in_shape->MutableData(), in_shape.data(), sizeof(int32_t) * input_rank);
    inputs.push_back(t_in_shape);

    auto *t_new_shape =
      new lite::Tensor(kNumberTypeInt32, {output_rank}, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, t_new_shape->MallocData(allocator_));
    ASSERT_NE(t_new_shape->MutableData(), nullptr);
    std::memcpy(t_new_shape->MutableData(), new_shape.data(), sizeof(int32_t) * output_rank);
    inputs.push_back(t_new_shape);

    auto *t_out_indices =
      new lite::Tensor(kNumberTypeInt32, {n, output_rank}, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, t_out_indices->MallocData(allocator_));
    ASSERT_NE(t_out_indices->MutableData(), nullptr);
    std::memset(t_out_indices->MutableData(), 0, sizeof(int32_t) * n * output_rank);
    outputs.push_back(t_out_indices);

    auto *t_out_shape =
      new lite::Tensor(kNumberTypeInt32, {output_rank}, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, t_out_shape->MallocData(allocator_));
    ASSERT_NE(t_out_shape->MutableData(), nullptr);
    std::memset(t_out_shape->MutableData(), 0, sizeof(int32_t) * output_rank);
    outputs.push_back(t_out_shape);

    std::vector<int32_t> ref_indices(static_cast<size_t>(n) * output_rank, 0);
    RefSparseReshape(indices_data, in_shape, out_shape, n, ref_indices.data());

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    auto *param = CreateParam();
    ASSERT_NE(param, nullptr);
    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_SparseReshape};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel = creator(inputs, outputs, param, ctx, key);
    ASSERT_NE(kernel, nullptr);

    ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
    ASSERT_EQ(kernel->Run(), lite::RET_OK);

    auto *actual_indices = reinterpret_cast<int32_t *>(t_out_indices->MutableData());
    for (int i = 0; i < n * output_rank; ++i) {
      EXPECT_EQ(ref_indices[i], actual_indices[i]) << "index element " << i;
    }
    auto *actual_shape = reinterpret_cast<int32_t *>(t_out_shape->MutableData());
    for (int d = 0; d < output_rank; ++d) {
      EXPECT_EQ(out_shape[d], actual_shape[d]) << "outshape dim " << d;
    }

    delete kernel;
    for (auto *t : inputs) {
      delete t;
    }
    for (auto *t : outputs) {
      delete t;
    }
    delete ctx;
    UninitDSPRuntime();
  }

  // creator must reject mismatched out-shape size so the op falls back to CPU.
  void ExpectRejected(const std::vector<int32_t> &in_shape, int new_shape_elems, int out_shape_elems) {
    InitDSPRuntime();
    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    const int input_rank = static_cast<int>(in_shape.size());
    inputs.push_back(
      new lite::Tensor(kNumberTypeInt32, {kRejectN, input_rank}, mindspore::NHWC, lite::Category::CONST_TENSOR));
    inputs.push_back(new lite::Tensor(kNumberTypeInt32, {input_rank}, mindspore::NHWC, lite::Category::CONST_TENSOR));
    inputs.push_back(
      new lite::Tensor(kNumberTypeInt32, {new_shape_elems}, mindspore::NHWC, lite::Category::CONST_TENSOR));
    outputs.push_back(
      new lite::Tensor(kNumberTypeInt32, {kRejectN, new_shape_elems}, mindspore::NHWC, lite::Category::CONST_TENSOR));
    outputs.push_back(
      new lite::Tensor(kNumberTypeInt32, {out_shape_elems}, mindspore::NHWC, lite::Category::CONST_TENSOR));

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    auto *param = CreateParam();
    ASSERT_NE(param, nullptr);
    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_SparseReshape};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel = creator(inputs, outputs, param, ctx, key);
    EXPECT_EQ(kernel, nullptr);

    delete kernel;
    for (auto *t : inputs) {
      delete t;
    }
    for (auto *t : outputs) {
      delete t;
    }
    delete ctx;
    UninitDSPRuntime();
  }
};

TEST_F(TestDSP_SparseReshape, Rank3_To_Rank2) { RunCase({2, 3, 4}, {6, 4}, 16); }
TEST_F(TestDSP_SparseReshape, Rank2_To_Rank3) { RunCase({4, 6}, {2, 3, 4}, 20); }
TEST_F(TestDSP_SparseReshape, Rank2_To_Rank2_SameShape) { RunCase({4, 6}, {4, 6}, 12); }
TEST_F(TestDSP_SparseReshape, Rank2_To_Rank2_WithUnknownDim) { RunCase({4, 6}, {-1, 4}, 10); }
TEST_F(TestDSP_SparseReshape, RejectShapeSizeMismatch) { ExpectRejected({4, 6}, 2, 3); }

}  // namespace mindspore::lite::dsp::test

#endif  // SUPPORT_FT04
