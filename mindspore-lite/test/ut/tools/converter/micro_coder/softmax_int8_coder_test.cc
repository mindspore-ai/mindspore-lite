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
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>
#include "common/common_test.h"
#include "include/model.h"
#include "src/tensor.h"
#include "nnacl_c/softmax_parameter.h"
#include "coder/config.h"
#include "coder/context.h"
#include "coder/allocator/allocator.h"
#include "coder/generator/component/component.h"
#include "coder/opcoders/nnacl/int8/softmax_int8_coder.h"

namespace mindspore {
class TestSoftmaxInt8Coder : public mindspore::CommonTest {
 public:
  TestSoftmaxInt8Coder() {}
  static void SetUpTestCase() {
    // CoderContext assigns std::string from these globals, which default to
    // nullptr unless the generator flow initializes them.
    lite::micro::InitGlobalVariable(0);
  }
};

// Regression guard for the on-board OOB (#830): the SoftmaxInt8 kernel indexes
// sum_data per outer row (sum_data[o * inner_size + c]) over the whole outer
// extent that micro emits in a single kernel call, so the coder's workspace
// must be outer_size * inner_size ints, not inner_size. An undersized buffer
// overflows the generated arena and corrupts adjacent memory on device while
// x86 simulation still passes (the kernel reads back its own OOB writes).
static void CheckSoftmaxInt8Workspace(const std::vector<int> &shape, int axis) {
  auto *allocator = lite::micro::MemoryAllocator::GetInstance();
  allocator->Free();

  auto axis_norm = axis < 0 ? axis + static_cast<int>(shape.size()) : axis;
  ASSERT_TRUE(axis_norm >= 0 && axis_norm < static_cast<int>(shape.size()));
  size_t elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
  size_t outer_size = 1;
  for (int i = 0; i < axis_norm; i++) {
    outer_size *= static_cast<size_t>(shape[i]);
  }
  size_t inner_size = 1;
  for (size_t i = axis_norm + 1; i < shape.size(); i++) {
    inner_size *= static_cast<size_t>(shape[i]);
  }
  size_t exp_bytes = elements * sizeof(int32_t);
  size_t sum_bytes = outer_size * inner_size * sizeof(int32_t);

  lite::Tensor in_tensor(kNumberTypeInt8, shape);
  lite::Tensor out_tensor(kNumberTypeInt8, shape);
  lite::LiteQuantParam in_quant;
  in_quant.scale = 0.0352941;
  in_quant.zeroPoint = -128;
  in_tensor.AddQuantParam(in_quant);
  lite::LiteQuantParam out_quant;
  out_quant.scale = 0.00392157;
  out_quant.zeroPoint = -128;
  out_tensor.AddQuantParam(out_quant);

  lite::LiteGraph::Node node;
  node.name_ = "Default/Softmax-op0";

  lite::micro::nnacl::SoftMaxInt8Coder coder({&in_tensor}, {&out_tensor}, &node, 0, lite::micro::Target::kX86);
  // ~OperatorCoder free()s the parameter, so it must be malloc'd, not new'd.
  auto *param = static_cast<SoftmaxParameter *>(malloc(sizeof(SoftmaxParameter)));
  ASSERT_NE(param, nullptr);
  memset(param, 0, sizeof(SoftmaxParameter));
  param->op_parameter_.type_ = schema::PrimitiveType_Softmax;
  param->axis_ = axis;
  coder.set_parameter(reinterpret_cast<OpParameter *>(param));

  lite::micro::CoderContext context(0);  // requires InitGlobalVariable() to have run
  ASSERT_EQ(lite::RET_OK, coder.Prepare(&context));
  EXPECT_EQ(allocator->total_buffer_size(), exp_bytes + sum_bytes);

  allocator->Free();
}

TEST_F(TestSoftmaxInt8Coder, SoftmaxInt8CoderWorkspaceSize8x1x66Axis2) { CheckSoftmaxInt8Workspace({8, 1, 66}, 2); }

TEST_F(TestSoftmaxInt8Coder, SoftmaxInt8CoderWorkspaceSize2x1536x2Axis2) { CheckSoftmaxInt8Workspace({2, 1536, 2}, 2); }

TEST_F(TestSoftmaxInt8Coder, SoftmaxInt8CoderWorkspaceSize4DAxis2) { CheckSoftmaxInt8Workspace({1, 2, 3, 4}, 2); }

TEST_F(TestSoftmaxInt8Coder, SoftmaxInt8CoderWorkspaceSizeNegativeAxis) { CheckSoftmaxInt8Workspace({2, 3, 4}, -1); }
}  // namespace mindspore
