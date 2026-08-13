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
#include <cstdlib>
#include <memory>
#include <vector>

#include "common/common_test.h"
#include "nnacl_c/random_parameter.h"
#include "schema/model_generated.h"
#include "src/litert/inner_context.h"
#include "src/litert/kernel_registry.h"

namespace mindspore {
class RandomStandardNormalTest : public mindspore::CommonTest {};

TEST_F(RandomStandardNormalTest, Int32ShapeInputKernelRegistration) {
  lite::Tensor shape_tensor(kNumberTypeInt32, {2});
  int32_t shape_data[] = {2, 3};
  shape_tensor.set_data(shape_data);
  lite::Tensor output_tensor(kNumberTypeFloat32, {2, 3});
  float output_data[6] = {0};
  output_tensor.set_data(output_data);
  std::vector<lite::Tensor *> inputs = {&shape_tensor};
  std::vector<lite::Tensor *> outputs = {&output_tensor};

  auto *parameter = reinterpret_cast<RandomNormalParam *>(calloc(1, sizeof(RandomNormalParam)));
  ASSERT_NE(parameter, nullptr);
  parameter->op_parameter_.type_ = schema::PrimitiveType_RandomStandardNormal;
  parameter->seed_ = 1.0f;
  parameter->mean_ = 0.0f;
  parameter->scale_ = 1.0f;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt32, NHWC,
                            schema::PrimitiveType_RandomStandardNormal};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto context = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(context->Init(), lite::RET_OK);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), context.get(), desc);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(kernel->Prepare(), lite::RET_OK);
  EXPECT_EQ(kernel->Run(), lite::RET_OK);

  shape_tensor.set_data(nullptr);
  output_tensor.set_data(nullptr);
  delete kernel;
}
}  // namespace mindspore
