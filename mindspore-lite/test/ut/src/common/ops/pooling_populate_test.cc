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
#include <memory>
#include "common/common_test.h"
#include "schema/inner/model_generated.h"
#include "src/common/ops/populate/populate_register.h"
#include "nnacl_c/pooling_parameter.h"

namespace mindspore {
namespace {
// Build a flatbuffer Primitive wrapping an AvgPoolFusion with the given
// kernel_size/strides vectors, mimicking what a (legacy) .ms carries.
const schema::Primitive *BuildAvgPoolPrimitive(flatbuffers::FlatBufferBuilder &builder,
                                               const std::vector<int64_t> &kernel,
                                               const std::vector<int64_t> &strides) {
  auto kernel_off = builder.CreateVector(kernel);
  auto strides_off = builder.CreateVector(strides);
  schema::AvgPoolFusionBuilder afb(builder);
  afb.add_kernel_size(kernel_off);
  afb.add_strides(strides_off);
  auto fusion_off = afb.Finish().o;
  schema::PrimitiveBuilder pb(builder);
  pb.add_value_type(schema::PrimitiveType_AvgPoolFusion);
  pb.add_value(fusion_off);
  builder.Finish(pb.Finish());
  // Copy out so the buffer outlives the builder.
  static std::vector<uint8_t> storage;
  storage.assign(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
  return flatbuffers::GetRoot<schema::Primitive>(storage.data());
}
}  // namespace

class PoolingPopulateTest : public mindspore::CommonTest {
 public:
  PoolingPopulateTest() = default;
};

// A legacy model may carry size-1 kernel/strides (1D pooling over W). The populate
// layer used to reject it ("strides is invalid!"); now the H slot takes 1.
TEST_F(PoolingPopulateTest, AvgPoolPopulate_accepts_size1_kernel_strides) {
  flatbuffers::FlatBufferBuilder builder;
  auto prim = BuildAvgPoolPrimitive(builder, {8}, {8});
  ASSERT_NE(prim, nullptr);
  auto creator = lite::PopulateRegistry::GetInstance()->GetParameterCreator(
    static_cast<int>(schema::PrimitiveType_AvgPoolFusion), lite::SCHEMA_CUR);
  ASSERT_NE(creator, nullptr);
  OpParameter *opaque = creator(prim);
  ASSERT_NE(opaque, nullptr);
  auto *param = reinterpret_cast<PoolingParameter *>(opaque);
  ASSERT_EQ(param->window_h_, 1);
  ASSERT_EQ(param->window_w_, 8);
  ASSERT_EQ(param->stride_h_, 1);
  ASSERT_EQ(param->stride_w_, 8);
  free(opaque);
}

// The regular 2-element form keeps the H/W mapping unchanged.
TEST_F(PoolingPopulateTest, AvgPoolPopulate_keeps_2element_mapping) {
  flatbuffers::FlatBufferBuilder builder;
  auto prim = BuildAvgPoolPrimitive(builder, {2, 8}, {2, 8});
  ASSERT_NE(prim, nullptr);
  auto creator = lite::PopulateRegistry::GetInstance()->GetParameterCreator(
    static_cast<int>(schema::PrimitiveType_AvgPoolFusion), lite::SCHEMA_CUR);
  ASSERT_NE(creator, nullptr);
  OpParameter *opaque = creator(prim);
  ASSERT_NE(opaque, nullptr);
  auto *param = reinterpret_cast<PoolingParameter *>(opaque);
  ASSERT_EQ(param->window_h_, 2);
  ASSERT_EQ(param->window_w_, 8);
  ASSERT_EQ(param->stride_h_, 2);
  ASSERT_EQ(param->stride_w_, 8);
  free(opaque);
}
}  // namespace mindspore
