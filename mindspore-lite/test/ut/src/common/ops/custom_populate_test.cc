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
#include <cstring>
#include <vector>
#include "common/common_test.h"
#include "schema/ops_generated.h"
#include "src/common/ops/populate/populate_register.h"
#include "nnacl_c/split_parameter.h"

namespace mindspore {
namespace {
// Wrap arbitrary bytes as the attr payload a SplitReduceConcatFusion Custom op
// carries, mirroring what the online fusion pass serializes for legit models.
const schema::Primitive *BuildSplitReduceConcatPrimitive(flatbuffers::FlatBufferBuilder &builder,
                                                         const std::vector<std::vector<uint8_t>> &attrs_data) {
  std::vector<flatbuffers::Offset<schema::Attribute>> attrs;
  int index = 0;
  for (const auto &data : attrs_data) {
    std::string name = std::to_string(index++);
    attrs.emplace_back(schema::CreateAttributeDirect(builder, name.c_str(), &data));
  }
  auto custom_off = schema::CreateCustomDirect(builder, "SplitReduceConcatFusion", &attrs);
  schema::PrimitiveBuilder pb(builder);
  pb.add_value_type(schema::PrimitiveType_Custom);
  pb.add_value(custom_off.o);
  builder.Finish(pb.Finish());
  // Copy out so the buffer outlives the builder.
  static std::vector<uint8_t> storage;
  storage.assign(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
  return flatbuffers::GetRoot<schema::Primitive>(storage.data());
}

std::vector<uint8_t> StructBytes(const SplitParameter &param) {
  const auto *begin = reinterpret_cast<const uint8_t *>(&param);
  return std::vector<uint8_t>(begin, begin + sizeof(SplitParameter));
}

lite::ParameterGen CustomCreator() {
  return lite::PopulateRegistry::GetInstance()->GetParameterCreator(static_cast<int>(schema::PrimitiveType_Custom),
                                                                    lite::SCHEMA_CUR);
}

void *kFakeFuncPtr = reinterpret_cast<void *>(static_cast<uintptr_t>(0x414141414141ull));
}  // namespace

class CustomPopulateTest : public mindspore::CommonTest {
 public:
  CustomPopulateTest() = default;
};

// The populate layer memcpy's attr[0].data over the whole SplitParameter. A
// crafted .ms model can plant a function pointer in OpParameter::destroy_func_
// (offset 120); the scheduler calls it on cleanup, giving arbitrary code
// execution. Populate must never let model bytes choose that pointer.
TEST_F(CustomPopulateTest, PopulateSplitReduceConcat_never_keeps_attacker_destroy_func) {
  SplitParameter fake{};
  fake.num_split_ = 4;
  fake.split_dim_ = 1;
  fake.op_parameter_.destroy_func_ = reinterpret_cast<void (*)(OpParameter *)>(kFakeFuncPtr);
  std::vector<int> split_sizes{1, 1, 1, 1};
  const auto *sizes_bytes = reinterpret_cast<const uint8_t *>(split_sizes.data());

  flatbuffers::FlatBufferBuilder builder;
  auto prim = BuildSplitReduceConcatPrimitive(
    builder, {StructBytes(fake), std::vector<uint8_t>(sizes_bytes, sizes_bytes + split_sizes.size() * sizeof(int))});
  ASSERT_NE(prim, nullptr);
  auto creator = CustomCreator();
  ASSERT_NE(creator, nullptr);
  auto *param = creator(prim);
  ASSERT_NE(param, nullptr);
  // The fake pointer planted by the model must not survive populate.
  ASSERT_NE(param->destroy_func_, reinterpret_cast<void (*)(OpParameter *)>(kFakeFuncPtr));
  param->destroy_func_(param);
  free(param);
}

// attr[1] shorter than num_split_ * sizeof(int) leaves the tail of the
// malloc'd split_sizes_ uninitialized; the fused kernel feeds those bytes
// into address arithmetic. Populate must reject a partial payload outright.
TEST_F(CustomPopulateTest, PopulateSplitReduceConcat_rejects_short_sizes_attr) {
  SplitParameter fake{};
  fake.num_split_ = 4;
  fake.op_parameter_.destroy_func_ = reinterpret_cast<void (*)(OpParameter *)>(kFakeFuncPtr);
  std::vector<int> split_sizes{1, 1};
  const auto *sizes_bytes = reinterpret_cast<const uint8_t *>(split_sizes.data());

  flatbuffers::FlatBufferBuilder builder;
  auto prim = BuildSplitReduceConcatPrimitive(
    builder, {StructBytes(fake), std::vector<uint8_t>(sizes_bytes, sizes_bytes + split_sizes.size() * sizeof(int))});
  ASSERT_NE(prim, nullptr);
  auto creator = CustomCreator();
  ASSERT_NE(creator, nullptr);
  auto *param = creator(prim);
  ASSERT_EQ(param, nullptr);
}

// A negative num_split_ is fed into size arithmetic; populate must reject it
// instead of wrapping through size_t.
TEST_F(CustomPopulateTest, PopulateSplitReduceConcat_rejects_negative_num_split) {
  SplitParameter fake{};
  fake.num_split_ = -1;
  fake.op_parameter_.destroy_func_ = reinterpret_cast<void (*)(OpParameter *)>(kFakeFuncPtr);
  std::vector<int> split_sizes{1, 1, 1, 1};
  const auto *sizes_bytes = reinterpret_cast<const uint8_t *>(split_sizes.data());

  flatbuffers::FlatBufferBuilder builder;
  auto prim = BuildSplitReduceConcatPrimitive(
    builder, {StructBytes(fake), std::vector<uint8_t>(sizes_bytes, sizes_bytes + split_sizes.size() * sizeof(int))});
  ASSERT_NE(prim, nullptr);
  auto creator = CustomCreator();
  ASSERT_NE(creator, nullptr);
  auto *param = creator(prim);
  ASSERT_EQ(param, nullptr);
}

// attr[1] (split sizes) is required; with only one attr present the populate
// must fail cleanly instead of reading past the attribute vector.
TEST_F(CustomPopulateTest, PopulateSplitReduceConcat_rejects_missing_sizes_attr) {
  SplitParameter fake{};
  fake.num_split_ = 4;
  flatbuffers::FlatBufferBuilder builder;
  auto prim = BuildSplitReduceConcatPrimitive(builder, {StructBytes(fake)});
  ASSERT_NE(prim, nullptr);
  auto creator = CustomCreator();
  ASSERT_NE(creator, nullptr);
  auto *param = creator(prim);
  ASSERT_EQ(param, nullptr);
}
}  // namespace mindspore
