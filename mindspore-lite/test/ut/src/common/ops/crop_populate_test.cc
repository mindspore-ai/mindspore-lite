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
#include <vector>
#include "common/common_test.h"
#include "schema/ops_generated.h"
#include "src/common/ops/populate/populate_register.h"
#include "nnacl_c/crop_parameter.h"

namespace mindspore {
namespace {
// Build a flatbuffer Primitive wrapping a Crop op with the given axis/offsets,
// mimicking what a (malicious or legacy) .ms model carries.
const schema::Primitive *BuildCropPrimitive(flatbuffers::FlatBufferBuilder &builder, int64_t axis,
                                            const std::vector<int64_t> &offsets) {
  auto offsets_off = builder.CreateVector(offsets);
  schema::CropBuilder cb(builder);
  cb.add_axis(axis);
  cb.add_offsets(offsets_off);
  auto crop_off = cb.Finish().o;
  schema::PrimitiveBuilder pb(builder);
  pb.add_value_type(schema::PrimitiveType_Crop);
  pb.add_value(crop_off);
  builder.Finish(pb.Finish());
  // Copy out so the buffer outlives the builder.
  static std::vector<uint8_t> storage;
  storage.assign(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
  return flatbuffers::GetRoot<schema::Primitive>(storage.data());
}

lite::ParameterGen CropCreator() {
  return lite::PopulateRegistry::GetInstance()->GetParameterCreator(static_cast<int>(schema::PrimitiveType_Crop),
                                                                    lite::SCHEMA_CUR);
}
}  // namespace

class CropPopulateTest : public mindspore::CommonTest {
 public:
  CropPopulateTest() = default;
};

// A malicious .ms model may carry negative offsets; PopulateCropParameter must
// reject them before they reach the kernel, where they wrap size_t index math
// into an out-of-bounds read.
TEST_F(CropPopulateTest, PopulateCrop_rejects_negative_offsets) {
  flatbuffers::FlatBufferBuilder builder;
  auto prim = BuildCropPrimitive(builder, 0, {-1, 0});
  ASSERT_NE(prim, nullptr);
  auto creator = CropCreator();
  ASSERT_NE(creator, nullptr);
  auto *param = creator(prim);
  ASSERT_EQ(param, nullptr);
}

TEST_F(CropPopulateTest, PopulateCrop_keeps_valid_offsets) {
  flatbuffers::FlatBufferBuilder builder;
  auto prim = BuildCropPrimitive(builder, 1, {1, 0, 0});
  ASSERT_NE(prim, nullptr);
  auto creator = CropCreator();
  ASSERT_NE(creator, nullptr);
  auto *param = creator(prim);
  ASSERT_NE(param, nullptr);
  auto *crop_param = reinterpret_cast<CropParameter *>(param);
  EXPECT_EQ(crop_param->axis_, 1);
  EXPECT_EQ(crop_param->offset_size_, 3);
  EXPECT_EQ(crop_param->offset_[0], 1);
  EXPECT_EQ(crop_param->offset_[1], 0);
  EXPECT_EQ(crop_param->offset_[2], 0);
  free(param);
}
}  // namespace mindspore
