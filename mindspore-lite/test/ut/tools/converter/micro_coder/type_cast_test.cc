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

#define USE_DEPRECATED_API
#include <vector>
#include "common/common_test.h"
#include "tools/converter/micro/coder/utils/type_cast.h"

namespace mindspore {
namespace lite {
namespace micro {

// EnumNameMSDataType feeds the generated model.c tensor declarations
// (input_tensors[i]->type = ...). MSDataType enum values mirror the TypeId values
// (kMSDataTypeNumberTypeInt64 = kNumberTypeInt64 = 35), and MSModelPredict compares the
// declared type against expect_types[], which is emitted as the raw TypeId number. A wrong
// name here therefore type-mismatches every predict call: for ConstantOfShape the shape
// input was declared kMSDataTypeNumberTypeUInt64 (40) while predict expected Int64 (35),
// TransformInput only converts fp32<->fp16, returned NULL, and MSModelPredict failed with
// kMSStatusLiteParamInvalid (-3). This table pins every mapping in the switch.
class TestMicroTypeCast : public mindspore::CommonTest {};

static const std::vector<std::pair<TypeId, std::string>> kMSDataTypeCases = {
  {kNumberTypeInt, "kMSDataTypeNumberTypeInt32"},  // legacy 32-bit int alias
  {kNumberTypeInt8, "kMSDataTypeNumberTypeInt8"},
  {kNumberTypeInt16, "kMSDataTypeNumberTypeInt16"},
  {kNumberTypeInt32, "kMSDataTypeNumberTypeInt32"},
  {kNumberTypeInt64, "kMSDataTypeNumberTypeInt64"},  // regression: used to emit UInt64
  {kNumberTypeUInt, "kMSDataTypeNumberTypeUInt32"},  // legacy 32-bit uint alias
  {kNumberTypeUInt8, "kMSDataTypeNumberTypeUInt8"},
  {kNumberTypeUInt16, "kMSDataTypeNumberTypeUInt16"},
  {kNumberTypeUInt32, "kMSDataTypeNumberTypeUInt32"},
  {kNumberTypeFloat, "kMSDataTypeNumberTypeFloat32"},  // legacy float alias
  {kNumberTypeFloat32, "kMSDataTypeNumberTypeFloat32"},
  {kNumberTypeFloat16, "kMSDataTypeNumberTypeFloat16"},
  {kNumberTypeFloat64, "kMSDataTypeNumberTypeFloat64"},
};

TEST_F(TestMicroTypeCast, MSDataTypeMappingMatchesTypeId) {
  for (auto &entry : kMSDataTypeCases) {
    EXPECT_STREQ(EnumNameMSDataType(entry.first).c_str(), entry.second.c_str())
      << "TypeId " << entry.first << " mapped to " << EnumNameMSDataType(entry.first);
  }
}

TEST_F(TestMicroTypeCast, Int64DeclaresInt64NotUInt64) {
  // The #43523 constantofshape ret:-3 root cause: UInt64 (40) != Int64 (35) makes
  // TransformInput reject every input of any Int64 graph input (e.g. shape tensors).
  EXPECT_STREQ(EnumNameMSDataType(kNumberTypeInt64).c_str(), "kMSDataTypeNumberTypeInt64");
}
}  // namespace micro
}  // namespace lite
}  // namespace mindspore
