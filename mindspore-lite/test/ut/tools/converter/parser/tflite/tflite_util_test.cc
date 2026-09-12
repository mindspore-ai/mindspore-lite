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
#include "gtest/gtest.h"
#include "tools/converter/parser/tflite/tflite_util.h"

namespace mindspore {
namespace lite {

class TfliteUtilTest : public ::testing::Test {};

TEST_F(TfliteUtilTest, GetBuiltinCodePrefersNonDefaultValue) {
  // Well-formed OperatorCode: both fields carry the same value.
  auto code = std::make_unique<tflite::OperatorCodeT>();
  code->builtin_code = tflite::BuiltinOperator_BATCH_MATMUL;
  code->deprecated_builtin_code = 126;
  ASSERT_EQ(GetBuiltinCode(code), tflite::BuiltinOperator_BATCH_MATMUL);
}

TEST_F(TfliteUtilTest, GetBuiltinCodeResumesBuiltinOnlyExport) {
  // Exporters that write builtin_code alone for old ops (<128) leave
  // deprecated_builtin_code at its default 0 (ADD); the op must not be
  // misparsed as ADD.
  auto code = std::make_unique<tflite::OperatorCodeT>();
  code->builtin_code = tflite::BuiltinOperator_BATCH_MATMUL;
  code->deprecated_builtin_code = 0;
  ASSERT_EQ(GetBuiltinCode(code), tflite::BuiltinOperator_BATCH_MATMUL);

  auto dequant = std::make_unique<tflite::OperatorCodeT>();
  dequant->builtin_code = tflite::BuiltinOperator_DEQUANTIZE;
  dequant->deprecated_builtin_code = 0;
  ASSERT_EQ(GetBuiltinCode(dequant), tflite::BuiltinOperator_DEQUANTIZE);
}

TEST_F(TfliteUtilTest, GetBuiltinCodeKeepsDeprecatedOnlyModels) {
  // Ancient models store the code only in deprecated_builtin_code.
  auto code = std::make_unique<tflite::OperatorCodeT>();
  code->builtin_code = tflite::BuiltinOperator_ADD;  // field default 0
  code->deprecated_builtin_code = 3;                 // CONV_2D
  ASSERT_EQ(GetBuiltinCode(code), tflite::BuiltinOperator_CONV_2D);
}

TEST_F(TfliteUtilTest, GetBuiltinCodeAddStaysAdd) {
  auto code = std::make_unique<tflite::OperatorCodeT>();
  code->builtin_code = tflite::BuiltinOperator_ADD;
  code->deprecated_builtin_code = 0;
  ASSERT_EQ(GetBuiltinCode(code), tflite::BuiltinOperator_ADD);
}
}  // namespace lite
}  // namespace mindspore
