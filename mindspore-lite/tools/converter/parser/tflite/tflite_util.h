/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_TFLITE_UTIL_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_TFLITE_UTIL_H_

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "src/common/log_adapter.h"
#include "schema/inner/model_generated.h"
#include "schema/schema_generated.h"
#include "schema/inner/ops_generated.h"
#include "ir/dtype/type_id.h"
#include "include/errorcode.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace lite {
std::string GetPadModeStr(tflite::Padding tflite_padmode);

mindspore::PadMode GetPadMode(tflite::Padding tflite_padmode);

size_t GetDataTypeSize(const TypeId &data_type);

mindspore::ActivationType GetActivationFunctionType(tflite::ActivationFunctionType tfliteAFType);

TypeId GetTfliteDataType(const tflite::TensorType &tflite_data_type);

STATUS getPaddingParam(const std::unique_ptr<tflite::TensorT> &tensor, mindspore::PadMode pad_mode, int strideH,
                       int strideW, int windowH, int windowW, std::vector<int64_t> *params);

inline tflite::BuiltinOperator GetBuiltinCode(const std::unique_ptr<tflite::OperatorCodeT> &opcode) {
  auto builtin_code = opcode->builtin_code;
  // Backward compatibility with old TFLite models:
  // Old models store the operator code at field 0 (now deprecated_builtin_code),
  // while the new builtin_code field at field 3 defaults to 0 (ADD) for old models.
  // For new operators (>=128), builtin_code has the correct value.
  // For old operators (<128), read from deprecated_builtin_code.
  if (builtin_code <= tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES) {
    return static_cast<tflite::BuiltinOperator>(opcode->deprecated_builtin_code);
  }
  return builtin_code;
}
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_TFLITE_UTIL_H_
