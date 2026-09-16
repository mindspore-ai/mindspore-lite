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
  // TFLite official schema_utils resolves with max(builtin_code, deprecated_builtin_code):
  // a well-formed OperatorCode stores the same value in both fields, so the max only picks
  // the non-default one. Some exporters write builtin_code alone for old ops (<128) and leave
  // deprecated_builtin_code at its default 0 (ADD); reading deprecated alone would misparse
  // e.g. BATCH_MATMUL(126) as ADD.
  auto builtin_code = static_cast<int32_t>(opcode->builtin_code);
  auto deprecated_code = static_cast<int32_t>(opcode->deprecated_builtin_code);
  return static_cast<tflite::BuiltinOperator>(builtin_code > deprecated_code ? builtin_code : deprecated_code);
}
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_TFLITE_UTIL_H_
