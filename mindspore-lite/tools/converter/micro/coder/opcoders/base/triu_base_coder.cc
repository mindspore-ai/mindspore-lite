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
#include "coder/opcoders/base/triu_base_coder.h"
#include "nnacl_c/fp32/triu_tril_fp32.h"
#include "nnacl_c/int8/trilu_int8.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_Triu;

namespace mindspore::lite::micro::nnacl {
namespace {
constexpr size_t kDiagonalInputIndex = 1;

// Read the diagonal offset from the constant k input (scalar int32/int64). Returns 0 when the k input is absent or
// carries no data.
int GetDiagonal(const std::vector<Tensor *> &input_tensors) {
  if (input_tensors.size() <= kDiagonalInputIndex) {
    return 0;
  }
  auto k_tensor = input_tensors.at(kDiagonalInputIndex);
  if (k_tensor == nullptr || k_tensor->data() == nullptr || k_tensor->ElementsNum() != 1) {
    return 0;
  }
  if (k_tensor->data_type() == kNumberTypeInt32) {
    return static_cast<int>(static_cast<const int32_t *>(k_tensor->data())[0]);
  }
  if (k_tensor->data_type() == kNumberTypeInt64) {
    return static_cast<int>(static_cast<const int64_t *>(k_tensor->data())[0]);
  }
  return 0;
}
}  // namespace

int TriuBaseCoder::DoCode(CoderContext *const context) {
  auto input_shape = input_tensor_->shape();
  if (input_shape.size() < 2) {
    MS_LOG(ERROR) << "Triu requires at least 2D input";
    return RET_ERROR;
  }
  int height = input_shape[input_shape.size() - 2];
  int width = input_shape[input_shape.size() - 1];
  // Product of leading dims: supports 2D..ND (each leading matrix is handled independently).
  int num = 1;
  for (size_t i = 0; i < input_shape.size() - 2; i++) {
    num *= input_shape[i];
  }
  int diagonal = GetDiagonal(input_tensors_);

  NNaclFp32Serializer code;
  if (input_tensor_->data_type() == kNumberTypeInt8) {
    // int8: upstream TriuByte1 writes literal 0 (not out_zp) for masked elements, collapsing accuracy when zp != 0.
    // Keep the custom TriuInt8 to requantize (masked elements -> out_zp, kept elements -> input-to-output rescale).
    Collect(context, {"nnacl_c/int8/trilu_int8.h"}, {"trilu_int8.c"});
    float in_scale = 1.0f;
    float out_scale = 1.0f;
    int in_zp = 0;
    int out_zp = 0;
    const auto &in_quant = input_tensor_->quant_params();
    const auto &out_quant = output_tensor_->quant_params();
    if (!in_quant.empty()) {
      in_scale = in_quant.front().scale;
      in_zp = in_quant.front().zeroPoint;
    }
    if (!out_quant.empty()) {
      out_scale = out_quant.front().scale;
      out_zp = out_quant.front().zeroPoint;
    }
    code.CodeFunction("TriuInt8", input_tensor_, height, width, diagonal, output_tensor_, num, in_scale, in_zp,
                      out_scale, out_zp);
  } else if (input_tensor_->data_type() == kNumberTypeFloat32) {
    // fp32: reuse the upstream TriuByte4 from triu_tril_fp32.c (correct, N-D, upstream-verified).
    Collect(context, {"nnacl_c/fp32/triu_tril_fp32.h"}, {"triu_tril_fp32.c"});
    code.CodeFunction("TriuByte4", input_tensor_, output_tensor_, diagonal, height, width, num);
  } else {
    MS_LOG(ERROR) << "Triu only support FP32/INT8. ";
    return RET_ERROR;
  }

  MS_LOG(DEBUG) << "TriuBaseCoder invoked";
  context->AppendCode(code.str());
  return lite::RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Triu, CPUOpCoderCreator<TriuBaseCoder>)
// INT8: Triu is a pure data-movement op; int8 data is handled in place by TriuInt8
// (masked elements are written as the output zero point). This is genuine int8, not an fp32 fallback.
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Triu, CPUOpCoderCreator<TriuBaseCoder>)
}  // namespace mindspore::lite::micro::nnacl
