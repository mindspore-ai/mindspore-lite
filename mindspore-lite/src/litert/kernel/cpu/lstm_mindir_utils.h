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

#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_LSTM_MINDIR_UTILS_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_LSTM_MINDIR_UTILS_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl_c/lstm_parameter.h"

namespace mindspore::kernel {

// Structure to hold GPU origin check result
struct GpuOriginCheckResult {
  bool gpu_orig_state{false};
  int bias_whole_size{0};
  bool is_valid{true};
};

/**
 * @brief Check if weights originate from GPU and validate weight configuration
 *
 * This helper function determines whether the LSTM weights originate from GPU
 * and validates the weight size configuration. It's shared between FP32 and FP16
 * LSTM MindIR implementations to eliminate code duplication.
 *
 * @param weight_tensor The weight tensor to check
 * @param lstm_param LSTM parameter containing size information
 * @param weight_segment_num Number of weight segments
 * @return GpuOriginCheckResult containing check results and bias size
 */
inline GpuOriginCheckResult CheckGpuOriginAndValidateWeights(lite::Tensor *weight_tensor, LstmParameter *lstm_param,
                                                             int weight_segment_num) {
  MS_ASSERT(weight_tensor != nullptr);
  MS_ASSERT(lstm_param != nullptr);
  GpuOriginCheckResult result;
  result.gpu_orig_state = false;
  result.is_valid = true;

  // Calculate input-hidden weight size
  if (INT_MUL_OVERFLOW(lstm_param->hidden_size_, lstm_param->input_size_) ||
      INT_MUL_OVERFLOW(weight_segment_num, lstm_param->hidden_size_ * lstm_param->input_size_)) {
    MS_LOG(ERROR) << "LSTM input-hidden weight size overflow";
    result.is_valid = false;
    return result;
  }
  int hi_unit_size = lstm_param->hidden_size_ * lstm_param->input_size_;
  int hi_whole_size = weight_segment_num * hi_unit_size;

  // Calculate hidden-hidden weight size
  if (INT_MUL_OVERFLOW(lstm_param->hidden_size_, lstm_param->output_size_) ||
      INT_MUL_OVERFLOW(weight_segment_num, lstm_param->hidden_size_ * lstm_param->output_size_)) {
    MS_LOG(ERROR) << "LSTM hidden-hidden weight size overflow";
    result.is_valid = false;
    return result;
  }
  int hh_unit_size = lstm_param->hidden_size_ * lstm_param->output_size_;
  int hh_whole_size = weight_segment_num * hh_unit_size;

  // Calculate hidden-project weight size
  int scale = lstm_param->bidirectional_ ? C2NUM : C1NUM;
  if (INT_MUL_OVERFLOW(lstm_param->hidden_size_, lstm_param->project_size_) ||
      INT_MUL_OVERFLOW(scale, lstm_param->hidden_size_ * lstm_param->project_size_)) {
    MS_LOG(ERROR) << "LSTM hidden-project weight size overflow";
    result.is_valid = false;
    return result;
  }
  int hp_unit_size = lstm_param->hidden_size_ * lstm_param->project_size_;
  int hp_whole_size = scale * hp_unit_size;

  // Calculate bias size
  if (INT_MUL_OVERFLOW(weight_segment_num * C2NUM, lstm_param->hidden_size_)) {
    MS_LOG(ERROR) << "LSTM bias size overflow";
    result.is_valid = false;
    return result;
  }
  result.bias_whole_size = weight_segment_num * C2NUM * lstm_param->hidden_size_;

  // Get total weight size
  auto whole_size = weight_tensor->ElementsNum();
  bool has_bias = (hi_whole_size + hh_whole_size + hp_whole_size < whole_size);

  // Determine GPU origin based on bias presence
  if (has_bias) {
    result.gpu_orig_state = (hi_whole_size + hh_whole_size + hp_whole_size + result.bias_whole_size == whole_size);
  } else {
    result.bias_whole_size = 0;
  }

  // Validate weight configuration for CPU origin
  if (!result.gpu_orig_state) {
    int adjusted_bias_size = result.bias_whole_size / C2NUM;
    if (hi_whole_size + hh_whole_size + hp_whole_size + adjusted_bias_size != whole_size) {
      MS_LOG(ERROR) << "LstmMindir is invalid when original model exports from CPU.";
      result.is_valid = false;
    }
  }

  return result;
}

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_LSTM_MINDIR_UTILS_H_
