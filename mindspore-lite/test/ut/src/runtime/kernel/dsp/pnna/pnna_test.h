/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TEST_UT_SRC_RUNTIME_KERNEL_DSP_DSP_TEST_H_
#define MINDSPORE_LITE_TEST_UT_SRC_RUNTIME_KERNEL_DSP_DSP_TEST_H_

#include <iostream>
#include <memory>
#include <limits>
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "include/api/model.h"
#include "schema/inner/model_generated.h"
#include "src/common/utils.h"
#include "common/common_test.h"
#include "nnacl_c/int8/quantize.h"

namespace mindspore::lite::dsp::test {
struct TensorInfo {
  float *data;
  float min;
  float max;
  int len;
};

inline void QuantProcess(float *input, int len, float min, float max, float *scale, int *zero_point, int8_t *output) {
  const int8_t qmin = std::numeric_limits<int8_t>::min();
  const int8_t qmax = std::numeric_limits<int8_t>::max();
  if (min == max) {
    *scale = 0.0f;
    *zero_point = 0;
    if (output) {
      Quantize(input, len, *scale, *zero_point, output);
    }
    return;
  }
  *scale = (max - min) / (qmax - qmin);
  float zero_point_from_min = qmin - min / (*scale);
  float zero_point_from_max = qmax - max / (*scale);
  float zero_point_from_min_error = std::abs(qmin) + std::abs(min / (*scale));
  float zero_point_from_max_error = std::abs(qmax) + std::abs(max / (*scale));
  float zero_point_double =
    zero_point_from_min_error < zero_point_from_max_error ? zero_point_from_min : zero_point_from_max;
  if (zero_point_double < qmin) {
    *zero_point = qmin;
  } else if (zero_point_double > qmax) {
    *zero_point = qmax;
  } else {
    *zero_point = static_cast<int>(std::round(zero_point_double));
  }
  if (output) {
    Quantize(input, len, *scale, *zero_point, output);
  }
}
}  // namespace mindspore::lite::dsp::test

#endif  // MINDSPORE_LITE_TEST_UT_SRC_RUNTIME_KERNEL_DSP_DSP_TEST_H_
