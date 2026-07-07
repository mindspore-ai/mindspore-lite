/**
 * Copyright 2021-2026 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_INT8_RESIZE_INT8_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_INT8_RESIZE_INT8_CODER_H_

#include <cstdint>
#include <cmath>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include "coder/opcoders/base/resize_base_coder.h"
#include "coder/log.h"
#include "nnacl_c/op_base.h"
#include "nnacl_c/int8/quantize.h"
#include "nnacl_c/int8/resize_int8.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"

namespace mindspore::lite::micro::nnacl {
class ResizeInt8Coder final : public ResizeBaseCoder {
 public:
  ResizeInt8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                  const LiteGraph::Node *node, size_t node_index, Target target)
      : ResizeBaseCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~ResizeInt8Coder() override;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  int ReSize();
  void FreeArgs();
  ResizeParameter *param_{nullptr};
  ::QuantArg *quant_in_{nullptr};
  ::QuantArg *quant_out_{nullptr};
  QuantMulArg *multiplier_{nullptr};
  ResizeQuantArg resize_quant_arg_ = {};
  ResizeFloatScaleQuantArg resize_float_quant_arg_ = {};
  int InitResizeQuantArg();
  int CalRatio();
  int CalInterpolationRange();
  int InitResizeBiLinear();
  int InitFloatResizeBiLinear();
  int InitResizeFloatQuantArg();
  int CalFloatRatio();
  int CalFloatInterpolationRange();
};

// Resize int8 coder helpers (single-file usage, merged from resize_int8_coder_utils.h).
constexpr unsigned int OFFSET_BASE = 10;
constexpr float HALF_PIXEL_OFFSET = 0.5f;

inline int InitResizeQuantArgCommon(int32_t *&x_axis_index, int32_t *&x_axis_lower, int32_t *&x_axis_upper,
                                    int32_t *&y_axis_index, int32_t *&y_axis_lower, int32_t *&y_axis_upper,
                                    const int *out_shape) {
  x_axis_index = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_W] * sizeof(int32_t)));
  if (x_axis_index == nullptr) {
    MS_LOG(ERROR) << "malloc x axis index array failed.";
    return RET_ERROR;
  }
  x_axis_lower = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_W] * sizeof(int32_t)));
  if (x_axis_lower == nullptr) {
    MS_LOG(ERROR) << "malloc x_axis_lower_ array failed.";
    return RET_ERROR;
  }
  x_axis_upper = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_W] * sizeof(int32_t)));
  if (x_axis_upper == nullptr) {
    MS_LOG(ERROR) << "malloc x_axis_upper_ array failed.";
    return RET_ERROR;
  }
  y_axis_index = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_H] * sizeof(int32_t)));
  if (y_axis_index == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_index_ array failed.";
    return RET_ERROR;
  }
  y_axis_lower = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_H] * sizeof(int32_t)));
  if (y_axis_lower == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_lower_ array failed.";
    return RET_ERROR;
  }
  y_axis_upper = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_H] * sizeof(int32_t)));
  if (y_axis_upper == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_upper_ array failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

inline int CalRatioCommon(int32_t &ratio_x, int32_t &ratio_y, int in_width, int in_height, int out_width,
                          int out_height, schema::CoordinateTransformMode coordinate_mode) {
  ratio_x = ((1 << OFFSET_BASE) * in_width + out_width / C2NUM) / out_width;
  ratio_y = ((1 << OFFSET_BASE) * in_height + out_height / C2NUM) / out_height;
  bool align_corners = coordinate_mode == schema::CoordinateTransformMode_ALIGN_CORNERS;
  if (align_corners && out_width > 1) {
    ratio_x = ((1 << OFFSET_BASE) * (in_width - 1) + (out_width - 1) / C2NUM) / (out_width - 1);
  }
  if (align_corners && out_height > 1) {
    ratio_y = ((1 << OFFSET_BASE) * (in_height - 1) + (out_height - 1) / C2NUM) / (out_height - 1);
  }
  return RET_OK;
}

inline int CalInterpolationRangeCommon(int32_t *x_axis_index, int32_t *x_axis_lower, int32_t *x_axis_upper,
                                       int32_t *y_axis_index, int32_t *y_axis_lower, int32_t *y_axis_upper,
                                       int32_t ratio_x, int32_t ratio_y, int in_height, int in_width, int out_height,
                                       int out_width) {
  for (int i = 0; i < out_height; ++i) {
    int32_t scaled_index = i * ratio_y;
    y_axis_index[i] = scaled_index;
    y_axis_lower[i] = std::max(scaled_index / (1 << OFFSET_BASE), 0);
    y_axis_upper[i] = std::min(scaled_index / (1 << OFFSET_BASE) + 1, in_height - 1);
  }
  for (int i = 0; i < out_width; ++i) {
    int32_t scaled_index = i * ratio_x;
    x_axis_index[i] = scaled_index;
    x_axis_lower[i] = std::max(scaled_index / (1 << OFFSET_BASE), 0);
    x_axis_upper[i] = std::min(scaled_index / (1 << OFFSET_BASE) + 1, in_width - 1);
  }
  return RET_OK;
}

inline int InitResizeFloatQuantArgCommon(float *&x_axis_index, int32_t *&x_axis_lower, int32_t *&x_axis_upper,
                                         float *&y_axis_index, int32_t *&y_axis_lower, int32_t *&y_axis_upper,
                                         const int *out_shape) {
  x_axis_index = reinterpret_cast<float *>(malloc(out_shape[kNHWC_W] * sizeof(float)));
  if (x_axis_index == nullptr) {
    MS_LOG(ERROR) << "malloc x axis index array failed.";
    return RET_ERROR;
  }
  x_axis_lower = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_W] * sizeof(int32_t)));
  if (x_axis_lower == nullptr) {
    MS_LOG(ERROR) << "malloc x_axis_lower_ array failed.";
    return RET_ERROR;
  }
  x_axis_upper = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_W] * sizeof(int32_t)));
  if (x_axis_upper == nullptr) {
    MS_LOG(ERROR) << "malloc x_axis_upper_ array failed.";
    return RET_ERROR;
  }
  y_axis_index = reinterpret_cast<float *>(malloc(out_shape[kNHWC_H] * sizeof(float)));
  if (y_axis_index == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_index_ array failed.";
    return RET_ERROR;
  }
  y_axis_lower = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_H] * sizeof(int32_t)));
  if (y_axis_lower == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_lower_ array failed.";
    return RET_ERROR;
  }
  y_axis_upper = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_H] * sizeof(int32_t)));
  if (y_axis_upper == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_upper_ array failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

inline int CalFloatRatioCommon(float &ratio_x, float &ratio_y, int in_width, int in_height, int out_width,
                               int out_height, schema::CoordinateTransformMode coordinate_mode) {
  ratio_x = static_cast<float>(in_width) / out_width;
  ratio_y = static_cast<float>(in_height) / out_height;
  bool align_corners = coordinate_mode == schema::CoordinateTransformMode_ALIGN_CORNERS;
  if (align_corners && out_width > 1) {
    ratio_x = static_cast<float>(in_width - 1) / (out_width - 1);
  }
  if (align_corners && out_height > 1) {
    ratio_y = static_cast<float>(in_height - 1) / (out_height - 1);
  }
  return RET_OK;
}

inline int CalFloatInterpolationRangeCommon(float *x_axis_index, int32_t *x_axis_lower, int32_t *x_axis_upper,
                                            float *y_axis_index, int32_t *y_axis_lower, int32_t *y_axis_upper,
                                            float ratio_x, float ratio_y, int in_height, int in_width, int out_height,
                                            int out_width, schema::CoordinateTransformMode coordinate_mode) {
  for (int i = 0; i < out_height; ++i) {
    float scaled_index = 0.0f;
    if (coordinate_mode == schema::CoordinateTransformMode_ASYMMETRIC ||
        coordinate_mode == schema::CoordinateTransformMode_ALIGN_CORNERS) {
      scaled_index = i * ratio_y;
    } else if (coordinate_mode == schema::CoordinateTransformMode_HALF_PIXEL) {
      scaled_index = (i + HALF_PIXEL_OFFSET) * ratio_y - HALF_PIXEL_OFFSET;
    } else {
      MS_LOG(ERROR) << "coordinate_transform_mode_ is invalid." << coordinate_mode;
      return RET_ERROR;
    }
    int lower_index = static_cast<int>(std::floor(scaled_index));
    y_axis_index[i] = scaled_index;
    y_axis_lower[i] = std::max(lower_index, 0);
    y_axis_upper[i] = std::min(lower_index + 1, in_height - 1);
  }
  for (int i = 0; i < out_width; ++i) {
    float scaled_index = 0.0f;
    if (coordinate_mode == schema::CoordinateTransformMode_ASYMMETRIC ||
        coordinate_mode == schema::CoordinateTransformMode_ALIGN_CORNERS) {
      scaled_index = i * ratio_x;
    } else if (coordinate_mode == schema::CoordinateTransformMode_HALF_PIXEL) {
      scaled_index = (i + HALF_PIXEL_OFFSET) * ratio_x - HALF_PIXEL_OFFSET;
    } else {
      MS_LOG(ERROR) << "coordinate_transform_mode_ is invalid." << coordinate_mode;
      return RET_ERROR;
    }
    int lower_index = static_cast<int>(std::floor(scaled_index));
    x_axis_index[i] = scaled_index;
    x_axis_lower[i] = std::max(lower_index, 0);
    x_axis_upper[i] = std::min(lower_index + 1, in_width - 1);
  }
  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_INT8_RESIZE_INT8_CODER_H_
