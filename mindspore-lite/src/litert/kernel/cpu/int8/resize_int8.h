/**
 * Copyright 2020-2026 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_RESIZE_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_RESIZE_INT8_H_

#include <cstdint>
#include <vector>
#include <algorithm>
#include "src/litert/lite_kernel.h"
#include "src/litert/kernel/cpu/base/resize_base.h"
#include "nnacl_c/int8/quantize.h"
#include "nnacl_c/int8/resize_int8.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"

using mindspore::schema::PrimitiveType_Resize;
using mindspore::schema::ResizeMethod;

namespace mindspore::kernel {
class ResizeInt8CPUKernel : public ResizeBaseCPUKernel {
 public:
  ResizeInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : ResizeBaseCPUKernel(parameter, inputs, outputs, ctx) {}

  ~ResizeInt8CPUKernel() override;

  int Prepare() override;
  int ReSize() override;
  int InitResizeBiLinear();
  int InitFloatResizeBiLinear();
  int InitResizeQuantArg();
  int CalRatio();
  int CalInterpolationRange();
  void FreeResizeBiLinear();
  int InitResizeFloatQuantArg();
  int CalFloatRatio();
  int CalFloatInterpolationRange();
  void FreeFloatResizeBiLinear();
  int Run() override;
  int RunImpl(int task_id);

 private:
  QuantArg *quant_in_{nullptr};
  QuantArg *quant_out_{nullptr};
  QuantMulArg *multiplier_{nullptr};
  ResizeQuantArg resize_quant_arg_ = {};
  ResizeFloatScaleQuantArg resize_float_quant_arg_ = {};
};

// Resize int8 helper templates (single-file usage, merged from resize_int8_utils.h).
template <typename Context, typename QuantArg>
inline int InitResizeQuantArgCommon(Context *ctx, QuantArg &arg, const int *out_shape) {
  (void)ctx;  // May be used in specialized versions
  arg.x_axis_index_ = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_W] * sizeof(int32_t)));
  if (arg.x_axis_index_ == nullptr) {
    MS_LOG(ERROR) << "malloc x axis index array failed.";
    return lite::RET_ERROR;
  }
  arg.x_axis_lower_ = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_W] * sizeof(int32_t)));
  if (arg.x_axis_lower_ == nullptr) {
    MS_LOG(ERROR) << "malloc x_axis_lower_ array failed.";
    return lite::RET_ERROR;
  }
  arg.x_axis_upper_ = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_W] * sizeof(int32_t)));
  if (arg.x_axis_upper_ == nullptr) {
    MS_LOG(ERROR) << "malloc x_axis_upper_ array failed.";
    return lite::RET_ERROR;
  }
  arg.y_axis_index_ = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_H] * sizeof(int32_t)));
  if (arg.y_axis_index_ == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_index_ array failed.";
    return lite::RET_ERROR;
  }
  arg.y_axis_lower_ = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_H] * sizeof(int32_t)));
  if (arg.y_axis_lower_ == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_lower_ array failed.";
    return lite::RET_ERROR;
  }
  arg.y_axis_upper_ = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_H] * sizeof(int32_t)));
  if (arg.y_axis_upper_ == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_upper_ array failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

template <typename QuantArg, int DIVISOR = 2>
inline int CalRatioCommon(QuantArg &arg, int in_width, int in_height, int out_width, int out_height,
                          int coordinate_mode) {
  constexpr unsigned int OFFSET_BASE = 10;
  arg.ratio_x_ = ((1 << OFFSET_BASE) * in_width + out_width / DIVISOR) / out_width;
  arg.ratio_y_ = ((1 << OFFSET_BASE) * in_height + out_height / DIVISOR) / out_height;
  bool align_corners = coordinate_mode == static_cast<int>(schema::CoordinateTransformMode_ALIGN_CORNERS);
  if (align_corners && out_width > 1) {
    arg.ratio_x_ = ((1 << OFFSET_BASE) * (in_width - 1) + (out_width - 1) / DIVISOR) / (out_width - 1);
  }
  if (align_corners && out_height > 1) {
    arg.ratio_y_ = ((1 << OFFSET_BASE) * (in_height - 1) + (out_height - 1) / DIVISOR) / (out_height - 1);
  }
  return lite::RET_OK;
}

template <typename QuantArg, typename InTensorAccessor>
inline int CalInterpolationRangeCommon(QuantArg &arg, const InTensorAccessor &in_accessor, int out_height,
                                       int out_width) {
  constexpr unsigned int OFFSET_BASE = 10;
  for (int i = 0; i < out_height; ++i) {
    int32_t scaled_index = i * arg.ratio_y_;
    arg.y_axis_index_[i] = scaled_index;
    arg.y_axis_lower_[i] = std::max(scaled_index / (1 << OFFSET_BASE), 0);
    arg.y_axis_upper_[i] = std::min(scaled_index / (1 << OFFSET_BASE) + 1, in_accessor.Height() - 1);
  }
  for (int i = 0; i < out_width; ++i) {
    int32_t scaled_index = i * arg.ratio_x_;
    arg.x_axis_index_[i] = scaled_index;
    arg.x_axis_lower_[i] = std::max(scaled_index / (1 << OFFSET_BASE), 0);
    arg.x_axis_upper_[i] = std::min(scaled_index / (1 << OFFSET_BASE) + 1, in_accessor.Width() - 1);
  }
  return lite::RET_OK;
}

template <typename Context, typename FloatQuantArg>
inline int InitResizeFloatQuantArgCommon(Context *ctx, FloatQuantArg &arg, const int *out_shape) {
  (void)ctx;  // May be used in specialized versions
  arg.x_axis_index_ = reinterpret_cast<float *>(malloc(out_shape[kNHWC_W] * sizeof(float)));
  if (arg.x_axis_index_ == nullptr) {
    MS_LOG(ERROR) << "malloc x axis index array failed.";
    return lite::RET_ERROR;
  }
  arg.x_axis_lower_ = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_W] * sizeof(int32_t)));
  if (arg.x_axis_lower_ == nullptr) {
    MS_LOG(ERROR) << "malloc x_axis_lower_ array failed.";
    return lite::RET_ERROR;
  }
  arg.x_axis_upper_ = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_W] * sizeof(int32_t)));
  if (arg.x_axis_upper_ == nullptr) {
    MS_LOG(ERROR) << "malloc x_axis_upper_ array failed.";
    return lite::RET_ERROR;
  }
  arg.y_axis_index_ = reinterpret_cast<float *>(malloc(out_shape[kNHWC_H] * sizeof(float)));
  if (arg.y_axis_index_ == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_index_ array failed.";
    return lite::RET_ERROR;
  }
  arg.y_axis_lower_ = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_H] * sizeof(int32_t)));
  if (arg.y_axis_lower_ == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_lower_ array failed.";
    return lite::RET_ERROR;
  }
  arg.y_axis_upper_ = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_H] * sizeof(int32_t)));
  if (arg.y_axis_upper_ == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_upper_ array failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

template <typename FloatQuantArg>
inline int CalFloatRatioCommon(FloatQuantArg &arg, int in_width, int in_height, int out_width, int out_height,
                               int coordinate_mode) {
  arg.ratio_x_ = static_cast<float>(in_width) / out_width;
  arg.ratio_y_ = static_cast<float>(in_height) / out_height;
  bool align_corners = coordinate_mode == static_cast<int>(schema::CoordinateTransformMode_ALIGN_CORNERS);
  if (align_corners && out_width > 1) {
    arg.ratio_x_ = static_cast<float>(in_width - 1) / (out_width - 1);
  }
  if (align_corners && out_height > 1) {
    arg.ratio_y_ = static_cast<float>(in_height - 1) / (out_height - 1);
  }
  return lite::RET_OK;
}

template <typename FloatQuantArg, typename InTensorAccessor>
inline int CalFloatInterpolationRangeCommon(FloatQuantArg &arg, const InTensorAccessor &in_accessor, int out_height,
                                            int out_width, int coordinate_mode) {
  for (int i = 0; i < out_height; ++i) {
    float scaled_index = i * arg.ratio_y_;
    int lower_index = static_cast<int>(std::floor(scaled_index));
    arg.y_axis_index_[i] = scaled_index;
    arg.y_axis_lower_[i] = std::max(lower_index, 0);
    arg.y_axis_upper_[i] = std::min(lower_index + 1, in_accessor.Height() - 1);
  }
  for (int i = 0; i < out_width; ++i) {
    float scaled_index = i * arg.ratio_x_;
    int lower_index = static_cast<int>(std::floor(scaled_index));
    arg.x_axis_index_[i] = scaled_index;
    arg.x_axis_lower_[i] = std::max(lower_index, 0);
    arg.x_axis_upper_[i] = std::min(lower_index + 1, in_accessor.Width() - 1);
  }
  return lite::RET_OK;
}

template <typename QuantArg>
inline void FreeResizeBiLinearCommon(QuantArg &arg) {
  free(arg.x_axis_index_);
  arg.x_axis_index_ = nullptr;
  free(arg.x_axis_lower_);
  arg.x_axis_lower_ = nullptr;
  free(arg.x_axis_upper_);
  arg.x_axis_upper_ = nullptr;
  free(arg.y_axis_index_);
  arg.y_axis_index_ = nullptr;
  free(arg.y_axis_lower_);
  arg.y_axis_lower_ = nullptr;
  free(arg.y_axis_upper_);
  arg.y_axis_upper_ = nullptr;
}
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_RESIZE_INT8_H_
