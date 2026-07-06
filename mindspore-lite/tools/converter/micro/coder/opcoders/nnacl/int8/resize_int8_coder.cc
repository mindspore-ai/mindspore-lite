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

#include "coder/opcoders/nnacl/int8/resize_int8_coder.h"
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "include/securec.h"
#include "nnacl_c/int8/quantize.h"
#include "coder/opcoders/parallel.h"
#include "nnacl_c/base/resize_base.h"

using mindspore::schema::PrimitiveType_Resize;

namespace mindspore::lite::micro::nnacl {

ResizeInt8Coder::~ResizeInt8Coder() { FreeArgs(); }

void ResizeInt8Coder::FreeArgs() {
  delete quant_out_;
  quant_out_ = nullptr;

  delete quant_in_;
  quant_in_ = nullptr;

  delete multiplier_;
  multiplier_ = nullptr;
}

int ResizeInt8Coder::Prepare(CoderContext *const context) {
  MS_CHECK_RET_CODE(ResizeBaseCoder::Init(), "init resize base failed");
  quant_in_ = new (std::nothrow)::QuantArg;
  MS_CHECK_PTR_WITH_EXE(quant_in_, FreeArgs());
  quant_out_ = new (std::nothrow)::QuantArg;
  MS_CHECK_PTR_WITH_EXE(quant_out_, FreeArgs());
  multiplier_ = new (std::nothrow) QuantMulArg;
  MS_CHECK_PTR_WITH_EXE(multiplier_, FreeArgs());
  MS_CHECK_TRUE_WITH_EXE(!input_tensors_.empty(), "input_tensors_ cannot be empty", FreeArgs());
  MS_CHECK_TRUE_WITH_EXE(!output_tensors_.empty(), "output_tensors_ cannot be empty", FreeArgs());
  quant_in_->zp_ = input_tensor_->quant_params().at(0).zeroPoint;
  quant_in_->scale_ = input_tensor_->quant_params().at(0).scale;
  quant_out_->zp_ = output_tensor_->quant_params().at(0).zeroPoint;
  quant_out_->scale_ = output_tensor_->quant_params().at(0).scale;

  QuantizeRoundParameterWithDoublePrecision(quant_in_->scale_ / quant_out_->scale_, &multiplier_->multiplier_,
                                            &multiplier_->left_shift_, &multiplier_->right_shift_);
  return ReSize();
}

int ResizeInt8Coder::ReSize() {
  if (method_ == static_cast<int>(schema::ResizeMethod_LINEAR)) {
    if (quant_in_->zp_ == 0) {
      return InitResizeBiLinear();
    } else {
      return InitFloatResizeBiLinear();
    }
  }
  if (input_tensors_.empty() || output_tensors_.empty()) {
    MS_LOG(ERROR) << "input_tensors_ or output_tensors_ is null.";
    return RET_ERROR;
  }
  if (input_tensors_.front()->quant_params().empty() || output_tensors_.front()->quant_params().empty() ||
      input_tensors_.front()->quant_params().front().zeroPoint !=
        output_tensors_.front()->quant_params().front().zeroPoint ||
      input_tensors_.front()->quant_params().front().zeroPoint > INT8_MAX ||
      input_tensors_.front()->quant_params().front().zeroPoint < INT8_MIN ||
      input_tensors_.front()->quant_params().front().scale < 0) {
    MS_LOG(ERROR) << "Resize quant param is invalid.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeInt8Coder::InitResizeQuantArg() {
  auto out_shape = output_tensors_.front()->shape();
  return InitResizeQuantArgCommon(resize_quant_arg_.x_axis_index_, resize_quant_arg_.x_axis_lower_,
                                  resize_quant_arg_.x_axis_upper_, resize_quant_arg_.y_axis_index_,
                                  resize_quant_arg_.y_axis_lower_, resize_quant_arg_.y_axis_upper_, out_shape.data());
}

int ResizeInt8Coder::CalRatio() {
  if (input_tensors_.empty() || output_tensors_.empty()) {
    MS_LOG(ERROR) << "input_tensors_ or output_tensors_ is null.";
    return RET_ERROR;
  }
  auto in_tensor = input_tensors_.front();
  auto in_width = in_tensor->Width();
  auto in_height = in_tensor->Height();
  auto out_tensor = output_tensors_.front();
  auto out_width = out_tensor->Width();
  auto out_height = out_tensor->Height();
  return CalRatioCommon(resize_quant_arg_.ratio_x_, resize_quant_arg_.ratio_y_, in_width, in_height, out_width,
                        out_height, static_cast<schema::CoordinateTransformMode>(coordinate_transform_mode_));
}

int ResizeInt8Coder::CalInterpolationRange() {
  auto in_tensor = input_tensors_.front();
  auto out_tensor = output_tensors_.front();
  return CalInterpolationRangeCommon(resize_quant_arg_.x_axis_index_, resize_quant_arg_.x_axis_lower_,
                                     resize_quant_arg_.x_axis_upper_, resize_quant_arg_.y_axis_index_,
                                     resize_quant_arg_.y_axis_lower_, resize_quant_arg_.y_axis_upper_,
                                     resize_quant_arg_.ratio_x_, resize_quant_arg_.ratio_y_, in_tensor->Height(),
                                     in_tensor->Width(), out_tensor->Height(), out_tensor->Width());
}

int ResizeInt8Coder::InitResizeFloatQuantArg() {
  auto out_shape = output_tensors_.front()->shape();
  return InitResizeFloatQuantArgCommon(resize_float_quant_arg_.x_axis_index_, resize_float_quant_arg_.x_axis_lower_,
                                       resize_float_quant_arg_.x_axis_upper_, resize_float_quant_arg_.y_axis_index_,
                                       resize_float_quant_arg_.y_axis_lower_, resize_float_quant_arg_.y_axis_upper_,
                                       out_shape.data());
}

int ResizeInt8Coder::CalFloatRatio() {
  if (input_tensors_.empty() || output_tensors_.empty()) {
    MS_LOG(ERROR) << "input_tensors_ or output_tensors_ is null.";
    return RET_ERROR;
  }
  auto in_tensor = input_tensors_.front();
  auto in_width = in_tensor->Width();
  auto in_height = in_tensor->Height();
  auto out_tensor = output_tensors_.front();
  auto out_width = out_tensor->Width();
  auto out_height = out_tensor->Height();
  return CalFloatRatioCommon(resize_float_quant_arg_.ratio_x_, resize_float_quant_arg_.ratio_y_, in_width, in_height,
                             out_width, out_height,
                             static_cast<schema::CoordinateTransformMode>(coordinate_transform_mode_));
}

int ResizeInt8Coder::CalFloatInterpolationRange() {
  MS_CHECK_TRUE_MSG(!output_tensors_.empty(), RET_ERROR, "Out tensors cannot be empty.");
  auto in_tensor = input_tensors_.front();
  auto out_tensor = output_tensors_.front();
  return CalFloatInterpolationRangeCommon(
    resize_float_quant_arg_.x_axis_index_, resize_float_quant_arg_.x_axis_lower_, resize_float_quant_arg_.x_axis_upper_,
    resize_float_quant_arg_.y_axis_index_, resize_float_quant_arg_.y_axis_lower_, resize_float_quant_arg_.y_axis_upper_,
    resize_float_quant_arg_.ratio_x_, resize_float_quant_arg_.ratio_y_, in_tensor->Height(), in_tensor->Width(),
    out_tensor->Height(), out_tensor->Width(),
    static_cast<schema::CoordinateTransformMode>(coordinate_transform_mode_));
}

int ResizeInt8Coder::InitResizeBiLinear() {
  auto ret = InitResizeQuantArg();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize Int8 Opcoder Resize Failed.";
    return ret;
  }
  ret = CalRatio();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize Int8 Opcoder Cal ratio Failed.";
    return ret;
  }
  ret = CalInterpolationRange();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize Int8 Opcoder Cal range of interpolation Failed.";
    return ret;
  }
  return RET_OK;
}

int ResizeInt8Coder::InitFloatResizeBiLinear() {
  auto ret = InitResizeFloatQuantArg();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize Int8 Opcoder Resize Failed.";
    return ret;
  }
  ret = CalFloatRatio();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize Int8 Opcoder Cal ratio Failed.";
    return ret;
  }
  ret = CalFloatInterpolationRange();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize Int8 Opcoder Cal range of interpolation Failed.";
    return ret;
  }
  return RET_OK;
}

int ResizeInt8Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl_c/int8/resize_int8.h",
            "wrapper/int8/resize_int8_wrapper.h",
            "nnacl_c/base/resize_base.h",
          },
          {"resize_int8.c", "common_func.c", "resize_int8_wrapper.c", "fixed_point.c", "resize_base.c"});

  nnacl::NNaclInt8Serializer code;
  code.CodeArray("input_shape", input_tensor_->shape().data(), input_tensor_->shape().size(), true);
  code.CodeArray("output_shape", output_tensor_->shape().data(), output_tensor_->shape().size(), true);
  switch (method_) {
    case static_cast<int>(schema::ResizeMethod_LINEAR): {
      auto input = input_tensors_.at(0);
      auto input_shape = input->shape();

      auto out_tensor = output_tensors_.front();
      auto out_c = out_tensor->Channel();
      int plane = out_tensor->Height() * out_tensor->Width();
      // Bilinear Resize OpCoder only support thread=1
      int num = UP_DIV(plane, kDefaultThreadNum);
      int start_index = kDefaultTaskId * num;
      int count = plane - start_index;
      count = count > num ? num : count;
      auto out_ptr = output_tensor_ + start_index * out_c;
      if (quant_in_->zp_ != 0) {
        code.CodeArray("x_axis_index", resize_float_quant_arg_.x_axis_index_, out_tensor->Width(), false);
        code.CodeArray("x_axis_lower", resize_float_quant_arg_.x_axis_lower_, out_tensor->Width(), false);
        code.CodeArray("x_axis_upper", resize_float_quant_arg_.x_axis_upper_, out_tensor->Width(), false);
        code.CodeArray("y_axis_index", resize_float_quant_arg_.y_axis_index_, out_tensor->Height(), false);
        code.CodeArray("y_axis_lower", resize_float_quant_arg_.y_axis_lower_, out_tensor->Height(), false);
        code.CodeArray("y_axis_upper", resize_float_quant_arg_.y_axis_upper_, out_tensor->Height(), false);
        code.CodeBaseStruct("ResizeFloatScaleQuantArg", "resize_float_quant", resize_float_quant_arg_.ratio_x_,
                            resize_float_quant_arg_.ratio_y_, "x_axis_index", "x_axis_lower", "x_axis_upper",
                            "y_axis_index", "y_axis_lower", "y_axis_upper");
        code.CodeFunction("ResizeBilinearWithFloatScaleInt8", input_tensor_, out_ptr, out_tensor->Batch(),
                          input->Height(), input->Width(), out_tensor->Height(), out_tensor->Width(), out_c,
                          start_index, count, "resize_float_quant");
      }
      break;
    }
    case static_cast<int>(schema::ResizeMethod_NEAREST): {
      bool same_zp = quant_in_->zp_ == quant_out_->zp_;
      bool same_scale = abs(quant_out_->scale_ - quant_in_->scale_) < 1e-6;
      bool align_corners =
        coordinate_transform_mode_ == static_cast<int>(schema::CoordinateTransformMode_ALIGN_CORNERS);
      if (same_zp && same_scale) {
        code.CodeBaseStruct("ResizeInt8Args", kRunArgs, input_tensor_, output_tensor_, "input_shape", "output_shape",
                            align_corners, coordinate_transform_mode_, nearest_method_, gThreadNum);
        if (support_parallel_) {
          code.CodeFunction(kParallelLaunch, "ResizeInt8Run", kRunArgsAddr, gThreadNum);
        } else {
          code.CodeFunction("ResizeInt8Run", kRunArgsAddr, kDefaultTaskId, kLhsScale, kRhsScale);
        }
      } else {
        MS_LOG(WARNING) << "unsupported parallel launch currently";
        code.CodeStruct("quant_in", *quant_in_);
        code.CodeStruct("quant_out", *quant_out_);
        code.CodeStruct("multiplier", *multiplier_);
        code.CodeFunction("ResizeNearestNeighborInt8", input_tensor_, output_tensor_, "input_shape", "output_shape",
                          align_corners, "&multiplier", "&quant_in", "&quant_out", coordinate_transform_mode_,
                          nearest_method_, kDefaultTaskId, gThreadNum);
      }
      break;
    }
    case schema::ResizeMethod_UNKNOWN:
    default: {
      MS_LOG(ERROR) << "Resize unknown method " << method_;
      return RET_ERROR;
    }
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Resize, CPUOpCoderCreator<ResizeInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
