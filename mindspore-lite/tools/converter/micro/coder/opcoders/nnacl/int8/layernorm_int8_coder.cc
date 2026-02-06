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

#include "coder/opcoders/nnacl/int8/layernorm_int8_coder.h"
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/utils/common.h"

using mindspore::schema::PrimitiveType_LayerNormFusion;

namespace mindspore::lite::micro::nnacl {
int LayerNormInt8Coder::SetQuantArgs() {
  CHECK_NULL_RETURN(input);
  CHECK_NULL_RETURN(gamma);
  CHECK_NULL_RETURN(beta);
  CHECK_NULL_RETURN(output);
  const auto input_params = input->quant_params();
  const auto gamma_params = gamma->quant_params();
  const auto beta_params = beta->quant_params();
  const auto output_params = output->quant_params();
  MS_CHECK_TRUE_MSG(!input_params.empty(), RET_ERROR, "Input quant param cannot be empty.");
  MS_CHECK_TRUE_MSG(!gamma_params.empty(), RET_ERROR, "Gamma quant param cannot be empty.");
  MS_CHECK_TRUE_MSG(!beta_params.empty(), RET_ERROR, "Beta quant param cannot be empty.");
  MS_CHECK_TRUE_MSG(!output_params.empty(), RET_ERROR, "Output quant param cannot be empty.");

  quant_param_.in_zp_ = input_params.front().zeroPoint;
  quant_param_.in_scale_ = input_params.front().scale;
  quant_param_.gamma_zp_ = gamma_params.front().zeroPoint;
  quant_param_.gamma_scale_ = gamma_params.front().scale;
  quant_param_.beta_zp_ = beta_params.front().zeroPoint;
  quant_param_.beta_scale_ = beta_params.front().scale;
  quant_param_.out_zp_ = output_params.front().zeroPoint;
  quant_param_.out_scale_ = output_params.front().scale;
  return RET_OK;
}

int LayerNormInt8Coder::ReSize() {
  CHECK_NULL_RETURN(input);
  auto shape = input->shape();
  compute_.begin_norm_axis_ = compute_.begin_norm_axis_ > 0
                                ? compute_.begin_norm_axis_
                                : compute_.begin_norm_axis_ + static_cast<int>(shape.size());
  compute_.begin_params_axis_ = compute_.begin_params_axis_ > 0
                                  ? compute_.begin_params_axis_
                                  : compute_.begin_params_axis_ + static_cast<int>(shape.size());

  compute_.norm_outer_size_ = 1;
  for (int i = 0; i < compute_.begin_norm_axis_; ++i) {
    compute_.norm_outer_size_ *= shape.at(i);
  }
  compute_.norm_inner_size_ = 1;
  for (int i = compute_.begin_norm_axis_; i < static_cast<int>(shape.size()); ++i) {
    compute_.norm_inner_size_ *= shape.at(i);
  }
  compute_.params_outer_size_ = 1;
  for (int i = 0; i < compute_.begin_params_axis_; ++i) {
    compute_.params_outer_size_ *= shape.at(i);
  }
  compute_.params_inner_size_ = 1;
  for (int i = compute_.begin_params_axis_; i < static_cast<int>(shape.size()); ++i) {
    compute_.params_inner_size_ *= shape.at(i);
  }
  return RET_OK;
}

int LayerNormInt8Coder::Prepare(CoderContext *const context) {
  layer_norm_param_ = reinterpret_cast<LayerNormParameter *>(parameter_);
  CHECK_NULL_RETURN(layer_norm_param_);
  CHECK_LESS_RETURN(input_tensors_.size(), C3NUM);
  CHECK_LESS_RETURN(output_tensors_.size(), C1NUM);
  input = input_tensors_.at(0);
  gamma = input_tensors_.at(Index1);
  beta = input_tensors_.at(Index2);
  output = output_tensors_.at(0);
  CHECK_NULL_RETURN(input);
  CHECK_NULL_RETURN(gamma);
  CHECK_NULL_RETURN(beta);
  CHECK_NULL_RETURN(output);
  layer_norm_param_->op_parameter_.thread_num_ = 1;
  is_const_ = gamma->IsConst() && beta->IsConst();
  compute_.epsilon_ = layer_norm_param_->epsilon_;
  compute_.elementwise_affine_ = layer_norm_param_->elementwise_affine_;
  compute_.begin_norm_axis_ = layer_norm_param_->begin_norm_axis_;
  compute_.begin_params_axis_ = layer_norm_param_->begin_params_axis_;

  auto ret = SetQuantArgs();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set QuantArgs failed.";
    return ret;
  }
  return ReSize();
}

int LayerNormInt8Coder::DoCode(CoderContext *context) {
  NNaclInt8Serializer code;
  CHECK_NULL_RETURN(layer_norm_param_);
  CHECK_NULL_RETURN(input);
  CHECK_NULL_RETURN(gamma);
  CHECK_NULL_RETURN(beta);
  CHECK_NULL_RETURN(output);
  code.CodeStruct("quant_param", quant_param_);
  code.CodeStruct("compute_param", compute_);
  if (is_const_) {
    MS_CHECK_GT(gamma->ElementsNum(), 0, RET_ERROR);
    auto gamma_ptr_ = reinterpret_cast<float *>(
      allocator_->Malloc(kNumberTypeFloat32, gamma->ElementsNum() * sizeof(float), kOfflinePackWeight));
    CHECK_NULL_RETURN(gamma_ptr_);
    MS_CHECK_RET_CODE(
      memset_s(gamma_ptr_, gamma->ElementsNum() * sizeof(float), 0, gamma->ElementsNum() * sizeof(float)),
      "memset_s gamma_ptr_addr failed.");
    int8_t *src_gamma = reinterpret_cast<int8_t *>(gamma->data());
    for (int i = 0; i < gamma->ElementsNum(); i++) {
      gamma_ptr_[i] = (src_gamma[i] - quant_param_.gamma_zp_) * quant_param_.gamma_scale_;
    }

    MS_CHECK_GT(beta->ElementsNum(), 0, RET_ERROR);
    auto beta_ptr_ = reinterpret_cast<float *>(
      allocator_->Malloc(kNumberTypeFloat32, beta->ElementsNum() * sizeof(float), kOfflinePackWeight));
    CHECK_NULL_RETURN(beta_ptr_);
    MS_CHECK_RET_CODE(memset_s(beta_ptr_, beta->ElementsNum() * sizeof(float), 0, beta->ElementsNum() * sizeof(float)),
                      "memset_s gamma_ptr_addr failed.");
    int32_t *src_beta = reinterpret_cast<int32_t *>(beta->data());
    for (int i = 0; i < beta->ElementsNum(); i++) {
      beta_ptr_[i] = src_beta[i] * quant_param_.in_scale_ * quant_param_.gamma_scale_;
    }
    code.CodeFunction("LayerNormInt8", input, gamma_ptr_, beta_ptr_, output, "&compute_param", "&quant_param", 0,
                      layer_norm_param_->op_parameter_.thread_num_);
  } else {
    code.CodeFunction("LayerNormDynamicInt8", input, gamma, beta, output, "&compute_param", "&quant_param", 0,
                      layer_norm_param_->op_parameter_.thread_num_);
  }

  Collect(context,
          {
            "nnacl_c/int8/layer_norm_int8.h",
          },
          {
            "layer_norm_int8.c",
          });
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_LayerNormFusion, CPUOpCoderCreator<LayerNormInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
