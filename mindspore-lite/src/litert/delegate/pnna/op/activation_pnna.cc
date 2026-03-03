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

#include "src/litert/delegate/pnna/op/activation_pnna.h"
#include <unordered_map>

namespace mindspore {
namespace lite {
#define CONVERT_PNNA_ACTIVATION(type, class_name)                                  \
  case schema::ActivationType_##type: {                                            \
    auto activation_op = graph->graph()->CreateOperation<pnna::ops::class_name>(); \
    activation_op->BindInputs({input_tensor});                                     \
    activation_op->BindOutputs({output_tensor});                                   \
  } break;

bool PNNAActivation::IsSupport() {
  if (act_type_ != schema::ActivationType_RELU && act_type_ != schema::ActivationType_RELU6 &&
      act_type_ != schema::ActivationType_SIGMOID && act_type_ != schema::ActivationType_TANH &&
      act_type_ != schema::ActivationType_SWISH && act_type_ != schema::ActivationType_LEAKY_RELU) {
    MS_LOG(WARNING) << "Unsupported activation type for activation op " << static_cast<int>(act_type_)
                    << " when running pnna";
    return false;
  }
  return in_tensors_.front().Shape().size() <= DIMENSION_4D;
}

int PNNAActivation::InitParams() {
  auto act = op_primitive_->value_as_Activation();
  MS_CHECK_TRUE_RET(act != nullptr, RET_ERROR);
  alpha_ = act->alpha();
  min_val_ = act->min_val();
  max_val_ = act->max_val();
  approximate_ = act->approximate();
  act_type_ = act->activation_type();
  return RET_OK;
}

int PNNAActivation::AddOpToPNNAModel(PNNASubGraph *graph) {
  MS_CHECK_TRUE_RET(graph->graph() != nullptr, RET_ERROR);

  auto input_tensor = graph->GetMappedTensor(&in_tensors_[0]);
  if (!input_tensor) {
    input_tensor = graph->ConvertOperand(&in_tensors_[0]);
  }
  auto output_tensor = graph->ConvertOperand(&out_tensors_[0]);
  switch (act_type_) {
    CONVERT_PNNA_ACTIVATION(RELU, Relu);
    CONVERT_PNNA_ACTIVATION(RELU6, Relu6);
    CONVERT_PNNA_ACTIVATION(SIGMOID, Sigmoid);
    CONVERT_PNNA_ACTIVATION(SWISH, Swish);
    CONVERT_PNNA_ACTIVATION(TANH, Tanh);

    case schema::ActivationType_LEAKY_RELU: {
      auto act_op = graph->graph()->CreateOperation<pnna::ops::LeakyRelu>(alpha_);
      act_op->BindInputs({input_tensor});
      act_op->BindOutputs({output_tensor});
    } break;
    default:
      MS_LOG(ERROR) << "Unsupported activation operation type " << static_cast<int>(act_type_) << " is found.";
      break;
  }
  return RET_OK;
}
#undef CONVERT_PNNA_ACTIVATION
}  // namespace lite
}  // namespace mindspore
