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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_POOLING_NPU_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_POOLING_NPU_H_

#include <vector>
#include <string>
#include "src/litert/delegate/npu/op/convolution_base_npu.h"
#include "src/litert/delegate/npu/npu_converter_utils.h"

namespace mindspore::lite {

enum class NPUPoolingType { MAX = 0, AVERAGE = 1 };

template <NPUPoolingType POOL_TYPE>
class PoolingNPUOp : public ConvolutionBaseNPUOp {
 public:
  PoolingNPUOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
               const std::vector<mindspore::MSTensor> &out_tensors, std::string name)
      : ConvolutionBaseNPUOp(primitive, in_tensors, out_tensors, name) {}

  virtual ~PoolingNPUOp() {
    if (pooling_ != nullptr) {
      delete pooling_;
      pooling_ = nullptr;
    }
  }

  int SetPoolingParam(const void *pooling_prim) {
    CHECK_NULL_RETURN(pooling_prim);
    pooling_->set_attr_mode(static_cast<int>(POOL_TYPE));

    auto *prim = static_cast<const schema::MaxPoolFusion *>(pooling_prim);

    if (prim->global()) {
      pooling_->set_attr_global_pooling(prim->global());
    } else {
      CHECK_NULL_RETURN(prim->kernel_size());
      CHECK_LESS_RETURN(prim->kernel_size()->size(), DIMENSION_2D);
      auto window_h = static_cast<int>(*(prim->kernel_size()->begin()));
      auto window_w = static_cast<int>(*(prim->kernel_size()->begin() + 1));
      pooling_->set_attr_window(ge::AttrValue::LIST_INT({window_h, window_w}));
    }

    CHECK_NULL_RETURN(prim->strides());
    CHECK_LESS_RETURN(prim->strides()->size(), DIMENSION_2D);
    auto stride_h = static_cast<int>(*(prim->strides()->begin()));
    auto stride_w = static_cast<int>(*(prim->strides()->begin() + 1));
    pooling_->set_attr_stride(ge::AttrValue::LIST_INT({stride_h, stride_w}));

    if (prim->pad_mode() == schema::PadMode_SAME) {
      pooling_->set_attr_pad_mode(PAD_SAME);
      pooling_->set_attr_pad({0, 0, 0, 0});
    } else if (prim->pad_mode() == schema::PadMode_VALID) {
      pooling_->set_attr_pad_mode(PAD_VALID);
      pooling_->set_attr_pad({0, 0, 0, 0});
    } else {
      pooling_->set_attr_pad_mode(0);
      CHECK_NULL_RETURN(prim->pad());
      CHECK_LESS_RETURN(prim->pad()->size(), DIMENSION_4D);
      auto pad_u = static_cast<int>(*(prim->pad()->begin() + PAD_UP));
      auto pad_d = static_cast<int>(*(prim->pad()->begin() + PAD_DOWN));
      auto pad_l = static_cast<int>(*(prim->pad()->begin() + PAD_LEFT));
      auto pad_r = static_cast<int>(*(prim->pad()->begin() + PAD_RIGHT));
      pooling_->set_attr_pad(ge::AttrValue::LIST_INT({pad_u, pad_d, pad_l, pad_r}));
    }

    if (prim->round_mode() == schema::RoundMode_FLOOR) {
      pooling_->set_attr_ceil_mode(0);
      pooling_->set_attr_data_mode(1);
    } else {
      pooling_->set_attr_ceil_mode(1);
      pooling_->set_attr_data_mode(0);
    }
    return RET_OK;
  }

  int Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
           const std::vector<mindspore::MSTensor> &out_tensors) override {
    pooling_ = new (std::nothrow) hiai::op::PoolingD(name_ + "_pooling");
    if (pooling_ == nullptr) {
      MS_LOG(ERROR) << "New pooling npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }

    auto *pooling_prim = GetPoolingPrimitive(primitive);
    if (pooling_prim == nullptr) {
      MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
      return RET_ERROR;
    }

    auto ret = SetPoolingParam(pooling_prim);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set npu op parameter for convolution op " << name_ << " failed.";
      return ret;
    }

    act_type_ = GetActivationType(pooling_prim);
    if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
      ret = SetActivation(pooling_, act_type_);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
        return ret;
      }
    }
    return RET_OK;
  }

  int SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                   const std::vector<mindspore::MSTensor> &out_tensors,
                   const std::vector<ge::Operator *> &npu_inputs) override {
    CHECK_LESS_RETURN(npu_inputs.size(), 1);
    pooling_->set_input_x(*npu_inputs[0]);
    return RET_OK;
  }

  ge::Operator *GetNPUOp() override {
    if (act_type_ == schema::ActivationType_NO_ACTIVATION) {
      return pooling_;
    } else {
      return act_;
    }
  }

 protected:
  /**
   * @brief Get the specific pooling primitive pointer
   * Subclasses must implement this method to return the corresponding primitive pointer
   */
  virtual const void *GetPoolingPrimitive(const schema::Primitive *primitive) const = 0;

  /**
   * @brief Get the activation type
   * Subclasses must implement this method
   */
  virtual schema::ActivationType GetActivationType(const void *pooling_prim) const = 0;

  schema::ActivationType act_type_ = schema::ActivationType_NO_ACTIVATION;
  hiai::op::PoolingD *pooling_ = nullptr;
};

}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_POOLING_NPU_H_
