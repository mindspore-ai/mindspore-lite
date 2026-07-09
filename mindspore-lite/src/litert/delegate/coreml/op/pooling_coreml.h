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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_POOLING_COREML_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_POOLING_COREML_H_

#include "src/litert/delegate/coreml/op/coreml_op.h"
#include "schema/model_generated.h"

namespace mindspore::lite {

enum class PoolingType { MAX, AVERAGE };

template <PoolingType POOL_TYPE>
class PoolingCoreMLOp : public CoreMLOp {
 public:
  PoolingCoreMLOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                  const std::vector<mindspore::MSTensor> &out_tensors, std::string name)
      : CoreMLOp(primitive, in_tensors, out_tensors, name) {}

  virtual ~PoolingCoreMLOp() = default;

  int BuildLayer() override {
    MS_ASSERT(op_ != nullptr);
    auto pooling_param = op_->mutable_pooling();

    if constexpr (POOL_TYPE == PoolingType::MAX) {
      pooling_param->set_type(CoreML::Specification::PoolingLayerParams::MAX);
    } else if constexpr (POOL_TYPE == PoolingType::AVERAGE) {
      pooling_param->set_type(CoreML::Specification::PoolingLayerParams::AVERAGE);
      pooling_param->set_avgpoolexcludepadding(true);
    }

    auto *pooling_prim = GetPoolingPrimitive();
    if (pooling_prim == nullptr) {
      MS_LOG(ERROR) << "Pooling primitive is null for op: " << name_;
      return RET_ERROR;
    }

    if (pooling_prim->global()) {
      pooling_param->set_globalpooling(true);
      pooling_param->mutable_valid();
      return RET_OK;
    }

    auto kernel_h = static_cast<int>(*(pooling_prim->kernel_size()->begin()));
    auto kernel_w = static_cast<int>(*(pooling_prim->kernel_size()->begin() + 1));
    auto stride_h = static_cast<int>(*(pooling_prim->strides()->begin()));
    auto stride_w = static_cast<int>(*(pooling_prim->strides()->begin() + 1));

    pooling_param->add_stride(stride_h);
    pooling_param->add_stride(stride_w);
    pooling_param->add_kernelsize(kernel_h);
    pooling_param->add_kernelsize(kernel_w);

    if (pooling_prim->pad_mode() == schema::PadMode_SAME) {
      pooling_param->mutable_same();
    } else {
      pooling_param->mutable_valid();
      if (pooling_prim->pad() != nullptr) {
        auto pad_u = static_cast<int>(*(pooling_prim->pad()->begin() + PAD_UP));
        auto pad_d = static_cast<int>(*(pooling_prim->pad()->begin() + PAD_DOWN));
        auto pad_l = static_cast<int>(*(pooling_prim->pad()->begin() + PAD_LEFT));
        auto pad_r = static_cast<int>(*(pooling_prim->pad()->begin() + PAD_RIGHT));
        auto ret = SetPadding({pad_u, pad_d, pad_l, pad_r});
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "Fail to set padding for op: " << name_;
          return RET_ERROR;
        }
      }
    }

    auto act_type = pooling_prim->activation_type();
    if (act_type != schema::ActivationType_NO_ACTIVATION) {
      auto ret = SetActivation(act_type);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Set pooling activation failed for op: " << name_;
        return RET_ERROR;
      }
    }

    return RET_OK;
  }

 protected:
  virtual const void *GetPoolingPrimitive() const = 0;
};

}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_POOLING_COREML_H_
