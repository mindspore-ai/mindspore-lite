/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_CONV3D_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_CONV3D_TENSORRT_H_
#include <string>
#include <vector>
#include <memory>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "infer/conv3d.h"

namespace mindspore::lite {
class Conv3DTensorRT : public TensorRTOp {
 public:
  Conv3DTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                 const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~Conv3DTensorRT() override;

  int AddInnerOp(TensorRTContext *ctx) override;

  bool IsWeightInputHanledInner() const override { return true; }

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;

 private:
  void SetAttributes(const std::shared_ptr<ops::Conv3D> &conv_op, nvinfer1::IConvolutionLayer *current_layer_);
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_CONV3D_TENSORRT_H_
