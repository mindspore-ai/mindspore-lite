/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_INT8_BATCHNORM_INT8_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_INT8_BATCHNORM_INT8_CODER_H_

#include <cstring>
#include <vector>
#include "coder/opcoders/op_coder.h"
#include "nnacl/batchnorm_parameter.h"

namespace mindspore::lite::micro::nnacl {
class BatchNormInt8Coder final : public OperatorCoder {
 public:
  BatchNormInt8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                     const LiteGraph::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {
    batchnorm_param_ = reinterpret_cast<BatchNormParameter *>(parameter_);
  }

  ~BatchNormInt8Coder() override = default;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *context) override;

 private:
  int InitConstTensor();
  int InitFusedConstTensor();

  float *alpha_addr_{nullptr};
  float *beta_addr_{nullptr};

  int unit_{0};
  int units_{0};
  int channel_{0};
  BatchNormParameter *batchnorm_param_;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_INT8_BATCHNORM_INT8_CODER_H_
