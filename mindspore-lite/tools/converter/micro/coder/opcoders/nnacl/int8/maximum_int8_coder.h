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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_INT8_MAXIMUM_INT8_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_INT8_MAXIMUM_INT8_CODER_H_

#include <string>
#include <memory>
#include <vector>
#include "coder/opcoders/op_coder.h"
#include "nnacl_c/arithmetic_parameter.h"

namespace mindspore::lite::micro::nnacl {

class MaximumInt8Coder : public OperatorCoder {
 public:
  MaximumInt8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                   const LiteGraph::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}
  ~MaximumInt8Coder() override = default;

  int Prepare(CoderContext *const context) override;
  int DoCode(CoderContext *const context) override;

 private:
  Tensor *input0_{nullptr};
  Tensor *input1_{nullptr};
  int8_t *tile0_data_{nullptr};
  int8_t *tile1_data_{nullptr};
  ArithmeticParameter *arith_para_{nullptr};
  float in0_scale_{1.0f};
  int32_t in0_zp_{0};
  float in1_scale_{1.0f};
  int32_t in1_zp_{0};
  float out_scale_{1.0f};
  int32_t out_zp_{0};
  int32_t element_num_{0};
};

class MinimumInt8Coder : public OperatorCoder {
 public:
  MinimumInt8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                   const LiteGraph::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}
  ~MinimumInt8Coder() override = default;

  int Prepare(CoderContext *const context) override;
  int DoCode(CoderContext *const context) override;

 private:
  Tensor *input0_{nullptr};
  Tensor *input1_{nullptr};
  int8_t *tile0_data_{nullptr};
  int8_t *tile1_data_{nullptr};
  ArithmeticParameter *arith_para_{nullptr};
  float in0_scale_{1.0f};
  int32_t in0_zp_{0};
  float in1_scale_{1.0f};
  int32_t in1_zp_{0};
  float out_scale_{1.0f};
  int32_t out_zp_{0};
  int32_t element_num_{0};
};

}  // namespace mindspore::lite::micro::nnacl
#endif
