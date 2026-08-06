/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_BASE_STACK_BASE_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_BASE_STACK_BASE_CODER_H_

#include <vector>
#include "coder/opcoders/op_coder.h"
#include "nnacl_c/stack_parameter.h"

namespace mindspore::lite::micro::nnacl {
// StackBaseCoder is dtype-agnostic: the nnacl `Stack` kernel is a `void*` byte-copy
// (signature: Stack(void **inputs, void *output, size_t input_num, size_t copy_size, ...)), and the
// per-element size is carried by `copy_size` (computed via DataTypeSize). So one coder serves every
// dtype (float32/int8/int32/float16) — same pattern as ReshapeBaseCoder. There is no int8-specific
// arithmetic, so no separate int8 coder is needed.
class StackBaseCoder final : public OperatorCoder {
 public:
  StackBaseCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                 const LiteGraph::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}
  ~StackBaseCoder() override = default;

  int Prepare(CoderContext *const context) override;
  int DoCode(CoderContext *const context) override;

 private:
  int ReSize();

  int axis_{0};
  StackParameter *stack_param_{nullptr};
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_BASE_STACK_BASE_CODER_H_
