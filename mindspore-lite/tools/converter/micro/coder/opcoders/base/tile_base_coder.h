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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_BASE_TILE_BASE_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_BASE_TILE_BASE_CODER_H_

#include <vector>
#include "coder/opcoders/op_coder.h"
#include "nnacl_c/base/tile_base.h"

namespace mindspore::lite::micro {
class TileBaseCoder : public OperatorCoder {
 public:
  TileBaseCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                const LiteGraph::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}
  ~TileBaseCoder() override = default;

 protected:
  void ComputeStrides(const int *shape, int *strides, int ndim) const;
  int TileDoubleInputScenes(TileStruct *tile);
  int Resize();
  TileStruct tile_struct_;
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_BASE_TILE_BASE_CODER_H_
