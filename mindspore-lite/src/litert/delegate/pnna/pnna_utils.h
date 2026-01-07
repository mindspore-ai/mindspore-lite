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
#ifndef MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_PNNA_UTILS_H_
#define MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_PNNA_UTILS_H_

#include <cmath>
#include <memory>
#include <utility>
#include <algorithm>
#include <limits>
#include <vector>
#include "include/api/types.h"
#include "pnna_core.h"  // NOLINT(build/include_subdir)
#include "src/litert/delegate/pnna/pnna_subgraph.h"

#define kNCHW_N 0
#define kNCHW_C 1
#define kNCHW_H 2
#define kNCHW_W 3
#define kNHWC_N 0
#define kNHWC_H 1
#define kNHWC_W 2
#define kNHWC_C 3

namespace mindspore {
namespace lite {

class PNNASubGraph;
enum PAD {
  PAD_UP = 0,
  PAD_DOWN = 1,
  PAD_LEFT = 2,
  PAD_RIGHT = 3,
};
pnna::DataType ConvertToPnnaDataType(DataType type_id);
pnna::DataLayout ConvertToPnnaDataLayout(Format input_layout);
pnna::ShapeType ConvertToPnnaShapeType(const std::vector<int64_t> &input_dimensions);
std::shared_ptr<pnna::Tensor> CreatePnnaTensor(pnna::Graph *graph, pnna::ShapeType shape, pnna::DataType data_type,
                                               const void *buffer = nullptr,
                                               pnna::DataLayout data_layout = pnna::DataLayout::WHCN,
                                               pnna::TensorAttribute tensor_attr = pnna::TensorAttribute::TRANSIENT,
                                               const float *quant_scale = nullptr, uint32_t quant_scale_count = 0,
                                               const int32_t *quant_zero_point = nullptr,
                                               uint32_t quant_channel_dim = 0);
std::shared_ptr<pnna::Tensor> CreatePnnaTensor(pnna::Graph *graph, MSTensor *tensor, pnna::TensorAttribute tensor_attr);
std::vector<uint32_t> ConvertToPnnaPerm(const int32_t *input_perm_data, size_t input_perm_count);
int32_t ConvertToPnnaAxis(int32_t axis, size_t dimension_count);
int HandleConstantInputs(PNNASubGraph *graph, std::vector<mindspore::MSTensor> &inputs);
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_LITERT_DELEGATE_PNNA_PNNA_UTILS_H_
