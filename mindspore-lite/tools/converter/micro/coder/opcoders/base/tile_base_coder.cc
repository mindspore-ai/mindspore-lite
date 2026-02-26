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

#include "coder/opcoders/base/tile_base_coder.h"
#include <string>
#include <type_traits>

namespace mindspore::lite::micro {
void TileBaseCoder::ComputeStrides(const int *shape, int *strides, int ndim) const {
  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
}

int TileBaseCoder::TileDoubleInputScenes(TileStruct *tile) {
  auto repeats = input_tensors()[SECOND_INPUT];
  if (repeats->data() == nullptr) {
    tile->resize_done_ = false;
    return RET_OK;
  }
  int *input1_addr = (int *)(repeats->data());
  for (int i = 0; i < repeats->ElementsNum(); ++i) {
    tile->dims_[i] = i;
    tile->multiples_[i] = input1_addr[i];
  }
  return RET_OK;
}

int TileBaseCoder::Resize() {
  tile_struct_.input_addr_ = (uint8_t *)(input_tensor_);
  tile_struct_.output_addr_ = (uint8_t *)(output_tensor_);
  tile_struct_.in_dim_ = static_cast<int>(input_tensor_->shape().size());
  tile_struct_.data_size_ = DataTypeSize(input_tensor_->data_type());
  if (input_tensors().size() == DIMENSION_2D) {
    int ret = TileDoubleInputScenes(&tile_struct_);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "tile double input scenes failed");
  }
  MS_CHECK_TRUE_MSG(input_tensor_ != nullptr, RET_ERROR, "input_tensor_ is nullptr");
  MS_CHECK_TRUE_MSG(output_tensor_ != nullptr, RET_ERROR, "output_tensor_ is nullptr");
  MS_CHECK_TRUE_MSG(parameter_ != nullptr, RET_ERROR, "parameter_ is nullptr");
  tile_struct_.in_dim_ = input_tensor_->shape().size();
  TileParameter *param = reinterpret_cast<TileParameter *>(parameter_);
  tile_struct_.dims_size_ = param->dims_size_;
  for (int i = 0; i < MAX_SHAPE_SIZE; i++) {
    tile_struct_.dims_[i] = param->dims_[i];
    tile_struct_.multiples_[i] = param->multiples_[i];
  }
  for (int i = 0; i < tile_struct_.in_dim_; ++i) {
    tile_struct_.in_shape_[i] = input_tensor_->shape().at(i);
    tile_struct_.out_shape_[i] = output_tensor_->shape().at(i);
  }
  ComputeStrides(tile_struct_.in_shape_, tile_struct_.in_strides_, tile_struct_.in_dim_);
  ComputeStrides(tile_struct_.out_shape_, tile_struct_.out_strides_, tile_struct_.in_dim_);
  return RET_OK;
}

}  // namespace mindspore::lite::micro
