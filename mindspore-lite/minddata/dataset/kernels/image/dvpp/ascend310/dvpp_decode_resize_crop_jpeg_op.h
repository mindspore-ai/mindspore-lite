/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_DVPP_DECODE_RESIZE_CROP_JPEG_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_DVPP_DECODE_RESIZE_CROP_JPEG_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "mindspore-lite/minddata/dataset/core/data_type.h"
#include "mindspore-lite/minddata/dataset/core/device_resource.h"
#include "mindspore-lite/minddata/dataset/core/tensor.h"
#include "mindspore-lite/minddata/dataset/kernels/image/dvpp/acl_adapter.h"
#include "mindspore-lite/minddata/dataset/kernels/image/dvpp/utils/ErrorCode.h"
#include "mindspore-lite/minddata/dataset/kernels/tensor_op.h"
#include "mindspore-lite/minddata/dataset/util/log_adapter.h"
#include "mindspore-lite/minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class DvppDecodeResizeCropJpegOp : public TensorOp {
 public:
  DvppDecodeResizeCropJpegOp(int32_t crop_height, int32_t crop_width, int32_t resized_height, int32_t resized_width)
      : crop_height_(crop_height),
        crop_width_(crop_width),
        resized_height_(resized_height),
        resized_width_(resized_width) {}

  /// \brief Destructor
  ~DvppDecodeResizeCropJpegOp() = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status Compute(const std::shared_ptr<DeviceTensor> &input, std::shared_ptr<DeviceTensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kDvppDecodeResizeCropJpegOp; }

  Status SetAscendResource(const std::shared_ptr<DeviceResource> &resource) override;

 private:
  int32_t crop_height_;
  int32_t crop_width_;
  int32_t resized_height_;
  int32_t resized_width_;
  std::shared_ptr<void> processor_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_DVPP_DECODE_RESIZE_CROP_JPEG_OP_H_
