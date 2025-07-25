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

#include "mindspore-lite/minddata/dataset/kernels/image/random_horizontal_flip_with_bbox_op.h"

#include "mindspore-lite/minddata/dataset/core/cv_tensor.h"
#include "mindspore-lite/minddata/dataset/kernels/image/bounding_box.h"
#include "mindspore-lite/minddata/dataset/kernels/image/image_utils.h"
#include "mindspore-lite/minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status RandomHorizontalFlipWithBBoxOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  RETURN_IF_NOT_OK(BoundingBox::ValidateBoundingBoxes(input));
  RETURN_IF_NOT_OK(ValidateImageDtype("RandomHorizontalFlipWithBBox", input[0]->type()));
  RETURN_IF_NOT_OK(ValidateImageRank("RandomHorizontalFlipWithBBox", input[0]->Rank()));

  if (distribution_(random_generator_)) {
    // To test bounding boxes algorithm, create random bboxes from image dims
    size_t num_of_boxes = input[1]->shape()[0];                         // set to give number of bboxes
    float img_center = static_cast<float>(input[0]->shape()[1]) / 2.F;  // get the center of the image
    for (int i = 0; i < num_of_boxes; i++) {
      std::shared_ptr<BoundingBox> bbox;
      RETURN_IF_NOT_OK(BoundingBox::ReadFromTensor(input[1], i, &bbox));
      // do the flip
      BoundingBox::bbox_float diff = img_center - bbox->x();   // get distance from min_x to center
      BoundingBox::bbox_float refl_min_x = diff + img_center;  // get reflection of min_x
      BoundingBox::bbox_float new_min_x =
        refl_min_x - bbox->width();  // subtract from the reflected min_x to get the new one
      bbox->SetX(new_min_x);
      RETURN_IF_NOT_OK(bbox->WriteToTensor(input[1], i));
    }
    (*output).resize(2);
    // move input to output pointer of bounding boxes
    (*output)[1] = input[1];
    // perform HorizontalFlip on the image
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input[0]);
    return HorizontalFlip(std::static_pointer_cast<Tensor>(input_cv), &(*output)[0]);
  }
  *output = input;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
