/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_UTILS_TENSOR_UTILS_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_UTILS_TENSOR_UTILS_H_

#include <vector>
#include <string>
#include <memory>
#include <functional>

#include "include/api/types.h"
#include "ir/tensor.h"
#include "common/device_address.h"
#include "common/utils.h"
#include "common/mutable_tensor_impl.h"
#include "common/kernel.h"
#include "src/tensor.h"
#include "infer/tensor.h"
#ifdef ENABLE_CLOUD_INFERENCE
#include "src/extendrt/delegate/ascend_acl/ascend_allocator_plugin.h"
#endif
namespace mindspore {
class CloudTensorUtils {
 public:
  /* lite tensor ---> Address */
  static kernel::AddressPtr LiteTensorToAddressPtr(const lite::Tensor *lite_tensor);
  static std::vector<mindspore::kernel::AddressPtr> LiteTensorToAddressPtrVec(
    const std::vector<lite::Tensor *> &lite_tensors);

  /* lite tensor ---> kernel tensor */
  static kernel::KernelTensor *LiteTensorToKernelTensorPtr(const lite::Tensor *lite_tensor);
  static std::vector<kernel::KernelTensor *> LiteTensorToKernelTensorPtrVec(
    const std::vector<lite::Tensor *> &lite_tensors);
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_UTILS_TENSOR_UTILS_H_
