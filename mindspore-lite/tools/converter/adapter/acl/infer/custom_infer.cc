/**
 * Copyright 2021-2026 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/infer/custom_infer.h"
#include "tools/converter/adapter/common/custom_infer_impl.h"
#include "include/registry/register_kernel_interface.h"
#include "tools/converter/adapter/acl/common/acl_types.h"

namespace mindspore {
namespace lite {

Status CustomInterface::Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                              const mindspore::schema::Primitive *primitive) {
  return CustomInferImpl::Infer(inputs, outputs, primitive);
}

Status CustomInterface::GetCustomAttr(const mindspore::schema::Custom *op, const std::string &attr_name,
                                      std::vector<char> *buf) {
  return CustomInferImpl::GetCustomAttr(op, attr_name, buf);
}

std::shared_ptr<mindspore::kernel::KernelInterface> CustomInferCreater() {
  return CustomInferImpl::CreateInfer<CustomInterface>();
}

}  // namespace lite
}  // namespace mindspore

namespace mindspore {
namespace kernel {
REGISTER_CUSTOM_KERNEL_INTERFACE(ACL, ACL, lite::CustomInferCreater);
}  // namespace kernel
}  // namespace mindspore
