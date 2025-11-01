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

#include "src/extendrt/lite_device_address.h"

#include <complex>
#include <string>
#include <utility>
#include <unordered_map>

#include "ir/device_address_maker.h"
#include "utils/ms_context.h"
#include "src/common/log_adapter.h"

namespace mindspore {
namespace lite {
namespace {
const char kDeviceName[] = "CPU";
DeviceAddressPtr CreateDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector, const Format &format,
                                     TypeId type_id, const std::string &device_name, uint32_t device_id,
                                     uint32_t stream_id, const UserDataPtr &user_data = nullptr) {
  MS_CHECK_TRUE_RET(ptr != nullptr, nullptr);
  return std::make_shared<TestDeviceAddress>(ptr, size, "fault", type_id, device_name);
}
DeviceAddressPtr MakeTestDeviceAddress(TypeId data_type, const ShapeVector &shape, void *data_ptr,
                                       DeviceAddressDeleter &&deleter) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto data_size = SizeOf(shape) * abstract::TypeIdSize(data_type);
  auto device_address =
    CreateDeviceAddress(data_ptr, data_size, shape, Format::DEFAULT_FORMAT, data_type, kDeviceName, device_id, 0);
  device_address->SetDevicePointerDeleter(std::move(deleter));
  return device_address;
}

REGISTER_DEVICE_ADDRESS_MAKER(device::DeviceType::kCPU, [](TypeId data_type, const ShapeVector &shape, void *data_ptr,
                                                           DeviceAddressDeleter &&deleter) {
  return MakeTestDeviceAddress(data_type, shape, data_ptr, std::move(deleter));
});
}  // namespace
}  // namespace lite
}  // namespace mindspore
