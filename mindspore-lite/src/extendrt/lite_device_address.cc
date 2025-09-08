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
template <typename T>
void ConvertSameType(T *dst, const T *src, size_t elem_num) {
  if (dst == nullptr || src == nullptr) {
    return;
  }
  for (size_t i = 0; i < elem_num; ++i) {
    dst[i] = src[i];
  }
}

void ConvertSameType(void *const dst, const void *src, size_t size, TypeId type) {
  if (type == kNumberTypeFloat16) {
    auto dst_data = static_cast<float16 *>(dst);
    auto src_data = static_cast<const float16 *>(src);
    ConvertSameType(dst_data, src_data, size >> 1);
  } else if (type == kNumberTypeFloat32) {
    auto dst_data = static_cast<float *>(dst);
    auto src_data = static_cast<const float *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(float));
  } else if (type == kNumberTypeFloat64) {
    auto dst_data = static_cast<double *>(dst);
    auto src_data = static_cast<const double *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(double));
  } else if (type == kNumberTypeBFloat16) {
    auto dst_data = static_cast<bfloat16 *>(dst);
    auto src_data = static_cast<const bfloat16 *>(src);
    ConvertSameType(dst_data, src_data, size >> 1);
  } else if (type == kNumberTypeHiFloat8) {
    auto dst_data = static_cast<hifloat8 *>(dst);
    auto src_data = static_cast<const hifloat8 *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(hifloat8));
  } else if (type == kNumberTypeFloat8E5M2) {
    auto dst_data = static_cast<float8_e5m2 *>(dst);
    auto src_data = static_cast<const float8_e5m2 *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(float8_e5m2));
  } else if (type == kNumberTypeFloat8E4M3FN) {
    auto dst_data = static_cast<float8_e4m3fn *>(dst);
    auto src_data = static_cast<const float8_e4m3fn *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(float8_e4m3fn));
  } else if (type == kNumberTypeInt8) {
    auto dst_data = static_cast<int8_t *>(dst);
    auto src_data = static_cast<const int8_t *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(int8_t));
  } else if (type == kNumberTypeInt16) {
    auto dst_data = static_cast<int16_t *>(dst);
    auto src_data = static_cast<const int16_t *>(src);
    ConvertSameType(dst_data, src_data, size >> 1);
  } else if (type == kNumberTypeInt32) {
    auto dst_data = static_cast<int *>(dst);
    auto src_data = static_cast<const int *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(int));
  } else if (type == kNumberTypeInt64) {
    auto dst_data = static_cast<int64_t *>(dst);
    auto src_data = static_cast<const int64_t *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(int64_t));
  } else if (type == kNumberTypeBool) {
    auto dst_data = static_cast<bool *>(dst);
    auto src_data = static_cast<const bool *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(bool));
  } else if (type == kNumberTypeUInt8) {
    auto dst_data = static_cast<uint8_t *>(dst);
    auto src_data = static_cast<const uint8_t *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(uint8_t));
  } else if (type == kNumberTypeUInt16) {
    auto dst_data = static_cast<uint16_t *>(dst);
    auto src_data = static_cast<const uint16_t *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(uint16_t));
  } else if (type == kNumberTypeUInt32) {
    auto dst_data = static_cast<uint32_t *>(dst);
    auto src_data = static_cast<const uint32_t *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(uint32_t));
  } else if (type == kNumberTypeUInt64) {
    auto dst_data = static_cast<uint64_t *>(dst);
    auto src_data = static_cast<const uint64_t *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(uint64_t));
  } else if (type == kNumberTypeComplex64) {
    auto dst_data = static_cast<std::complex<float> *>(dst);
    auto src_data = static_cast<const std::complex<float> *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(std::complex<float>));
  } else if (type == kNumberTypeComplex128) {
    auto dst_data = static_cast<std::complex<double> *>(dst);
    auto src_data = static_cast<const std::complex<double> *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(std::complex<double>));
  } else {
    MS_LOG(EXCEPTION) << "Invalid Type: " << TypeIdLabel(type);
  }
}
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

// clang-format off
#define FOR_EACH_TYPE_BASE(M)                    \
  M(kNumberTypeBool, bool)                       \
  M(kNumberTypeUInt8, uint8_t)                   \
  M(kNumberTypeInt4, int8_t)                     \
  M(kNumberTypeInt8, int8_t)                     \
  M(kNumberTypeInt16, int16_t)                   \
  M(kNumberTypeInt32, int32_t)                   \
  M(kNumberTypeInt64, int64_t)                   \
  M(kNumberTypeUInt16, uint16_t)                 \
  M(kNumberTypeUInt32, uint32_t)                 \
  M(kNumberTypeUInt64, uint64_t)                 \
  M(kNumberTypeFloat16, float16)                 \
  M(kNumberTypeFloat32, float)                   \
  M(kNumberTypeFloat64, double)                  \
  M(kNumberTypeFloat8E4M3FN, float8_e4m3fn)      \
  M(kNumberTypeFloat8E5M2, float8_e5m2)          \
  M(kNumberTypeHiFloat8, hifloat8)               \
  M(kNumberTypeComplex64, ComplexStorage<float>) \
  M(kNumberTypeComplex128, ComplexStorage<double>)

#ifndef KERNEL_EXECUTOR_ANDROID
#define FOR_EACH_TYPE_EXTRA(M) M(kNumberTypeBFloat16, bfloat16)
#else
#define FOR_EACH_TYPE_EXTRA(M)
#endif

#define FOR_EACH_TYPE(M) \
  FOR_EACH_TYPE_BASE(M)  \
  FOR_EACH_TYPE_EXTRA(M)

#define REGISTER_SIZE(address_type_id, address_type) { address_type_id, sizeof(address_type) },

static const std::unordered_map<TypeId, size_t> kTypeSizeMap = {
  FOR_EACH_TYPE(REGISTER_SIZE)
};

size_t GetTypeSize(TypeId tid) {
  return kTypeSizeMap.at(tid);
}

template <typename T>
using DstCopyFunc = void (*)(T *src_ptr, void *dst_ptr, size_t size);

template <typename T>
static const std::unordered_map<TypeId, DstCopyFunc<T>> g_dst_copy_map = {
#define REGISTER_DST(dst_type_id, dst_type)                     \
  {dst_type_id, +[](T *src_ptr, void *dst_ptr, size_t size) {   \
    auto buf = static_cast<dst_type *>(dst_ptr);                \
    return tensor::TransDataType<dst_type>(src_ptr, buf, size); \
    }},
  FOR_EACH_TYPE(REGISTER_DST)
#undef REGISTER_DST
};

template <typename T>
void CopyData(T *src_ptr, size_t size, void *dst_ptr, TypeId dst_type_id) {
  auto &m = g_dst_copy_map<T>;
  auto it = m.find(dst_type_id);
  if (it == m.end()) {
    MS_LOG(EXCEPTION) << "Cannot construct Tensor because of unsupported dst data type: " << dst_type_id << ".";
  }
  it->second(src_ptr, dst_ptr, size);
}

using SrcCopyFunc = std::function<void(void *src_ptr, void *dst_ptr, size_t size, TypeId dst_type_id)>;

static const std::unordered_map<TypeId, SrcCopyFunc> g_src_copy_map = {
#define REGISTER_SRC(src_type_id, src_type)                                          \
  {src_type_id, +[](void *src_ptr, void *dst_ptr, size_t size, TypeId dst_type_id) { \
    auto buf = static_cast<src_type *>(src_ptr);                                     \
    return CopyData<src_type>(buf, size, dst_ptr, dst_type_id);                      \
    }},
  FOR_EACH_TYPE(REGISTER_SRC)
#undef REGISTER_SRC
};

#undef FOR_EACH_TYPE
#undef FOR_EACH_TYPE_BASE
#undef FOR_EACH_TYPE_EXTRA
#undef REGISTER_SIZE
// clang-format on

void CopyData(const DeviceAddress *src_device_address, const DeviceAddress *dst_device_address) {
  MS_EXCEPTION_IF_NULL(src_device_address);
  MS_EXCEPTION_IF_NULL(dst_device_address);

  TypeId src_type_id = src_device_address->type_id();
  TypeId dst_type_id = dst_device_address->type_id();
  auto src_size = src_device_address->GetSize() / GetTypeSize(src_type_id);
  auto dst_size = dst_device_address->GetSize() / GetTypeSize(dst_type_id);
  if (src_size != dst_size) {
    MS_LOG(EXCEPTION) << "Not same shape in device address:" << src_device_address->ToString()
                      << " and:" << dst_device_address->ToString();
  }

  void *src_ptr = src_device_address->GetMutablePtr();
  void *dst_ptr = dst_device_address->GetMutablePtr();
  MS_EXCEPTION_IF_NULL(src_ptr);
  MS_EXCEPTION_IF_NULL(dst_ptr);

  auto it = g_src_copy_map.find(src_type_id);
  if (it == g_src_copy_map.end()) {
    MS_LOG(EXCEPTION) << "Unsupported conversion from " << src_type_id << " to " << dst_type_id;
  }
  it->second(src_ptr, dst_ptr, src_size, dst_type_id);
}
}  // namespace

bool LiteAsyncCopy(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync, size_t stream_id) {
  const auto &dst_device_address = dynamic_cast<const TestDeviceAddress *>(dst_device_sync.get());
  const auto &src_device_address = dynamic_cast<const TestDeviceAddress *>(src_device_sync.get());
  MS_EXCEPTION_IF_NULL(dst_device_address);
  MS_EXCEPTION_IF_NULL(src_device_address);
  if (dst_device_address->GetSize() == 0 || src_device_address->GetSize() == 0) {
    MS_LOG(INFO) << "No need sync for dst device address: " << dst_device_address->ToString()
                 << " and src device address: " << src_device_address->ToString();
    return true;
  }

  if (dst_device_address->format() != src_device_address->format()) {
    MS_LOG(ERROR) << "Format is different, src(format:" << src_device_address->format()
                  << "), dst(format:" << dst_device_address->format() << ") for device address:" << dst_device_address;
    return false;
  }
  auto dst_ptr = dst_device_address->GetMutablePtr();
  auto src_ptr = src_device_address->GetMutablePtr();
  MS_EXCEPTION_IF_NULL(src_device_address->GetMutablePtr());
  MS_EXCEPTION_IF_NULL(dst_device_address->GetMutablePtr());
  if (dst_ptr == src_ptr) {
    MS_LOG(DEBUG) << "host_ptr is equal to device ptr, request ignored.";
    return true;
  }
  auto dst_type_id = dst_device_address->type_id();
  auto src_type_id = src_device_address->type_id();

  if (src_type_id == dst_type_id) {
    if (src_device_address->GetSize() > dst_device_address->GetSize()) {
      MS_LOG(WARNING) << "Please check whether need sync data, src size: " << src_device_address->GetSize()
                      << ", dst size: " << dst_device_address->GetSize();
      return true;
    }
    auto ret_code = memcpy_s(dst_ptr, src_device_address->GetSize(), src_ptr, src_device_address->GetSize());
    // Return ERANGE when the copy size is larger than SECUREC_MEM_MAX_LEN.
    if (ret_code == ERANGE) {
      ConvertSameType(dst_device_address->GetMutablePtr(), src_device_address->GetMutablePtr(),
                      dst_device_address->GetSize(), src_type_id);
    } else if (ret_code != EOK) {
      MS_LOG(ERROR) << "Failed to copy tensor from device address:" << src_device_address->ToString()
                    << " to :" << dst_device_address->ToString();
      return false;
    } else {
      return true;
    }
  }

  MS_LOG(INFO) << "Types not match. src type: " << TypeIdLabel(src_type_id)
               << ", dst type: " << TypeIdLabel(dst_type_id) << " device_address:" << dst_device_address << " !";
  CopyData(src_device_address, dst_device_address);
  return true;
}

bool LiteSyncCopy(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync, size_t stream_id) {
  return LiteAsyncCopy(dst_device_sync, src_device_sync, stream_id);
}

MS_REGISTER_HAL_COPY_FUNC(DeviceType::kCPU,
                          ([](const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                              size_t stream_id) { return LiteSyncCopy(dst_device_sync, src_device_sync, stream_id); }),
                          ([](const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                              size_t stream_id,
                              bool) { return LiteSyncCopy(dst_device_sync, src_device_sync, stream_id); }),
                          ([](void *dst, const void *src, uint64_t size, size_t stream_id) { return true; }));

}  // namespace lite
}  // namespace mindspore
