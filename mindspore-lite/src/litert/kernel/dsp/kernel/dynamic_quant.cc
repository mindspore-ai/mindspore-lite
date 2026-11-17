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

#include "src/litert/kernel/dsp/kernel/dynamic_quant.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include "src/litert/kernel_registry.h"
#include "schema/inner/model_generated.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DynamicQuant;

namespace mindspore::kernel {

namespace {
constexpr int kInt8Min = -128;
constexpr int kInt8Max = 127;
constexpr int kInt8MinSymmetric = -127;
constexpr int kAffineRange = 255;     // affine (asymmetric) int8 range: -128..127
constexpr int kSymmetricRange = 254;  // symmetric int8 range: -127..127
constexpr float kDefaultRange = 0.01f;

float Fp16ToFp32Scalar(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
  uint32_t exp = (static_cast<uint32_t>(h) & 0x7C00u) >> 10;
  uint32_t mant = static_cast<uint32_t>(h) & 0x03FFu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      exp = 1;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x03FFu;
      f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    f = sign | 0x7F800000u | (mant << 13);
  } else {
    f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

struct QuantParams {
  float scale{0.0f};
  int zp{0};
  int min_value{kInt8Min};
  int max_value{kInt8Max};
};

// Per-tensor affine/symmetric quantization parameters from min/max.
QuantParams ComputeQuantParams(float real_min, float real_max, bool symmetric) {
  QuantParams q;
  if (symmetric) {
    float absmax = std::max(std::fabs(real_max), std::fabs(real_min));
    // An all-zero input would produce scale==0 and make the whole inference fail.
    // Fall back to a positive scale like the affine branch does (CPU keeps running).
    if (absmax <= 0.0f) {
      absmax = kDefaultRange;
    }
    q.scale = 2.0f * absmax / static_cast<float>(kSymmetricRange);
    q.zp = 0;
    q.min_value = kInt8MinSymmetric;
  } else {
    float range = real_max - real_min;
    if (range <= 0.0f) {
      range = kDefaultRange;
    }
    q.scale = range / static_cast<float>(kAffineRange);
    q.zp = static_cast<int>(std::round(kInt8Min - real_min / q.scale));
  }
  return q;
}
}  // namespace

int DynamicQuantDSPKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_1 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "DynamicQuant unexpected io sizes, in: " << in_tensors_.size() << ", out: " << out_tensors_.size();
    return RET_ERROR;
  }
  auto dtype = in_tensors_[0]->data_type();
  if (dtype != kNumberTypeFloat32 && dtype != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "DynamicQuant unsupported input dtype: " << static_cast<int>(dtype);
    return RET_ERROR;
  }
  param_ = reinterpret_cast<DynamicQuantParameter *>(op_parameter_);
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "DynamicQuant param is null.";
    return RET_ERROR;
  }
  if (param_->dst_type_ != kNumberTypeInt8 || out_tensors_[0]->data_type() != kNumberTypeInt8) {
    MS_LOG(ERROR) << "DynamicQuant only supports int8 output, got dst_type=" << param_->dst_type_;
    return RET_ERROR;
  }
  if (param_->axis_num_ != 0) {
    MS_LOG(ERROR) << "DynamicQuant DSP only supports per-tensor quantization (axis_num=0), got " << param_->axis_num_;
    return RET_ERROR;
  }
  if (in_tensors_[0]->ElementsNum() <= 0) {
    MS_LOG(ERROR) << "DynamicQuant empty input.";
    return RET_ERROR;
  }
  return RET_OK;
}

int DynamicQuantDSPKernel::Prepare() { return RET_OK; }

int DynamicQuantDSPKernel::FindMinMax(float *min_value, float *max_value) {
  auto *data = in_tensors_[0]->data();
  if (data == nullptr) {
    MS_LOG(ERROR) << "DynamicQuant input data is null.";
    return RET_ERROR;
  }
  const size_t count = static_cast<size_t>(in_tensors_[0]->ElementsNum());
  *min_value = FLT_MAX;
  *max_value = -FLT_MAX;
  if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
    const auto *src = reinterpret_cast<const float *>(data);
    for (size_t i = 0; i < count; ++i) {
      *min_value = std::min(*min_value, src[i]);
      *max_value = std::max(*max_value, src[i]);
    }
  } else {
    const auto *src = reinterpret_cast<const uint16_t *>(data);
    for (size_t i = 0; i < count; ++i) {
      const float v = Fp16ToFp32Scalar(src[i]);
      *min_value = std::min(*min_value, v);
      *max_value = std::max(*max_value, v);
    }
  }
  return RET_OK;
}

int DynamicQuantDSPKernel::Run() {
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "DynamicQuant not checked.";
    return RET_ERROR;
  }
  float real_min = 0.0f;
  float real_max = 0.0f;
  if (FindMinMax(&real_min, &real_max) != RET_OK) {
    return RET_ERROR;
  }

  const QuantParams qp = ComputeQuantParams(real_min, real_max, param_->symmetric_);
  if (!(qp.scale > 0.0f)) {
    MS_LOG(ERROR) << "DynamicQuant invalid scale " << qp.scale;
    return RET_ERROR;
  }

  lite::LiteQuantParam quant_param;
  quant_param.scale = qp.scale;
  quant_param.zeroPoint = qp.zp;
  quant_param.bitNum = 8;
  quant_param.inited = true;
  out_tensors_[0]->set_quant_params({quant_param});

  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "DynamicQuant allocator is null.";
    return RET_ERROR;
  }
  uint64_t input_dev = allocator->GetDeviceMemPtr(in_tensors_[0]->data());
  uint64_t output_dev = allocator->GetDeviceMemPtr(out_tensors_[0]->data());
  if (input_dev == 0 || output_dev == 0) {
    MS_LOG(ERROR) << "DynamicQuant device ptr is null.";
    return RET_ERROR;
  }
  const int dtype = static_cast<int>(in_tensors_[0]->data_type());
  const int mtype = static_cast<int>(GetMemType());
  int ret = dynamic_quant_func(input_dev, output_dev, static_cast<uint64_t>(in_tensors_[0]->ElementsNum()), qp.scale,
                               qp.zp, qp.min_value, qp.max_value, core_mask_, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << this->name() << " Run failed!";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_DynamicQuant, DSPKernelCreator<DynamicQuantDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_DynamicQuant, DSPKernelCreator<DynamicQuantDSPKernel>)
#endif
}  // namespace mindspore::kernel
