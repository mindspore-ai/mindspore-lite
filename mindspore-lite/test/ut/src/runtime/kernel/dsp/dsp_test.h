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

#ifndef MINDSPORE_LITE_TEST_UT_SRC_RUNTIME_KERNEL_DSP_DSP_TEST_H_
#define MINDSPORE_LITE_TEST_UT_SRC_RUNTIME_KERNEL_DSP_DSP_TEST_H_

#include <iostream>
#include <memory>
#include "schema/inner/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/dsp/dsp_subgraph.h"
#include "common/common_test.h"
#include "nnacl_c/arithmetic_parameter.h"

namespace mindspore::lite::dsp::test {

class DSPCommonTest : public CommonTest {
 public:
  void InitDSPRuntime() {
    dsp_runtime_wrapper_ = new (std::nothrow) dsp::DSPRuntimeInnerWrapper();
    if (dsp_runtime_wrapper_ == nullptr) {
      MS_LOG(ERROR) << "create DSPRuntimeInnerWrapper failed.";
    }
    auto dsp_runtime = dsp_runtime_wrapper_->GetInstance();
    if (dsp_runtime->Init() != RET_OK) {
      MS_LOG(ERROR) << "Init DSP runtime failed.";
    }
    allocator_ = dsp_runtime->GetAllocator();
  }

  void UninitDSPRuntime() {
    delete dsp_runtime_wrapper_;
    dsp_runtime_wrapper_ = nullptr;
  }

  // Local IEEE754 half <-> float converters to avoid any linkage/impl mismatch in tests.
  float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (static_cast<uint32_t>(h) & 0x7C00u) >> 10;
    uint32_t mant = static_cast<uint32_t>(h & 0x03FFu);
    uint32_t f;
    if (exp == 0) {
      if (mant == 0) {
        f = sign;  // zero
      } else {
        // subnormal -> normalize
        exp = 1;
        while ((mant & 0x0400u) == 0) {
          mant <<= 1;
          --exp;
        }
        mant &= 0x03FFu;
        uint32_t fexp = (exp + (127 - 15)) << 23;
        f = sign | fexp | (mant << 13);
      }
    } else if (exp == 0x1Fu) {  // Inf/NaN
      f = sign | 0x7F800000u | (mant << 13);
    } else {
      uint32_t fexp = (exp + (127 - 15)) << 23;
      f = sign | fexp | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
  }

  uint16_t fp32_to_fp16(float val) {
    uint32_t fbits;
    std::memcpy(&fbits, &val, sizeof(fbits));
    uint32_t sign = (fbits >> 16) & 0x8000u;
    uint32_t fexp = (fbits >> 23) & 0xFFu;
    uint32_t fmant = fbits & 0x007FFFFFu;

    // NaN/Inf handling
    if (fexp == 0xFFu) {
      if (fmant != 0) {
        // NaN: keep a quiet NaN in half
        return static_cast<uint16_t>(sign | 0x7C00u | 0x0001u);
      }
      // Inf
      return static_cast<uint16_t>(sign | 0x7C00u);
    }

    // Rebias exponent for half
    int32_t hexp = static_cast<int32_t>(fexp) - 127 + 15;

    if (hexp <= 0) {
      // Subnormal or underflow to zero in half
      if (hexp < -10) {
        return static_cast<uint16_t>(sign);  // Underflow to zero
      }
      // Make implicit leading 1 explicit
      uint32_t mant = fmant | 0x00800000u;
      // Shift to align to half subnormal mantissa (10 bits)
      int shift = 1 - hexp;  // shift in [1..10]
      // Compute mantissa with round-to-nearest-even
      uint32_t mant_rounded = mant >> (shift + 13);
      uint32_t round_bit = (mant >> (shift + 12)) & 1u;
      uint32_t sticky = (mant & ((1u << (shift + 12)) - 1u)) != 0u;
      mant_rounded += (round_bit & (sticky | (mant_rounded & 1u)));
      return static_cast<uint16_t>(sign | static_cast<uint16_t>(mant_rounded));
    }

    if (hexp >= 0x1F) {
      // Overflow to half inf
      return static_cast<uint16_t>(sign | 0x7C00u);
    }

    // Normal case: build exponent and mantissa with round-to-nearest-even
    uint16_t hexp_field = static_cast<uint16_t>(hexp) << 10;
    uint32_t mant = fmant;
    uint32_t mant_rounded = mant >> 13;
    uint32_t round_bit = (mant >> 12) & 1u;
    uint32_t sticky = (mant & 0xFFFu) != 0u;
    mant_rounded += (round_bit & (sticky | (mant_rounded & 1u)));
    if (mant_rounded == 0x400u) {
      // Mantissa overflow after rounding; bump exponent, zero mantissa
      mant_rounded = 0;
      hexp_field = static_cast<uint16_t>(hexp_field + 0x0400u);
      if (hexp_field >= 0x7C00u) {
        // Exponent overflow -> inf
        return static_cast<uint16_t>(sign | 0x7C00u);
      }
    }
    return static_cast<uint16_t>(sign | hexp_field | static_cast<uint16_t>(mant_rounded));
  }

 protected:
  dsp::DSPRuntimeInnerWrapper *dsp_runtime_wrapper_{nullptr};
  std::shared_ptr<DSPAllocator> allocator_;
};
}  // namespace mindspore::lite::dsp::test

#endif  // MINDSPORE_LITE_TEST_UT_SRC_RUNTIME_KERNEL_DSP_DSP_TEST_H_
