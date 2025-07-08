/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/base/cast_base.h"
#include "nnacl/cast_base_simd.h"

typedef union float32_bits {
  unsigned int u;
  float f;
} float32_bits;

uint16_t Float32ToFloat16_(float f) {
  float32_bits hbit;
  hbit.f = f;
  uint16_t hbits = 0;
  // Extract the sign bit
  uint16_t sign = (hbit.u >> FP16_BIT_SIZE) & 0x8000;  // Get the sign (1 bit) ox8000
  // Extract the exponent
  uint32_t exponent = (hbit.u >> FP32_SIGNIFICAND) & 0xFF;  // Extract the exponent (8 bits) 0xFF
  // Handle special cases (NaN, Inf, 0)
  if (exponent == 0xFF) {    // NaN or Infinity 0xFF
    hbits |= sign | 0x7FFF;  // Set to max float16 value (Infinity)
    return hbits;
  } else if (exponent == 0) {  // Zero or denormalized number
    // In float16, we treat zero the same way
    hbits |= sign;  // Preserve sign for zero
    return hbits;
  }
  // Adjust the exponent to fit float16
  exponent -= FP32_EXPONENT_BIAS;  // Remove float32 bias
  exponent += FP16_EXPONENT_BIAS;  // Add float16 bias
  // Check for overflow
  if (exponent >= 0x1F) {    // 0X1F
    hbits |= sign | 0x7FFF;  // Set to max float16 value (Infinity) 0x7FFF
    return hbits;
  }
  if (exponent == 0) {
    // Handle underflow (too small to represent)
    return sign;  // Return zero with the correct sign
  }
  // Shift the mantissa:
  // Extract the mantissa (23 bits), shift right by 13 (10-exp)
  uint32_t mantissa = (hbit.u & 0x7FFFFF) >> FP16_SHIFT;  // 0x7FFFFF
  // Combine sign, exponent, and mantissa into hbits
  hbits |=
    sign | ((uint16_t)exponent << FP16_SIGNIFICAND) | (mantissa & 0x3FF);  // combine sign exponent and mantissa 0x3FF
  return hbits;
}

void Int32ToFloat32(const int32_t *input, float *output, int number) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(Int32ToFloat32, index, input, output, number);

  for (; index < number; ++index) {
    output[index] = (float)input[index];
  }
}

void Float32ToInt32(const float *input, int32_t *output, int number) {
  int index = 0;

  SIMD_RUN_X86_NO_SCALAR(Float32ToInt32, index, input, output, number);

  for (; index < number; ++index) {
    output[index] = (int32_t)input[index];
  }
}

void BoolToFloat32(const bool *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float)input[i];
  }
}

void Uint8ToFloat32(const uint8_t *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float)input[i];
  }
}

void Int32ToFloat32(const int32_t *input, float *output, int number);

void Int64ToFloat32(const int64_t *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float)input[i];
  }
}

#ifdef ENABLE_FP16
void Int64ToFp16(const int64_t *input, float16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float16_t)input[i];
  }
}

void Int32ToFp16(const int32_t *input, float16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float16_t)input[i];
  }
}

void BoolToFp16(const bool *input, float16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float16_t)input[i];
  }
}

void Uint8ToFp16(const uint8_t *input, float16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float16_t)input[i];
  }
}

void Float32ToFp16(const float *input, float16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float16_t)(input[i]);
  }
}

void Fp16ToFloat32(const float16_t *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float)(input[i]);
  }
}
#else
void Fp16ToFloat32(const uint16_t *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = ShortToFloat32(input[i]);
  }
}

void Float32ToFp16(const float *input, uint16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = Float32ToFloat16_(input[i]);
  }
}
#endif

void Float32ToInt32(const float *input, int32_t *output, int number);

void Float32ToInt64(const float *input, int64_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (int64_t)input[i];
  }
}

void Int32ToInt64(const int32_t *input, int64_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (int64_t)input[i];
  }
}

void Int64ToInt32(const int64_t *input, int32_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (int32_t)input[i];
  }
}

void Float32ToInt16(const float *input, int16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (int16_t)input[i];
  }
}

void BoolToInt32(const bool *input, int32_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    if (input[i]) {
      output[i] = 1;
    } else {
      output[i] = 0;
    }
  }
}

void Float32ToBool(const float *input, bool *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (bool)input[i];
  }
}

void Float32ToUint8(const float *input, uint8_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (uint8_t)input[i];
  }
}
