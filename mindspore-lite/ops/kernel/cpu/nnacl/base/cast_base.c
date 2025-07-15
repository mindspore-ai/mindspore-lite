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

uint16_t Float32ToFloat16_(float input) {
  float32_bits hbit;
  hbit.f = input;
  uint16_t output = 0;
  // Extract the sign bit, exponent, and mantissa
  uint32_t sign = (hbit.u >> BITS_SHIFT_SIZE_31) & 0x1;
  int32_t exponent = ((hbit.u >> BITS_SHIFT_SIZE_23) & 0xFF) - FP32_EXPONENT_BIAS;  // The exponent bias of float32
  uint32_t mantissa = hbit.u & 0x007FFFFF;                                          // 23-digit tail number

  // Handle special cases (Inf/NaN/Zero)
  if (exponent == EXPONENT_BIAS_VALUE_128) {
    output = (sign << BITS_SHIFT_SIZE_15) | 0x7C00;  // float16's Inf/NaN
    if (mantissa != 0) {
      output |= 0x0200;  // Retain some NaN information
    }
    return output;
  } else if (exponent < (-1 * EXPONENT_BIAS_VALUE_14)) {  // Too small, change to 0
    output = (sign << BITS_SHIFT_SIZE_15) | 0x0000;
    return output;
  } else if (exponent > EXPONENT_BIAS_VALUE_15) {  // Too large, convert to Inf
    output = (sign << BITS_SHIFT_SIZE_15) | 0x7C00;
    return output;
  }

  // Adjust the index (the offset for float16 is 15)
  int32_t new_exponent = exponent + EXPONENT_BIAS_VALUE_15;

  // Trailing digit rounding (23 digits -> 10 digits)
  uint32_t new_mantissa = mantissa >> BITS_SHIFT_SIZE_13;          // Retain the top 10 bits
  uint32_t rounding_bit = (mantissa >> BITS_SHIFT_SIZE_12) & 0x1;  // The 11th digit (rounding position)
  uint32_t sticky_bits = mantissa & 0x0FFF;  // The lower 12 bits (to determine if a carry is needed)

  if (rounding_bit && ((new_mantissa & 0x01) || sticky_bits)) {
    new_mantissa++;                // Rounding
    if (new_mantissa == 0x0400) {  // Carry in the mantissa causes exponent overflow
      new_mantissa = 0;
      new_exponent++;
      if (new_exponent > EXPONENT_BIAS_VALUE_30) {  // Index overflow converts to Inf
        output = (sign << BITS_SHIFT_SIZE_15) | 0x7C00;
        return output;
      }
    }
  }
  output = (sign << BITS_SHIFT_SIZE_15) | (new_exponent << BITS_SHIFT_SIZE_10) | new_mantissa;
  return output;
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
