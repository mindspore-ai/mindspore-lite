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

#include "backend/nnrt/nnrt_embedding_dequant.h"

#include <cmath>
#include <cstring>

namespace mslite {
namespace backend {
namespace nnrt {
namespace {

/// Reinterpret-cast between trivially copyable types of equal size.  C++17
/// has no std::bit_cast, so use memcpy — the portable, optimizer-transparent
/// equivalent — instead of union type-punning.
template <typename To, typename From>
To BitCast(const From &src) {
  static_assert(sizeof(To) == sizeof(From), "BitCast requires equal sizes");
  To dst;
  std::memcpy(&dst, &src, sizeof(dst));
  return dst;
}

inline float Fp32FromBits(uint32_t w) { return BitCast<float>(w); }

inline uint32_t Fp32ToBits(float f) { return BitCast<uint32_t>(f); }

}  // namespace

uint16_t Fp32ToFp16Bits(float value) {
  // IEEE-754 binary16 conversion via standard math decomposition (frexp /
  // ldexp / round-to-nearest-even), unlike the bit-twiddling trick used by
  // ggml/llama.cpp (scale_to_inf/scale_to_zero/bias folding).  Results are
  // bit-identical: round-to-nearest-even, overflow -> +/-inf, NaN canonical.
  const uint16_t sign = static_cast<uint16_t>((Fp32ToBits(value) & 0x80000000u) >> 16);

  if (std::isnan(value)) {
    return static_cast<uint16_t>(sign | 0x7E00u);  // canonical fp16 NaN
  }
  if (std::isinf(value)) {
    return static_cast<uint16_t>(sign | 0x7C00u);
  }
  if (value == 0.0f) {
    return sign;  // signed zero
  }

  // value = mant * 2^exp, mant in [0.5, 1); the fp16 leading-bit exponent is
  // exp - 1 and normal fp16 covers 2^-14 .. 2^15*(2 - 2^-10).
  int exp = 0;
  const float abs_mant = std::fabs(std::frexp(value, &exp));
  const int lead_exp = exp - 1;

  if (lead_exp < -14) {
    // Subnormal: |value| = m * 2^-24, m in (0, 1024); round m to nearest-even
    // on the magnitude, re-attach the sign below.
    const double m_d = std::ldexp(static_cast<double>(std::fabs(value)), 24);
    const uint32_t m = static_cast<uint32_t>(std::nearbyint(m_d));
    if (m >= 1024) {  // rounded up into the smallest normal (exp=-14, mant=0)
      return static_cast<uint16_t>(sign | 0x0400u);
    }
    return static_cast<uint16_t>(sign | m);
  }

  // Normal: 10-bit mantissa from the fractional part, rounded to nearest-even;
  // a round-up carry bumps the exponent, overflow saturates to +/-inf.
  const double frac = static_cast<double>(abs_mant) - 0.5;  // in [0, 0.5)
  uint32_t m = static_cast<uint32_t>(std::nearbyint(frac * 2048.0));
  int16_t fexp = static_cast<int16_t>(lead_exp);
  if (m >= 1024) {
    m = 0;
    ++fexp;
  }
  if (fexp > 15) {
    return static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> inf
  }
  const uint16_t exp_bits = static_cast<uint16_t>((fexp + 15) << 10);
  return static_cast<uint16_t>(sign | exp_bits | static_cast<uint16_t>(m));
}

float Fp16BitsToFp32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x03FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // +/- zero
    } else {
      // fp16 subnormal: value = mant * 2^-24. Normalize into fp32.
      int shift = 0;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        ++shift;
      }
      mant &= 0x03FFu;
      const uint32_t fexp = static_cast<uint32_t>(127 - 15 + 1 - shift);
      bits = sign | (fexp << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
  } else {
    bits = sign | ((exp + 112) << 23) | (mant << 13);  // rebias 15 -> 127
  }
  return Fp32FromBits(bits);
}

void DequantizeEmbeddingRow(const uint8_t *packed, const uint16_t *scales_fp16, int hidden, int group_size,
                            uint16_t *out_fp16) {
  if (packed == nullptr || scales_fp16 == nullptr || out_fp16 == nullptr || hidden <= 0 || group_size <= 0 ||
      group_size % 2 != 0) {
    return;
  }
  const int half = group_size / 2;  // packed bytes per group
  const int num_groups = (hidden + group_size - 1) / group_size;
  for (int g = 0; g < num_groups; ++g) {
    const float scale = Fp16BitsToFp32(scales_fp16[g]);
    const int base = g * group_size;
    for (int j = 0; j < half; ++j) {
      const uint8_t byte = packed[g * half + j];
      const int i0 = base + j;         // low nibble
      const int i1 = base + j + half;  // high nibble
      if (i0 < hidden) {
        out_fp16[i0] = Fp32ToFp16Bits((static_cast<float>(byte & 0x0Fu) - 8.0f) * scale);
      }
      if (i1 < hidden) {
        out_fp16[i1] = Fp32ToFp16Bits((static_cast<float>(byte >> 4) - 8.0f) * scale);
      }
    }
  }
}

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite
