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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

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

// IEEE-754 layout constants: binary16 (fp16) carries a 10-bit mantissa and a
// 5-bit exponent biased by 15; binary32 (fp32) carries a 23-bit mantissa and
// an 8-bit exponent biased by 127.
constexpr int kFp16MinNormalExp = -14;          // normals cover 2^-14 .. 2^15 * (2 - 2^-10)
constexpr int kFp16MaxExp = 15;                 // overflow saturates to inf; NOT the encoding bias 15
constexpr uint32_t kFp16MantissaCarry = 1024;   // 2^10: 10-bit mantissa round-up carry bound
constexpr int kFp16SubnormalMantExp = 24;       // subnormal |v| = m * 2^-24
constexpr double kFp16MantQuantScale = 2048.0;  // frac in [0, 0.5) * 2048 -> mantissa in [0, 1024)
constexpr int kFp32MantissaLsb = 23;            // binary32 mantissa occupies bits [22:0]
constexpr int kFp16ToFp32MantShift = 13;        // 23 - 10
constexpr int kFp16ToFp32ExpBiasDelta = 112;    // 127 - 15

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

  if (lead_exp < kFp16MinNormalExp) {
    // Subnormal: |value| = m * 2^-24, m in (0, 1024); round m to nearest-even
    // on the magnitude, re-attach the sign below.
    const double m_d = std::ldexp(static_cast<double>(std::fabs(value)), kFp16SubnormalMantExp);
    const uint32_t m = static_cast<uint32_t>(std::nearbyint(m_d));
    if (m >= kFp16MantissaCarry) {  // rounded up into the smallest normal (exp=-14, mant=0)
      return static_cast<uint16_t>(sign | 0x0400u);
    }
    return static_cast<uint16_t>(sign | m);
  }

  // Normal: 10-bit mantissa from the fractional part, rounded to nearest-even;
  // a round-up carry bumps the exponent, overflow saturates to +/-inf.
  const double frac = static_cast<double>(abs_mant) - 0.5;  // in [0, 0.5)
  uint32_t m = static_cast<uint32_t>(std::nearbyint(frac * kFp16MantQuantScale));
  int16_t fexp = static_cast<int16_t>(lead_exp);
  if (m >= kFp16MantissaCarry) {
    m = 0;
    ++fexp;
  }
  if (fexp > kFp16MaxExp) {
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
      bits = sign | (fexp << kFp32MantissaLsb) | (mant << kFp16ToFp32MantShift);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (mant << kFp16ToFp32MantShift);  // inf / nan
  } else {
    // rebias 15 -> 127
    bits = sign | ((exp + kFp16ToFp32ExpBiasDelta) << kFp32MantissaLsb) | (mant << kFp16ToFp32MantShift);
  }
  return Fp32FromBits(bits);
}

bool Q4NzfEmbeddingSize(int rows, int hidden, size_t *packed_bytes, size_t *total_bytes) {
  if (rows <= 0 || hidden <= 0 || rows % kQ4NzFractalRows != 0 || hidden % kQ4GroupElems != 0 ||
      packed_bytes == nullptr || total_bytes == nullptr) {
    return false;
  }
  const size_t groups_per_row = static_cast<size_t>(hidden) / kQ4GroupElems;
  if (static_cast<size_t>(rows) > std::numeric_limits<size_t>::max() / kQ4BytesPerGroup / groups_per_row) {
    return false;
  }
  const size_t groups = static_cast<size_t>(rows) * groups_per_row;
  *packed_bytes = groups * kQ4PackedBytesPerGroup;
  *total_bytes = groups * kQ4BytesPerGroup;
  return true;
}

bool DequantizeEmbeddingRow(const uint8_t *blob, size_t blob_size, Q4EmbeddingShape shape, int row,
                            uint16_t *out_fp16) {
  size_t packed_bytes = 0;
  size_t total_bytes = 0;
  if (blob == nullptr || out_fp16 == nullptr || row < 0 || row >= shape.rows ||
      !Q4NzfEmbeddingSize(shape.rows, shape.hidden, &packed_bytes, &total_bytes) || blob_size != total_bytes) {
    return false;
  }
  const size_t hidden = static_cast<size_t>(shape.hidden);
  const size_t n = static_cast<size_t>(row);
  const size_t n0 = n / 64 * 64;
  const size_t nc = std::min(size_t{64}, static_cast<size_t>(shape.rows) - n0);
  for (size_t k_base = 0; k_base < hidden; k_base += kQ4GroupElems) {
    const size_t k0 = k_base / 1024 * 1024;
    const size_t kc = std::min(size_t{1024}, hidden - k0);
    // Divide before multiplying: the compact byte count may fit size_t even
    // when the logical element count does not (notably on 32-bit hosts).
    const size_t cell_offset = n0 * (hidden / 2) + nc * (k0 / 2);
    const size_t scale_offset =
      packed_bytes + n0 * (hidden / 16) + nc * (k0 / 16) + (n - n0) * (kc / 16) + (k_base - k0) / 16;
    const uint16_t scale_bits =
      static_cast<uint16_t>(blob[scale_offset]) | (static_cast<uint16_t>(blob[scale_offset + 1]) << 8);
    const float scale = Fp16BitsToFp32(scale_bits);
    for (size_t lane = 0; lane < kQ4GroupElems; ++lane) {
      const size_t k = k_base + lane;
      const size_t fractal = ((k - k0) / 16) * (nc / 16) + (n - n0) / 16;
      const size_t flat = (n % 16) * 16 + k % 16;
      const size_t packed_index = flat / 2;
      const size_t phase4_index = 4 * (packed_index % 32) + packed_index / 32;
      const uint8_t byte = blob[cell_offset + fractal * 128 + phase4_index];
      const int nibble = (byte >> ((flat % 2) * 4)) & 15;
      const int value = nibble < 8 ? nibble : nibble - 16;
      out_fp16[k] = Fp32ToFp16Bits(static_cast<float>(value) * scale);
    }
  }
  return true;
}

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite
