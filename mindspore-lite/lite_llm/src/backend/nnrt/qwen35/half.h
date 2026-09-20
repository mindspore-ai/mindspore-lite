// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI half-precision conversions.
#pragma once
#include <cstdint>
#include <cstring>
namespace mslite_llm::qwen35 {
inline float HalfToFloat(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  const uint32_t exponent = (bits >> 10) & 0x1fu;
  const uint32_t mantissa = bits & 0x03ffu;
  uint32_t value = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      value = sign;
    } else {
      uint32_t normalized = mantissa;
      int32_t shift = 0;
      while ((normalized & 0x0400u) == 0) {
        normalized <<= 1;
        ++shift;
      }
      value = sign | (static_cast<uint32_t>(127 - 15 - shift + 1) << 23) | ((normalized & 0x03ffu) << 13);
    }
  } else if (exponent == 31) {
    value = sign | 0x7f800000u | (mantissa << 13);
  } else {
    value = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  float result = 0.0f;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

inline uint16_t FloatToHalf(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t mantissa = bits & 0x7fffffu;
  const uint32_t raw_exponent = (bits >> 23) & 0xffu;
  if (raw_exponent == 0xffu) {
    return static_cast<uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
  }
  int32_t exponent = static_cast<int32_t>(raw_exponent) - 127 + 15;
  if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<uint16_t>(sign);
    const uint32_t normalized = mantissa | 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint32_t half_mantissa = normalized >> shift;
    const uint32_t remainder = normalized & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u) != 0)) {
      ++half_mantissa;
    }
    return static_cast<uint16_t>(sign | half_mantissa);
  }
  uint32_t half = (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13);
  const uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u) != 0)) {
    ++half;
  }
  return static_cast<uint16_t>(sign | half);
}

}  // namespace mslite_llm::qwen35
