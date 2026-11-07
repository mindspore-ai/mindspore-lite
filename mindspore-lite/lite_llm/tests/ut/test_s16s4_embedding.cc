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

#include <gtest/gtest.h>

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <limits>
#include <vector>
#include "backend/nnrt/nnrt_embedding.h"

using mslite::backend::nnrt::DequantizeS16S4EmbeddingRow;
using mslite::backend::nnrt::Fp32ToFp16Bits;
using mslite::backend::nnrt::S16S4EmbeddingView;

class S16S4RoundingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    restore_environment_ = std::feholdexcept(&original_environment_) == 0;
    ASSERT_TRUE(restore_environment_);
  }
  void TearDown() override {
    if (restore_environment_) {
      EXPECT_EQ(std::fesetenv(&original_environment_), 0);
    }
  }
  std::fenv_t original_environment_{};
  bool restore_environment_{false};
};

TEST_F(S16S4RoundingTest, UnusedValuesDoNotRaiseOverflow) {
  constexpr int kSize = 128;
  std::vector<uint8_t> weights(kSize * kSize / 2);
  std::vector<uint8_t> scales(kSize * 8, 0);
  S16S4EmbeddingView view{weights.data(), weights.size(), scales.data(), scales.size(), kSize, kSize, 1.0f};
  std::vector<uint16_t> output(kSize);
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
  for (uint8_t packed : {0x00, 0x11, 0xff}) {
    std::fill(weights.begin(), weights.end(), packed);
    const int nibble = packed & 15;
    const int value = nibble < 8 ? nibble : nibble - 16;
    for (float scale : {8188.0f, 8192.0f, 16384.0f, -16384.0f, std::numeric_limits<float>::max()}) {
      const uint16_t expected = Fp32ToFp16Bits(value * scale);
      std::memcpy(scales.data(), &scale, sizeof(scale));
      ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
      ASSERT_TRUE(DequantizeS16S4EmbeddingRow(view, 0, output.data()));
      EXPECT_EQ(std::fetestexcept(FE_OVERFLOW), 0) << "packed=" << static_cast<int>(packed) << ", scale=" << scale;
      EXPECT_TRUE(std::all_of(output.begin(), output.end(), [expected](uint16_t bits) { return bits == expected; }));
    }
  }
}

TEST_F(S16S4RoundingTest, UsedValuesRetainScalarOverflow) {
  constexpr int kSize = 128;
  std::vector<uint8_t> weights(kSize * kSize / 2, 0x22);
  std::vector<uint8_t> scales(kSize * 8, 0);
  const float scale = std::numeric_limits<float>::max();
  std::memcpy(scales.data(), &scale, sizeof(scale));
  S16S4EmbeddingView view{weights.data(), weights.size(), scales.data(), scales.size(), kSize, kSize, 1.0f};
  std::vector<uint16_t> output(kSize);
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
  ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
  ASSERT_TRUE(DequantizeS16S4EmbeddingRow(view, 0, output.data()));
  EXPECT_NE(std::fetestexcept(FE_OVERFLOW), 0);
  EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](uint16_t bits) { return bits == 0x7c00; }));
}

TEST_F(S16S4RoundingTest, UnusedValuesDoNotRaiseUnderflowOrInexact) {
  constexpr int kSize = 128;
  std::vector<uint8_t> weights(kSize * kSize / 2, 0);
  std::vector<uint8_t> scales(kSize * 8, 0);
  S16S4EmbeddingView view{weights.data(), weights.size(), scales.data(), scales.size(), kSize, kSize, 1.0f};
  std::vector<uint16_t> output(kSize);
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
  for (float scale : {0x1p-25f, 0x1p-149f, 1.0f / 3}) {
    std::memcpy(scales.data(), &scale, sizeof(scale));
    ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    ASSERT_TRUE(DequantizeS16S4EmbeddingRow(view, 0, output.data()));
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), 0) << "scale=" << scale;
    EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](uint16_t bits) { return bits == 0; }));
  }
}

TEST_F(S16S4RoundingTest, LookupPreservesScalarRoundingModes) {
  constexpr int kSize = 128;
  std::vector<uint8_t> weights(kSize * kSize / 2, 0);
  for (int k = 0; k < kSize; k += 2) {
    const size_t offset = (k / 64 * kSize) * 32 + k % 64 / 2;
    weights[offset] = static_cast<uint8_t>((k % 16) | (((k + 1) % 16) << 4));
  }
  std::vector<uint8_t> scales(kSize * 8, 0);
  S16S4EmbeddingView view{weights.data(), weights.size(), scales.data(), scales.size(), kSize, kSize, 1.0f};
  std::vector<uint16_t> output(kSize);
  for (int mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(mode), 0);
    for (float scale : {1.0f + 0x1p-11f, 0x1p-25f}) {
      std::memcpy(scales.data(), &scale, sizeof(scale));
      ASSERT_TRUE(DequantizeS16S4EmbeddingRow(view, 0, output.data()));
      for (int k = 0; k < kSize; ++k) {
        const int nibble = k % 16;
        const int value = nibble < 8 ? nibble : nibble - 16;
        EXPECT_EQ(output[k], Fp32ToFp16Bits(value * scale)) << "mode=" << mode << ", k=" << k;
      }
      if (mode == FE_UPWARD && scale > 1.0f) {
        EXPECT_EQ(output[15], 0xbc01);
      }
    }
  }
}

TEST(S16S4Embedding, LookupMatchesScalarForRoundingAndSpecialScales) {
  constexpr int kSize = 128;
  std::vector<uint8_t> weights(kSize * kSize / 2, 0);
  for (int k = 0; k < kSize; k += 2) {
    const size_t offset = (k / 64 * kSize) * 32 + k % 64 / 2;
    weights[offset] = static_cast<uint8_t>((k % 16) | (((k + 1) % 16) << 4));
  }
  std::vector<uint8_t> scales(kSize * 8, 0);
  S16S4EmbeddingView view{weights.data(), weights.size(), scales.data(), scales.size(), kSize, kSize, 1.0f};
  std::vector<uint16_t> output(kSize);
  for (float scale : {1.0f / 3, -1.0f / 3, 0x1p-24f, 1.0e-8f, 65504.0f, 0.0f, -0.0f}) {
    std::memcpy(scales.data(), &scale, sizeof(scale));
    ASSERT_TRUE(DequantizeS16S4EmbeddingRow(view, 0, output.data()));
    for (int k = 0; k < kSize; ++k) {
      const int nibble = k % 16;
      const int value = nibble < 8 ? nibble : nibble - 16;
      EXPECT_EQ(output[k], Fp32ToFp16Bits(value * scale)) << "scale=" << scale << ", k=" << k;
    }
  }
  for (float invalid : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    std::memcpy(scales.data(), &invalid, sizeof(invalid));
    EXPECT_FALSE(DequantizeS16S4EmbeddingRow(view, 0, output.data()));
  }
}

TEST(S16S4Embedding, TileBoundariesAndSignedNibbles) {
  constexpr int kHidden = 256;
  constexpr int kVocab = 384;
  std::vector<uint8_t> weights;
  for (int nt = 0; nt < kVocab / 128; ++nt) {
    for (int kt = 0; kt < kHidden / 64; ++kt) {
      for (int row = 0; row < 128; ++row) {
        for (int col = 0; col < 64; col += 2) {
          int token = nt * 128 + row;
          int k = kt * 64 + col;
          int low = (token * 7 + k * 3) % 16 - 8;
          int high = (token * 7 + (k + 1) * 3) % 16 - 8;
          weights.push_back(static_cast<uint8_t>((low & 15) | ((high & 15) << 4)));
        }
      }
    }
  }
  std::vector<uint8_t> scales(kVocab * kHidden / 16, 0);
  for (size_t i = 0; i < scales.size(); i += 8) {
    const float scale = 1.0f / 1024.0f;
    std::memcpy(scales.data() + i, &scale, sizeof(scale));
  }
  S16S4EmbeddingView view{weights.data(), weights.size(), scales.data(), scales.size(), kHidden, kVocab, 1024.0f};
  std::vector<uint16_t> output(kHidden);
  // Exact FP16 encodings for integer magnitudes 0..8.
  const uint16_t magnitude[] = {0, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800};
  for (int token : {0, 1, 127, 128, 255, 256, 383}) {
    ASSERT_TRUE(DequantizeS16S4EmbeddingRow(view, token, output.data()));
    for (int k = 0; k < kHidden; ++k) {
      const int value = (token * 7 + k * 3) % 16 - 8;
      const uint16_t expected = value < 0 ? magnitude[-value] | 0x8000 : magnitude[value];
      EXPECT_EQ(output[k], expected) << token << ":" << k;
    }
  }
  EXPECT_FALSE(DequantizeS16S4EmbeddingRow(view, -1, output.data()));
  EXPECT_FALSE(DequantizeS16S4EmbeddingRow(view, kVocab, output.data()));
  // A128 scales encode the same embedding values with activation scale 128.
  view.activation_scale = 128.0f;
  for (size_t i = 0; i < scales.size(); i += 8) {
    const float scale = 1.0f / view.activation_scale;
    std::memcpy(scales.data() + i, &scale, sizeof(scale));
  }
  ASSERT_TRUE(DequantizeS16S4EmbeddingRow(view, 128, output.data()));
  for (int k = 0; k < kHidden; ++k) {
    const int value = (128 * 7 + k * 3) % 16 - 8;
    EXPECT_EQ(output[k], value < 0 ? magnitude[-value] | 0x8000 : magnitude[value]);
  }
  --view.weight_bytes;
  EXPECT_FALSE(DequantizeS16S4EmbeddingRow(view, 0, output.data()));
}
