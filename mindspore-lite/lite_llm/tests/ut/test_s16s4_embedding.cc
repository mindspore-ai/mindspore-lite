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

#include <cstring>
#include <vector>
#include "backend/nnrt/nnrt_embedding.h"
#include "gtest/gtest.h"

using mslite::backend::nnrt::DequantizeS16S4EmbeddingRow;
using mslite::backend::nnrt::S16S4EmbeddingView;

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
  S16S4EmbeddingView view{weights.data(), weights.size(), scales.data(), scales.size(), kHidden, kVocab};
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
  --view.weight_bytes;
  EXPECT_FALSE(DequantizeS16S4EmbeddingRow(view, 0, output.data()));
}
