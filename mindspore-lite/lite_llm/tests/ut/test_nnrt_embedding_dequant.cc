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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "backend/nnrt/nnrt_embedding_dequant.h"

namespace mslite {
namespace backend {
namespace nnrt {
namespace {

int QuantValue(int row, int k) { return (row * 5 + k * 3 + k / 32 + row / 64 + k / 1024) % 16 - 8; }
int ScaleExponent(int row, int group) { return (row * 3 + group + row / 64 + group / 32) % 4 - 2; }
bool NegativeScale(int row, int group) { return (row + group + row / 64 + group / 32) % 2 != 0; }

// All fixture scales are signed powers of two, exactly representable in fp16.
uint16_t ScaleBits(int row, int group) {
  return static_cast<uint16_t>(((15 + ScaleExponent(row, group)) << 10) | (NegativeScale(row, group) ? 0x8000 : 0));
}

// Scalar oracle: encode the exact dyadic product using known integer fp16
// encodings, without the decoder's indexing or either runtime fp16 converter.
uint16_t ExpectedBits(int row, int k) {
  constexpr uint16_t kIntegerBits[] = {0, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800};
  const int q = QuantValue(row, k);
  const int magnitude = q < 0 ? -q : q;
  const uint16_t sign = ((q < 0) != NegativeScale(row, k / 32)) ? 0x8000 : 0;
  return static_cast<uint16_t>(sign |
                               (magnitude == 0 ? 0 : kIntegerBits[magnitude] + ScaleExponent(row, k / 32) * 1024));
}

// Serialize a 16x16 tile in physical phase4 order, independently of the
// decoder's random-access offset expressions.
void AppendTile(int row_base, int k_base, std::vector<uint8_t> *blob) {
  for (int row_in_phase = 0; row_in_phase < 4; ++row_in_phase) {
    for (int pair = 0; pair < 8; ++pair) {
      for (int phase = 0; phase < 4; ++phase) {
        const int row = row_base + phase * 4 + row_in_phase;
        const int k = k_base + pair * 2;
        blob->push_back(static_cast<uint8_t>((QuantValue(row, k) & 15) | ((QuantValue(row, k + 1) & 15) << 4)));
      }
    }
  }
}

// Traverse compact physical axes [nt, ks, k16, n16], then [nt, ks, row, g32].
// Terminal cells retain only complete logical n16/g32 groups, never padding.
std::vector<uint8_t> MakeNzfBlob(int rows, int hidden) {
  std::vector<uint8_t> blob;
  blob.reserve(static_cast<size_t>(rows) * (hidden / 32) * 18);
  for (int n0 = 0; n0 < rows; n0 += 64) {
    for (int k0 = 0; k0 < hidden; k0 += 1024) {
      for (int k = k0; k < std::min(k0 + 1024, hidden); k += 16) {
        for (int n = n0; n < std::min(n0 + 64, rows); n += 16) {
          AppendTile(n, k, &blob);
        }
      }
    }
  }
  for (int n0 = 0; n0 < rows; n0 += 64) {
    for (int k0 = 0; k0 < hidden; k0 += 1024) {
      for (int row = n0; row < std::min(n0 + 64, rows); ++row) {
        for (int k = k0; k < std::min(k0 + 1024, hidden); k += 32) {
          const uint16_t bits = ScaleBits(row, k / 32);
          blob.push_back(static_cast<uint8_t>(bits & 255));
          blob.push_back(static_cast<uint8_t>(bits >> 8));
        }
      }
    }
  }
  return blob;
}

TEST(NnrtEmbeddingDequant, SizesExcludePaddingAndSeparateScales) {
  struct Case {
    int rows;
    int hidden;
  };
  for (const auto &item : {Case{16, 32}, Case{64, 1024}, Case{80, 1056}, Case{144, 2080}}) {
    SCOPED_TRACE(::testing::Message() << item.rows << "x" << item.hidden);
    size_t packed = 0;
    size_t total = 0;
    ASSERT_TRUE(Q4NzfEmbeddingSize(item.rows, item.hidden, &packed, &total));
    EXPECT_EQ(packed, static_cast<size_t>(item.rows) * item.hidden / 2);
    EXPECT_EQ(total, static_cast<size_t>(item.rows) * (item.hidden / 32) * 18);
  }
}

TEST(NnrtEmbeddingDequant, DecodesSignedNibblesAndPerRowGroupScalesAcrossTiles) {
  for (const auto shape : {Q4EmbeddingShape{80, 1056}, {112, 1152}, {144, 2080}}) {
    SCOPED_TRACE(::testing::Message() << shape.rows << "x" << shape.hidden);
    const auto blob = MakeNzfBlob(shape.rows, shape.hidden);
    ASSERT_EQ(blob.size(), static_cast<size_t>(shape.rows) * (shape.hidden / 32) * 18);
    // Full rows include every g32 boundary, K1024 and the final compact group.
    // Rows straddle n16 subtiles and N64, including both logical edge rows.
    for (int row : {0, 1, 15, 16, 31, 32, 47, 48, 63, 64, shape.rows - 1}) {
      SCOPED_TRACE(row);
      std::vector<uint16_t> output(shape.hidden + 2, 0x7bff);
      ASSERT_TRUE(DequantizeEmbeddingRow(blob.data(), blob.size(), shape, row, output.data() + 1));
      EXPECT_EQ(output.front(), 0x7bff);
      EXPECT_EQ(output.back(), 0x7bff);
      for (int k = 0; k < shape.hidden; ++k) {
        ASSERT_EQ(output[k + 1], ExpectedBits(row, k)) << "column " << k;
      }
    }
  }
}

TEST(NnrtEmbeddingDequant, DecodesIndependentPhaseInterleavedWireBytes) {
  // A complete 16x16 tile begins with four interleaved rows 0, 4, 8, 12.
  // Distinct signed nibbles make the old adjacent-byte indexing fail.
  std::vector<uint8_t> blob(16 * 18, 0);
  const uint8_t wire[] = {0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x0f};
  for (size_t i = 0; i < sizeof(wire); ++i) {
    blob[i] = wire[i];
  }
  const uint16_t expected[][4] = {
    {0x3c00, 0x4000, 0xc700, 0xc600},  // 1, 2, -7, -6
    {0x4200, 0x4400, 0xc500, 0xc400},  // 3, 4, -5, -4
    {0x4500, 0x4600, 0xc200, 0xc000},  // 5, 6, -3, -2
    {0x4700, 0xc800, 0xbc00, 0x0000},  // 7, -8, -1, 0
  };
  for (int phase = 0; phase < 4; ++phase) {
    const int row = phase * 4;
    blob[256 + row * 2 + 1] = 0x3c;  // g32 scale = fp16 1.
    uint16_t output[32] = {};
    ASSERT_TRUE(DequantizeEmbeddingRow(blob.data(), blob.size(), {16, 32}, row, output));
    for (int k = 0; k < 32; ++k) {
      EXPECT_EQ(output[k], k < 4 ? expected[phase][k] : 0) << "row " << row << ", column " << k;
    }
  }
}

TEST(NnrtEmbeddingDequant, RoundsProductsToFp16InLastCompactGroup) {
  // Last row/group of the minimal compact cell: scale 0x3555, q=7.
  // Exact product rounds from 2.332763671875 to fp16 0x40aa (ties to even).
  std::vector<uint8_t> blob(16 * 18, 0x77);
  blob[286] = 0x55;
  blob[287] = 0x35;
  uint16_t output[32] = {};
  ASSERT_TRUE(DequantizeEmbeddingRow(blob.data(), blob.size(), {16, 32}, 15, output));
  for (uint16_t value : output) {
    EXPECT_EQ(value, 0x40aa);
  }
}

TEST(NnrtEmbeddingDequant, ChecksSizeOverflowBeforeMultiplication) {
  constexpr int kRows = std::numeric_limits<int>::max() / 16 * 16;
  constexpr int kHidden = std::numeric_limits<int>::max() / 32 * 32;
  constexpr uint64_t kBytes = static_cast<uint64_t>(kRows) * (kHidden / 32) * 18;
  size_t packed = 123;
  size_t total = 456;
  const bool fits = kBytes <= std::numeric_limits<size_t>::max();
  EXPECT_EQ(Q4NzfEmbeddingSize(kRows, kHidden, &packed, &total), fits);
  EXPECT_EQ(packed, fits ? static_cast<size_t>(kBytes / 18 * 16) : 123);
  EXPECT_EQ(total, fits ? static_cast<size_t>(kBytes) : 456);
}

TEST(NnrtEmbeddingDequant, RejectsInvalidDimensionsAndNullSizeOutputs) {
  size_t packed = 123;
  size_t total = 456;
  for (const auto &dims :
       {std::pair<int, int>{0, 32}, {-1, 32}, {64, 0}, {64, -1}, {1, 32}, {17, 32}, {64, 16}, {64, 33}, {65, 1057}}) {
    SCOPED_TRACE(::testing::Message() << dims.first << "x" << dims.second);
    EXPECT_FALSE(Q4NzfEmbeddingSize(dims.first, dims.second, &packed, &total));
    EXPECT_EQ(packed, 123u);
    EXPECT_EQ(total, 456u);
    uint8_t blob = 0;
    uint16_t output = 0x7bff;
    EXPECT_FALSE(DequantizeEmbeddingRow(&blob, 1, {dims.first, dims.second}, 0, &output));
    EXPECT_EQ(output, 0x7bff);
  }
  EXPECT_FALSE(Q4NzfEmbeddingSize(64, 1024, nullptr, &total));
  EXPECT_FALSE(Q4NzfEmbeddingSize(64, 1024, &packed, nullptr));
}

TEST(NnrtEmbeddingDequant, RejectsInvalidRowsPointersAndNonExactBlobSizesWithoutWriting) {
  auto blob = MakeNzfBlob(80, 1056);
  blob.push_back(0);
  const size_t valid_size = blob.size() - 1;
  std::vector<uint16_t> output(1056, 0x7bff);
  const auto untouched = output;
  for (int row : {-1, 80, 81}) {
    EXPECT_FALSE(DequantizeEmbeddingRow(blob.data(), valid_size, {80, 1056}, row, output.data()));
    EXPECT_EQ(output, untouched);
  }
  for (size_t size : {size_t{0}, valid_size - 1, valid_size + 1, size_t{4 * 36864}}) {
    EXPECT_FALSE(DequantizeEmbeddingRow(blob.data(), size, {80, 1056}, 0, output.data()));
    EXPECT_EQ(output, untouched);
  }
  EXPECT_FALSE(DequantizeEmbeddingRow(nullptr, valid_size, {80, 1056}, 0, output.data()));
  EXPECT_EQ(output, untouched);
  EXPECT_FALSE(DequantizeEmbeddingRow(blob.data(), valid_size, {80, 1056}, 0, nullptr));
}

}  // namespace
}  // namespace nnrt
}  // namespace backend
}  // namespace mslite
