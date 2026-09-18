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

#ifndef MSLLM_NNRT_EMBEDDING_DEQUANT_H
#define MSLLM_NNRT_EMBEDDING_DEQUANT_H

#include <cstddef>
#include <cstdint>

namespace mslite {
namespace backend {
namespace nnrt {

// IEEE fp16 <-> fp32 bit-level conversions (fp16 values are passed as raw
// uint16 bits).
uint16_t Fp32ToFp16Bits(float value);
float Fp16BitsToFp32(uint16_t bits);

// Compact Q4_0 phase4 NZF: packed cells followed by per-row g32 fp16 scales.
// Cells are at most 64x1024, with no padding. Rows must be a multiple of 16,
// hidden a multiple of 32; total storage is rows * hidden / 32 * 18 bytes.
bool Q4NzfEmbeddingSize(int rows, int hidden, size_t *packed_bytes, size_t *total_bytes);

struct Q4EmbeddingShape {
  int rows = 0;
  int hidden = 0;
};

// Decode only the requested logical row directly into hidden fp16 output
// values. Adjacent K lanes occupy low/high signed two's-complement nibbles.
// Each 128-byte 16x16 tile interleaves four 32-byte phases: byte j of the
// adjacent-byte layout is at 4 * (j % 32) + j / 32. Scale bits are unchanged.
bool DequantizeEmbeddingRow(const uint8_t *blob, size_t blob_size, Q4EmbeddingShape shape, int row, uint16_t *out_fp16);

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite

#endif  // MSLLM_NNRT_EMBEDDING_DEQUANT_H
