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

#include <cstdint>

namespace mslite {
namespace backend {
namespace nnrt {

// IEEE fp16 <-> fp32 bit-level conversions (fp16 values are passed as raw uint16 bits).
uint16_t Fp32ToFp16Bits(float value);
float Fp16BitsToFp32(uint16_t bits);

// Dequantize one row of W4A16 int4-packed embedding weights to fp16.
// The format matches quantize_Q4_N_0_V1_reference (douyin kernel): symmetric
// quantization, one fp16 scale per group of group_size weights,
// w = (q - 8) * scale. Within a group, byte j packs the quantized value of
// element base+j in its low nibble and of element base+j+group_size/2 in its
// high nibble (SPLIT order).
//   packed:      this row's hidden/2 packed int4 bytes
//   scales_fp16: this row's hidden/group_size fp16 scales
//   out_fp16:    output buffer for hidden fp16 values (raw uint16 bits)
void DequantizeEmbeddingRow(const uint8_t *packed, const uint16_t *scales_fp16, int hidden, int group_size,
                            uint16_t *out_fp16);

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite

#endif  // MSLLM_NNRT_EMBEDDING_DEQUANT_H
