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
#ifndef MSLLM_MSL_FORMAT_H
#define MSLLM_MSL_FORMAT_H

/// .msl v1 format contract — the runtime-side single source of truth.
///
/// Mirrored by the Python packer ``export/msl_pack.py`` and pinned
/// byte-for-byte by ``tests/data/golden_v1.msl`` (C++ side:
/// ``tests/ut/test_msl_golden.cpp``; Python side:
/// ``tests/py/test_msl_golden.py``).  Layout changes must land on all
/// four sides together or CI goes red.
///
/// Layout (little-endian):
///   MslHeader(24B):  magic[4] ".MSL" | version u32 | kv_count u32
///                    | resource_count u32 | alignment u32 | reserved u32
///   KV region:       key_len u32 | key | type u32 | value_len u32 | value
///   Resource table:  name[64] | offset u64 | size u64 | access u32 | reserved u32
///   Data region:     payloads, each starting at an offset aligned to
///                    ``alignment``.
///
/// KV value types form a closed v1 set: unknown types are rejected
/// (layout contract), unknown keys are skipped (forward-compatible
/// metadata extension).  Adding a value type or changing the layout
/// requires a new ``version``.

#include <cstddef>
#include <cstdint>

namespace mslite_llm {
namespace msl_format {

constexpr char kMagic[4] = {'.', 'M', 'S', 'L'};
constexpr uint32_t kVersion = 1;
constexpr size_t kHeaderSize = 24;
constexpr size_t kEntrySize = 88;
constexpr size_t kNameSize = 64;
constexpr uint32_t kDefaultAlignment = 4096;

// KV value types (v1 closed set).
enum KvType : uint32_t {
  kTypeBool = 0,
  kTypeUint32 = 1,
  kTypeUint64 = 2,
  kTypeFloat32 = 3,
  kTypeString = 4,
  kTypeStringArray = 5,
};

// Resource access modes.
constexpr uint32_t kAccessMmap = 0;  // consumer may mmap the payload
constexpr uint32_t kAccessRead = 1;  // consumer should read/pread the payload

// Layout reference structs — parsed with memcpy, never reinterpret_cast'ed.
#pragma pack(push, 1)
struct MslHeader {
  uint8_t magic[4];
  uint32_t version;
  uint32_t kv_count;
  uint32_t resource_count;
  uint32_t alignment;
  uint32_t reserved;
};
#pragma pack(pop)
static_assert(sizeof(MslHeader) == kHeaderSize, "MslHeader must be 24 bytes");

#pragma pack(push, 1)
struct MslResourceEntry {
  uint8_t name[kNameSize];
  uint64_t offset;
  uint64_t size;
  uint32_t access;
  uint32_t reserved;
};
#pragma pack(pop)
static_assert(sizeof(MslResourceEntry) == kEntrySize, "MslResourceEntry must be 88 bytes");

/// v1 KV keys consumed by the runtime.  Adding a key does NOT bump the
/// version (readers skip unknown keys); only keys with runtime consumers
/// live here (audited against nnrt_backend / llm.cpp).
namespace key {
inline constexpr const char kModelName[] = "model.name";
inline constexpr const char kModelVersion[] = "model.version";
inline constexpr const char kModelFormatVersion[] = "model.format_version";
inline constexpr const char kModelDtype[] = "model.dtype";

inline constexpr const char kArchNumLayers[] = "arch.num_layers";
inline constexpr const char kArchHiddenSize[] = "arch.hidden_size";
inline constexpr const char kArchIntermediateSize[] = "arch.intermediate_size";
inline constexpr const char kArchNumHeads[] = "arch.num_heads";
inline constexpr const char kArchNumKvHeads[] = "arch.num_kv_heads";
inline constexpr const char kArchHeadDim[] = "arch.head_dim";
inline constexpr const char kArchVocabSize[] = "arch.vocab_size";
inline constexpr const char kArchMaxPositionEmbeddings[] = "arch.max_position_embeddings";
inline constexpr const char kArchTieWordEmbeddings[] = "arch.tie_word_embeddings";
inline constexpr const char kArchRopeTheta[] = "arch.rope_theta";
inline constexpr const char kArchNormEps[] = "arch.norm_eps";

inline constexpr const char kLitertPrefillPath[] = "litert.prefill.path";
inline constexpr const char kLitertPrefillSeqLen[] = "litert.prefill.seq_len";
inline constexpr const char kLitertDecodePath[] = "litert.decode.path";
inline constexpr const char kLitertDecodeDynamicPastLen[] = "litert.decode.dynamic_past_len";
inline constexpr const char kLitertDecodePastLen[] = "litert.decode.past_len";
inline constexpr const char kLitertDecodeMaxPastLen[] = "litert.decode.max_past_len";
inline constexpr const char kLitertDecodeVariants[] = "litert.decode_variants";

inline constexpr const char kNpuMaxLength[] = "npu.max_length";
inline constexpr const char kNpuChunkSize[] = "npu.chunk_size";
inline constexpr const char kNpuEmbeddingQuant[] = "npu.embedding_quant";
inline constexpr const char kNpuScaleGpSize[] = "npu.scale_gp_size";

inline constexpr const char kAssetTokenizer[] = "asset.tokenizer";
inline constexpr const char kAssetEmbedding[] = "asset.embedding";
inline constexpr const char kAssetEmbeddingFp16[] = "asset.embedding_fp16";
inline constexpr const char kAssetRopeSin[] = "asset.rope_sin";
inline constexpr const char kAssetRopeCos[] = "asset.rope_cos";
inline constexpr const char kAssetAttentionMask[] = "asset.attention_mask";

inline constexpr const char kGenEosTokenId[] = "gen.eos_token_id";
}  // namespace key

}  // namespace msl_format
}  // namespace mslite_llm

#endif  // MSLLM_MSL_FORMAT_H
