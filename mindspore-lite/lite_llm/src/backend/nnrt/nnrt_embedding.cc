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

#include "backend/nnrt/nnrt_embedding.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include "backend/nnrt/nnrt_config.h"
#include "backend/nnrt/nnrt_log.h"
#include "backend/nnrt/nnrt_wrapper.h"
#include "manifest/msl_package_reader.h"

#include "include/securec.h"

namespace mslite {
namespace backend {
namespace nnrt {

namespace {
constexpr int kBitsPerByte = 8;
constexpr int kWeightBits = 4;
constexpr int kWeightsPerByte = kBitsPerByte / kWeightBits;
constexpr int kWeightSignBit = 1 << (kWeightBits - 1);
constexpr int kWeightValueCount = 1 << kWeightBits;
constexpr int kWeightMask = kWeightValueCount - 1;
constexpr int kQwen3SharedHeadLayers = 36;
constexpr float kQwen3SharedHeadScale = 128.0f;
constexpr float kDefaultActivationScale = 1024.0f;

// S16S4 uses [N/128, K/64, 128, 32] packed weights and
// [N/128, K/128, 128, 8] FixPipe scale records.
constexpr int kS16S4RowsPerTile = 128;
constexpr int kS16S4ColumnsPerTile = 64;
constexpr int kS16S4GroupSize = 128;
constexpr int kS16S4PackedRowBytes = kS16S4ColumnsPerTile / kWeightsPerByte;
constexpr int kS16S4ScaleRecordBytes = 8;

// Q4 NZF stores one FP16 scale per Group32 and phase-interleaved 16x16 tiles.
constexpr int kQ4ElementsPerScaleByte = kQ4GroupElems / kQ4ScaleBytesPerGroup;
constexpr int kQ4FractalBytes = kQ4NzFractalRows * kQ4NzFractalRows / kWeightsPerByte;
constexpr int kQ4RowsPerTile = 64;
constexpr int kQ4ColumnsPerTile = 1024;
constexpr int kQ4PackingPhases = 4;
constexpr int kQ4BytesPerPhase = kQ4FractalBytes / kQ4PackingPhases;
}  // namespace

bool NnrtEmbedding::Configure(const NnrtConfig &config) {
  if (config.vocab_size <= 0 || config.hidden_size <= 0) {
    MS_LOG(ERROR) << "Embedding dimensions must be positive";
    return false;
  }
  vocab_size_ = config.vocab_size;
  hidden_size_ = config.hidden_size;
  quantized_ = config.embedding_quant;
  format_ = config.embedding_format;
  // Temporary RTN shared-head policy; keep it aligned with the Qwen3 exporter.
  activation_scale_ = config.architecture_name == "qwen3" && config.num_layers == kQwen3SharedHeadLayers
                        ? kQwen3SharedHeadScale
                        : kDefaultActivationScale;
  path_ = config.embedding_path;
  package_reader_ = config.package_reader;
  single_file_ = config.single_file;
  if (!InitStorage()) {
    MS_LOG(ERROR) << "Unsupported embedding format, dimensions or storage size";
    return false;
  }
  return true;
}

bool NnrtEmbedding::Load(const ReadAsset &read_asset) {
  size_t bytes = 0;
  if (single_file_) {
    if (package_reader_ == nullptr || !package_reader_->Mmap(path_, &weights_, &bytes)) {
      MS_LOG(ERROR) << "Failed to map embedding asset";
      return false;
    }
  } else {
    if (!read_asset(path_, &upload_buffer_)) {
      MS_LOG(ERROR) << "Failed to read embedding asset";
      return false;
    }
    weights_ = upload_buffer_.data();
    bytes = upload_buffer_.size();
  }
  if (weights_ == nullptr || bytes != weight_bytes_ + scale_bytes_) {
    MS_LOG(ERROR) << "Embedding asset size does not match its format and dimensions";
    return false;
  }
  scales_ = scale_bytes_ == 0 ? nullptr : weights_ + weight_bytes_;
  return true;
}

bool NnrtEmbedding::InitStorage() {
  // Derive storage once from the encoding and logical dimensions. Use wide
  // arithmetic before narrowing to size_t or NNRT's signed dimensions.
  const uint64_t elements = static_cast<uint64_t>(vocab_size_) * hidden_size_;
  uint64_t weight_bytes = 0;
  uint64_t scale_bytes = 0;
  input_count_ = 1;
  if (mslite_llm::IsS16S4Embedding(format_)) {
    if (!quantized_ || vocab_size_ % kS16S4RowsPerTile != 0 || hidden_size_ % kS16S4GroupSize != 0 ||
        !std::isfinite(activation_scale_) || activation_scale_ <= 0.0f) {
      return false;
    }
    input_count_ = 2;
    weight_bytes = elements / kWeightsPerByte;
    scale_bytes = elements / kS16S4GroupSize * kS16S4ScaleRecordBytes;
  } else if (format_ == EmbeddingFormat::kFP16) {
    if (quantized_) {
      return false;
    }
    weight_bytes = elements * sizeof(uint16_t);
  } else if (format_ == EmbeddingFormat::kW4A16) {
    size_t packed_bytes = 0;
    size_t total_bytes = 0;
    if (!quantized_ || !Q4NzfEmbeddingSize(vocab_size_, hidden_size_, &packed_bytes, &total_bytes)) {
      return false;
    }
    weight_bytes = total_bytes;
  } else {
    return false;
  }
  if (weight_bytes > std::numeric_limits<int32_t>::max() || scale_bytes > std::numeric_limits<int32_t>::max()) {
    return false;
  }
  weight_bytes_ = static_cast<size_t>(weight_bytes);
  scale_bytes_ = static_cast<size_t>(scale_bytes);
  return true;
}

std::vector<int32_t> NnrtEmbedding::InputShape(size_t input) const {
  if (input == 1) {
    return {vocab_size_ / kS16S4RowsPerTile, hidden_size_ / kS16S4GroupSize, kS16S4RowsPerTile, kS16S4ScaleRecordBytes};
  }
  if (mslite_llm::IsS16S4Embedding(format_)) {
    return {vocab_size_, hidden_size_ / kWeightsPerByte};
  }
  const size_t element_bytes = quantized_ ? 1 : sizeof(uint16_t);
  // Preserve original ONNX rank; NNRT may report a padded four-dimensional desc.
  return {static_cast<int32_t>(weight_bytes_ / element_bytes)};
}

bool NnrtEmbedding::CreateInputs(const CreateTensor &create, const UploadTensor &upload,
                                 std::vector<NN_Tensor *> *inputs) {
  constexpr size_t kFirstEmbeddingInput = 6;
  if (inputs == nullptr || inputs->size() < kFirstEmbeddingInput + input_count_ || weights_ == nullptr) {
    return false;
  }
  const int32_t dtype = !quantized_ ? kOhNnFloat16 : (mslite_llm::IsS16S4Embedding(format_) ? kOhNnInt8 : kOhNnUint8);
  const uint8_t *buffers[2] = {weights_, scales_};
  const size_t sizes[2] = {weight_bytes_, scale_bytes_};
  for (size_t i = 0; i < input_count_; ++i) {
    auto *tensor = create(kFirstEmbeddingInput + i, InputShape(i), dtype, sizes[i]);
    (*inputs)[kFirstEmbeddingInput + i] = tensor;
    if (tensor == nullptr) {
      return false;
    }
    buffers[i] = upload(tensor, buffers[i], sizes[i]);
    if (buffers[i] == nullptr) {
      return false;
    }
  }
  weights_ = buffers[0];
  scales_ = buffers[1];
  ReleaseUploadSource();
  return true;
}

void NnrtEmbedding::ReleaseUploadSource() {
  std::vector<uint8_t>().swap(upload_buffer_);
  if (single_file_ && package_reader_ != nullptr && !package_reader_->Reclaim(path_)) {
    MS_LOG(WARNING) << "Failed to reclaim embedding package pages";
  }
}

bool NnrtEmbedding::Row(int token, uint16_t *output) const {
  if (token < 0 || token >= vocab_size_ || output == nullptr || weights_ == nullptr) {
    return false;
  }
  if (!quantized_) {
    const size_t row_bytes = static_cast<size_t>(hidden_size_) * sizeof(uint16_t);
    if (memcpy_s(output, row_bytes, weights_ + static_cast<size_t>(token) * row_bytes, row_bytes) != EOK) {
      return false;
    }
    return true;
  }
  if (mslite_llm::IsS16S4Embedding(format_)) {
    return DequantizeS16S4EmbeddingRow(
      {weights_, weight_bytes_, scales_, scale_bytes_, hidden_size_, vocab_size_, activation_scale_}, token, output);
  }
  return DequantizeEmbeddingRow(weights_, weight_bytes_, {vocab_size_, hidden_size_}, token, output);
}

namespace {

/// Reinterpret-cast between trivially copyable types of equal size.  C++17
/// has no std::bit_cast, so use memcpy — the portable, optimizer-transparent
/// equivalent — instead of union type-punning.
template <typename To, typename From>
To BitCast(const From &src) {
  static_assert(sizeof(To) == sizeof(From), "BitCast requires equal sizes");
  To dst;
  // count == destMax is pinned by the static_assert above, so this cannot fail.
  (void)memcpy_s(&dst, sizeof(dst), &src, sizeof(dst));
  return dst;
}

inline float Fp32FromBits(uint32_t w) { return BitCast<float>(w); }

inline uint32_t Fp32ToBits(float f) { return BitCast<uint32_t>(f); }

// fp16 conversion constants shared by Fp32ToFp16Bits / Fp16BitsToFp32.
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

bool ValidateS16S4Embedding(const S16S4EmbeddingView &view) {
  if (view.weights == nullptr || view.scales == nullptr || view.hidden <= 0 || view.vocab <= 0 ||
      view.hidden % kS16S4GroupSize != 0 || view.vocab % kS16S4RowsPerTile != 0 ||
      (!std::isfinite(view.activation_scale) || view.activation_scale <= 0.0f)) {
    return false;
  }
  const size_t hidden = static_cast<size_t>(view.hidden);
  const size_t vocab = static_cast<size_t>(view.vocab);
  if (vocab > std::numeric_limits<size_t>::max() / hidden) {
    return false;
  }
  return view.weight_bytes == vocab * hidden / kWeightsPerByte &&
         view.scale_bytes == vocab * (hidden / kS16S4GroupSize) * kS16S4ScaleRecordBytes;
}

bool DequantizeS16S4EmbeddingRow(const S16S4EmbeddingView &view, int token, uint16_t *output) {
  if (output == nullptr || token < 0 || token >= view.vocab || !ValidateS16S4Embedding(view)) {
    return false;
  }
  const size_t hidden = static_cast<size_t>(view.hidden);
  const size_t row = static_cast<size_t>(token);
  const size_t row_tile = row / kS16S4RowsPerTile;
  const size_t row_in_tile = row % kS16S4RowsPerTile;
  const size_t groups_per_row = hidden / kS16S4GroupSize;
  const size_t column_tiles = hidden / kS16S4ColumnsPerTile;
  for (size_t group = 0; group < groups_per_row; ++group) {
    const size_t scale_offset =
      ((row_tile * groups_per_row + group) * kS16S4RowsPerTile + row_in_tile) * kS16S4ScaleRecordBytes;
    float effective_scale = 0.0f;
    if (memcpy_s(&effective_scale, sizeof(effective_scale), view.scales + scale_offset, sizeof(effective_scale)) !=
        EOK) {
      return false;
    }
    // Recover the group scale from the activation-scaled FixPipe encoding.
    const float scale = effective_scale * view.activation_scale;
    if (!std::isfinite(scale)) {
      return false;
    }
    for (size_t lane = 0; lane < kS16S4GroupSize; ++lane) {
      const size_t k = group * kS16S4GroupSize + lane;
      const size_t offset = ((row_tile * column_tiles + k / kS16S4ColumnsPerTile) * kS16S4RowsPerTile + row_in_tile) *
                              kS16S4PackedRowBytes +
                            k % kS16S4ColumnsPerTile / kWeightsPerByte;
      const uint8_t packed = view.weights[offset];
      const int nibble = (packed >> ((k % kWeightsPerByte) * kWeightBits)) & kWeightMask;
      const int signed_weight = nibble >= kWeightSignBit ? nibble - kWeightValueCount : nibble;
      output[k] = Fp32ToFp16Bits(static_cast<float>(signed_weight) * scale);
    }
  }
  return true;
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
  const size_t n0 = n / kQ4RowsPerTile * kQ4RowsPerTile;
  const size_t nc = std::min(size_t{kQ4RowsPerTile}, static_cast<size_t>(shape.rows) - n0);
  for (size_t k_base = 0; k_base < hidden; k_base += kQ4GroupElems) {
    const size_t k0 = k_base / kQ4ColumnsPerTile * kQ4ColumnsPerTile;
    const size_t kc = std::min(size_t{kQ4ColumnsPerTile}, hidden - k0);
    // Divide before multiplying: the compact byte count may fit size_t even
    // when the logical element count does not (notably on 32-bit hosts).
    const size_t cell_offset = n0 * (hidden / kWeightsPerByte) + nc * (k0 / kWeightsPerByte);
    const size_t scale_offset = packed_bytes + n0 * (hidden / kQ4ElementsPerScaleByte) +
                                nc * (k0 / kQ4ElementsPerScaleByte) + (n - n0) * (kc / kQ4ElementsPerScaleByte) +
                                (k_base - k0) / kQ4ElementsPerScaleByte;
    const uint16_t scale_bits =
      static_cast<uint16_t>(blob[scale_offset]) | (static_cast<uint16_t>(blob[scale_offset + 1]) << kBitsPerByte);
    const float scale = Fp16BitsToFp32(scale_bits);
    for (size_t lane = 0; lane < kQ4GroupElems; ++lane) {
      const size_t k = k_base + lane;
      const size_t fractal = ((k - k0) / kQ4NzFractalRows) * (nc / kQ4NzFractalRows) + (n - n0) / kQ4NzFractalRows;
      const size_t flat = (n % kQ4NzFractalRows) * kQ4NzFractalRows + k % kQ4NzFractalRows;
      const size_t packed_index = flat / kWeightsPerByte;
      const size_t phase4_index =
        kQ4PackingPhases * (packed_index % kQ4BytesPerPhase) + packed_index / kQ4BytesPerPhase;
      const uint8_t byte = blob[cell_offset + fractal * kQ4FractalBytes + phase4_index];
      const int nibble = (byte >> ((flat % kWeightsPerByte) * kWeightBits)) & kWeightMask;
      const int value = nibble < kWeightSignBit ? nibble : nibble - kWeightValueCount;
      out_fp16[k] = Fp32ToFp16Bits(static_cast<float>(value) * scale);
    }
  }
  return true;
}

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite
