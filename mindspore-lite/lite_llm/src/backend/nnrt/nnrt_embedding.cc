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

namespace mslite {
namespace backend {
namespace nnrt {

bool NnrtEmbedding::Configure(const NnrtConfig &config) {
  if (config.vocab_size <= 0 || config.hidden_size <= 0) {
    MS_LOG(ERROR) << "Embedding dimensions must be positive";
    return false;
  }
  vocab_size_ = config.vocab_size;
  hidden_size_ = config.hidden_size;
  quantized_ = config.embedding_quant;
  format_ = config.embedding_format;
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
  if (format_ == EmbeddingFormat::kS16S4NzV1) {
    if (!quantized_ || vocab_size_ % 128 != 0 || hidden_size_ % 128 != 0) {
      return false;
    }
    input_count_ = 2;
    weight_bytes = elements / 2;
    scale_bytes = elements / 16;
  } else if (format_ == EmbeddingFormat::kW4A16) {
    if (quantized_) {
      size_t packed_bytes = 0;
      size_t total_bytes = 0;
      if (!Q4NzfEmbeddingSize(vocab_size_, hidden_size_, &packed_bytes, &total_bytes)) {
        return false;
      }
      weight_bytes = total_bytes;
    } else {
      weight_bytes = elements * sizeof(uint16_t);
    }
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
    return {vocab_size_ / 128, hidden_size_ / 128, 128, 8};
  }
  if (format_ == EmbeddingFormat::kS16S4NzV1) {
    return {vocab_size_, hidden_size_ / 2};
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
  const int32_t dtype = !quantized_ ? kOhNnFloat16 : (format_ == EmbeddingFormat::kS16S4NzV1 ? kOhNnInt8 : kOhNnUint8);
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
    std::memcpy(output, weights_ + static_cast<size_t>(token) * row_bytes, row_bytes);
    return true;
  }
  if (format_ == EmbeddingFormat::kS16S4NzV1) {
    return DequantizeS16S4EmbeddingRow({weights_, weight_bytes_, scales_, scale_bytes_, hidden_size_, vocab_size_},
                                       token, output);
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

bool ValidateS16S4Embedding(const S16S4EmbeddingView &view) {
  if (view.weights == nullptr || view.scales == nullptr || view.hidden <= 0 || view.vocab <= 0 ||
      view.hidden % 128 != 0 || view.vocab % 128 != 0) {
    return false;
  }
  const size_t hidden = static_cast<size_t>(view.hidden);
  const size_t vocab = static_cast<size_t>(view.vocab);
  if (vocab > std::numeric_limits<size_t>::max() / hidden) {
    return false;
  }
  return view.weight_bytes == vocab * hidden / 2 && view.scale_bytes == vocab * (hidden / 128) * 8;
}

bool DequantizeS16S4EmbeddingRow(const S16S4EmbeddingView &view, int token, uint16_t *output) {
  if (output == nullptr || token < 0 || token >= view.vocab || !ValidateS16S4Embedding(view)) {
    return false;
  }
  const size_t hidden = static_cast<size_t>(view.hidden);
  const size_t row = static_cast<size_t>(token);
  for (size_t group = 0; group < hidden / 128; ++group) {
    const size_t scale_offset = ((row / 128 * (hidden / 128) + group) * 128 + row % 128) * 8;
    float effective_scale = 0.0f;
    std::memcpy(&effective_scale, view.scales + scale_offset, sizeof(effective_scale));
    // S16S4_NZ_V1 FixPipe records contain the FP16 group scale divided by 1024.
    const float scale = effective_scale * 1024.0f;
    if (!std::isfinite(scale)) {
      return false;
    }
    for (size_t lane = 0; lane < 128; ++lane) {
      const size_t k = group * 128 + lane;
      const size_t offset = ((row / 128 * (hidden / 64) + k / 64) * 128 + row % 128) * 32 + k % 64 / 2;
      const uint8_t packed = view.weights[offset];
      const int nibble = (k % 2 == 0 ? packed & 15 : packed >> 4);
      const int signed_weight = nibble >= 8 ? nibble - 16 : nibble;
      output[k] = Fp32ToFp16Bits(static_cast<float>(signed_weight) * scale);
    }
  }
  return true;
}

bool Q4NzfEmbeddingSize(int rows, int hidden, size_t *packed_bytes, size_t *total_bytes) {
  if (rows <= 0 || hidden <= 0 || rows % 16 != 0 || hidden % 32 != 0 || packed_bytes == nullptr ||
      total_bytes == nullptr) {
    return false;
  }
  const size_t groups_per_row = static_cast<size_t>(hidden) / 32;
  if (static_cast<size_t>(rows) > std::numeric_limits<size_t>::max() / 18 / groups_per_row) {
    return false;
  }
  const size_t groups = static_cast<size_t>(rows) * groups_per_row;
  *packed_bytes = groups * 16;
  *total_bytes = groups * 18;
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
  for (size_t k_base = 0; k_base < hidden; k_base += 32) {
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
    for (size_t lane = 0; lane < 32; ++lane) {
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
