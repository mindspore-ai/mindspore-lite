// Copyright 2026 Huawei Technologies Co., Ltd
#include "tokenizer/qwen35/qwen_tokenizer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <string_view>
#include <utility>

#include "manifest/artifact_file.h"

namespace mslite_llm::qwen35_tokenizer {
namespace {

constexpr uint32_t kSBase = 0xac00;
constexpr uint32_t kLBase = 0x1100;
constexpr uint32_t kVBase = 0x1161;
constexpr uint32_t kTBase = 0x11a7;
constexpr uint32_t kLCount = 19;
constexpr uint32_t kVCount = 21;
constexpr uint32_t kTCount = 28;
constexpr uint32_t kNCount = kVCount * kTCount;
constexpr uint32_t kSCount = kLCount * kNCount;

constexpr char kSeparator = '\x1f';
constexpr char kMagic[] = "QWEN_BPE_TOKENIZER_V1";
constexpr char kBinaryMagic[] = "QWENBPE2";
constexpr uint32_t kBinaryVersion = 2;
constexpr uint32_t kHeaderSize = 48;
constexpr uint32_t kSectionByteTokenIds = 1;
constexpr uint32_t kSectionLetterRanges = 2;
constexpr uint32_t kSectionMarkRanges = 3;
constexpr uint32_t kSectionNumberRanges = 4;
constexpr uint32_t kSectionWhitespaceRanges = 5;
constexpr uint32_t kSectionNfdIndex = 6;
constexpr uint32_t kSectionNfdValues = 7;
constexpr uint32_t kSectionCcc = 8;
constexpr uint32_t kSectionComposition = 9;
constexpr uint32_t kSectionMerges = 10;
constexpr uint32_t kSectionAddedTokenIndex = 11;
constexpr uint32_t kSectionAddedTokenBytes = 12;
constexpr uint32_t kSectionTokenIndex = 13;
constexpr uint32_t kSectionTokenBytes = 14;
constexpr uint32_t kSectionTokenOffsets = 15;
constexpr uint32_t kTokenSpecial = 1u << 0;
constexpr uint32_t kPackedOffsetMask = 0x0fffffffu;
constexpr uint32_t kPackedFlagShift = 28;
constexpr size_t kChunkCacheSize = 512;
constexpr size_t kMaxCachedChunkBytes = 256;

uint32_t ReadU32(const std::vector<uint8_t> &data, size_t offset) {
  return static_cast<uint32_t>(data[offset]) | (static_cast<uint32_t>(data[offset + 1]) << 8) |
         (static_cast<uint32_t>(data[offset + 2]) << 16) | (static_cast<uint32_t>(data[offset + 3]) << 24);
}

int32_t ReadI32(const std::vector<uint8_t> &data, size_t offset) { return static_cast<int32_t>(ReadU32(data, offset)); }

uint64_t ReadU64(const std::vector<uint8_t> &data, size_t offset) {
  return static_cast<uint64_t>(ReadU32(data, offset)) | (static_cast<uint64_t>(ReadU32(data, offset + 4)) << 32);
}

uint64_t HashBytes(std::string_view value) {
  uint64_t hash = 1469598103934665603ULL;
  for (unsigned char byte : value) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::atomic<uint64_t> g_next_instance_id{1};

std::string PairKey(const std::string &left, const std::string &right) {
  std::string key;
  key.reserve(left.size() + right.size() + 1);
  key.append(left);
  key.push_back(kSeparator);
  key.append(right);
  return key;
}

uint64_t CompositionKey(uint32_t first, uint32_t second) { return (static_cast<uint64_t>(first) << 32) | second; }

bool Base64Decode(std::string_view input, std::string *output) {
  static constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  int accumulator = 0;
  int bits = 0;
  output->clear();
  for (char c : input) {
    if (c == '=') break;
    const char *found = std::find(std::begin(kAlphabet), std::end(kAlphabet) - 1, c);
    if (found == std::end(kAlphabet) - 1) return false;
    accumulator = (accumulator << 6) | static_cast<int>(found - kAlphabet);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      output->push_back(static_cast<char>((accumulator >> bits) & 0xff));
    }
  }
  return bits < 6;
}

bool ParseInt(std::string_view value, int32_t *result) {
  if (value.empty()) return false;
  size_t position = 0;
  bool negative = false;
  if (value.front() == '-') {
    negative = true;
    position = 1;
  }
  if (position == value.size()) return false;
  int64_t number = 0;
  for (; position < value.size(); ++position) {
    if (value[position] < '0' || value[position] > '9') return false;
    number = number * 10 + value[position] - '0';
    if (number > std::numeric_limits<int32_t>::max() + (negative ? 1LL : 0LL)) {
      return false;
    }
  }
  if (negative) number = -number;
  *result = static_cast<int32_t>(number);
  return true;
}

bool ParseUint32(const std::string_view &value, uint32_t *result) {
  int32_t parsed = 0;
  if (!ParseInt(value, &parsed) || parsed < 0) return false;
  *result = static_cast<uint32_t>(parsed);
  return true;
}

std::vector<std::string_view> SplitTabs(std::string_view line) {
  std::vector<std::string_view> fields;
  size_t begin = 0;
  while (true) {
    size_t end = line.find('\t', begin);
    if (end == std::string_view::npos) {
      fields.push_back(line.substr(begin));
      return fields;
    }
    fields.push_back(line.substr(begin, end - begin));
    begin = end + 1;
  }
}

uint32_t DecodeUtf8(std::string_view text, size_t offset, size_t *length) {
  const auto byte = [&text](size_t index) { return static_cast<uint8_t>(text[index]); };
  if (offset >= text.size()) {
    *length = 0;
    return 0xfffd;
  }
  const uint8_t first = byte(offset);
  if (first < 0x80) {
    *length = 1;
    return first;
  }
  const int count = (first & 0xe0) == 0xc0 ? 2 : (first & 0xf0) == 0xe0 ? 3 : (first & 0xf8) == 0xf0 ? 4 : 0;
  if (count == 0 || offset + static_cast<size_t>(count) > text.size()) {
    *length = 1;
    return 0xfffd;
  }
  uint32_t codepoint = first & ((1u << (8 - count - 1)) - 1u);
  for (int i = 1; i < count; ++i) {
    const uint8_t next = byte(offset + i);
    if ((next & 0xc0) != 0x80) {
      *length = 1;
      return 0xfffd;
    }
    codepoint = (codepoint << 6) | (next & 0x3f);
  }
  *length = static_cast<size_t>(count);
  return codepoint;
}

size_t ValidUtf8Length(std::string_view text, size_t offset) {
  if (offset >= text.size()) return 0;
  const uint8_t first = static_cast<uint8_t>(text[offset]);
  if (first < 0x80) return 1;
  const size_t length = (first & 0xe0) == 0xc0 ? 2 : (first & 0xf0) == 0xe0 ? 3 : (first & 0xf8) == 0xf0 ? 4 : 1;
  if (length == 1 || offset + length > text.size()) return 1;
  for (size_t i = 1; i < length; ++i) {
    if ((static_cast<uint8_t>(text[offset + i]) & 0xc0) != 0x80) return 1;
  }
  uint32_t codepoint = first & ((1u << (7 - static_cast<unsigned>(length))) - 1u);
  for (size_t i = 1; i < length; ++i) {
    codepoint = (codepoint << 6) | (static_cast<uint8_t>(text[offset + i]) & 0x3f);
  }
  static constexpr uint32_t kMinimumCodepoint[] = {0, 0, 0x80, 0x800, 0x10000};
  if (codepoint < kMinimumCodepoint[length] || (codepoint >= 0xd800 && codepoint <= 0xdfff) || codepoint > 0x10ffff) {
    return 1;
  }
  return length;
}

std::string ReplaceInvalidUtf8(std::string_view bytes) {
  std::string result;
  result.reserve(bytes.size());
  for (size_t offset = 0; offset < bytes.size();) {
    const size_t length = ValidUtf8Length(bytes, offset);
    if (length == 1 && static_cast<uint8_t>(bytes[offset]) >= 0x80) {
      result.append("\xef\xbf\xbd");
      ++offset;
    } else {
      result.append(bytes.data() + offset, length);
      offset += length;
    }
  }
  return result;
}

std::string EncodeUtf8(uint32_t codepoint) {
  std::string result;
  if (codepoint <= 0x7f) {
    result.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7ff) {
    result.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
    result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else if (codepoint <= 0xffff) {
    result.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
    result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else {
    result.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
    result.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
    result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  }
  return result;
}

bool IsCrLf(uint32_t codepoint) { return codepoint == '\r' || codepoint == '\n'; }

bool IsAsciiContraction(std::string_view text, size_t offset, size_t *length) {
  if (offset >= text.size() || text[offset] != '\'') return false;
  static constexpr std::string_view kSuffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
  for (std::string_view suffix : kSuffixes) {
    const std::string_view candidate = text.substr(offset + 1, suffix.size());
    if (candidate.size() != suffix.size()) continue;
    bool equal = true;
    for (size_t i = 0; i < suffix.size(); ++i) {
      if (std::tolower(static_cast<unsigned char>(candidate[i])) != suffix[i]) {
        equal = false;
        break;
      }
    }
    if (equal) {
      *length = suffix.size() + 1;
      return true;
    }
  }
  return false;
}

bool InRanges(const std::vector<CodepointRange> &ranges, uint32_t codepoint) {
  auto it = std::upper_bound(ranges.begin(), ranges.end(), codepoint,
                             [](uint32_t value, const CodepointRange &range) { return value < range.begin; });
  if (it == ranges.begin()) return false;
  --it;
  return codepoint <= it->end;
}

}  // namespace

void QwenTokenizer::Reset() {
  id_to_token_.clear();
  token_to_id_.clear();
  special_token_to_id_.clear();
  merge_rank_.clear();
  letter_ranges_.clear();
  mark_ranges_.clear();
  number_ranges_.clear();
  whitespace_ranges_.clear();
  nfd_.clear();
  combining_class_.clear();
  compositions_.clear();
  binary_data_.clear();
  sections_.fill({});
  byte_token_ids_.fill(-1);
  added_tokens_.clear();
  token_views_.clear();
  merge_hash_.clear();
  binary_mode_ = false;
  add_prefix_space_ = false;
  normalize_nfc_ = false;
  include_marks_ = false;
  compact_token_index_ = false;
  number_chunk_size_ = 3;
  vocab_size_ = 0;
  unk_id_ = bos_id_ = eos_id_ = -1;
  instance_id_ = 0;
}

bool QwenTokenizer::Load(const std::string &path, std::string *error) {
  Reset();
  if (IsArtifactPackagePath(path)) {
    const bool ok = LoadBinary(path, error);
    if (ok) instance_id_ = g_next_instance_id.fetch_add(1, std::memory_order_relaxed);
    return ok;
  }
  std::ifstream probe(path, std::ios::binary);
  if (!probe) {
    if (error) *error = "cannot open tokenizer file: " + path;
    return false;
  }
  std::array<char, 8> magic{};
  probe.read(magic.data(), magic.size());
  const bool binary = probe.gcount() == static_cast<std::streamsize>(magic.size()) &&
                      std::memcmp(magic.data(), kBinaryMagic, magic.size()) == 0;
  probe.close();
  const bool ok = binary ? LoadBinary(path, error) : LoadText(path, error);
  if (ok) instance_id_ = g_next_instance_id.fetch_add(1, std::memory_order_relaxed);
  return ok;
}

const QwenTokenizer::SectionView *QwenTokenizer::FindSection(uint32_t section_type) const {
  if (section_type >= sections_.size() || sections_[section_type].stride == 0) {
    return nullptr;
  }
  return &sections_[section_type];
}

bool QwenTokenizer::LoadBinary(const std::string &path, std::string *error) {
  if (!ReadArtifactBytes(path, &binary_data_, error)) return false;
  uint32_t header_size = 0, flags = 0, section_count = 0;
  if (!LoadBinaryHeader(header_size, flags, section_count, error)) return false;
  if (!LoadSectionDirectory(header_size, section_count, error)) return false;
  LoadRanges(kSectionLetterRanges, &letter_ranges_);
  LoadRanges(kSectionMarkRanges, &mark_ranges_);
  LoadRanges(kSectionNumberRanges, &number_ranges_);
  LoadRanges(kSectionWhitespaceRanges, &whitespace_ranges_);

  const SectionView *bytes = FindSection(kSectionByteTokenIds);
  const SectionView *merges = FindSection(kSectionMerges);
  const SectionView *added_index = FindSection(kSectionAddedTokenIndex);
  const SectionView *added_bytes = FindSection(kSectionAddedTokenBytes);
  if (!bytes || bytes->count != 256 || !merges || !added_index || !added_bytes) {
    if (error) *error = "QWENBPE2 is missing required sections";
    return false;
  }
  for (size_t byte = 0; byte < 256; ++byte) {
    byte_token_ids_[byte] = ReadI32(binary_data_, bytes->offset + byte * 4);
    if (byte_token_ids_[byte] < 0 || byte_token_ids_[byte] >= vocab_size_) {
      if (error) *error = "invalid byte token id in QWENBPE2";
      return false;
    }
  }
  if (!LoadMergeTable(merges, error)) return false;
  if (!LoadAddedTokens(added_index, added_bytes, error)) return false;
  if (!LoadDecodeTable(error)) return false;
  const uint64_t merge_end =
    static_cast<uint64_t>(merges->offset) + static_cast<uint64_t>(merges->count) * merges->stride;
  if (merge_end == binary_data_.size()) {
    binary_data_.resize(merges->offset);
    binary_data_.shrink_to_fit();
    sections_[kSectionMerges] = {};
  }
  add_prefix_space_ = (flags & 1u) != 0;
  normalize_nfc_ = (flags & 2u) != 0;
  include_marks_ = (flags & 4u) != 0;
  binary_mode_ = true;
  return true;
}

bool QwenTokenizer::LoadText(const std::string &path, std::string *error) {
  std::ifstream file(path);
  if (!file) {
    if (error) *error = "cannot open tokenizer file: " + path;
    return false;
  }
  std::string line;
  size_t line_number = 0;
  bool first = true;
  while (std::getline(file, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    if (first) {
      first = false;
      if (line != kMagic) {
        if (error) *error = "invalid tokenizer magic at line 1";
        return false;
      }
      continue;
    }

    const auto fields = SplitTabs(line);
    if (fields.empty()) continue;
    if (!LoadTextRecord(fields, line_number, error)) return false;
  }
  if (first || id_to_token_.empty()) {
    if (error) *error = "tokenizer file has no vocabulary";
    return false;
  }
  vocab_size_ = std::max(vocab_size_, static_cast<int32_t>(id_to_token_.size()));
  return true;
}

QwenTokenizer::TokenView QwenTokenizer::GetTokenView(int32_t id) const {
  if (id < 0 || id >= vocab_size_) return {};
  if (!token_views_.empty()) return token_views_[id];
  const SectionView *offsets = FindSection(kSectionTokenOffsets);
  if (!compact_token_index_ || !offsets) return {};
  const uint32_t packed = ReadU32(binary_data_, offsets->offset + static_cast<size_t>(id) * 4);
  const uint32_t next = ReadU32(binary_data_, offsets->offset + static_cast<size_t>(id + 1) * 4);
  const uint32_t offset = packed & kPackedOffsetMask;
  return {offset, (next & kPackedOffsetMask) - offset, packed >> kPackedFlagShift};
}

bool QwenTokenizer::BinaryInRanges(uint32_t section_type, uint32_t codepoint) const {
  const SectionView *section = FindSection(section_type);
  if (!section) return false;
  uint32_t low = 0, high = section->count;
  while (low < high) {
    const uint32_t mid = low + (high - low) / 2;
    const size_t row = section->offset + static_cast<size_t>(mid) * section->stride;
    if (ReadU32(binary_data_, row) <= codepoint)
      low = mid + 1;
    else
      high = mid;
  }
  if (low == 0) return false;
  const size_t row = section->offset + static_cast<size_t>(low - 1) * section->stride;
  return codepoint <= ReadU32(binary_data_, row + 4);
}

bool QwenTokenizer::IsLetter(uint32_t codepoint) const {
  if (!letter_ranges_.empty()) return InRanges(letter_ranges_, codepoint);
  if (binary_mode_) return BinaryInRanges(kSectionLetterRanges, codepoint);
  return (codepoint >= 'A' && codepoint <= 'Z') || (codepoint >= 'a' && codepoint <= 'z') ||
         (codepoint >= 0x3400 && codepoint <= 0x9fff);
}

bool QwenTokenizer::IsMark(uint32_t codepoint) const {
  if (!mark_ranges_.empty()) return InRanges(mark_ranges_, codepoint);
  return binary_mode_ && BinaryInRanges(kSectionMarkRanges, codepoint);
}

bool QwenTokenizer::IsNumber(uint32_t codepoint) const {
  if (!number_ranges_.empty()) return InRanges(number_ranges_, codepoint);
  if (binary_mode_) return BinaryInRanges(kSectionNumberRanges, codepoint);
  return codepoint >= '0' && codepoint <= '9';
}

bool QwenTokenizer::IsWhitespace(uint32_t codepoint) const {
  if (!whitespace_ranges_.empty()) return InRanges(whitespace_ranges_, codepoint);
  if (binary_mode_) return BinaryInRanges(kSectionWhitespaceRanges, codepoint);
  return codepoint == ' ' || codepoint == '\t' || codepoint == '\n' || codepoint == '\r' || codepoint == '\v' ||
         codepoint == '\f';
}

uint8_t QwenTokenizer::CombiningClass(uint32_t codepoint) const {
  if (!binary_mode_) {
    auto it = combining_class_.find(codepoint);
    return it == combining_class_.end() ? 0 : it->second;
  }
  const SectionView *section = FindSection(kSectionCcc);
  if (!section) return 0;
  uint32_t low = 0, high = section->count;
  while (low < high) {
    const uint32_t mid = low + (high - low) / 2;
    const size_t row = section->offset + static_cast<size_t>(mid) * section->stride;
    const uint32_t value = ReadU32(binary_data_, row);
    if (value < codepoint)
      low = mid + 1;
    else
      high = mid;
  }
  if (low >= section->count) return 0;
  const size_t row = section->offset + static_cast<size_t>(low) * section->stride;
  return ReadU32(binary_data_, row) == codepoint ? static_cast<uint8_t>(ReadU32(binary_data_, row + 4)) : 0;
}

bool QwenTokenizer::AppendDecomposition(uint32_t codepoint, std::vector<uint32_t> *output) const {
  if (!binary_mode_) {
    auto it = nfd_.find(codepoint);
    if (it == nfd_.end()) return false;
    output->insert(output->end(), it->second.begin(), it->second.end());
    return true;
  }
  const SectionView *index = FindSection(kSectionNfdIndex);
  const SectionView *values = FindSection(kSectionNfdValues);
  if (!index || !values) return false;
  uint32_t low = 0, high = index->count;
  while (low < high) {
    const uint32_t mid = low + (high - low) / 2;
    const size_t row = index->offset + static_cast<size_t>(mid) * index->stride;
    if (ReadU32(binary_data_, row) < codepoint)
      low = mid + 1;
    else
      high = mid;
  }
  if (low >= index->count) return false;
  const size_t row = index->offset + static_cast<size_t>(low) * index->stride;
  if (ReadU32(binary_data_, row) != codepoint) return false;
  const uint32_t value_offset = ReadU32(binary_data_, row + 4);
  const uint32_t count = ReadU32(binary_data_, row + 8);
  if (static_cast<uint64_t>(value_offset) + count > values->count) return false;
  output->reserve(output->size() + count);
  for (uint32_t i = 0; i < count; ++i) {
    output->push_back(ReadU32(binary_data_, values->offset + static_cast<size_t>(value_offset + i) * 4));
  }
  return true;
}

bool QwenTokenizer::FindComposition(uint32_t first, uint32_t second, uint32_t *result) const {
  const uint64_t key = CompositionKey(first, second);
  if (!binary_mode_) {
    auto it = compositions_.find(key);
    if (it == compositions_.end()) return false;
    *result = it->second;
    return true;
  }
  const SectionView *section = FindSection(kSectionComposition);
  if (!section) return false;
  uint32_t low = 0, high = section->count;
  while (low < high) {
    const uint32_t mid = low + (high - low) / 2;
    const size_t row = section->offset + static_cast<size_t>(mid) * section->stride;
    if (ReadU64(binary_data_, row) < key)
      low = mid + 1;
    else
      high = mid;
  }
  if (low >= section->count) return false;
  const size_t row = section->offset + static_cast<size_t>(low) * section->stride;
  if (ReadU64(binary_data_, row) != key) return false;
  *result = ReadU32(binary_data_, row + 8);
  return true;
}

std::string QwenTokenizer::NormalizeNfc(const std::string &text) const {
  if (!normalize_nfc_) return text;

  std::vector<uint32_t> decomposed;
  for (size_t offset = 0; offset < text.size();) {
    size_t length = 0;
    uint32_t cp = DecodeUtf8(text, offset, &length);
    offset += length ? length : 1;
    if (cp >= kSBase && cp < kSBase + kSCount) {
      const uint32_t index = cp - kSBase;
      decomposed.push_back(kLBase + index / kNCount);
      decomposed.push_back(kVBase + (index % kNCount) / kTCount);
      if (index % kTCount != 0) {
        decomposed.push_back(kTBase + index % kTCount);
      }
    } else if (!AppendDecomposition(cp, &decomposed)) {
      decomposed.push_back(cp);
    }
  }

  OrderCombiningMarks(&decomposed);
  std::vector<uint32_t> composed;
  composed.reserve(decomposed.size());
  size_t starter_position = 0;
  uint32_t starter = 0;
  uint8_t last_class = 0;
  for (uint32_t cp : decomposed) {
    if (composed.empty()) {
      composed.push_back(cp);
      starter = cp;
      continue;
    }
    const uint8_t current_class = CombiningClass(cp);
    uint32_t replacement = 0;

    replacement = ComposeCodepoint(starter, cp);

    if (replacement != 0 && (last_class == 0 || last_class < current_class)) {
      composed[starter_position] = replacement;
      starter = replacement;
      continue;
    }

    if (current_class == 0) {
      starter_position = composed.size();
      starter = cp;
    }
    last_class = current_class;
    composed.push_back(cp);
  }

  const std::string result =
    std::accumulate(composed.begin(), composed.end(), std::string{}, [](std::string value, uint32_t cp) {
      value += EncodeUtf8(cp);
      return value;
    });
  return result;
}

std::vector<std::string> QwenTokenizer::PreTokenize(const std::string &input, bool add_prefix_space) const {
  std::string text = NormalizeNfc(input);
  if (add_prefix_space_ && add_prefix_space && !text.empty() && text.front() != ' ') {
    text.insert(text.begin(), ' ');
  }

  std::vector<std::string> pieces;
  size_t offset = 0;
  while (offset < text.size()) {
    size_t contraction_length = 0;
    if (IsAsciiContraction(text, offset, &contraction_length)) {
      pieces.emplace_back(text, offset, contraction_length);
      offset += contraction_length;
      continue;
    }

    size_t cp_length = 0;
    const uint32_t cp = DecodeUtf8(text, offset, &cp_length);
    if (cp_length == 0) break;

    size_t end = LetterEnd(text, offset, cp, cp_length);
    if (end != offset) {
      pieces.emplace_back(text, offset, end - offset);
      offset = end;
      continue;
    }

    // \p{N} or the Qwen3 variant \p{N}{1,3}.
    if (IsNumber(cp)) {
      end = offset + cp_length;
      for (int count = 1; count < number_chunk_size_ && end < text.size(); ++count) {
        size_t length = 0;
        const uint32_t next = DecodeUtf8(text, end, &length);
        if (length == 0 || !IsNumber(next)) break;
        end += length;
      }
      pieces.emplace_back(text, offset, end - offset);
      offset = end;
      continue;
    }

    end = PunctuationEnd(text, offset, cp, cp_length);
    if (end != offset) {
      pieces.emplace_back(text, offset, end - offset);
      offset = end;
      continue;
    }

    if (IsWhitespace(cp)) {
      const size_t whitespace_end = WhitespaceEnd(text, offset, cp_length);
      pieces.emplace_back(text, offset, whitespace_end - offset);
      offset = whitespace_end;
      continue;
    }

    // Defensive fallback for malformed/unassigned Unicode code points.
    pieces.emplace_back(text, offset, cp_length);
    offset += cp_length;
  }
  return pieces;
}

std::vector<std::string> QwenTokenizer::ByteEncode(const std::string &text) const {
  bool included[256] = {};
  for (int b = '!'; b <= '~'; ++b) included[b] = true;
  for (int b = 0xa1; b <= 0xac; ++b) included[b] = true;
  for (int b = 0xae; b <= 0xff; ++b) included[b] = true;
  uint32_t mapped[256];
  int next = 0;
  for (int b = 0; b < 256; ++b) {
    mapped[b] = included[b] ? static_cast<uint32_t>(b) : 256u + static_cast<uint32_t>(next++);
  }
  std::vector<std::string> result;
  result.reserve(text.size());
  std::transform(text.begin(), text.end(), std::back_inserter(result),
                 [&mapped](unsigned char byte) { return EncodeUtf8(mapped[byte]); });
  return result;
}

bool QwenTokenizer::FindMerge(int32_t left, int32_t right, MergeValue *value) const {
  if (!binary_mode_ || merge_hash_.empty()) return false;
  const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32) | static_cast<uint32_t>(right);
  uint64_t mixed = key;
  mixed ^= mixed >> 33;
  mixed *= 0xff51afd7ed558ccdULL;
  mixed ^= mixed >> 33;
  const size_t mask = merge_hash_.size() - 1;
  size_t slot = static_cast<size_t>(mixed) & mask;
  while (true) {
    const MergeHashEntry &entry = merge_hash_[slot];
    if (entry.key == key) {
      value->result_id = entry.result_id;
      value->rank = entry.rank;
      return true;
    }
    if (entry.key == UINT64_MAX) return false;
    slot = (slot + 1) & mask;
  }
}

std::vector<int32_t> QwenTokenizer::EncodeChunkUncached(const std::string &text) const {
  if (binary_mode_) return EncodeBinaryChunk(text);

  std::vector<std::string> symbols = ByteEncode(text);
  while (symbols.size() > 1) {
    int32_t best_rank = std::numeric_limits<int32_t>::max();
    size_t best_index = symbols.size();
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      auto it = merge_rank_.find(PairKey(symbols[i], symbols[i + 1]));
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_index = i;
      }
    }
    if (best_index == symbols.size()) break;
    std::vector<std::string> merged;
    merged.reserve(symbols.size());
    for (size_t i = 0; i < symbols.size();) {
      const bool merge = CanMergeText(symbols, i, best_rank);
      if (merge) {
        merged.push_back(symbols[i] + symbols[i + 1]);
        i += 2;
      } else {
        merged.push_back(std::move(symbols[i++]));
      }
    }
    symbols.swap(merged);
  }
  std::vector<int32_t> ids;
  ids.reserve(symbols.size());
  for (const std::string &symbol : symbols) {
    auto it = token_to_id_.find(symbol);
    ids.push_back(it == token_to_id_.end() ? unk_id_ : it->second);
  }
  return ids;
}

std::vector<int32_t> QwenTokenizer::EncodeChunk(const std::string &text) const {
  struct CacheEntry {
    uint64_t instance_id = 0;
    uint64_t hash = 0;
    std::string text;
    std::vector<int32_t> ids;
  };
  thread_local std::array<CacheEntry, kChunkCacheSize> cache;
  if (text.size() > kMaxCachedChunkBytes) return EncodeChunkUncached(text);
  const uint64_t hash = HashBytes(text);
  CacheEntry &entry = cache[hash % cache.size()];
  if (entry.instance_id == instance_id_ && entry.hash == hash && entry.text == text) {
    return entry.ids;
  }
  std::vector<int32_t> result = EncodeChunkUncached(text);
  entry.instance_id = instance_id_;
  entry.hash = hash;
  entry.text = text;
  entry.ids = result;
  return result;
}

bool QwenTokenizer::EncodeSpecialAt(const std::string &text, size_t offset, size_t *length, int32_t *id) const {
  bool found = false;
  size_t best = 0;
  if (binary_mode_) {
    const SectionView *bytes = FindSection(kSectionAddedTokenBytes);
    if (!bytes) return false;
    for (const AddedTokenView &item : added_tokens_) {
      if (item.length <= best || offset + item.length > text.size()) continue;
      const char *token = reinterpret_cast<const char *>(binary_data_.data() + bytes->offset + item.offset);
      if (std::memcmp(text.data() + offset, token, item.length) != 0) continue;
      found = true;
      best = item.length;
      *id = item.id;
    }
  } else {
    for (const auto &item : special_token_to_id_) {
      if (item.first.size() <= best || text.compare(offset, item.first.size(), item.first) != 0) continue;
      found = true;
      best = item.first.size();
      *id = item.second;
    }
  }
  if (found) *length = best;
  return found;
}

std::vector<int32_t> QwenTokenizer::Encode(std::string_view text_view, const EncodeOptions &options) const {
  const std::string text(text_view);
  std::vector<int32_t> result;
  if (options.add_bos && bos_id_ >= 0) result.push_back(bos_id_);

  size_t offset = 0;
  size_t ordinary_begin = 0;
  while (offset < text.size()) {
    size_t special_length = 0;
    int32_t special_id = -1;
    if (EncodeSpecialAt(text, offset, &special_length, &special_id)) {
      if (offset > ordinary_begin) {
        for (const auto &piece :
             PreTokenize(text.substr(ordinary_begin, offset - ordinary_begin), ordinary_begin == 0)) {
          auto ids = EncodeChunk(piece);
          result.insert(result.end(), ids.begin(), ids.end());
        }
      }
      result.push_back(special_id);
      offset += special_length;
      ordinary_begin = offset;
    } else {
      size_t length = 0;
      DecodeUtf8(text, offset, &length);
      offset += length ? length : 1;
    }
  }
  if (ordinary_begin < text.size()) {
    for (const auto &piece : PreTokenize(text.substr(ordinary_begin), ordinary_begin == 0)) {
      auto ids = EncodeChunk(piece);
      result.insert(result.end(), ids.begin(), ids.end());
    }
  }

  if (options.add_eos && eos_id_ >= 0) result.push_back(eos_id_);
  return result;
}

std::string QwenTokenizer::Decode(const std::vector<int32_t> &ids, const DecodeOptions &options) const {
  static const auto unicode_to_byte = UnicodeToByte();

  std::string result;
  for (int32_t id : ids) {
    if (id < 0 || id >= vocab_size_) continue;
    std::string_view token;
    uint32_t flags = 0;
    if (binary_mode_) {
      if (!SupportsDecode()) continue;
      const SectionView *bytes = FindSection(kSectionTokenBytes);
      if (!bytes) continue;
      const TokenView view = GetTokenView(id);
      flags = view.flags;
      token = {reinterpret_cast<const char *>(binary_data_.data() + bytes->offset + view.offset), view.length};
    } else if (id < static_cast<int32_t>(id_to_token_.size())) {
      token = id_to_token_[id];
      flags = special_token_to_id_.count(std::string(token)) ? kTokenSpecial : 0;
    }
    if (options.skip_special_tokens && (flags & kTokenSpecial)) continue;
    if (flags & kTokenSpecial) {
      result.append(token);
      continue;
    }
    for (size_t offset = 0; offset < token.size();) {
      size_t length = 0;
      const uint32_t cp = DecodeUtf8(token, offset, &length);
      if (length == 0) break;
      if (cp < unicode_to_byte.size() && unicode_to_byte[cp] >= 0) {
        result.push_back(static_cast<char>(unicode_to_byte[cp]));
      } else {
        result.append(token.data() + offset, length);
      }
      offset += length;
    }
  }
  return ReplaceInvalidUtf8(result);
}

bool QwenTokenizer::LoadSectionDirectory(uint32_t header_size, uint32_t section_count, std::string *error) {
  static constexpr uint32_t kExpectedStride[16] = {0, 4, 8, 8, 8, 8, 12, 4, 8, 12, 16, 16, 1, 12, 1, 4};
  for (uint32_t index = 0; index < section_count; ++index) {
    const size_t entry = kHeaderSize + static_cast<size_t>(index) * 16;
    const uint32_t type = ReadU32(binary_data_, entry);
    const uint32_t offset = ReadU32(binary_data_, entry + 4);
    const uint32_t count = ReadU32(binary_data_, entry + 8);
    const uint32_t stride = ReadU32(binary_data_, entry + 12);
    const uint64_t end = static_cast<uint64_t>(offset) + static_cast<uint64_t>(count) * stride;
    if (type == 0 || type >= sections_.size() || stride != kExpectedStride[type] || sections_[type].stride != 0 ||
        offset < header_size || end > binary_data_.size()) {
      if (error) *error = "malformed QWENBPE2 section directory";
      return false;
    }
    sections_[type] = {offset, count, stride};
  }
  return true;
}

bool QwenTokenizer::LoadMergeTable(const SectionView *merges, std::string *error) {
  uint64_t previous_merge_key = 0;
  for (uint32_t i = 0; i < merges->count; ++i) {
    const size_t row = merges->offset + static_cast<size_t>(i) * merges->stride;
    const uint64_t key = ReadU64(binary_data_, row);
    const int32_t result = ReadI32(binary_data_, row + 8);
    if ((i && key <= previous_merge_key) || result < 0 || result >= vocab_size_) {
      if (error) *error = "unsorted or invalid merge table in QWENBPE2";
      return false;
    }
    previous_merge_key = key;
  }
  size_t hash_capacity = 1;
  while (hash_capacity < static_cast<size_t>(merges->count) * 2) {
    hash_capacity <<= 1;
  }
  merge_hash_.assign(hash_capacity, {});
  const size_t hash_mask = hash_capacity - 1;
  for (uint32_t i = 0; i < merges->count; ++i) {
    const size_t row = merges->offset + static_cast<size_t>(i) * merges->stride;
    const uint64_t key = ReadU64(binary_data_, row);
    uint64_t mixed = key;
    mixed ^= mixed >> 33;
    mixed *= 0xff51afd7ed558ccdULL;
    mixed ^= mixed >> 33;
    size_t slot = static_cast<size_t>(mixed) & hash_mask;
    while (merge_hash_[slot].key != UINT64_MAX) slot = (slot + 1) & hash_mask;
    merge_hash_[slot] = {key, ReadI32(binary_data_, row + 8), ReadI32(binary_data_, row + 12)};
  }

  return true;
}

bool QwenTokenizer::LoadAddedTokens(const SectionView *added_index, const SectionView *added_bytes,
                                    std::string *error) {
  added_tokens_.reserve(added_index->count);
  for (uint32_t i = 0; i < added_index->count; ++i) {
    const size_t row = added_index->offset + static_cast<size_t>(i) * added_index->stride;
    AddedTokenView token{ReadI32(binary_data_, row), ReadU32(binary_data_, row + 4), ReadU32(binary_data_, row + 8),
                         ReadU32(binary_data_, row + 12)};
    if (token.id < 0 || token.id >= vocab_size_ ||
        static_cast<uint64_t>(token.offset) + token.length > added_bytes->count) {
      if (error) *error = "invalid added token table in QWENBPE2";
      return false;
    }
    added_tokens_.push_back(token);
  }
  return true;
}

bool QwenTokenizer::LoadDecodeTable(std::string *error) {
  const SectionView *token_index = FindSection(kSectionTokenIndex);
  const SectionView *token_bytes = FindSection(kSectionTokenBytes);
  const SectionView *token_offsets = FindSection(kSectionTokenOffsets);
  if ((token_index && token_offsets) || ((token_index || token_offsets) != (token_bytes != nullptr))) {
    if (error) *error = "incomplete QWENBPE2 decode table";
    return false;
  }
  if (token_index) {
    if (!LoadTokenViews(token_index, token_bytes, error)) return false;
  } else if (token_offsets) {
    if (token_offsets->count != static_cast<uint32_t>(vocab_size_) + 1) {
      if (error) *error = "invalid QWENBPE2 compact token index size";
      return false;
    }
    uint32_t previous = 0;
    for (int32_t id = 0; id <= vocab_size_; ++id) {
      const uint32_t packed = ReadU32(binary_data_, token_offsets->offset + static_cast<size_t>(id) * 4);
      const uint32_t offset = packed & kPackedOffsetMask;
      if (offset < previous || offset > token_bytes->count) {
        if (error) *error = "invalid QWENBPE2 compact token offsets";
        return false;
      }
      previous = offset;
    }
    compact_token_index_ = true;
  }

  return true;
}

bool QwenTokenizer::LoadTokenViews(const SectionView *token_index, const SectionView *token_bytes, std::string *error) {
  if (token_index->count != static_cast<uint32_t>(vocab_size_)) {
    if (error) *error = "invalid QWENBPE2 token index size";
    return false;
  }
  token_views_.resize(vocab_size_);
  for (int32_t id = 0; id < vocab_size_; ++id) {
    const size_t row = token_index->offset + static_cast<size_t>(id) * 12;
    TokenView view{ReadU32(binary_data_, row), ReadU32(binary_data_, row + 4), ReadU32(binary_data_, row + 8)};
    if (static_cast<uint64_t>(view.offset) + view.length > token_bytes->count) {
      if (error) *error = "invalid QWENBPE2 token bytes";
      return false;
    }
    token_views_[id] = view;
  }
  return true;
}

bool QwenTokenizer::LoadTextMeta(const std::vector<std::string_view> &fields) {
  if (fields.size() != 3) return false;
  std::string key;
  std::string value;
  if (!Base64Decode(fields[1], &key) || !Base64Decode(fields[2], &value)) return false;
  int32_t parsed = -1;
  const bool numeric = ParseInt(value, &parsed);
  if (key == "add_prefix_space") {
    add_prefix_space_ = value == "true" || value == "1";
  } else if (key == "vocab_size" && numeric) {
    vocab_size_ = parsed;
  } else if (key == "unk_id" && numeric) {
    unk_id_ = parsed;
  } else if (key == "bos_id" && numeric) {
    bos_id_ = parsed;
  } else if (key == "eos_id" && numeric) {
    eos_id_ = parsed;
  } else if (key == "normalizer") {
    normalize_nfc_ = value == "NFC";
  } else if (key == "split_regex") {
    include_marks_ = value.find("\\p{M}") != std::string::npos;
    number_chunk_size_ = value.find("\\p{N}{1,3}") != std::string::npos ? 3 : 1;
  }
  return true;
}

bool QwenTokenizer::LoadTextRange(const std::vector<std::string_view> &fields) {
  if (fields.size() != 4) return false;
  uint32_t begin = 0, end = 0;
  if (!ParseUint32(fields[2], &begin) || !ParseUint32(fields[3], &end) || begin > end) return false;
  CodepointRange range{begin, end};
  if (fields[1] == "L")
    letter_ranges_.push_back(range);
  else if (fields[1] == "M")
    mark_ranges_.push_back(range);
  else if (fields[1] == "N")
    number_ranges_.push_back(range);
  else if (fields[1] == "S")
    whitespace_ranges_.push_back(range);
  else
    return false;
  return true;
}

bool QwenTokenizer::LoadTextNfd(const std::vector<std::string_view> &fields) {
  if (fields.size() < 3) return false;
  uint32_t codepoint = 0;
  if (!ParseUint32(fields[1], &codepoint)) return false;
  std::vector<uint32_t> values;
  for (size_t i = 2; i < fields.size(); ++i) {
    uint32_t value = 0;
    if (!ParseUint32(fields[i], &value)) return false;
    values.push_back(value);
  }
  nfd_[codepoint] = std::move(values);
  return true;
}

bool QwenTokenizer::LoadTextCcc(const std::vector<std::string_view> &fields) {
  if (fields.size() != 3) return false;
  uint32_t codepoint = 0;
  int32_t value = 0;
  if (!ParseUint32(fields[1], &codepoint) || !ParseInt(fields[2], &value) || value < 0 || value > 255) return false;
  combining_class_[codepoint] = static_cast<uint8_t>(value);
  return true;
}

bool QwenTokenizer::LoadTextCompose(const std::vector<std::string_view> &fields) {
  if (fields.size() != 4) return false;
  uint32_t first_cp = 0, second_cp = 0, result = 0;
  if (!ParseUint32(fields[1], &first_cp) || !ParseUint32(fields[2], &second_cp) || !ParseUint32(fields[3], &result))
    return false;
  compositions_[CompositionKey(first_cp, second_cp)] = result;
  return true;
}

bool QwenTokenizer::LoadTextToken(const std::vector<std::string_view> &fields) {
  if (fields.size() != 3) return false;
  int32_t id = -1;
  std::string token;
  if (!ParseInt(fields[1], &id) || !Base64Decode(fields[2], &token) || id < 0) return false;
  if (static_cast<size_t>(id) >= id_to_token_.size()) id_to_token_.resize(static_cast<size_t>(id) + 1);
  id_to_token_[id] = token;
  token_to_id_[token] = id;
  if (fields[0] == "SPECIAL") special_token_to_id_[token] = id;
  return true;
}

bool QwenTokenizer::LoadTextMerge(const std::vector<std::string_view> &fields) {
  if (fields.size() != 4) return false;
  std::string left, right;
  int32_t rank = -1;
  if (!Base64Decode(fields[1], &left) || !Base64Decode(fields[2], &right) || !ParseInt(fields[3], &rank) || rank < 0)
    return false;
  merge_rank_[PairKey(left, right)] = rank;
  return true;
}

uint32_t QwenTokenizer::ComposeCodepoint(uint32_t starter, uint32_t codepoint) const {
  uint32_t replacement = 0;
  const uint32_t l_index = starter - kLBase;
  const uint32_t v_index = codepoint - kVBase;
  if (l_index < kLCount && v_index < kVCount) {
    replacement = kSBase + (l_index * kVCount + v_index) * kTCount;
  } else {
    const uint32_t s_index = starter - kSBase;
    const uint32_t t_index = codepoint - kTBase;
    if (s_index < kSCount && s_index % kTCount == 0 && t_index > 0 && t_index < kTCount) {
      replacement = starter + t_index;
    } else {
      FindComposition(starter, codepoint, &replacement);
    }
  }

  return replacement;
}

size_t QwenTokenizer::LetterEnd(const std::string &text, size_t offset, uint32_t cp, size_t cp_length) const {
  const auto is_letter_mark = [this](uint32_t cp) { return IsLetter(cp) || (include_marks_ && IsMark(cp)); };
  size_t end = offset;
  if (is_letter_mark(cp)) {
    end += cp_length;
  } else if (!IsCrLf(cp) && !IsLetter(cp) && !IsNumber(cp)) {
    size_t next_length = 0;
    const uint32_t next = DecodeUtf8(text, offset + cp_length, &next_length);
    if (next_length != 0 && is_letter_mark(next)) {
      end = offset + cp_length + next_length;
    }
  }
  if (end != offset) {
    while (end < text.size()) {
      size_t length = 0;
      const uint32_t next = DecodeUtf8(text, end, &length);
      if (length == 0 || !is_letter_mark(next)) break;
      end += length;
    }
  }
  return end;
}

size_t QwenTokenizer::PunctuationEnd(const std::string &text, size_t offset, uint32_t cp, size_t cp_length) const {
  const auto is_letter_mark = [this](uint32_t cp) { return IsLetter(cp) || (include_marks_ && IsMark(cp)); };
  const auto is_punctuation = [this, &is_letter_mark](uint32_t cp) {
    return !IsWhitespace(cp) && !is_letter_mark(cp) && !IsNumber(cp);
  };

  //  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
  size_t end = offset;
  if (cp == ' ') {
    size_t next_length = 0;
    const uint32_t next = DecodeUtf8(text, offset + cp_length, &next_length);
    if (next_length != 0 && is_punctuation(next)) {
      end = offset + cp_length + next_length;
    }
  } else if (is_punctuation(cp)) {
    end = offset + cp_length;
  }
  if (end != offset) {
    while (end < text.size()) {
      size_t length = 0;
      const uint32_t next = DecodeUtf8(text, end, &length);
      if (length == 0 || !is_punctuation(next)) break;
      end += length;
    }
    while (end < text.size()) {
      size_t length = 0;
      const uint32_t next = DecodeUtf8(text, end, &length);
      if (length == 0 || !IsCrLf(next)) break;
      end += length;
    }
  }
  return end;
}

std::vector<int32_t> QwenTokenizer::EncodeBinaryChunk(const std::string &text) const {
  std::vector<int32_t> symbols;
  symbols.reserve(text.size());
  std::transform(text.begin(), text.end(), std::back_inserter(symbols),
                 [this](unsigned char byte) { return byte_token_ids_[byte]; });
  std::vector<int32_t> merged;
  merged.reserve(symbols.size());
  while (symbols.size() > 1) {
    int32_t best_rank = std::numeric_limits<int32_t>::max();
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      MergeValue value;
      if (FindMerge(symbols[i], symbols[i + 1], &value) && value.rank < best_rank) {
        best_rank = value.rank;
      }
    }
    if (best_rank == std::numeric_limits<int32_t>::max()) break;
    merged.clear();
    for (size_t i = 0; i < symbols.size();) {
      MergeValue value;
      if (i + 1 < symbols.size() && FindMerge(symbols[i], symbols[i + 1], &value) && value.rank == best_rank) {
        merged.push_back(value.result_id);
        i += 2;
      } else {
        merged.push_back(symbols[i++]);
      }
    }
    symbols.swap(merged);
  }
  return symbols;
}

std::array<int16_t, 0x144> QwenTokenizer::UnicodeToByte() {
  std::array<int16_t, 0x144> table{};
  table.fill(-1);
  std::vector<int> values;
  for (int i = '!'; i <= '~'; ++i) values.push_back(i);
  for (int i = 0xa1; i <= 0xac; ++i) values.push_back(i);
  for (int i = 0xae; i <= 0xff; ++i) values.push_back(i);
  std::vector<int> chars = values;
  int missing = 0;
  for (int byte = 0; byte < 256; ++byte) {
    if (std::find(values.begin(), values.end(), byte) == values.end()) {
      values.push_back(byte);
      chars.push_back(256 + missing++);
    }
  }
  for (size_t i = 0; i < values.size(); ++i) table[chars[i]] = values[i];
  return table;
}

bool QwenTokenizer::LoadBinaryHeader(uint32_t &header_size, uint32_t &flags, uint32_t &section_count,
                                     std::string *error) {
  if (binary_data_.size() < kHeaderSize || std::memcmp(binary_data_.data(), kBinaryMagic, 8) != 0) {
    if (error) *error = "invalid QWENBPE2 header";
    return false;
  }
  const uint32_t version = ReadU32(binary_data_, 8);
  header_size = ReadU32(binary_data_, 12);
  flags = ReadU32(binary_data_, 16);
  vocab_size_ = ReadI32(binary_data_, 20);
  unk_id_ = ReadI32(binary_data_, 24);
  bos_id_ = ReadI32(binary_data_, 28);
  eos_id_ = ReadI32(binary_data_, 32);
  number_chunk_size_ = static_cast<int>(ReadU32(binary_data_, 36));
  section_count = ReadU32(binary_data_, 40);
  if (version != kBinaryVersion || header_size < kHeaderSize || header_size > binary_data_.size() ||
      section_count > 64 || kHeaderSize + static_cast<uint64_t>(section_count) * 16 > header_size || vocab_size_ <= 0 ||
      (number_chunk_size_ != 1 && number_chunk_size_ != 3)) {
    if (error) *error = "unsupported or malformed QWENBPE2 header";
    return false;
  }

  return true;
}

bool QwenTokenizer::LoadTextRecord(const std::vector<std::string_view> &fields, size_t line_number,
                                   std::string *error) {
  using Loader = bool (QwenTokenizer::*)(const std::vector<std::string_view> &);
  static const std::unordered_map<std::string_view, Loader> loaders = {
    {"META", &QwenTokenizer::LoadTextMeta},       {"RANGE", &QwenTokenizer::LoadTextRange},
    {"NFD", &QwenTokenizer::LoadTextNfd},         {"CCC", &QwenTokenizer::LoadTextCcc},
    {"COMPOSE", &QwenTokenizer::LoadTextCompose}, {"TOKEN", &QwenTokenizer::LoadTextToken},
    {"SPECIAL", &QwenTokenizer::LoadTextToken},   {"MERGE", &QwenTokenizer::LoadTextMerge}};
  const auto loader = loaders.find(fields[0]);
  if (loader == loaders.end()) {
    if (error) *error = "unknown record at line " + std::to_string(line_number);
    return false;
  }
  if (!(this->*loader->second)(fields)) {
    if (error) *error = "malformed tokenizer record at line " + std::to_string(line_number);
    return false;
  }
  return true;
}

void QwenTokenizer::OrderCombiningMarks(std::vector<uint32_t> *decomposed) const {
  // Canonical ordering by combining class, preserving equal-class order.
  for (size_t i = 1; i < decomposed->size(); ++i) {
    const uint8_t current_class = CombiningClass((*decomposed)[i]);
    if (current_class == 0) continue;
    size_t j = i;
    while (j > 0) {
      const uint8_t previous_class = CombiningClass((*decomposed)[j - 1]);
      if (previous_class == 0 || previous_class <= current_class) break;
      std::swap((*decomposed)[j], (*decomposed)[j - 1]);
      --j;
    }
  }
}

size_t QwenTokenizer::WhitespaceEnd(const std::string &text, size_t offset, size_t cp_length) const {
  // \s*[\r\n]+. Match through the final CR/LF in the current
  // whitespace run and leave following spaces for the next alternative.
  size_t scan = offset;
  size_t last_newline_end = offset;
  std::vector<size_t> boundaries;
  while (scan < text.size()) {
    size_t length = 0;
    const uint32_t next = DecodeUtf8(text, scan, &length);
    if (length == 0 || !IsWhitespace(next)) break;
    scan += length;
    boundaries.push_back(scan);
    if (IsCrLf(next)) last_newline_end = scan;
  }
  if (last_newline_end != offset) {
    return last_newline_end;
  }

  // \s+(?!\S) backtracks by one whitespace before non-whitespace. This
  // lets the optional-prefix word alternative consume the final space.
  if (scan == text.size()) {
    return scan;
  }
  if (boundaries.size() >= 2) {
    const size_t before_last = boundaries[boundaries.size() - 2];
    return before_last;
  }

  // Final \s+ fallback.
  return offset + cp_length;
}

void QwenTokenizer::LoadRanges(uint32_t type, std::vector<CodepointRange> *output) {
  const SectionView *section = FindSection(type);
  if (!section) return;
  output->reserve(section->count);
  for (uint32_t i = 0; i < section->count; ++i) {
    const size_t row = section->offset + static_cast<size_t>(i) * section->stride;
    output->push_back({ReadU32(binary_data_, row), ReadU32(binary_data_, row + 4)});
  }
}

bool QwenTokenizer::CanMergeText(const std::vector<std::string> &symbols, size_t index, int32_t rank) const {
  if (index + 1 >= symbols.size()) return false;
  const auto pair = merge_rank_.find(PairKey(symbols[index], symbols[index + 1]));
  return pair != merge_rank_.end() && pair->second == rank;
}

}  // namespace mslite_llm::qwen35_tokenizer
