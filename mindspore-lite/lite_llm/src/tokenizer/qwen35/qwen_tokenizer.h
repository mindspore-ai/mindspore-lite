// Copyright 2026 Huawei Technologies Co., Ltd
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "tokenizer/qwen35/tokenizer.h"

namespace mslite_llm::qwen35_tokenizer {

struct CodepointRange {
  uint32_t begin;
  uint32_t end;
};

// Dependency-free Qwen byte-level BPE tokenizer. Load() auto-detects the
// compact QWENBPE2 binary format and the legacy QWEN_BPE_TOKENIZER_V1 text
// format. The binary path keeps tables in one read-only buffer and performs
// BPE entirely on token ids, avoiding vocabulary strings and merge hashing.
class QwenTokenizer final : public Tokenizer {
 public:
  QwenTokenizer() = default;
  QwenTokenizer(const QwenTokenizer &) = delete;
  QwenTokenizer &operator=(const QwenTokenizer &) = delete;
  QwenTokenizer(QwenTokenizer &&) = delete;
  QwenTokenizer &operator=(QwenTokenizer &&) = delete;

  bool Load(const std::string &path, std::string *error = nullptr);

  std::vector<int32_t> Encode(std::string_view text, const EncodeOptions &options = {}) const override;
  std::string Decode(const std::vector<int32_t> &ids, const DecodeOptions &options = {}) const override;

  TokenizerFamily Family() const override { return TokenizerFamily::kQwenByteLevelBpe; }
  const char *FamilyName() const override { return "qwen_bytelevel_bpe"; }
  int32_t VocabSize() const override { return vocab_size_; }
  int32_t UnknownId() const override { return unk_id_; }
  int32_t BosId() const override { return bos_id_; }
  int32_t EosId() const override { return eos_id_; }
  bool SupportsDecode() const override { return !binary_mode_ || !token_views_.empty() || compact_token_index_; }
  bool IsBinary() const { return binary_mode_; }

 private:
  struct SectionView {
    uint32_t offset = 0;
    uint32_t count = 0;
    uint32_t stride = 0;
  };
  struct AddedTokenView {
    int32_t id = -1;
    uint32_t offset = 0;
    uint32_t length = 0;
    uint32_t flags = 0;
  };
  struct TokenView {
    uint32_t offset = 0;
    uint32_t length = 0;
    uint32_t flags = 0;
  };
  struct MergeValue {
    int32_t result_id = -1;
    int32_t rank = -1;
  };
  struct MergeHashEntry {
    uint64_t key = UINT64_MAX;
    int32_t result_id = -1;
    int32_t rank = -1;
  };

  bool LoadText(const std::string &path, std::string *error);
  bool LoadBinary(const std::string &path, std::string *error);
  bool LoadSectionDirectory(uint32_t header_size, uint32_t section_count, std::string *error);
  bool LoadMergeTable(const SectionView *merges, std::string *error);
  bool LoadAddedTokens(const SectionView *added_index, const SectionView *added_bytes, std::string *error);
  bool LoadDecodeTable(std::string *error);
  bool LoadTokenViews(const SectionView *token_index, const SectionView *token_bytes, std::string *error);
  bool LoadTextMeta(const std::vector<std::string_view> &fields);
  bool LoadTextRange(const std::vector<std::string_view> &fields);
  bool LoadTextNfd(const std::vector<std::string_view> &fields);
  bool LoadTextCcc(const std::vector<std::string_view> &fields);
  bool LoadTextCompose(const std::vector<std::string_view> &fields);
  bool LoadTextToken(const std::vector<std::string_view> &fields);
  bool LoadTextMerge(const std::vector<std::string_view> &fields);
  uint32_t ComposeCodepoint(uint32_t starter, uint32_t codepoint) const;
  size_t LetterEnd(const std::string &text, size_t offset, uint32_t cp, size_t cp_length) const;
  size_t PunctuationEnd(const std::string &text, size_t offset, uint32_t cp, size_t cp_length) const;
  std::vector<int32_t> EncodeBinaryChunk(const std::string &text) const;
  static std::array<int16_t, 0x144> UnicodeToByte();
  bool LoadBinaryHeader(uint32_t &header_size, uint32_t &flags, uint32_t &section_count, std::string *error);
  bool LoadTextRecord(const std::vector<std::string_view> &fields, size_t line_number, std::string *error);
  void OrderCombiningMarks(std::vector<uint32_t> *decomposed) const;
  size_t WhitespaceEnd(const std::string &text, size_t offset, size_t cp_length) const;
  void LoadRanges(uint32_t type, std::vector<CodepointRange> *output);
  bool CanMergeText(const std::vector<std::string> &symbols, size_t index, int32_t rank) const;
  void Reset();

  std::vector<std::string> PreTokenize(const std::string &text, bool add_prefix_space) const;
  std::vector<std::string> ByteEncode(const std::string &text) const;
  std::vector<int32_t> EncodeChunk(const std::string &text) const;
  std::vector<int32_t> EncodeChunkUncached(const std::string &text) const;
  bool EncodeSpecialAt(const std::string &text, size_t offset, size_t *length, int32_t *id) const;

  bool IsLetter(uint32_t codepoint) const;
  bool IsMark(uint32_t codepoint) const;
  bool IsNumber(uint32_t codepoint) const;
  bool IsWhitespace(uint32_t codepoint) const;
  uint8_t CombiningClass(uint32_t codepoint) const;
  bool FindComposition(uint32_t first, uint32_t second, uint32_t *result) const;
  bool AppendDecomposition(uint32_t codepoint, std::vector<uint32_t> *output) const;
  bool FindMerge(int32_t left, int32_t right, MergeValue *value) const;
  bool BinaryInRanges(uint32_t section_type, uint32_t codepoint) const;
  const SectionView *FindSection(uint32_t section_type) const;
  TokenView GetTokenView(int32_t id) const;
  std::string NormalizeNfc(const std::string &text) const;

  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int32_t> token_to_id_;
  std::unordered_map<std::string, int32_t> special_token_to_id_;
  std::unordered_map<std::string, int32_t> merge_rank_;

  std::vector<CodepointRange> letter_ranges_;
  std::vector<CodepointRange> mark_ranges_;
  std::vector<CodepointRange> number_ranges_;
  std::vector<CodepointRange> whitespace_ranges_;
  std::unordered_map<uint32_t, std::vector<uint32_t>> nfd_;
  std::unordered_map<uint32_t, uint8_t> combining_class_;
  std::unordered_map<uint64_t, uint32_t> compositions_;

  std::vector<uint8_t> binary_data_;
  std::array<SectionView, 16> sections_{};
  std::array<int32_t, 256> byte_token_ids_{};
  std::vector<AddedTokenView> added_tokens_;
  std::vector<TokenView> token_views_;
  std::vector<MergeHashEntry> merge_hash_;

  bool binary_mode_ = false;
  bool add_prefix_space_ = false;
  bool normalize_nfc_ = false;
  bool include_marks_ = false;
  bool compact_token_index_ = false;
  int number_chunk_size_ = 3;
  int32_t vocab_size_ = 0;
  int32_t unk_id_ = -1;
  int32_t bos_id_ = -1;
  int32_t eos_id_ = -1;
  uint64_t instance_id_ = 0;
};

}  // namespace mslite_llm::qwen35_tokenizer
