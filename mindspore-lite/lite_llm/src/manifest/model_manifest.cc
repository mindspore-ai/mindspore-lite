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
#include "manifest/model_manifest.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_set>
#include <utility>

#include "manifest/msl_format.h"
#include "manifest/msl_package_reader.h"

namespace mslite_llm {

namespace {

std::string Lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string Trim(std::string value) {
  auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(),
                                          [&](char c) { return !is_space(static_cast<unsigned char>(c)); }));
  value.erase(
    std::find_if(value.rbegin(), value.rend(), [&](char c) { return !is_space(static_cast<unsigned char>(c)); }).base(),
    value.end());
  return value;
}

struct JsonValue {
  enum class Type {
    kNull,
    kBool,
    kNumber,
    kString,
    kArray,
    kObject,
  };

  Type type = Type::kNull;
  bool bool_value = false;
  double number_value = 0.0;
  std::string string_value;
  std::vector<JsonValue> array_value;
  std::map<std::string, JsonValue> object_value;

  bool IsObject() const { return type == Type::kObject; }
  bool IsArray() const { return type == Type::kArray; }
  bool IsString() const { return type == Type::kString; }
  bool IsNumber() const { return type == Type::kNumber; }
  bool IsBool() const { return type == Type::kBool; }
};

class JsonParser {
 public:
  explicit JsonParser(std::string text) : text_(std::move(text)) {}

  bool Parse(JsonValue *out, std::string *error) {
    SkipWhitespace();
    if (!ParseValue(out, error)) {
      return false;
    }
    SkipWhitespace();
    if (pos_ != text_.size()) {
      SetError(error, "unexpected trailing content");
      return false;
    }
    return true;
  }

 private:
  void SkipWhitespace() {
    while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_])) != 0) {
      ++pos_;
    }
  }

  bool Consume(char expected) {
    SkipWhitespace();
    if (pos_ >= text_.size() || text_[pos_] != expected) {
      return false;
    }
    ++pos_;
    return true;
  }

  bool MatchLiteral(const char *literal) {
    SkipWhitespace();
    const std::string word(literal);
    if (text_.compare(pos_, word.size(), word) != 0) {
      return false;
    }
    pos_ += word.size();
    return true;
  }

  void SetError(std::string *error, const std::string &message) const {
    if (error != nullptr) {
      *error = message + " at byte " + std::to_string(pos_);
    }
  }

  bool ParseValue(JsonValue *out, std::string *error) {
    SkipWhitespace();
    if (pos_ >= text_.size()) {
      SetError(error, "unexpected end of JSON");
      return false;
    }

    const char ch = text_[pos_];
    if (ch == '{') {
      return ParseObject(out, error);
    }
    if (ch == '[') {
      return ParseArray(out, error);
    }
    if (ch == '"') {
      out->type = JsonValue::Type::kString;
      return ParseString(&out->string_value, error);
    }
    if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch)) != 0) {
      return ParseNumber(out, error);
    }
    if (MatchLiteral("true")) {
      out->type = JsonValue::Type::kBool;
      out->bool_value = true;
      return true;
    }
    if (MatchLiteral("false")) {
      out->type = JsonValue::Type::kBool;
      out->bool_value = false;
      return true;
    }
    if (MatchLiteral("null")) {
      out->type = JsonValue::Type::kNull;
      return true;
    }
    SetError(error, "unexpected JSON token");
    return false;
  }

  bool ParseObject(JsonValue *out, std::string *error) {
    if (!Consume('{')) {
      SetError(error, "expected object");
      return false;
    }
    out->type = JsonValue::Type::kObject;
    out->object_value.clear();

    SkipWhitespace();
    if (Consume('}')) {
      return true;
    }

    while (pos_ < text_.size()) {
      std::string key;
      if (!ParseString(&key, error)) {
        return false;
      }
      if (!Consume(':')) {
        SetError(error, "expected ':' after object key");
        return false;
      }
      JsonValue value;
      if (!ParseValue(&value, error)) {
        return false;
      }
      out->object_value[key] = std::move(value);

      SkipWhitespace();
      if (Consume('}')) {
        return true;
      }
      if (!Consume(',')) {
        SetError(error, "expected ',' or '}' in object");
        return false;
      }
    }

    SetError(error, "unterminated object");
    return false;
  }

  bool ParseArray(JsonValue *out, std::string *error) {
    if (!Consume('[')) {
      SetError(error, "expected array");
      return false;
    }
    out->type = JsonValue::Type::kArray;
    out->array_value.clear();

    SkipWhitespace();
    if (Consume(']')) {
      return true;
    }

    while (pos_ < text_.size()) {
      JsonValue value;
      if (!ParseValue(&value, error)) {
        return false;
      }
      out->array_value.push_back(std::move(value));

      SkipWhitespace();
      if (Consume(']')) {
        return true;
      }
      if (!Consume(',')) {
        SetError(error, "expected ',' or ']' in array");
        return false;
      }
    }

    SetError(error, "unterminated array");
    return false;
  }

  bool ParseString(std::string *out, std::string *error) {
    SkipWhitespace();
    if (pos_ >= text_.size() || text_[pos_] != '"') {
      SetError(error, "expected string");
      return false;
    }
    ++pos_;
    out->clear();

    while (pos_ < text_.size()) {
      const char ch = text_[pos_++];
      if (ch == '"') {
        return true;
      }
      if (ch != '\\') {
        out->push_back(ch);
        continue;
      }
      if (pos_ >= text_.size()) {
        SetError(error, "unterminated string escape");
        return false;
      }
      const char esc = text_[pos_++];
      switch (esc) {
        case '"':
        case '\\':
        case '/':
          out->push_back(esc);
          break;
        case 'b':
          out->push_back('\b');
          break;
        case 'f':
          out->push_back('\f');
          break;
        case 'n':
          out->push_back('\n');
          break;
        case 'r':
          out->push_back('\r');
          break;
        case 't':
          out->push_back('\t');
          break;
        case 'u':
          if (pos_ + 4 > text_.size()) {
            SetError(error, "short unicode escape");
            return false;
          }
          out->push_back('?');
          pos_ += 4;
          break;
        default:
          SetError(error, "unsupported string escape");
          return false;
      }
    }

    SetError(error, "unterminated string");
    return false;
  }

  bool ParseNumber(JsonValue *out, std::string *error) {
    SkipWhitespace();
    const size_t start = pos_;
    if (pos_ < text_.size() && text_[pos_] == '-') {
      ++pos_;
    }
    while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0) {
      ++pos_;
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+')) {
        ++pos_;
      }
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0) {
        ++pos_;
      }
    }

    const std::string token = text_.substr(start, pos_ - start);
    errno = 0;
    char *end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (errno != 0 || end == token.c_str() || *end != '\0') {
      SetError(error, "invalid number");
      return false;
    }
    out->type = JsonValue::Type::kNumber;
    out->number_value = value;
    return true;
  }

  std::string text_;
  size_t pos_ = 0;
};

const JsonValue *Find(const JsonValue &object, const std::string &key) {
  if (!object.IsObject()) {
    return nullptr;
  }
  auto it = object.object_value.find(key);
  if (it == object.object_value.end()) {
    return nullptr;
  }
  return &it->second;
}

std::string GetString(const JsonValue &object, const std::string &key, const std::string &fallback = {}) {
  const JsonValue *value = Find(object, key);
  if (value == nullptr) {
    return fallback;
  }
  if (value->IsString()) {
    return value->string_value;
  }
  if (value->IsNumber()) {
    const int64_t as_int = static_cast<int64_t>(value->number_value);
    if (static_cast<double>(as_int) == value->number_value) {
      return std::to_string(as_int);
    }
  }
  if (value->IsBool()) {
    return value->bool_value ? "true" : "false";
  }
  return fallback;
}

bool GetInt(const JsonValue &object, const std::string &key, int32_t *out) {
  const JsonValue *value = Find(object, key);
  if (value == nullptr) {
    return false;
  }
  if (value->IsNumber()) {
    *out = static_cast<int32_t>(value->number_value);
    return true;
  }
  if (value->IsString()) {
    try {
      *out = std::stoi(value->string_value);
      return true;
    } catch (...) {
      return false;
    }
  }
  return false;
}

bool GetFloat(const JsonValue &object, const std::string &key, float *out) {
  const JsonValue *value = Find(object, key);
  if (value == nullptr) {
    return false;
  }
  if (value->IsNumber()) {
    *out = static_cast<float>(value->number_value);
    return true;
  }
  if (value->IsString()) {
    try {
      *out = std::stof(value->string_value);
      return true;
    } catch (...) {
      return false;
    }
  }
  return false;
}

bool GetBool(const JsonValue &object, const std::string &key, bool *out) {
  const JsonValue *value = Find(object, key);
  if (value == nullptr) {
    return false;
  }
  if (value->IsBool()) {
    *out = value->bool_value;
    return true;
  }
  if (value->IsNumber()) {
    *out = value->number_value != 0.0;
    return true;
  }
  if (value->IsString()) {
    const std::string normalized = Lower(Trim(value->string_value));
    if (normalized == "true" || normalized == "1") {
      *out = true;
      return true;
    }
    if (normalized == "false" || normalized == "0") {
      *out = false;
      return true;
    }
  }
  return false;
}

void ReadIntAlias(const JsonValue &object, const std::string &first, const std::string &second, int32_t *out) {
  if (!GetInt(object, first, out)) {
    GetInt(object, second, out);
  }
}

void ParseArchitectureFrom(const JsonValue &source, ModelArchitecture *arch) {
  ReadIntAlias(source, "num_layers", "num_hidden_layers", &arch->num_layers);
  GetInt(source, "hidden_size", &arch->hidden_size);
  GetInt(source, "intermediate_size", &arch->intermediate_size);
  ReadIntAlias(source, "num_heads", "num_attention_heads", &arch->num_heads);
  ReadIntAlias(source, "num_kv_heads", "num_key_value_heads", &arch->num_kv_heads);
  GetInt(source, "head_dim", &arch->head_dim);
  GetInt(source, "vocab_size", &arch->vocab_size);
  GetInt(source, "max_position_embeddings", &arch->max_position_embeddings);
  GetFloat(source, "rope_theta", &arch->rope_theta);
  if (!GetFloat(source, "norm_eps", &arch->norm_eps)) {
    GetFloat(source, "rms_norm_eps", &arch->norm_eps);
  }

  bool tie = false;
  if (GetBool(source, "tie_word_embeddings", &tie)) {
    arch->tie_word_embeddings = tie ? 1 : 0;
  } else {
    GetInt(source, "tie_word_embeddings", &arch->tie_word_embeddings);
  }

  arch->present = arch->num_layers > 0 || arch->hidden_size > 0 || arch->vocab_size > 0;
}

bool ParseTokenIdArray(const JsonValue &policy, const std::string &field, std::vector<int32_t> *out,
                       std::string *error) {
  const JsonValue *value = Find(policy, field);
  if (value == nullptr) {
    return true;
  }
  if (!value->IsArray()) {
    if (error != nullptr) {
      *error = "generation." + field + " must be an array";
    }
    return false;
  }

  std::unordered_set<int32_t> seen;
  for (const auto &entry : value->array_value) {
    const double number = entry.number_value;
    if (!entry.IsNumber() || !std::isfinite(number) || std::trunc(number) != number || number < 0.0 ||
        number > static_cast<double>(std::numeric_limits<int32_t>::max())) {
      if (error != nullptr) {
        *error = "generation." + field + " must contain non-negative integer token IDs";
      }
      return false;
    }
    const int32_t token_id = static_cast<int32_t>(number);
    if (!seen.insert(token_id).second) {
      if (error != nullptr) {
        *error = "generation." + field + " contains duplicate token IDs";
      }
      return false;
    }
    out->push_back(token_id);
  }
  return true;
}

bool ParseGenerationPolicy(const JsonValue &root, ModelManifest *manifest, std::string *error) {
  const JsonValue *generation = Find(root, "generation");
  if (generation == nullptr) {
    return true;
  }
  if (!generation->IsObject()) {
    if (error != nullptr) {
      *error = "manifest generation policy must be an object";
    }
    return false;
  }

  auto &policy = manifest->generation;
  policy.present = true;
  if (!ParseTokenIdArray(*generation, "stop_token_ids", &policy.stop_token_ids, error) ||
      !ParseTokenIdArray(*generation, "suppress_token_ids", &policy.suppress_token_ids, error)) {
    return false;
  }

  std::unordered_set<int32_t> stop_ids(policy.stop_token_ids.begin(), policy.stop_token_ids.end());
  for (int32_t token_id : policy.suppress_token_ids) {
    if (stop_ids.find(token_id) != stop_ids.end()) {
      if (error != nullptr) {
        *error = "generation stop and suppress token IDs must not overlap";
      }
      return false;
    }
  }

  if (manifest->architecture.vocab_size > 0) {
    auto out_of_range = [&](int32_t token_id) { return token_id >= manifest->architecture.vocab_size; };
    if (std::any_of(policy.stop_token_ids.begin(), policy.stop_token_ids.end(), out_of_range) ||
        std::any_of(policy.suppress_token_ids.begin(), policy.suppress_token_ids.end(), out_of_range)) {
      if (error != nullptr) {
        *error = "generation token ID exceeds architecture.vocab_size";
      }
      return false;
    }
  }
  return true;
}

bool ParseLiteRtManifest(const JsonValue &root, ModelManifest *manifest, std::string *error) {
  const JsonValue *litert = Find(root, "litert");
  if (litert == nullptr || !litert->IsObject()) {
    return true;
  }

  auto &out = manifest->litert;
  out.present = true;
  out.precision = manifest->dtype;
  const std::string precision = GetString(*litert, "precision");
  if (!precision.empty() && !ParseDTypeName(precision, &out.precision)) {
    if (error != nullptr) {
      *error = "unsupported litert.precision: " + precision;
    }
    return false;
  }

  const std::string top_prefill = GetString(*litert, "prefill");
  if (!top_prefill.empty()) {
    out.has_prefill = true;
    out.prefill_path = top_prefill;
  }
  const std::string top_decode = GetString(*litert, "decode");
  if (!top_decode.empty()) {
    out.has_decode = true;
    out.decode_path = top_decode;
  }

  const JsonValue *capabilities = Find(*litert, "capabilities");
  if (capabilities != nullptr && capabilities->IsObject()) {
    const JsonValue *prefill = Find(*capabilities, "prefill");
    if (prefill != nullptr && prefill->IsObject()) {
      const std::string path = GetString(*prefill, "path");
      if (!path.empty()) {
        out.has_prefill = true;
        out.prefill_path = path;
      }
      GetInt(*prefill, "seq_len", &out.prefill_seq_len);
    }

    const JsonValue *decode = Find(*capabilities, "decode");
    if (decode != nullptr && decode->IsObject()) {
      const std::string path = GetString(*decode, "path");
      if (!path.empty()) {
        out.has_decode = true;
        out.decode_path = path;
      }
      bool dynamic = false;
      if (GetBool(*decode, "dynamic_past_len", &dynamic)) {
        out.decode_dynamic_past_len = dynamic;
      }
      GetInt(*decode, "past_len", &out.decode_past_len);
      GetInt(*decode, "max_past_len", &out.decode_max_past_len);
    }

    const JsonValue *variants = Find(*capabilities, "decode_variants");
    if (variants != nullptr && variants->IsArray()) {
      for (const auto &entry : variants->array_value) {
        if (!entry.IsObject()) {
          continue;
        }
        LiteRtDecodeVariant variant;
        GetInt(entry, "past_len", &variant.past_len);
        variant.path = GetString(entry, "path");
        if (variant.past_len >= 0 && !variant.path.empty()) {
          out.decode_variants.push_back(std::move(variant));
        }
      }
    }
  }

  const JsonValue *top_variants = Find(*litert, "decode_variants");
  if (top_variants != nullptr && top_variants->IsArray()) {
    for (const auto &entry : top_variants->array_value) {
      if (!entry.IsObject()) {
        continue;
      }
      LiteRtDecodeVariant variant;
      GetInt(entry, "past_len", &variant.past_len);
      variant.path = GetString(entry, "path");
      if (variant.past_len >= 0 && !variant.path.empty()) {
        out.decode_variants.push_back(std::move(variant));
      }
    }
  }

  if ((out.has_prefill && !IsPackageRelativePath(out.prefill_path)) ||
      (out.has_decode && !IsPackageRelativePath(out.decode_path))) {
    if (error != nullptr) {
      *error = "LiteRT graph paths must be relative to the model package";
    }
    return false;
  }
  if (std::any_of(out.decode_variants.begin(), out.decode_variants.end(),
                  [](const auto &variant) { return !IsPackageRelativePath(variant.path); })) {
    if (error != nullptr) {
      *error = "LiteRT decode variant paths must be relative to the model package";
    }
    return false;
  }

  return true;
}

std::string ReadTextFile(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return {};
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

}  // namespace

bool ModelArchitecture::IsComplete() const {
  return num_layers > 0 && hidden_size > 0 && intermediate_size > 0 && num_heads > 0 && num_kv_heads > 0 &&
         head_dim > 0 && vocab_size > 0 && max_position_embeddings > 0 && rope_theta > 0.0f && norm_eps > 0.0f;
}

bool ParseDTypeName(const std::string &raw, MSLlmDType *out) {
  if (out == nullptr) {
    return false;
  }
  const std::string value = Lower(Trim(raw));
  if (value.empty()) {
    return false;
  }
  if (value == "float32" || value == "fp32") {
    *out = MSLLM_DTYPE_FLOAT32;
    return true;
  }
  if (value == "float16" || value == "fp16") {
    *out = MSLLM_DTYPE_FLOAT16;
    return true;
  }
  if (value == "int8") {
    *out = MSLLM_DTYPE_INT8;
    return true;
  }
  if (value == "int4") {
    *out = MSLLM_DTYPE_INT4;
    return true;
  }
  if (value == "bfloat16" || value == "bf16") {
    *out = MSLLM_DTYPE_BFLOAT16;
    return true;
  }
  return false;
}

MSLlmStatus ParseManifest(const std::string &content, ModelManifest *manifest, std::string *error_message) {
  if (manifest == nullptr) {
    if (error_message != nullptr) {
      *error_message = "manifest output is null";
    }
    return MSLLM_ERROR_INVALID_ARGS;
  }

  if (content.empty()) {
    if (error_message != nullptr) {
      *error_message = "manifest content is empty";
    }
    return MSLLM_ERROR_IO;
  }

  JsonValue root;
  std::string parse_error;
  if (!JsonParser(content).Parse(&root, &parse_error) || !root.IsObject()) {
    if (error_message != nullptr) {
      *error_message = "invalid manifest JSON: " + parse_error;
    }
    return MSLLM_ERROR_INVALID_ARGS;
  }

  ModelManifest parsed;
  parsed.model_name = GetString(root, "model_name");
  parsed.version = GetString(root, "version");
  parsed.format_version = GetString(root, "format_version");
  if (!parsed.format_version.empty() && parsed.format_version != "1.0") {
    if (error_message != nullptr) {
      *error_message = "unsupported manifest format_version: " + parsed.format_version;
    }
    return MSLLM_ERROR_NOT_SUPPORTED;
  }

  bool precision_declared = false;
  auto apply_precision = [&](const std::string &value, const std::string &field) {
    if (value.empty()) {
      return true;
    }
    MSLlmDType candidate = MSLLM_DTYPE_FLOAT32;
    if (!ParseDTypeName(value, &candidate)) {
      if (error_message != nullptr) {
        *error_message = "unsupported " + field + ": " + value;
      }
      return false;
    }
    if (precision_declared && candidate != parsed.dtype) {
      if (error_message != nullptr) {
        *error_message = "conflicting precision declaration in " + field;
      }
      return false;
    }
    parsed.dtype = candidate;
    precision_declared = true;
    return true;
  };

  if (!apply_precision(GetString(root, "dtype"), "manifest dtype")) {
    return MSLLM_ERROR_INVALID_ARGS;
  }
  const JsonValue *pipeline_config = Find(root, "pipeline_config");
  if (pipeline_config != nullptr && pipeline_config->IsObject()) {
    if (!apply_precision(GetString(*pipeline_config, "precision"), "pipeline_config.precision")) {
      return MSLLM_ERROR_INVALID_ARGS;
    }
  }
  const JsonValue *litert_config = Find(root, "litert");
  if (litert_config != nullptr && litert_config->IsObject() &&
      !apply_precision(GetString(*litert_config, "precision"), "litert.precision")) {
    return MSLLM_ERROR_INVALID_ARGS;
  }

  const JsonValue *architecture = Find(root, "architecture");
  if (architecture != nullptr && architecture->IsObject()) {
    ParseArchitectureFrom(*architecture, &parsed.architecture);
  } else {
    ParseArchitectureFrom(root, &parsed.architecture);
  }

  if (!ParseGenerationPolicy(root, &parsed, error_message)) {
    return MSLLM_ERROR_INVALID_ARGS;
  }

  if (!ParseLiteRtManifest(root, &parsed, error_message)) {
    return MSLLM_ERROR_INVALID_ARGS;
  }
  if (parsed.litert.present) {
    parsed.dtype = parsed.litert.precision;
  }

  // ── Assets ──────────────────────────────────────────────────────────
  const JsonValue *assets = Find(root, "assets");
  if (assets != nullptr && assets->IsObject()) {
    auto &out = parsed.assets;
    out.present = true;
    out.tokenizer = GetString(*assets, "tokenizer");
    out.embedding = GetString(*assets, "embedding");
    out.embedding_fp16 = GetString(*assets, "embedding_fp16");
    out.rope_sin = GetString(*assets, "rope_sin");
    out.rope_cos = GetString(*assets, "rope_cos");
    out.attention_mask = GetString(*assets, "attention_mask");
  }

  // ── NPU (NNRT) runtime params ───────────────────────────────────────
  const JsonValue *npu = Find(root, "npu");
  if (npu != nullptr && npu->IsObject()) {
    auto &out = parsed.npu;
    out.present = true;
    GetInt(*npu, "max_length", &out.max_length);
    GetInt(*npu, "chunk_size", &out.chunk_size);
    GetBool(*npu, "embedding_quant", &out.embedding_quant);
    GetInt(*npu, "scale_gp_size", &out.scale_gp_size);
    if (out.max_length <= 0 || out.chunk_size <= 0) {
      if (error_message != nullptr) {
        *error_message = "npu.max_length and npu.chunk_size must be positive";
      }
      return MSLLM_ERROR_INVALID_ARGS;
    }
    if (out.max_length % out.chunk_size != 0) {
      if (error_message != nullptr) {
        *error_message = "npu.max_length must be a multiple of npu.chunk_size";
      }
      return MSLLM_ERROR_INVALID_ARGS;
    }
    if (out.scale_gp_size <= 0) {
      out.scale_gp_size = 32;
    }
  }

  *manifest = std::move(parsed);
  return MSLLM_SUCCESS;
}

MSLlmStatus LoadModelManifest(const std::string &manifest_path, ModelManifest *manifest, std::string *error_message) {
  if (manifest == nullptr) {
    if (error_message != nullptr) {
      *error_message = "manifest output is null";
    }
    return MSLLM_ERROR_INVALID_ARGS;
  }

  const std::string content = ReadTextFile(manifest_path);
  if (content.empty()) {
    if (error_message != nullptr) {
      *error_message = "manifest is missing or empty: " + manifest_path;
    }
    return MSLLM_ERROR_IO;
  }

  return ParseManifest(content, manifest, error_message);
}

bool IsPackageRelativePath(const std::string &path) {
  if (path.empty() || path.front() == '/' || path.front() == '\\' || path.find(':') != std::string::npos) {
    return false;
  }
  size_t start = 0;
  while (start <= path.size()) {
    const size_t end = path.find_first_of("/\\", start);
    const std::string component = path.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (component == "..") {
      return false;
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return true;
}

bool ResolvePackagePath(const std::string &package_root, const std::string &candidate, std::string *resolved) {
  if (resolved == nullptr || package_root.empty()) {
    return false;
  }
  if (!IsPackageRelativePath(candidate)) {
    return false;
  }
  // Build a clean concatenation: package_root + "/" + candidate.
  std::string combined = package_root;
  if (combined.back() != '/' && combined.back() != '\\') {
    combined.push_back('/');
  }
  combined.append(candidate);
  // Canonicalise (realpath) to resolve symlinks and "."
  char *real = ::realpath(combined.c_str(), nullptr);
  if (real == nullptr) {
    return false;  // path does not exist or is inaccessible
  }
  *resolved = real;
  std::free(real);
  // Verify the resolved path is inside package_root.
  // Canonicalise the root separately.
  char *root_real = ::realpath(package_root.c_str(), nullptr);
  if (root_real != nullptr) {
    std::string root_str(root_real);
    std::free(root_real);
    if (!root_str.empty() && root_str.back() != '/') {
      root_str.push_back('/');
    }
    bool inside = resolved->size() >= root_str.size() && resolved->compare(0, root_str.size(), root_str) == 0;
    return inside;
  }
  // Fallback: prefix match against the non-canonicalised root.
  return resolved->size() >= package_root.size() && resolved->compare(0, package_root.size(), package_root) == 0;
}

MSLlmStatus BuildModelManifestFromKv(const MslPackageReader &reader, ModelManifest *manifest,
                                     std::string *error_message) {
  if (manifest == nullptr) {
    if (error_message != nullptr) {
      *error_message = "manifest output is null";
    }
    return MSLLM_ERROR_INVALID_ARGS;
  }
  ModelManifest &m = *manifest;

  reader.GetKvString(msl_format::key::kModelName, &m.model_name);
  reader.GetKvString(msl_format::key::kModelVersion, &m.version);
  reader.GetKvString(msl_format::key::kModelFormatVersion, &m.format_version);
  std::string dtype_name;
  if (reader.GetKvString(msl_format::key::kModelDtype, &dtype_name) && !ParseDTypeName(dtype_name, &m.dtype)) {
    if (error_message != nullptr) {
      *error_message = "unknown dtype in KV metadata: " + dtype_name;
    }
    return MSLLM_ERROR_NOT_SUPPORTED;
  }

  // ── architecture ───────────────────────────────────────────────────────
  uint32_t u32 = 0;
  if (reader.GetKvUint32(msl_format::key::kArchNumLayers, &u32)) m.architecture.num_layers = u32;
  if (reader.GetKvUint32(msl_format::key::kArchHiddenSize, &u32)) m.architecture.hidden_size = u32;
  if (reader.GetKvUint32(msl_format::key::kArchIntermediateSize, &u32)) m.architecture.intermediate_size = u32;
  if (reader.GetKvUint32(msl_format::key::kArchNumHeads, &u32)) m.architecture.num_heads = u32;
  if (reader.GetKvUint32(msl_format::key::kArchNumKvHeads, &u32)) m.architecture.num_kv_heads = u32;
  if (reader.GetKvUint32(msl_format::key::kArchHeadDim, &u32)) m.architecture.head_dim = u32;
  if (reader.GetKvUint32(msl_format::key::kArchVocabSize, &u32)) m.architecture.vocab_size = u32;
  if (reader.GetKvUint32(msl_format::key::kArchMaxPositionEmbeddings, &u32)) {
    m.architecture.max_position_embeddings = u32;
  }
  if (reader.GetKvUint32(msl_format::key::kArchTieWordEmbeddings, &u32)) m.architecture.tie_word_embeddings = u32;
  float f32 = 0.0f;
  if (reader.GetKvFloat32(msl_format::key::kArchRopeTheta, &f32)) m.architecture.rope_theta = f32;
  if (reader.GetKvFloat32(msl_format::key::kArchNormEps, &f32)) m.architecture.norm_eps = f32;
  m.architecture.present = m.architecture.num_layers > 0;

  // ── NPU runtime params ─────────────────────────────────────────────────
  if (reader.GetKvUint32(msl_format::key::kNpuMaxLength, &u32)) m.npu.max_length = u32;
  if (reader.GetKvUint32(msl_format::key::kNpuChunkSize, &u32)) m.npu.chunk_size = u32;
  if (reader.GetKvUint32(msl_format::key::kNpuScaleGpSize, &u32)) m.npu.scale_gp_size = u32;
  bool flag = false;
  if (reader.GetKvBool(msl_format::key::kNpuEmbeddingQuant, &flag)) m.npu.embedding_quant = flag;
  m.npu.present = m.npu.max_length > 0;

  // ── generation (eos token id; NNRTBackend reads stop_token_ids.front) ──
  if (reader.GetKvUint32(msl_format::key::kGenEosTokenId, &u32)) {
    m.generation.present = true;
    m.generation.stop_token_ids.assign(1, static_cast<int32_t>(u32));
  }

  // ── LiteRT graphs ──────────────────────────────────────────────────────
  std::string str;
  if (reader.GetKvString(msl_format::key::kLitertPrefillPath, &str)) {
    m.litert.present = true;
    m.litert.has_prefill = true;
    m.litert.prefill_path = str;
  }
  if (reader.GetKvUint32(msl_format::key::kLitertPrefillSeqLen, &u32)) m.litert.prefill_seq_len = u32;
  if (reader.GetKvString(msl_format::key::kLitertDecodePath, &str)) {
    m.litert.present = true;
    m.litert.has_decode = true;
    m.litert.decode_path = str;
  }
  if (reader.GetKvBool(msl_format::key::kLitertDecodeDynamicPastLen, &flag)) m.litert.decode_dynamic_past_len = flag;
  if (reader.GetKvUint32(msl_format::key::kLitertDecodePastLen, &u32)) m.litert.decode_past_len = u32;
  if (reader.GetKvUint32(msl_format::key::kLitertDecodeMaxPastLen, &u32)) m.litert.decode_max_past_len = u32;

  std::string variants_json;
  if (reader.GetKvString(msl_format::key::kLitertDecodeVariants, &variants_json) && !variants_json.empty()) {
    JsonValue root;
    std::string parse_error;
    if (JsonParser(variants_json).Parse(&root, &parse_error) && root.IsArray()) {
      for (const auto &entry : root.array_value) {
        if (!entry.IsObject()) {
          continue;
        }
        LiteRtDecodeVariant variant;
        GetInt(entry, "past_len", &variant.past_len);
        variant.path = GetString(entry, "path");
        if (variant.past_len >= 0 && !variant.path.empty()) {
          m.litert.decode_variants.push_back(std::move(variant));
        }
      }
    } else if (error_message != nullptr) {
      *error_message = "invalid litert.decode_variants JSON: " + parse_error;
      return MSLLM_ERROR_INVALID_ARGS;
    }
  }

  // ── assets (resolved against the resource table by the loader) ────────
  if (reader.GetKvString(msl_format::key::kAssetTokenizer, &str)) {
    m.assets.present = true;
    m.assets.tokenizer = str;
  }
  if (reader.GetKvString(msl_format::key::kAssetEmbedding, &str)) m.assets.embedding = str;
  if (reader.GetKvString(msl_format::key::kAssetEmbeddingFp16, &str)) m.assets.embedding_fp16 = str;
  if (reader.GetKvString(msl_format::key::kAssetRopeSin, &str)) m.assets.rope_sin = str;
  if (reader.GetKvString(msl_format::key::kAssetRopeCos, &str)) m.assets.rope_cos = str;
  if (reader.GetKvString(msl_format::key::kAssetAttentionMask, &str)) m.assets.attention_mask = str;

  return MSLLM_SUCCESS;
}

}  // namespace mslite_llm
