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

#include "gtest/gtest.h"
#include "include/errorcode.h"
#include "tools/converter/config_parser/config_file_parser.h"

namespace mindspore {
namespace lite {
TEST(ConfigFileParserTest, ParseRegistryEnableConstFoldOff) {
  ConfigFileParser parser;
  std::map<std::string, std::map<std::string, std::string>> maps{{"registry", {{"enable_const_fold", "off"}}}};
  ASSERT_EQ(parser.ParseConfigParam(&maps), RET_OK);
  ASSERT_EQ(parser.GetRegistryInfoString().enable_const_fold, "off");
}

TEST(ConfigFileParserTest, ParseRegistryEnableConstFoldOn) {
  ConfigFileParser parser;
  std::map<std::string, std::map<std::string, std::string>> maps{{"registry", {{"enable_const_fold", "on"}}}};
  ASSERT_EQ(parser.ParseConfigParam(&maps), RET_OK);
  ASSERT_EQ(parser.GetRegistryInfoString().enable_const_fold, "on");
}

TEST(ConfigFileParserTest, ParseRegistryEnableConstFoldNotConfigured) {
  ConfigFileParser parser;
  std::map<std::string, std::map<std::string, std::string>> maps{{"registry", {{"disable_fusion", "off"}}}};
  ASSERT_EQ(parser.ParseConfigParam(&maps), RET_OK);
  // Not configured: stays empty, converter keeps the default (enable_const_fold = true).
  ASSERT_EQ(parser.GetRegistryInfoString().enable_const_fold, "");
}
}  // namespace lite
}  // namespace mindspore
