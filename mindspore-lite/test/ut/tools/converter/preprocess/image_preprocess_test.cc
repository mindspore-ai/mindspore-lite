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
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "tools/converter/preprocess/image_preprocess.h"
#include "tools/converter/preprocess/preprocess_param.h"

namespace mindspore {
namespace lite {
namespace {
// Prepare one calibrate bin file and return its absolute path.
std::string WriteCalibFile(const std::string &name, const std::string &content) {
  auto path = "/tmp/lite_ut_preprocess_" + name + ".bin";
  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  ofs << content;
  ofs.close();
  return path;
}

preprocess::DataPreProcessParam MakeBinParam(
  const std::map<std::string, std::vector<std::string>> &calibrate_path_vector) {
  preprocess::DataPreProcessParam param;
  param.calibrate_path_vector = calibrate_path_vector;
  param.calibrate_size = 1;
  param.input_type = preprocess::BIN;
  return param;
}
}  // namespace

class ImagePreProcessTest : public ::testing::Test {};

// Single-input model: a calibrate_path key that differs from the tensor name
// (e.g. cfg uses the generic "input" while the model input is "x") must fall
// back to the only configured entry instead of failing the quantization.
TEST_F(ImagePreProcessTest, SingleEntryFallsBackToOnlyCalibratePath) {
  auto path = WriteCalibFile("fallback", std::string("calib-data-fallback", 20));
  auto param = MakeBinParam({{"input", {path}}});
  void *data = nullptr;
  size_t size = 0;
  auto ret = preprocess::PreProcess(param, "x", 0, &data, &size);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(size, 20);
  ASSERT_EQ(memcmp(data, "calib-data-fallback", 20), 0);
  delete[] reinterpret_cast<char *>(data);
  std::remove(path.c_str());
}

// Exact tensor-name match keeps working.
TEST_F(ImagePreProcessTest, ExactNameMatchReadsFile) {
  auto path = WriteCalibFile("exact", std::string("calib-data-exact!!", 18));
  auto param = MakeBinParam({{"x", {path}}});
  void *data = nullptr;
  size_t size = 0;
  auto ret = preprocess::PreProcess(param, "x", 0, &data, &size);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(size, 18);
  delete[] reinterpret_cast<char *>(data);
  std::remove(path.c_str());
}

// Multi-input cfg stays strict: a missing tensor name must not silently pick
// another input's calibrate data.
TEST_F(ImagePreProcessTest, MultiEntryKeepsStrictNameMatch) {
  auto path_a = WriteCalibFile("strictA", std::string("a", 1));
  auto path_b = WriteCalibFile("strictB", std::string("b", 1));
  auto param = MakeBinParam({{"foo", {path_a}}, {"bar", {path_b}}});
  void *data = nullptr;
  size_t size = 0;
  auto ret = preprocess::PreProcess(param, "x", 0, &data, &size);
  ASSERT_EQ(ret, RET_INPUT_PARAM_INVALID);
  std::remove(path_a.c_str());
  std::remove(path_b.c_str());
}
}  // namespace lite
}  // namespace mindspore
