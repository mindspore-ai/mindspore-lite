/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <iostream>
#include "schema/inner/model_generated.h"
#include "src/tensor.h"
#include "src/litert/tensor_category.h"
#include "common/common_test.h"
#include "src/common/file_utils.h"
#include "nnacl/pad_parameter.h"
#include "src/litert/kernel/cpu/int8/pad_int8.h"

namespace mindspore {
using mindspore::lite::LiteQuantParam;
using mindspore::lite::Tensor;
class TestPadInt8 : public mindspore::CommonTest {
 public:
  TestPadInt8() {}
};

int PadInt8TestInit1(std::vector<Tensor *> *inputs_, std::vector<Tensor *> *outputs_, PadParameter *pad_param,
                     int8_t **correct) {
  Tensor *in_t = new Tensor(kNumberTypeInt8, {3}, mindspore::NHWC, lite::Category::VAR);
  in_t->MallocData();
  int8_t in[] = {1, 1, 1};
  memcpy(in_t->MutableData(), in, sizeof(int8_t) * in_t->ElementsNum());
  LiteQuantParam *in_quant_arg = new LiteQuantParam();
  in_quant_arg->zeroPoint = 10, in_quant_arg->scale = 0.31228156;
  in_t->AddQuantParam(*in_quant_arg);
  delete in_quant_arg;
  inputs_->push_back(in_t);

  Tensor *in_t_pad = new Tensor(kNumberTypeInt8, {1}, mindspore::NHWC, lite::Category::VAR);
  in_t->MallocData();
  inputs_->push_back(in_t_pad);

  Tensor *out_t = new Tensor(kNumberTypeInt8, {7}, mindspore::NHWC, lite::Category::VAR);
  out_t->MallocData();
  LiteQuantParam *out_quant_arg = new LiteQuantParam();
  out_quant_arg->zeroPoint = 10, out_quant_arg->scale = 0.31228156;
  out_t->AddQuantParam(*out_quant_arg);
  delete out_quant_arg;
  outputs_->push_back(out_t);

  *correct = reinterpret_cast<int8_t *>(malloc(out_t->ElementsNum() * sizeof(int8_t)));
  int8_t co[] = {10, 10, 1, 1, 1, 10, 10};
  memcpy(*correct, co, out_t->ElementsNum() * sizeof(int8_t));

  int padding[] = {0, 0, 0, 0, 0, 0, 2, 2};
  memcpy(pad_param->paddings_, padding, std::min(sizeof(padding), MAX_PAD_SIZE * sizeof(int)));
  pad_param->constant_value_ = 0;
  pad_param->padding_length = in_t->ConvertToTensorC()->shape_size_ * Num2;

  return out_t->ElementsNum();
}

TEST_F(TestPadInt8, PadInt8Test1) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  PadParameter *pad_param = reinterpret_cast<PadParameter *>(malloc(sizeof(PadParameter)));
  ASSERT_NE(pad_param, nullptr);
  memset(pad_param, 0, sizeof(PadParameter));
  lite::InnerContext *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  int8_t *correct;
  int total_size = PadInt8TestInit1(&inputs_, &outputs_, pad_param, &correct);
  kernel::PadInt8CPUKernel *pad = new kernel::PadInt8CPUKernel(&pad_param->op_parameter_, inputs_, outputs_, ctx);

  ASSERT_EQ(lite::RET_OK, pad->Prepare());
  ASSERT_EQ(lite::RET_OK, pad->Run());

  int8_t *output_data = reinterpret_cast<int8_t *>(outputs_[0]->MutableData());
  ASSERT_EQ(0, CompareOutputData(output_data, correct, total_size, 0));
  for (auto &in_t : inputs_) {
    delete in_t;
  }
  for (auto &out_t : outputs_) {
    delete out_t;
  }
  delete pad;
  delete ctx;
  free(correct);
}

int PadInt8TestInit2(std::vector<Tensor *> *inputs_, std::vector<Tensor *> *outputs_, PadParameter *pad_param,
                     int8_t **correct) {
  Tensor *in_t = new Tensor(kNumberTypeInt8, {6, 2}, mindspore::NHWC, lite::Category::VAR);
  in_t->MallocData();
  int8_t in[] = {18, 71, 99, -6, 5, -119, 86, 13, 15, -85, -41, -77};
  memcpy(in_t->MutableData(), in, sizeof(int8_t) * in_t->ElementsNum());
  LiteQuantParam *in_quant_arg = new LiteQuantParam();
  in_quant_arg->zeroPoint = 10, in_quant_arg->scale = 0.31228156;
  in_t->AddQuantParam(*in_quant_arg);
  delete in_quant_arg;
  inputs_->push_back(in_t);

  Tensor *in_t_pad = new Tensor(kNumberTypeInt8, {1}, mindspore::NHWC, lite::Category::VAR);
  in_t->MallocData();
  inputs_->push_back(in_t_pad);

  Tensor *out_t = new Tensor(kNumberTypeInt8, {10, 5}, mindspore::NHWC, lite::Category::VAR);
  out_t->MallocData();
  LiteQuantParam *out_quant_arg = new LiteQuantParam();
  out_quant_arg->zeroPoint = 10, out_quant_arg->scale = 0.31228156;
  out_t->AddQuantParam(*out_quant_arg);
  delete out_quant_arg;
  outputs_->push_back(out_t);

  *correct = reinterpret_cast<int8_t *>(malloc(out_t->ElementsNum() * sizeof(int8_t)));
  int8_t co[] = {10, 10, 10, 10,  10, 10, 10, 10,  10,  10, 10,   10, 10, 10, 10, 10, 18,
                 71, 10, 10, 10,  99, -6, 10, 10,  10,  5,  -119, 10, 10, 10, 86, 13, 10,
                 10, 10, 15, -85, 10, 10, 10, -41, -77, 10, 10,   10, 10, 10, 10, 10};
  memcpy(*correct, co, out_t->ElementsNum() * sizeof(int8_t));

  int padding[] = {0, 0, 0, 0, 3, 1, 1, 2};
  memcpy(pad_param->paddings_, padding, std::min(sizeof(padding), MAX_PAD_SIZE * sizeof(int)));
  pad_param->constant_value_ = 0;
  pad_param->padding_length = in_t->ConvertToTensorC()->shape_size_ * Num2;

  return out_t->ElementsNum();
}

TEST_F(TestPadInt8, PadInt8Test2) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  PadParameter *pad_param = reinterpret_cast<PadParameter *>(malloc(sizeof(PadParameter)));
  ASSERT_NE(pad_param, nullptr);
  memset(pad_param, 0, sizeof(PadParameter));
  lite::InnerContext *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  int8_t *correct;
  int total_size = PadInt8TestInit2(&inputs_, &outputs_, pad_param, &correct);
  kernel::PadInt8CPUKernel *pad = new kernel::PadInt8CPUKernel(&pad_param->op_parameter_, inputs_, outputs_, ctx);

  ASSERT_EQ(lite::RET_OK, pad->Prepare());
  ASSERT_EQ(lite::RET_OK, pad->Run());

  int8_t *output_data = reinterpret_cast<int8_t *>(outputs_[0]->MutableData());
  ASSERT_EQ(0, CompareOutputData(output_data, correct, total_size, 0));

  for (auto &in_t : inputs_) {
    delete in_t;
  }
  for (auto &out_t : outputs_) {
    delete out_t;
  }
  delete pad;
  delete ctx;
  free(correct);
}

int PadInt8TestInit4(std::vector<Tensor *> *inputs_, std::vector<Tensor *> *outputs_, PadParameter *pad_param,
                     int8_t **correct) {
  Tensor *in_t = new Tensor(kNumberTypeInt8, {2, 3, 2, 1}, mindspore::NHWC, lite::Category::VAR);
  in_t->MallocData();
  int8_t in[] = {73, 24, 7, -31, -109, -2, 69, -64, 51, -45, 38, 53};
  memcpy(in_t->MutableData(), in, sizeof(int8_t) * in_t->ElementsNum());
  LiteQuantParam *in_quant_arg = new LiteQuantParam();
  in_quant_arg->zeroPoint = 10, in_quant_arg->scale = 0.31228156;
  in_t->AddQuantParam(*in_quant_arg);
  delete in_quant_arg;
  inputs_->push_back(in_t);

  Tensor *in_t_pad = new Tensor(kNumberTypeInt8, {1}, mindspore::NHWC, lite::Category::VAR);
  in_t->MallocData();
  inputs_->push_back(in_t_pad);

  Tensor *out_t = new Tensor(kNumberTypeInt8, {6, 6, 4, 3}, mindspore::NHWC, lite::Category::VAR);
  out_t->MallocData();
  LiteQuantParam *out_quant_arg = new LiteQuantParam();
  out_quant_arg->zeroPoint = 10, out_quant_arg->scale = 0.31228156;
  out_t->AddQuantParam(*out_quant_arg);
  delete out_quant_arg;
  outputs_->push_back(out_t);

  *correct = reinterpret_cast<int8_t *>(malloc(out_t->ElementsNum() * sizeof(int8_t)));
  int8_t co[] = {
    10, 10, 10, 10, 10, 10, 10, 10,  10, 10, 10, 10, 10, 10,  10, 10, 10,   10, 10, 10, 10, 10, 10, 10, 10, 10,  10,
    10, 10, 10, 10, 10, 10, 10, 10,  10, 10, 10, 10, 10, 10,  10, 10, 10,   10, 10, 10, 10, 10, 10, 10, 10, 10,  10,
    10, 10, 10, 10, 10, 10, 10, 10,  10, 10, 10, 10, 10, 10,  10, 10, 10,   10, 10, 10, 10, 10, 10, 10, 10, 10,  10,
    10, 10, 10, 10, 10, 10, 10, 10,  10, 10, 10, 10, 10, 10,  10, 10, 10,   10, 10, 10, 10, 10, 10, 10, 10, 10,  10,
    10, 10, 10, 10, 10, 10, 10, 10,  10, 10, 10, 10, 10, 10,  10, 10, 10,   10, 10, 10, 10, 10, 10, 10, 10, 10,  10,
    10, 10, 10, 10, 10, 10, 10, 10,  10, 10, 10, 10, 10, 10,  10, 10, 10,   10, 10, 10, 10, 10, 10, 10, 10, 10,  10,
    10, 10, 10, 10, 10, 10, 10, 10,  10, 10, 10, 10, 10, 10,  10, 10, 10,   10, 10, 10, 10, 10, 10, 10, 10, 10,  10,
    10, 10, 10, 10, 10, 10, 10, 10,  10, 10, 10, 10, 10, 10,  10, 10, 10,   10, 10, 10, 10, 10, 10, 10, 10, 10,  10,
    10, 10, 10, 10, 10, 10, 10, 10,  10, 10, 10, 10, 10, 10,  10, 10, 10,   10, 10, 73, 10, 10, 24, 10, 10, 10,  10,
    10, 10, 10, 10, 7,  10, 10, -31, 10, 10, 10, 10, 10, 10,  10, 10, -109, 10, 10, -2, 10, 10, 10, 10, 10, 10,  10,
    10, 10, 10, 10, 10, 10, 10, 10,  10, 10, 10, 10, 10, 10,  10, 10, 10,   10, 10, 10, 10, 10, 10, 10, 10, 10,  10,
    10, 10, 10, 10, 10, 10, 10, 10,  10, 10, 69, 10, 10, -64, 10, 10, 10,   10, 10, 10, 10, 10, 51, 10, 10, -45, 10,
    10, 10, 10, 10, 10, 10, 10, 38,  10, 10, 53, 10, 10, 10,  10, 10, 10,   10, 10, 10, 10, 10, 10, 10, 10, 10,  10,
    10, 10, 10, 10, 10, 10, 10, 10,  10, 10, 10, 10, 10, 10,  10, 10, 10,   10, 10, 10, 10, 10, 10, 10, 10, 10,  10,
    10, 10, 10, 10, 10, 10, 10, 10,  10, 10, 10, 10, 10, 10,  10, 10, 10,   10, 10, 10, 10, 10, 10, 10, 10, 10,  10,
    10, 10, 10, 10, 10, 10, 10, 10,  10, 10, 10, 10, 10, 10,  10, 10, 10,   10, 10, 10, 10, 10, 10, 10, 10, 10,  10};
  memcpy(*correct, co, out_t->ElementsNum() * sizeof(int8_t));

  int padding[] = {3, 1, 1, 2, 2, 0, 1, 1};
  memcpy(pad_param->paddings_, padding, std::min(sizeof(padding), MAX_PAD_SIZE * sizeof(int)));
  pad_param->constant_value_ = 0;
  pad_param->padding_length = in_t->ConvertToTensorC()->shape_size_ * Num2;

  return out_t->ElementsNum();
}

TEST_F(TestPadInt8, PadInt8TestInit4) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  PadParameter *pad_param = reinterpret_cast<PadParameter *>(malloc(sizeof(PadParameter)));
  ASSERT_NE(pad_param, nullptr);
  memset(pad_param, 0, sizeof(PadParameter));
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  int8_t *correct;
  int total_size = PadInt8TestInit2(&inputs_, &outputs_, pad_param, &correct);
  kernel::PadInt8CPUKernel *pad = new kernel::PadInt8CPUKernel(&pad_param->op_parameter_, inputs_, outputs_, ctx);

  ASSERT_EQ(lite::RET_OK, pad->Prepare());
  ASSERT_EQ(lite::RET_OK, pad->Run());

  int8_t *output_data = reinterpret_cast<int8_t *>(outputs_[0]->MutableData());
  ASSERT_EQ(0, CompareOutputData(output_data, correct, total_size, 0));

  for (auto &in_t : inputs_) {
    delete in_t;
  }
  for (auto &out_t : outputs_) {
    delete out_t;
  }
  delete pad;
  delete ctx;
  free(correct);
}
}  // namespace mindspore
