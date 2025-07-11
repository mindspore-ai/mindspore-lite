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
#include <memory>
#include <vector>
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "src/common/utils.h"
#include "src/common/file_utils.h"
#include "src/litert/kernel/cpu/fp32_grad/softmax_grad.h"
#include "nnacl/fp32_grad/softmax_grad.h"
#include "src/litert/kernel_registry.h"

namespace mindspore {
class TestSoftmaxGradFp32 : public mindspore::CommonTest {
 public:
  TestSoftmaxGradFp32() {}
};

TEST_F(TestSoftmaxGradFp32, SoftmaxGradAxis0) {
  auto softmax_param = new SoftmaxParameter();
  ASSERT_NE(softmax_param, nullptr);

  softmax_param->axis_ = 0;
  int element_size_ = 1188;
  int n_dim_ = 4;
  int input_shape_[] = {1, 9, 11, 12};

  int inner_size = 1;
  if (softmax_param->axis_ == -1) softmax_param->axis_ = n_dim_ - 1;
  for (int i = softmax_param->axis_ + 1; i < n_dim_; i++) {
    inner_size *= input_shape_[i];
  }
  float *sum_data = new (std::nothrow) float[inner_size];
  ASSERT_NE(sum_data, nullptr);
  float *sum_mul = new (std::nothrow) float[inner_size * input_shape_[softmax_param->axis_]];
  ASSERT_NE(sum_mul, nullptr);
  std::vector<int> shape = {1, 9, 11, 12};
  size_t input_size;
  std::string input_path = "./softmax/softmaxgrad_yinput.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);
  std::string yt_path = "./softmax/softmaxgrad_yt_input.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  ASSERT_NE(yt_data, nullptr);
  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  auto out_data = new float[element_size_];
  ASSERT_NE(out_data, nullptr);
  // warm up loop
  for (int i = 0; i < 3; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, input_shape_, n_dim_, element_size_,
                softmax_param->axis_);
  }

  int loop_count = 3;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, input_shape_, n_dim_, element_size_,
                softmax_param->axis_);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./softmax/softmaxgrad_out.bin";

  auto res = CompareRelativeOutput(out_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] yt_data;
  delete[] out_data;
  delete[] sum_data;
  delete[] sum_mul;

  delete softmax_param;

  MS_LOG(INFO) << "SoftmaxGradAxis0 passed";
}

TEST_F(TestSoftmaxGradFp32, SoftmaxGradAxis1) {
  auto softmax_param = new SoftmaxParameter();
  ASSERT_NE(softmax_param, nullptr);

  softmax_param->axis_ = 1;
  int element_size_ = 1188;
  int n_dim_ = 4;
  int input_shape_[] = {1, 9, 11, 12};

  int inner_size = 1;
  if (softmax_param->axis_ == -1) softmax_param->axis_ = n_dim_ - 1;
  for (int i = softmax_param->axis_ + 1; i < n_dim_; i++) {
    inner_size *= input_shape_[i];
  }
  float *sum_data = new (std::nothrow) float[inner_size];
  ASSERT_NE(sum_data, nullptr);
  float *sum_mul = new (std::nothrow) float[inner_size * input_shape_[softmax_param->axis_]];
  ASSERT_NE(sum_mul, nullptr);

  std::vector<int> shape = {1, 9, 11, 12};
  size_t input_size;
  std::string input_path = "./softmax/softmaxgrad_1_yinput.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);

  std::string yt_path = "./softmax/softmaxgrad_1_yt_input.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  ASSERT_NE(yt_data, nullptr);

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  auto out_data = new float[element_size_];
  ASSERT_NE(out_data, nullptr);

  // warm up loop
  for (int i = 0; i < 3; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, input_shape_, n_dim_, element_size_,
                softmax_param->axis_);
  }

  int loop_count = 3;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, input_shape_, n_dim_, element_size_,
                softmax_param->axis_);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./softmax/softmaxgrad_1_out.bin";
  // auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  auto res = CompareRelativeOutput(out_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] yt_data;
  delete[] out_data;
  delete[] sum_data;
  delete[] sum_mul;

  delete softmax_param;

  MS_LOG(INFO) << "SoftmaxGradAxis1 passed";
}

TEST_F(TestSoftmaxGradFp32, SoftmaxGradAxis2) {
  auto softmax_param = new SoftmaxParameter();
  ASSERT_NE(softmax_param, nullptr);

  softmax_param->axis_ = 2;
  int element_size_ = 1188;
  int n_dim_ = 4;
  int input_shape_[] = {1, 9, 11, 12};

  int inner_size = 1;
  if (softmax_param->axis_ == -1) softmax_param->axis_ = n_dim_ - 1;
  for (int i = softmax_param->axis_ + 1; i < n_dim_; i++) {
    inner_size *= input_shape_[i];
  }
  float *sum_data = new (std::nothrow) float[inner_size];
  ASSERT_NE(sum_data, nullptr);
  float *sum_mul = new (std::nothrow) float[inner_size * input_shape_[softmax_param->axis_]];
  ASSERT_NE(sum_mul, nullptr);

  std::vector<int> shape = {1, 9, 11, 12};
  size_t input_size;
  std::string input_path = "./softmax/softmaxgrad_2_yinput.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);

  std::string yt_path = "./softmax/softmaxgrad_2_yt_input.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  ASSERT_NE(yt_data, nullptr);

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  auto out_data = new float[element_size_];
  ASSERT_NE(out_data, nullptr);

  // warm up loop
  for (int i = 0; i < 3; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, input_shape_, n_dim_, element_size_,
                softmax_param->axis_);
  }

  int loop_count = 3;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, input_shape_, n_dim_, element_size_,
                softmax_param->axis_);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./softmax/softmaxgrad_2_out.bin";
  // auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  auto res = CompareRelativeOutput(out_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] yt_data;
  delete[] out_data;
  delete[] sum_data;
  delete[] sum_mul;

  delete softmax_param;

  MS_LOG(INFO) << "SoftmaxGradAxis2 passed";
}

TEST_F(TestSoftmaxGradFp32, SoftmaxGradAxis3) {
  auto softmax_param = new SoftmaxParameter();
  ASSERT_NE(softmax_param, nullptr);

  softmax_param->axis_ = 3;
  int element_size_ = 1188;
  int n_dim_ = 4;
  int input_shape_[] = {1, 9, 11, 12};

  int inner_size = 1;
  if (softmax_param->axis_ == -1) softmax_param->axis_ = n_dim_ - 1;
  for (int i = softmax_param->axis_ + 1; i < n_dim_; i++) {
    inner_size *= input_shape_[i];
  }
  float *sum_data = new (std::nothrow) float[inner_size];
  ASSERT_NE(sum_data, nullptr);
  float *sum_mul = new (std::nothrow) float[inner_size * input_shape_[softmax_param->axis_]];
  ASSERT_NE(sum_mul, nullptr);

  std::vector<int> shape = {1, 9, 11, 12};
  size_t input_size;
  std::string input_path = "./softmax/softmaxgrad_3_yinput.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);
  std::string yt_path = "./softmax/softmaxgrad_3_yt_input.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  ASSERT_NE(yt_data, nullptr);

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  auto out_data = new float[element_size_];
  ASSERT_NE(out_data, nullptr);

  // warm up loop
  for (int i = 0; i < 3; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, input_shape_, n_dim_, element_size_,
                softmax_param->axis_);
  }

  int loop_count = 3;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, input_shape_, n_dim_, element_size_,
                softmax_param->axis_);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./softmax/softmaxgrad_3_out.bin";
  // auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  auto res = CompareRelativeOutput(out_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] yt_data;
  delete[] out_data;
  delete[] sum_data;
  delete[] sum_mul;

  delete softmax_param;

  MS_LOG(INFO) << "SoftmaxGradAxis3 passed";
}

TEST_F(TestSoftmaxGradFp32, SoftmaxGradAxisMinus1) {
  auto softmax_param = new SoftmaxParameter();
  ASSERT_NE(softmax_param, nullptr);

  softmax_param->axis_ = -1;
  int element_size_ = 1188;
  int n_dim_ = 4;
  int input_shape_[] = {1, 9, 11, 12};

  int inner_size = 1;
  if (softmax_param->axis_ == -1) softmax_param->axis_ = n_dim_ - 1;
  for (int i = softmax_param->axis_ + 1; i < n_dim_; i++) {
    inner_size *= input_shape_[i];
  }
  float *sum_data = new (std::nothrow) float[inner_size];
  ASSERT_NE(sum_data, nullptr);
  float *sum_mul = new (std::nothrow) float[inner_size * input_shape_[softmax_param->axis_]];
  ASSERT_NE(sum_mul, nullptr);

  std::vector<int> shape = {1, 9, 11, 12};
  size_t input_size;
  std::string input_path = "./softmax/softmaxgrad_-1_yinput.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);

  std::string yt_path = "./softmax/softmaxgrad_-1_yt_input.bin";
  auto yt_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  ASSERT_NE(yt_data, nullptr);
  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  auto out_data = new float[element_size_];
  ASSERT_NE(out_data, nullptr);

  // warm up loop
  for (int i = 0; i < 3; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, input_shape_, n_dim_, element_size_,
                softmax_param->axis_);
  }

  int loop_count = 3;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    SoftmaxGrad(input_data, yt_data, out_data, sum_data, sum_mul, input_shape_, n_dim_, element_size_,
                softmax_param->axis_);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::string output_path = "./softmax/softmaxgrad_-1_out.bin";
  // auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  auto res = CompareRelativeOutput(out_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] yt_data;
  delete[] out_data;
  delete[] sum_data;
  delete[] sum_mul;

  delete softmax_param;

  MS_LOG(INFO) << "SoftmaxGradAxisMinus1 passed";
}
}  // namespace mindspore
