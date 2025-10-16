/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>
#include <string>
#include "schema/inner/model_generated.h"
#include "common/common_test.h"
#include "include/api/context.h"
#include "include/api/model.h"
#include "src/common/log_adapter.h"
#include "src/litert/lite_session.h"
#include "include/registry/register_kernel_interface.h"
#include "include/registry/register_kernel.h"
#include "src/common/tensor_util.h"
#include "ut/src/runtime/kernel/dsp/dsp_test.h"

using mindspore::kernel::Kernel;
using mindspore::kernel::KernelInterface;

namespace mindspore::lite::dsp::test {

class TestDSP_FFT : public DSPCommonTest {
 public:
  void w_init(float *w, int n) {
    int i;
    const float PI = 3.14159;
    for (i = 0; i < n; i++) {
      w[i * 2] = cos(2 * PI * i / n);
      w[i * 2 + 1] = -sin(2 * PI * i / n);
    }
  }

  void bitrev_r(std::complex<float> *Input, int N) {
    int i, j, k;
    std::complex<float> temp;
    j = 0;
    for (i = 0; i < N - 1; i++) {
      if (i < j) {
        temp = Input[i];
        Input[i] = Input[j];
        Input[j] = temp;
      }
      k = N >> 1;
      while (k <= j) {
        j = j - k;
        k >>= 1;
      }
      j = j + k;
    }
  }

  void fft(float *src, float *src_w, int n) {
    if (n <= 0 || (n & (n - 1)) != 0) {
      MS_LOG(ERROR) << "Input size must be a power of 2";
      return;
    }
    int log_n = log2(n);
    float *src_w_tmp = new float[n * 2];
    if (src_w_tmp == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      return;
    }
    memcpy(src_w_tmp, src_w, n * 2 * sizeof(float));
    int segment_size, segment_num;
    int offset = 0;
    float real, imag, real_up, imag_up, real_down, imag_down;
    for (int stage = 0; stage < log_n; ++stage) {
      segment_size = 1 << (log_n - stage);
      segment_num = n / segment_size;
      const int segment_step = segment_size * 2;
      offset = -segment_step;
      for (int group = 0; group < segment_num; ++group) {
        offset += segment_step;
        for (int idx = 0; idx < segment_size; idx += 2) {
          const int upper_offset = offset + idx;
          const int lower_offset = upper_offset + segment_size;
          real_up = src[upper_offset] + src[lower_offset];
          imag_up = src[upper_offset + 1] + src[lower_offset + 1];
          real = src[upper_offset] - src[lower_offset];
          imag = src[upper_offset + 1] - src[lower_offset + 1];
          const int w_idx = idx;
          real_down = src_w_tmp[w_idx] * real - src_w_tmp[w_idx + 1] * imag;
          imag_down = src_w_tmp[w_idx] * imag + src_w_tmp[w_idx + 1] * real;
          src[upper_offset] = real_up;
          src[upper_offset + 1] = imag_up;
          src[lower_offset] = real_down;
          src[lower_offset + 1] = imag_down;
        }
      }
      for (int idx = 0; idx < segment_size / 2; idx += 2) {
        src_w_tmp[idx] = src_w_tmp[2 * idx];
        src_w_tmp[idx + 1] = src_w_tmp[2 * idx + 1];
      }
    }
    delete[] src_w_tmp;
  }
};

TEST_F(TestDSP_FFT, 16K_Cplx64) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input0_shape = {16 * 1024};
  std::vector<int> output_shape = {16 * 1024};
  int num = input0_shape[0];
  auto x = new lite::Tensor(kNumberTypeComplex64, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  x->MallocData(allocator_);
  inputs_.push_back(x);
  auto out_t = new lite::Tensor(kNumberTypeComplex64, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  out_t->MallocData(allocator_);
  outputs_.push_back(out_t);
  auto input_x = reinterpret_cast<float *>(x->MutableData());
  memset(input_x, 0, sizeof(float) * num * 2);
  auto output = reinterpret_cast<float *>(out_t->MutableData());
  memset(output, 0, sizeof(float) * num * 2);
  flatbuffers::FlatBufferBuilder fbb(1024);
  // create custom kernel
  auto val_offset = schema::CreateCustomDirect(fbb, "Custom_FT_FFT");
  flatbuffers::Offset<mindspore::schema::Primitive> primitive =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(PrimType::PrimType_Custom), val_offset.o);
  fbb.Finish(primitive);
  const mindspore::schema::Primitive *primitive_ptr =
    flatbuffers::GetRoot<mindspore::schema::Primitive>(fbb.GetBufferPointer());
  auto data_type = x->data_type();
  auto prim_type = mindspore::schema::PrimitiveType::PrimitiveType_Custom;
  std::string kernel_arch = "DSP";
  std::string provider = "FTMatrix";
  auto arch = kernel::KERNEL_ARCH::kDSP;
  kernel::KernelKey key{arch, data_type, NHWC, prim_type, kernel_arch, provider};
  registry::KernelDesc desc{static_cast<DataType>(key.data_type), key.type, key.kernel_arch, key.provider};
  auto creator = registry::RegisterKernel::GetCreator(primitive_ptr, &desc);
  if (creator == nullptr) {
    MS_LOG(ERROR) << "creator Custom_FT_FFT kernel error";
    return;
  }
  float *input_x_s = new float[num * 2];
  float *input_w = new float[num * 2];
  w_init(input_w, num);
  for (int i = 0; i < num; ++i) {
    input_x[i * 2] = (i % 50) * 0.01 + 0.6;
    input_x[i * 2 + 1] = (i % 50) * 0.01 - 0.7;
    input_x_s[i * 2] = input_x[i * 2];
    input_x_s[i * 2 + 1] = input_x[i * 2 + 1];
  }
  auto context = new mindspore::Context();
  auto ms_in_tensors = LiteTensorsToMSTensors(inputs_);
  auto ms_out_tensors = LiteTensorsToMSTensors(outputs_);
  auto base_kernel = creator(ms_in_tensors, ms_out_tensors, primitive_ptr, context);
  ASSERT_NE(nullptr, base_kernel);
  auto *kernel_exec = new (std::nothrow) kernel::KernelExec(base_kernel);
  auto mskernel = kernel_exec->kernel();
  mskernel->set_name("Custom_FT_FFT");
  mskernel->Prepare();
  mskernel->Execute();
  fft(input_x_s, input_w, num);
  bitrev_r(reinterpret_cast<std::complex<float> *>(input_x_s), num);
  ASSERT_EQ(0, CompareOutputData(output, input_x_s, num * 2, 0.01));
  UninitDSPRuntime();
  delete[] input_x_s;
  delete[] input_w;
  delete context;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel_exec;
}
}  // namespace mindspore::lite::dsp::test
