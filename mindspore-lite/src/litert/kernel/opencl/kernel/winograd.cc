/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/opencl/kernel/winograd.h"
#include <memory>
#include "src/litert/kernel/opencl/cl/winograd.cl.inc"
#include "nnacl/base/minimal_filtering_generator.h"
#include "nnacl/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
void Align(const std::vector<int> &global, const std::vector<int> &local, cl::NDRange *global_range,
           cl::NDRange *local_range) {
  *local_range = cl::NDRange(local[CLIDX_X], local[CLIDX_Y], local[CLIDX_Z]);
  *global_range = cl::NDRange(UP_ROUND(global[CLIDX_X], local[CLIDX_X]), UP_ROUND(global[CLIDX_Y], local[CLIDX_Y]),
                              UP_ROUND(global[CLIDX_Z], local[CLIDX_Z]));
}

constexpr float Gt[] = {1.0000000000, 1.0000000000, 1.0000000000,  1.0000000000, 1.0000000000,  0.0000000000,
                        0.0000000000, 0.7071067691, -0.7071067691, 1.4142135382, -1.4142135382, 0.0000000000,
                        0.0000000000, 0.4999999702, 0.4999999702,  1.9999998808, 1.9999998808,  1.0000000000};

constexpr float G[] = {1.0000000000, 0.0000000000,  0.0000000000, 1.0000000000, 0.7071067691, 0.4999999702,
                       1.0000000000, -0.7071067691, 0.4999999702, 1.0000000000, 1.4142135382, 1.9999998808,
                       1.0000000000, -1.4142135382, 1.9999998808, 0.0000000000, 0.0000000000, 1.0000000000};
const int F43_IH = 3;
const int F43_IW = 3;
const int F43_OH = 6;
const int F43_OW = 6;

int GenerateWinogradF43Filter(void *src, TypeId dtype, size_t CO, size_t CI, void *dst) {
  // OHWI -> O66I
  if (src == nullptr) {
    MS_LOG(ERROR) << "GenerateWinogradF43Filter src is nullptr.";
    return RET_ERROR;
  }
  if (dtype == kNumberTypeFloat32) {
    int trans_ret = WinogradWeightTransform(reinterpret_cast<const float *>(src), reinterpret_cast<float *>(dst),
                                            nullptr, Gt, 1, F43_OH, F43_IH, CI, CO, false);
    if (trans_ret != NNACL_OK) {
      MS_LOG(ERROR) << "WinogradWeightTransform failed.";
      return RET_ERROR;
    }
    return RET_OK;
  } else if (dtype == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
    auto src_fp16 = reinterpret_cast<float16_t *>(src);
    auto dst_fp32 = reinterpret_cast<float *>(dst);
    std::function<float(int)> access_func = [=](int idx) { return static_cast<float>(src_fp16[idx]); };

    for (size_t co = 0; co < CO; ++co) {
      for (size_t ci = 0; ci < CI; ++ci) {
        float in_vals[F43_IH * F43_IW];
        for (int kh = 0; kh < F43_IH; ++kh) {
          for (int kw = 0; kw < F43_IW; ++kw) {
            const int f_index = ((co * F43_IH + kh) * F43_IW + kw) * CI + ci;
            in_vals[kh * F43_IW + kw] = access_func(f_index);
          }
        }
        auto temp_vals = MatrixMultiply(G, in_vals, F43_OH, F43_IH, F43_IW);
        auto out_vals = MatrixMultiply(temp_vals.data(), Gt, F43_OH, F43_IH, F43_OW);
        for (int kh = 0; kh < F43_OH; ++kh) {
          for (int kw = 0; kw < F43_OW; ++kw) {
            const int f_index = ((co * F43_OH + kh) * F43_OW + kw) * CI + ci;
            dst_fp32[f_index] = out_vals[kh * F43_OW + kw];
          }
        }
      }
    }
#else
    MS_LOG(ERROR) << "filter data type is fp16 but fp16 not enable.";
    return RET_ERROR;
#endif
  } else {
    MS_LOG(ERROR) << "filter dytpe = " << dtype << " is no support.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace

int WinogradOpenCLKernel::BuildKernel() {
  const std::string program_name = "winograd";
  if (!ocl_runtime_->LoadSource(program_name, GetActDefines() + winograd_source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  auto build_options_ext = CreateBuildOptionsExtByDType(this->registry_data_type_);
  auto ret = ocl_runtime_->BuildKernel(kernel_4x4to36_, program_name, "Winograd4x4To36", build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  ret = ocl_runtime_->BuildKernel(
    kernel_, program_name, filter_type_ == MemType::IMG ? "WinogradConv2D_Img" : "WinogradConv2D", build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  ret = ocl_runtime_->BuildKernel(kernel_36to4x4_, program_name, "Winograd36To4x4", build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  return RET_OK;
}

int WinogradOpenCLKernel::InitFilter() {
  if (packed_filter_ != nullptr) {
    MS_LOG(DEBUG) << "filter is already inited";
    return RET_OK;
  }

  auto allocator = ocl_runtime_->GetAllocator();

  // allocate opencl memory: buffer or image2d
  size_t size = 0;
  int Ogroup = 2;
  if (filter_type_ == MemType::IMG) {
    size_t width = F43_OH * F43_OW * UP_ROUND(CI_, CI_TILE);
    size_t height = static_cast<size_t>(CO_SLICES_);
    size_t dtype = use_fp16_ ? CL_HALF_FLOAT : CL_FLOAT;
    size = width * height * CO_TILE * sizeof_FLT_;
    packed_filter_ = allocator->Malloc({width, height, dtype});
    if (packed_filter_ == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return RET_ERROR;
    }
  } else {
    size = UP_DIV(CO_SLICES_, Ogroup) * F43_OH * F43_OW * CI_SLICES_ * Ogroup * CI_TILE * CO_TILE * sizeof_FLT_;
    packed_filter_ = allocator->Malloc(size, MemType::BUF);
    if (packed_filter_ == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return RET_ERROR;
    }
  }

  // rearrange filter
  auto filter_tensor = in_tensors_.at(1);
  void *src_filter_data = stored_filter_ == nullptr ? filter_tensor->data() : stored_filter_;
  MS_ASSERT(src_filter_data);

  std::vector<float> winograd_filter(CO_ * F43_OH * F43_OW * CI_);
  auto ret = GenerateWinogradF43Filter(src_filter_data, filter_tensor->data_type(), CO_, CI_, winograd_filter.data());
  if (ret != RET_OK) {
    return ret;
  }
  void *src_data = winograd_filter.data();

  auto src_dtype = kNumberTypeFloat32;
  auto dst_dtype = use_fp16_ ? kNumberTypeFloat16 : kNumberTypeFloat32;
  std::vector<char> tmp(size, 0);
  if (filter_type_ == MemType::IMG) {
    ConvertFilter(src_data, tmp.data(), src_dtype, dst_dtype, OHWI, OHWIOgroupI4O4, CO_, F43_OH, F43_OW, CI_);
  } else {
    ConvertFilter(src_data, tmp.data(), src_dtype, dst_dtype, OHWI, OHWIOgroupI4O4, CO_, F43_OH, F43_OW, CI_, Ogroup);
  }

  // unmap
  if (filter_type_ == MemType::IMG) {
    ocl_runtime_->WriteImage(packed_filter_, tmp.data());
  } else {
    if (allocator->MapBuffer(packed_filter_, CL_MAP_WRITE, nullptr, true) == nullptr) {
      MS_LOG(ERROR) << "Map Buffer failed.";
      return RET_ERROR;
    }
    memcpy(packed_filter_, tmp.data(), size);
    if (allocator->UnmapBuffer(packed_filter_) != RET_OK) {
      MS_LOG(ERROR) << "UnmapBuffer failed.";
      return RET_ERROR;
    }
  }
  FreeStoredData(stored_filter_);
  return RET_OK;
}

int WinogradOpenCLKernel::AllocateMemory() {
  auto allocator = ocl_runtime_->GetAllocator();
  size_t img_dtype = use_fp16_ ? CL_HALF_FLOAT : CL_FLOAT;

  size_t width = static_cast<size_t>(TILE_HW_);
  size_t height = CI_SLICES_ * F43_OH * F43_OW;
  winograd_mem0_ = allocator->Malloc({width, height, img_dtype});
  if (winograd_mem0_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }

  width = static_cast<size_t>(TILE_HW_);
  height = CO_SLICES_ * F43_OH * F43_OW;
  winograd_mem1_ = allocator->Malloc({width, height, img_dtype});
  if (winograd_mem1_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int WinogradOpenCLKernel::SetConstArgs() {
  int ret = AllocateMemory();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "AllocateMemory failed.";
    return ret;
  }

  int arg_cn = 1;
  cl_int4 input_shape = {batch_size_, OH_, OW_, CI_SLICES_};  // maybe pad=0, so use OH/OW
  if (ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn++, winograd_mem0_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn++, input_shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn++, TILE_HW_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn++, param_->pad_u_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn, param_->pad_l_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }

  arg_cn = 0;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, winograd_mem0_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, winograd_mem1_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, packed_filter_, (filter_type_ == lite::opencl::MemType::BUF)) !=
      CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, TILE_HW_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, CI_SLICES_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn, CO_SLICES_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }

  arg_cn = CLARGSINDEX2;
  cl_int4 output_shape = {batch_size_, OH_, OW_, CO_SLICES_};
  if (ocl_runtime_->SetKernelArg(kernel_36to4x4_, CLARGSINDEX0, winograd_mem1_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, packed_bias_, true) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, output_shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, TILE_HW_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, param_->act_type_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn, alpha_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int WinogradOpenCLKernel::SetGlobalLocal() {
  Align({TILE_HW_, F43_OH, CI_SLICES_}, {8, F43_OW, 4}, &global_4x4to36_, &local_4x4to36_);  // local {8, 6, 4}
  Align({UP_DIV(TILE_HW_, C2NUM), F43_OH * F43_OW, UP_DIV(CO_SLICES_, C2NUM)}, {8, 3, 8},    // local {8, 3, 8}
        &global_range_, &local_range_);
  Align({TILE_HW_, 4, CO_SLICES_}, {4, 4, 8}, &global_36to4x4_, &local_36to4x4_);  // local {4, 4, 8}

  return RET_OK;
}

int WinogradOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " winograd Running!";
  MS_LOG(DEBUG) << "winograd kernel0 Running!";
  if (ocl_runtime_->SetKernelArg(kernel_4x4to36_, CLARGSINDEX0, in_tensors_.front()->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(kernel_4x4to36_, global_4x4to36_, local_4x4to36_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }

  MS_LOG(DEBUG) << "winograd kernel1 Running!";
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &kernel2_event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }

  MS_LOG(DEBUG) << "winograd kernel2 Running!";
  if (ocl_runtime_->SetKernelArg(kernel_36to4x4_, CLARGSINDEX1, out_tensors_.front()->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(kernel_36to4x4_, global_36to4x4_, local_36to4x4_, nullptr, &kernel3_event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

double WinogradOpenCLKernel::GetProfilingTimeMs() {
  if (!ocl_runtime_->isProfiling()) {
    return MAX_PROFILING_TIME_MILLI_SECOND;
  }
  cl_ulong time_start = 0;
  cl_ulong time_end = 0;
  if (event_.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start) != CL_SUCCESS) {
    MS_LOG(ERROR) << "event_ getProfilingInfo CL_PROFILING_COMMAND_START failed, time_start is untrustable.";
  }
  if (event_.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end) != CL_SUCCESS) {
    MS_LOG(ERROR) << "event_ getProfilingInfo CL_PROFILING_COMMAND_END failed, time_end is untrustable.";
  }
  cl_ulong time_ns = time_end - time_start;
  if (kernel2_event_.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start) != CL_SUCCESS) {
    MS_LOG(ERROR) << "kernel2_event_ getProfilingInfo CL_PROFILING_COMMAND_START failed, time_start is untrustable.";
  }
  if (kernel2_event_.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end) != CL_SUCCESS) {
    MS_LOG(ERROR) << "kernel2_event_ getProfilingInfo CL_PROFILING_COMMAND_END failed, time_end is untrustable.";
  }
  time_ns += time_end - time_start;
  if (kernel3_event_.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start) != CL_SUCCESS) {
    MS_LOG(ERROR) << "kernel3_event_ getProfilingInfo CL_PROFILING_COMMAND_START failed, time_start is untrustable.";
  }
  if (kernel3_event_.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end) != CL_SUCCESS) {
    MS_LOG(ERROR) << "evekernel3_event_nt_ getProfilingInfo CL_PROFILING_COMMAND_END failed, time_end is untrustable.";
  }
  time_ns += time_end - time_start;
  return static_cast<double>(time_ns) * 1e-6;
}
}  // namespace mindspore::kernel
