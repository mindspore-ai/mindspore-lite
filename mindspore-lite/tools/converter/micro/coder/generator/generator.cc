/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "coder/generator/generator.h"
#include <sys/stat.h>
#include <set>
#include <fstream>
#include "coder/generator/component/component.h"
#include "coder/generator/component/cmake_component.h"
#include "coder/generator/component/weight_component.h"
#include "coder/generator/component/allocator_component.h"
#include "coder/generator/component/common_component.h"
#include "coder/generator/component/const_blocks/cmake_lists.h"
#include "coder/generator/component/const_blocks/debug_utils.h"
#include "coder/generator/component/const_blocks/load_input.h"
#include "coder/generator/component/const_blocks/calib_output.h"
#include "coder/generator/component/const_blocks/msession.h"
#include "coder/generator/component/const_blocks/mtensor.h"
#include "coder/generator/component/const_blocks/mlog.h"
#include "coder/generator/component/const_blocks/mcontext.h"
#include "coder/generator/component/const_blocks/benchmark.h"
#include "coder/generator/component/const_blocks/benchmark_train.h"
#include "coder/generator/component/const_blocks/license.h"
#include "coder/generator/component/train_component.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/kernel_registry.h"
#include "coder/utils/dir_utils.h"

namespace mindspore::lite::micro {
const char micro_model_define_source[] = R"RAW(
typedef struct {
  void *runtime_buffer;
  bool train_mode;  // true: train mode, false: eval mode
  MSTensorHandleArray inputs;
  MSTensorHandleArray outputs;
)RAW";

const char micro_model_method_source[] = R"RAW(
  ModelBuild build;
  ModelResize resize;
  ModelSetWorkspace set_work_space;
  ModelCalcWorkspaceSize calc_work_space;
  FreeResource free_resource;
)RAW";

const char set_workspace_state[] = R"RAW(
typedef void (*ModelSetWorkspace)(MSModelHandle model, void *workspace, size_t workspace_size);
)RAW";

const char calc_workspace_state[] = R"RAW(
typedef size_t (*ModelCalcWorkspaceSize)(MSModelHandle model);
)RAW";

const char model_resize[] = R"RAW(
typedef MSStatus (*ModelResize)(MSModelHandle model,
                                const MSTensorHandleArray inputs,
                                MSShapeInfo *shape_infos,
                                size_t shape_info_num);
)RAW";

int WriteContentToFile(const std::string &file, const std::string &content) {
  std::ofstream of(file);
  if (of.bad()) {
    MS_LOG(ERROR) << "open file error " << file;
    return RET_ERROR;
  }
  MS_LOG(INFO) << "write " << file;
  of << content;
  of.close();
  return RET_OK;
}

Generator::Generator(std::unique_ptr<CoderContext> ctx) {
  ctx_ = std::move(ctx);
  this->config_ = Configurator::GetInstance();
  this->net_include_file_path_ = config_->code_path() + std::string(kSlash) + kIncludePath + std::string(kSlash);
  this->net_src_file_path_ = config_->code_path() + std::string(kSlash) + kSourcePath + std::string(kSlash);
  this->net_main_file_path_ = config_->code_path() + std::string(kSlash) + kBenchmarkPath + std::string(kSlash);
  this->model_dir_ =
    this->net_src_file_path_ + "model" + std::to_string(ctx_->GetCurModelIndex()) + std::string(kSlash);
  this->net_inc_hfile_ = "net" + std::to_string(ctx_->GetCurModelIndex()) + ".h";
  this->net_src_cfile_ = "net" + std::to_string(ctx_->GetCurModelIndex()) + ".c";
  this->net_weight_hfile_ = "weight" + std::to_string(ctx_->GetCurModelIndex()) + ".h";
  this->net_weight_cfile_ = "weight" + std::to_string(ctx_->GetCurModelIndex()) + ".c";
  this->net_model_cfile_ = "model" + std::to_string(ctx_->GetCurModelIndex()) + ".c";
  origin_umask_ = umask(user_umask_);
  MS_LOG(DEBUG) << "origin umask: " << origin_umask_ << ", user umask: " << user_umask_;
}

Generator::~Generator() { (void)umask(origin_umask_); }

int Generator::CodeSourceCMakeFile() {
  std::string src_cmake_file = net_src_file_path_ + cmake_file_name_;
  std::ofstream ofs(src_cmake_file);
  MS_CHECK_TRUE(!ofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << src_cmake_file;
  MS_CHECK_PTR(ctx_);
  CodeCMakeNetLibrary(ofs, ctx_, config_);
  ofs.close();
  return RET_OK;
}

int Generator::CodeDataCFile() {
  std::string cfile = net_main_file_path_ + "data.c";
  std::ofstream cofs(cfile);
  MS_CHECK_TRUE(!cofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << cfile;
  cofs << g_hwLicense;
  cofs << "#include \"data.h\"\n";
  MS_CHECK_PTR(ctx_);
  auto inputs_num = ctx_->graph_inputs().size();
  auto outputs_num = ctx_->graph_outputs().size();

  cofs << "#define NET_INPUTS_NUM " << inputs_num << "\n";
  cofs << "#define NET_OUTPUTS_NUM " << outputs_num << "\n";
  std::stringstream data_def;
  std::stringstream calib_data_def;
  std::stringstream input_tensors_def;
  std::stringstream output_tensors_def;
  std::stringstream calib_input_tensors_def;
  std::stringstream calib_output_tensors_def;
  for (size_t i = 0; i < inputs_num; i++) {
    Tensor *tensor = ctx_->graph_inputs()[i];
    cofs << "#define NET_INPUT" << i << "_SIZE " << tensor->ElementsNum() << "\n";
    data_def << "float input" << i << "_data[NET_INPUT" << i << "_SIZE];\n";
    calib_data_def << "float calib_input" << i << "_data[NET_INPUT" << i << "_SIZE] = {};\n";
    input_tensors_def << "  {\n"
                      << "    \"" << tensor->tensor_name() << "\",\n"
                      << "    NET_INPUT" << i << "_SIZE,\n"
                      << "    NET_INPUT" << i << "_SIZE * sizeof(float),\n"
                      << "    input" << i << "_data,\n"
                      << "  },\n";
    calib_input_tensors_def << "  {\n"
                            << "    \"" << tensor->tensor_name() << "\",\n"
                            << "    NET_INPUT" << i << "_SIZE,\n"
                            << "    NET_INPUT" << i << "_SIZE * sizeof(float),\n"
                            << "    calib_input" << i << "_data,\n"
                            << "  },\n";
  }
  for (size_t i = 0; i < outputs_num; i++) {
    Tensor *tensor = ctx_->graph_outputs()[i];
    cofs << "#define NET_OUTPUT" << i << "_SIZE " << tensor->ElementsNum() << "\n";
    data_def << "float output" << i << "_data[NET_OUTPUT" << i << "_SIZE];\n";
    calib_data_def << "float calib_output" << i << "_data[NET_OUTPUT" << i << "_SIZE] = {};\n";
    output_tensors_def << "  {\n"
                       << "    \"" << tensor->tensor_name() << "\",\n"
                       << "    NET_OUTPUT" << i << "_SIZE,\n"
                       << "    NET_OUTPUT" << i << "_SIZE * sizeof(float),\n"
                       << "    output" << i << "_data,\n"
                       << "  },\n";
    calib_output_tensors_def << "  {\n"
                             << "    \"" << tensor->tensor_name() << "\",\n"
                             << "    NET_OUTPUT" << i << "_SIZE,\n"
                             << "    NET_OUTPUT" << i << "_SIZE * sizeof(float),\n"
                             << "    calib_output" << i << "_data,\n"
                             << "  },\n";
  }

  cofs << "\n" << data_def.str() << calib_data_def.str();
  cofs << "\nTensor inputs_tensors[NET_INPUTS_NUM] = {\n" << input_tensors_def.str() << "};\n";
  cofs << "\nTensor outputs_tensors[NET_OUTPUTS_NUM] = {\n" << output_tensors_def.str() << "};\n";
  cofs << "\nTensor calib_inputs_tensors[NET_INPUTS_NUM] = {\n" << calib_input_tensors_def.str() << "};\n";
  cofs << "\nTensor calib_outputs_tensors[NET_OUTPUTS_NUM] = {\n" << calib_output_tensors_def.str() << "};\n";

  cofs << "TensorArray g_inputs = {inputs_tensors, NET_INPUTS_NUM};\n";
  cofs << "TensorArray g_outputs = {outputs_tensors, NET_OUTPUTS_NUM};\n";
  cofs << "TensorArray g_calib_inputs = {calib_inputs_tensors, NET_INPUTS_NUM};\n";
  cofs << "TensorArray g_calib_outputs = {calib_outputs_tensors, NET_OUTPUTS_NUM};\n";

  cofs.close();
  return RET_OK;
}

int Generator::CodeBenchmarkHFile(const std::string &file) {
  std::ofstream of(file);
  if (of.bad()) {
    MS_LOG(ERROR) << "open file error " << file;
    return RET_ERROR;
  }
  MS_LOG(INFO) << "write " << file;
  of << R"RAW(/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_MICRO_BENCHMARK_H_
#define MINDSPORE_LITE_MICRO_BENCHMARK_H_
#ifdef __cplusplus
extern "C" {
#endif

)RAW";
  size_t shape_size = 0;
  MS_CHECK_PTR(ctx_);
  std::vector<Tensor *> inputs = ctx_->graph_inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    MS_CHECK_PTR(inputs[i]);
    shape_size += inputs[i]->shape().size();
  }
  std::vector<Tensor *> outputs = ctx_->graph_outputs();
  for (size_t i = 0; i < outputs.size(); ++i) {
    MS_CHECK_PTR(outputs[i]);
    shape_size += outputs[i]->shape().size();
  }
  constexpr int kMicroTensorSize = 36;  // sizeof(MicroTensor) in generated file tensor.h, need to check and update
  size_t workspace_size = UP_ROUND(ctx_->total_buffer_size(), C4NUM) + UP_ROUND(ctx_->weight_buffer_size(), C4NUM) +
                          shape_size * sizeof(int64_t) +
                          (kMicroTensorSize) * (ctx_->graph_inputs().size() + ctx_->graph_outputs().size());

  of << "#define WORK_SPACE_SIZE " << workspace_size << "\n";
  of << R"RAW(

int benchmark();

#ifdef __cplusplus
}
#endif
#endif //MINDSPORE_LITE_MICRO_BENCHMARK_H_

)RAW";
  of.close();
  return RET_OK;
}

void Generator::SelectCMakeListsContent(bool isStaticAlloc, std::string *bench_cmake_lists_txt,
                                        std::string *src_cmake_lists_txt) {
  *bench_cmake_lists_txt = bench_cmake_lists;
  *src_cmake_lists_txt = src_cmake_lists;
  if (config_->target() == kRiscV) {
    if (config_->debug_mode()) {
      *bench_cmake_lists_txt = isStaticAlloc ? bench_cmake_lists_riscv_static_debug : bench_cmake_lists_riscv_debug;
    } else {
      *bench_cmake_lists_txt = isStaticAlloc ? bench_cmake_lists_riscv_static : bench_cmake_lists_riscv;
    }
    *src_cmake_lists_txt = src_cmake_lists_riscv;
  }
}

void Generator::SelectContextAndTensorContent(bool isStaticAlloc, std::string *context_header_txt,
                                              std::string *context_source_txt, std::string *tensor_header_txt,
                                              std::string *tensor_source_txt) {
  if (isStaticAlloc) {
    *context_header_txt = CONTEXT_HEADER_STATIC;
    *context_source_txt = CONTEXT_SOURCE_STATIC;
    *tensor_header_txt = TENSOR_HEADER_STATIC;
    *tensor_source_txt = TENSOR_SOURCE_STATIC;
  } else {
    *context_header_txt = context_header;
    *context_source_txt = context_source;
    *tensor_header_txt = tensor_header;
    *tensor_source_txt = tensor_source;
  }
}

void Generator::ReplaceAllPlaceholders(const std::string &placeholder, const std::string &replacement,
                                       std::string *content) {
  auto pos = content->find(placeholder);
  while (pos != std::string::npos) {
    content->replace(pos, placeholder.length(), replacement);
    pos = content->find(placeholder, pos);
  }
}

void Generator::ApplyStaticAllocPlaceholders(std::string *tensor_source_txt) {
  size_t max_tensor_data_size = 0;
  size_t max_name_len = 0;
  size_t max_shape_dim = 0;
  for (auto *tensor : ctx_->graph_inputs()) {
    if (tensor != nullptr) {
      max_tensor_data_size = std::max(max_tensor_data_size, tensor->Size());
      max_name_len = std::max(max_name_len, tensor->tensor_name().length());
      max_shape_dim = std::max(max_shape_dim, tensor->shape().size());
    }
  }
  for (auto *tensor : ctx_->graph_outputs()) {
    if (tensor != nullptr) {
      max_tensor_data_size = std::max(max_tensor_data_size, tensor->Size());
      max_name_len = std::max(max_name_len, tensor->tensor_name().length());
      max_shape_dim = std::max(max_shape_dim, tensor->shape().size());
    }
  }
  if (max_tensor_data_size == 0) {
    max_tensor_data_size = 1;
  }
  if (max_name_len == 0) {
    max_name_len = 1;
  }
  if (max_shape_dim == 0) {
    max_shape_dim = 1;
  }
  ReplaceAllPlaceholders("__MICRO_TENSOR_STATIC_BUF_SIZE__", std::to_string(max_tensor_data_size), tensor_source_txt);
  ReplaceAllPlaceholders("__MICRO_TENSOR_NAME_MAX_SIZE__", std::to_string(max_name_len + 1), tensor_source_txt);
  ReplaceAllPlaceholders("__MICRO_TENSOR_SHAPE_MAX_SIZE__", std::to_string(max_shape_dim), tensor_source_txt);
}

void Generator::SelectTargetSpecificContent(bool isStaticAlloc, std::string *bench_cmake_lists_txt,
                                            std::string *calib_header_txt, std::string *calib_source_txt,
                                            std::string *load_input_h_txt, std::string *load_input_c_txt,
                                            std::string *benchmark_source_txt, std::string *context_source_txt,
                                            std::string *log_header_txt, std::string *log_source_txt) {
  if (config_->target() == kRiscV && config_->debug_mode()) {
    *log_header_txt = log_header;
    *log_source_txt = log_source;
  }
  if (config_->code_mode() == CodeMode::Train) {
    *benchmark_source_txt = GetBenchmarkTrainSource();
  }
  if (config_->target() == kCortex_M) {
    *bench_cmake_lists_txt = bench_cmake_lists_cortex;
    *calib_header_txt = calib_header_cortex;
    *calib_source_txt = calib_source_cortex;
    *load_input_h_txt = load_input_h_cortex;
    *load_input_c_txt = load_input_c_cortex;
    *benchmark_source_txt = GetBenchmarkSourceCortex();
    *context_source_txt = context_source_cortex;
  } else if (!config_->support_parallel()) {
    *context_source_txt = isStaticAlloc ? CONTEXT_SOURCE_NO_PARALLEL_STATIC : context_source_no_parallel;
  }
}

int Generator::CollectAndWriteConstBlocks(bool isStaticAlloc, const std::string &bench_cmake_lists_txt,
                                          const std::string &calib_header_txt, const std::string &calib_source_txt,
                                          const std::string &load_input_h_txt, const std::string &load_input_c_txt,
                                          const std::string &benchmark_source_txt,
                                          const std::string &src_cmake_lists_txt, const std::string &context_header_txt,
                                          const std::string &context_source_txt, const std::string &tensor_header_txt,
                                          const std::string &tensor_source_txt, const std::string &log_header_txt,
                                          const std::string &log_source_txt) {
  std::vector<std::pair<std::string, std::string>> const_blocks = {
    {config_->code_path() + "/" + "CMakeLists.txt", bench_cmake_lists_txt},
    {net_main_file_path_ + "calib_output.h", calib_header_txt},
    {net_main_file_path_ + "calib_output.c", calib_source_txt},
    {net_main_file_path_ + "load_input.h", load_input_h_txt},
    {net_main_file_path_ + "load_input.c", load_input_c_txt},
    {net_main_file_path_ + "benchmark.c", benchmark_source_txt},
    {net_src_file_path_ + "CMakeLists.txt", src_cmake_lists_txt},
    {net_src_file_path_ + "context.h", context_header_txt},
    {net_src_file_path_ + "context.c", context_source_txt},
    {net_src_file_path_ + "tensor.h", tensor_header_txt},
    {net_src_file_path_ + "tensor.c", tensor_source_txt}};
  if (config_->target() == kRiscV && config_->debug_mode()) {
    const_blocks.emplace_back(std::make_pair(net_src_file_path_ + "log.h", log_header_txt));
    const_blocks.emplace_back(std::make_pair(net_src_file_path_ + "log.c", log_source_txt));
  }
  if (config_->target() == kRiscV && config_->debug_mode()) {
    const_blocks.emplace_back(std::make_pair(net_src_file_path_ + "debug_utils.h", debug_utils_h));
    const_blocks.emplace_back(std::make_pair(net_src_file_path_ + "debug_utils.c", debug_utils_c));
  }
  if (config_->target() == kCortex_M) {
    CodeBenchmarkHFile(net_main_file_path_ + "benchmark.h");
    const_blocks.emplace_back(std::make_pair(net_main_file_path_ + "data.h", data_h_cortex));
    const_blocks.emplace_back(
      std::make_pair(config_->code_path() + "/" + "cortex-m7.toolchain.cmake", cortex_m7_toolchain));
    const_blocks.emplace_back(std::make_pair(config_->code_path() + "/" + "build.sh", cortex_build_sh));
    MS_CHECK_RET_CODE(CodeDataCFile(), "code data c file failed.");
  }
  for (const auto &static_block : const_blocks) {
    MS_CHECK_RET_CODE(WriteContentToFile(static_block.first, static_block.second), "write file failed");
  }
  return RET_OK;
}

int Generator::CodeStaticContent() {
  bool isStaticAlloc = (config_->allocate_memory_type() == AllocateMemoryType::kStatic);

  std::string bench_cmake_lists_txt;
  std::string src_cmake_lists_txt;
  SelectCMakeListsContent(isStaticAlloc, &bench_cmake_lists_txt, &src_cmake_lists_txt);

  std::string context_header_txt;
  std::string context_source_txt;
  std::string tensor_header_txt;
  std::string tensor_source_txt;
  SelectContextAndTensorContent(isStaticAlloc, &context_header_txt, &context_source_txt, &tensor_header_txt,
                                &tensor_source_txt);

  if (isStaticAlloc) {
    ApplyStaticAllocPlaceholders(&tensor_source_txt);
  }

  std::string calib_header_txt = calib_header;
  std::string calib_source_txt = calib_source;
  std::string load_input_h_txt = load_input_h;
  std::string load_input_c_txt = load_input_c;
  std::string benchmark_source_txt = GetBenchmarkSource();
  std::string log_header_txt;
  std::string log_source_txt;
  SelectTargetSpecificContent(isStaticAlloc, &bench_cmake_lists_txt, &calib_header_txt, &calib_source_txt,
                              &load_input_h_txt, &load_input_c_txt, &benchmark_source_txt, &context_source_txt,
                              &log_header_txt, &log_source_txt);

  return CollectAndWriteConstBlocks(isStaticAlloc, bench_cmake_lists_txt, calib_header_txt, calib_source_txt,
                                    load_input_h_txt, load_input_c_txt, benchmark_source_txt, src_cmake_lists_txt,
                                    context_header_txt, context_source_txt, tensor_header_txt, tensor_source_txt,
                                    log_header_txt, log_source_txt);
}

int Generator::CodeCommonModelFile() {
  // model header file
  bool isStaticAlloc = (config_->allocate_memory_type() == AllocateMemoryType::kStatic);
  std::string hfile = net_src_file_path_ + "model.h";
  std::ofstream hofs(hfile);
  MS_CHECK_TRUE(!hofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << hfile;
  hofs << g_hwLicense;
  hofs << "#ifndef MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_MODEL_H_\n"
       << "#define MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_MODEL_H_\n\n";
  if (isStaticAlloc) {
    hofs << "#include \"ai_mcu.h\"\n";
  } else {
    hofs << "#include \"c_api/model_c.h\"\n";
  }
  CodeMSModelBuildState(hofs);
  MS_CHECK_PTR(config_);
  CodeMSModelPredictState(hofs);
  CodeMSModelRunStepState(hofs);
  CodeMSModelSetTrainModeState(hofs);
  CodeMSModelExportWeightState(hofs);
  CodeFreeResourceState(hofs, false);
  hofs << set_workspace_state;
  hofs << calc_workspace_state;
  hofs << model_resize;
  hofs << micro_model_define_source;
  hofs << micro_model_method_source;
  hofs << "  ModelPredict predict;\n";
  hofs << "  ModelRunStep run_step;\n";
  hofs << "  ModelSetTrainMode set_train_mode;\n";
  hofs << "  ModelExportWeight export_weight;\n";
  hofs << "} MicroModel;\n";

  hofs << "void MSTensorHandleArrayDestroy(MSTensorHandleArray inputs);\n";
  if (isStaticAlloc) {
    hofs << "size_t MSModelCalcWorkspaceSize(MSModelHandle model);\n";
    hofs << "void MSModelSetWorkspace(MSModelHandle model, void *workspace, size_t workspace_size);\n";
    hofs << "MSModelHandle MSModelCreate(void);\n";
    hofs << "MSStatus MSModelResize(MSModelHandle model, const MSTensorHandleArray inputs,"
         << " MSShapeInfo *shape_infos,\n"
         << "                       size_t shape_info_num);\n"
         << "MSStatus MSModelBuild(MSModelHandle model, const void *model_data,\n"
         << "                      size_t data_size, MSModelType model_type,\n"
         << "                      const MSContextHandle model_context);\n"
         << "MSTensorHandleArray MSModelGetInputs(const MSModelHandle model);\n"
         << "MSTensorHandleArray MSModelGetOutputs(const MSModelHandle model);\n"
         << "MSTensorHandle MSModelGetInputByTensorName(const MSModelHandle model, const char *tensor_name);\n"
         << "MSTensorHandle MSModelGetOutputByTensorName(const MSModelHandle model, const char *tensor_name);\n"
         << "MSStatus MSModelPredict(MSModelHandle model, const MSTensorHandleArray inputs,"
         << " MSTensorHandleArray *outputs,\n"
         << "                        const MSKernelCallBackC before, const MSKernelCallBackC after);\n"
         << "void MSModelDestroy(MSModelHandle *model);\n";
  }
  hofs << "#endif // MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_MODEL_H_\n\n";

  // model source file
  std::string cfile = net_src_file_path_ + "model.c";
  std::ofstream cofs(cfile);
  MS_CHECK_TRUE(!cofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << cfile;
  cofs << g_hwLicense;
  if (isStaticAlloc) {
    cofs << "#include \"model.h\"\n"
         << "#include \"model_handle.h\"\n"
         << "#include \"context.h\"\n"
         << "#include \"tensor.h\"\n"
         << "#include \"string.h\"\n"
         << "#include \"ai_mcu.h\"\n";
  } else {
    cofs << "#include \"src/model.h\"\n"
         << "#include \"c_api/model_c.h\"\n"
         << "#include \"include/model_handle.h\"\n"
         << "#include \"malloc.h\"\n"
         << "#include \"src/context.h\"\n"
         << "#include \"src/tensor.h\"\n"
         << "#include \"string.h\"\n";
  }
  if (config_->support_parallel()) {
    cofs << "#include \"" << kThreadWrapper << "\"\n";
  }
  if (config_->target() != kCortex_M && !config_->dynamic_shape()) {
    if (isStaticAlloc) {
      cofs << "#include \"allocator.h\"\n";
    } else {
      cofs << "#include \"src/allocator.h\"\n";
    }
  }
  CodeMSModelCalcWorkspaceSize(cofs, ctx_, *config_);
  CodeMSModelSetWorkspace(cofs, ctx_, *config_);
  CodeMSModelCreateDefault(cofs, isStaticAlloc);
  CodeMSModelBuildCommon(cofs, *config_);
  cofs << model_runtime_other_source;
  CodeMSModelPredictCommon(cofs);
  CodeMSModelRunStepCommon(cofs);
  CodeMSModelSetTrainModeCommon(cofs);
  CodeMSModelExportWeightCommon(cofs);
  CodeMSModelDestory(cofs, config_);
  return RET_OK;
}

int Generator::CodeModelHandleHFile() {
  bool isStaticAlloc = (config_->allocate_memory_type() == AllocateMemoryType::kStatic);
  std::string handle_file = net_include_file_path_ + "model_handle.h";
  std::ofstream ofs(handle_file);
  MS_CHECK_TRUE(!ofs.bad(), "failed to open file");
  MS_LOG(INFO) << "write " << handle_file;
  ofs << g_hwLicense;
  ofs << "#ifndef MINDSPORE_LITE_MICRO_LIBRARY_INCLUDE_MODEL_HANDLE_H_\n"
         "#define MINDSPORE_LITE_MICRO_LIBRARY_INCLUDE_MODEL_HANDLE_H_\n\n";
  if (!isStaticAlloc) {
    ofs << "#include \"c_api/model_c.h\"\n\n";
  }
  for (int i = 0; i <= ctx_->GetCurModelIndex(); ++i) {
    ofs << "extern MSModelHandle model" << std::to_string(i) << "; // " << ctx_->model_name() << "\n";
  }
  ofs << "\n#endif  // MINDSPORE_LITE_MICRO_LIBRARY_INCLUDE_MODEL_HANDLE_H_\n";
  return RET_OK;
}

int Generator::CodeLogRegFile() {
  std::string handle_file = net_include_file_path_ + "log_reg.h";
  std::ofstream ofs(handle_file);
  MS_CHECK_TRUE(!ofs.bad(), "failed to open file");
  MS_LOG(INFO) << "write " << handle_file;
  ofs << g_hwLicense;
  ofs << "#ifndef MINDSPORE_LITE_MICRO_LIBRARY_INCLUDE_LOG_REG_H_\n"
         "#define MINDSPORE_LITE_MICRO_LIBRARY_INCLUDE_LOG_REG_H_\n\n"
      << "typedef void (*MSLogPrintFunc)(const char *);\n\n"
      << "void MSRegisterLogFunc(MSLogPrintFunc func);\n\n";
  ofs << "\n#endif  // MINDSPORE_LITE_MICRO_LIBRARY_INCLUDE_LOG_REG_H_\n";
  return RET_OK;
}

void Generator::CodeMSModelDeclarations(std::ofstream &ofs, bool isStaticAlloc) {
  ofs << "MSStatus MSModelBuild" << ctx_->GetCurModelIndex() << "(MSModelHandle model, const void *model_data,\n"
      << "                       size_t data_size, const MSContextHandle model_context);\n";
  ofs << "MSStatus MSModelResize" << ctx_->GetCurModelIndex() << "(MSModelHandle model, \n"
      << "                       const MSTensorHandleArray inputs, MSShapeInfo *shape_infos, size_t shape_info_num);\n";
  if (config_->code_mode() == CodeMode::Inference) {
    ofs << "MSStatus MSModelPredict" << ctx_->GetCurModelIndex()
        << "(MSModelHandle model, const MSTensorHandleArray inputs,\n"
        << "                         MSTensorHandleArray *output,\n"
        << "                         const MSKernelCallBackC before,\n"
        << "                         const MSKernelCallBackC after);\n";
  } else {
    ofs << "MSStatus MSModelRunStep" << ctx_->GetCurModelIndex()
        << "(MSModelHandle model,\n"
           "                       const MSKernelCallBackC before,\n"
           "                       const MSKernelCallBackC after);\n";
    ofs << "MSStatus MSModelSetTrainMode" << ctx_->GetCurModelIndex() << "(MSModelHandle model, bool train);\n";
    ofs << "MSStatus MSModelExportWeight" << ctx_->GetCurModelIndex()
        << "(MSModelHandle model, const char *export_path);\n";
  }
  if (isStaticAlloc) {
    ofs << "void Reset" << ctx_->GetCurModelIndex() << "(void);\n";
  } else {
    ofs << "void Reset" << ctx_->GetCurModelIndex() << "();\n";
  }
  ofs << "void MSModelSetWorkspace" << ctx_->GetCurModelIndex()
      << "(MSModelHandle model, void *workspace, size_t workspace_size);\n";
  ofs << "size_t MSModelCalcWorkspaceSize" << ctx_->GetCurModelIndex() << "(MSModelHandle model);\n";
}

int Generator::CodeMSModelImplement() {
  std::string cfile = model_dir_ + net_model_cfile_;
  std::ofstream ofs(cfile);
  bool isStaticAlloc = (config_->allocate_memory_type() == AllocateMemoryType::kStatic);
  MS_CHECK_TRUE(!ofs.bad(), "failed to open file");
  MS_LOG(INFO) << "write " << cfile;
  ofs << g_hwLicense;
  if (isStaticAlloc) {
    ofs << "#include \"tensor.h\"\n";
    ofs << "#include \"context.h\"\n";
    ofs << "#include \"model.h\"\n";
    ofs << "#include \"" << net_inc_hfile_ << "\"\n";
    if (config_->target() != kCortex_M && !config_->dynamic_shape()) {
      ofs << "#include \"allocator.h\"\n";
    }
    ofs << "#include \"" << net_weight_hfile_ << "\"\n\n";
  } else {
    ofs << "#include \"src/tensor.h\"\n";
    ofs << "#include \"src/context.h\"\n";
    ofs << "#include \"c_api/model_c.h\"\n";
    ofs << "#include \"src/model.h\"\n";
    ofs << "#include \"src/model" << ctx_->GetCurModelIndex() << "/" << net_inc_hfile_ << "\"\n";
    if (config_->target() != kCortex_M && !config_->dynamic_shape()) {
      ofs << "#include \"src/allocator.h\"\n";
    }
    if (config_->support_parallel()) {
      ofs << "#include \"" << kThreadWrapper << "\"\n";
    }
    ofs << "#include \"src/model" << ctx_->GetCurModelIndex() << "/" << net_weight_hfile_ << "\"\n\n";
  }

  if (config_->target() == kCortex_M) {
    ofs << "#define GRAPH_INPUTS_SIZE " << ctx_->graph_inputs().size() << "\n";
    ofs << "#define GRAPH_OUTPUTS_SIZE " << ctx_->graph_outputs().size() << "\n";
    ofs << "#define WEIGHT_BUF_SIZE " << ctx_->weight_buffer_size() << "\n";
  }
  CodeMSModelDeclarations(ofs, isStaticAlloc);
  ofs << "static MicroModel gModel" << ctx_->GetCurModelIndex() << " = {.runtime_buffer = NULL,\n"
      << "                             .train_mode = false,\n"
      << "                             .inputs = {" << ctx_->graph_inputs().size() << ", NULL},\n"
      << "                             .outputs = {" << ctx_->graph_outputs().size() << ", NULL},\n";
  ofs << "                             .build = MSModelBuild" << ctx_->GetCurModelIndex() << ",\n"
      << "                             .resize = MSModelResize" << ctx_->GetCurModelIndex() << ",\n";
  if (config_->code_mode() == CodeMode::Inference) {
    ofs << "                             .predict = MSModelPredict" << ctx_->GetCurModelIndex() << ",\n";
  } else {
    ofs << "                             .run_step = MSModelRunStep" << ctx_->GetCurModelIndex() << ",\n"
        << "                             .set_train_mode = MSModelSetTrainMode" << ctx_->GetCurModelIndex() << ",\n"
        << "                             .export_weight = MSModelExportWeight" << ctx_->GetCurModelIndex() << ",\n";
  }
  if (config_->target() == kCortex_M) {
    ofs << "                             .set_work_space = MSModelSetWorkspace" << ctx_->GetCurModelIndex() << ",\n"
        << "                             .calc_work_space = MSModelCalcWorkspaceSize" << ctx_->GetCurModelIndex()
        << ",\n";
  } else {
    ofs << "                             .set_work_space = NULL,\n"
        << "                             .calc_work_space = NULL,\n";
  }
  if (config_->allocate_memory_type() == AllocateMemoryType::kStatic) {
    ofs << "                             .free_resource = NULL};\n";
  } else {
    ofs << "                             .free_resource = Reset" << ctx_->GetCurModelIndex() << "};\n";
  }
  ofs << "MSModelHandle model" << ctx_->GetCurModelIndex() << " = &gModel" << ctx_->GetCurModelIndex() << ";\n\n";
  auto &dynamic_symbols = config_->dynamic_symbols();
  for (size_t i = 0; i < dynamic_symbols.size(); ++i) {
    ofs << "static int store" << ctx_->GetCurModelIndex() << "_" << i << " = -1;\n";
  }
  CodeResetImplement(ofs, ctx_, *config_);
  CodeMSModelCreate(ofs, ctx_, *config_);
  CodeMSModelBuild(ofs, ctx_->GetCurModelIndex(), weight_size_, *config_);
  CodeMSModelResize(ofs, ctx_, *config_);
  CodeCopyOutputsImplement(ofs, ctx_, *config_);
  if (config_->target() == kCortex_M) {
    CodeCortexCalcWorkspaceSize(ofs, ctx_);
    CodeCortexSetWorkspace(ofs, ctx_);
  }
  if (config_->code_mode() == CodeMode::Train) {
    CodeMSModelRunStep(ofs, ctx_);
    CodeMSModelSetTrainMode(ofs, ctx_);
    CodeMSModelExportWeight(ofs, ctx_->GetCurModelIndex());
  } else {
    CodeMSModelPredict(ofs, ctx_, *config_);
  }
  return RET_OK;
}

int Generator::CodeWeightFile() {
  // weight header file
  std::string hfile = model_dir_ + net_weight_hfile_;
  std::ofstream hofs(hfile);
  MS_CHECK_TRUE(!hofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << hfile;
  CodeWeightFileHeader(hofs, ctx_, *config_);

  // weight source file
  std::string cfile = model_dir_ + net_weight_cfile_;
  std::ofstream cofs(cfile);
  MS_CHECK_TRUE(!cofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << cfile;
  cofs << g_hwLicense;
  if (config_->allocate_memory_type() == AllocateMemoryType::kStatic) {
    cofs << "#include \"" << net_weight_hfile_ << "\"\n";
  } else {
    cofs << "#include \"src/model" << ctx_->GetCurModelIndex() << "/" << net_weight_hfile_ << "\"\n";
  }
  cofs << "#include <stdio.h>\n\n";
  cofs << "int  " << gThreadNum << " = 1; \n";
  std::vector<Tensor *> inputs = ctx_->graph_inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    cofs << "extern const unsigned char *" << ctx_->input_name() + std::to_string(i) << ";\n";
  }
  if ((config_->target() != kCortex_M) && (config_->target() != kRiscV)) {
    cofs << "unsigned char *" << ctx_->buffer_name() << " = 0; \n";
    cofs << "unsigned char *" << ctx_->weight_name() << " = 0; \n";
    cofs << "int *" << kShapePrefixName << " = 0; \n";
    cofs << "int *" << kOffsetPrefixName << " = 0; \n";
    std::string net_file = model_dir_ + "net" + std::to_string(ctx_->GetCurModelIndex()) + ".bin";
    SaveDataToNet(ctx_, net_file, config_->keep_original_weight(), &weight_size_);
  } else {
    if (!ctx_->weight_buffer_size_code_blocks().empty()) {
      MS_LOG(ERROR) << "Weight init code generation error ";
      return RET_ERROR;
    }
    cofs << "int __errno; \n";
    cofs << "unsigned char * " << ctx_->buffer_name() << " = NULL; \n";
    cofs << "unsigned char * " << ctx_->weight_name() << " = NULL; \n";
    CodeModelParamsData(cofs, ctx_->saved_weights());
  }
  CodeModelParamsForNet(hofs, cofs, ctx_, *config_);
  CodeInitWeightState(hofs, ctx_->GetCurModelIndex());
  if (config_->code_mode() == CodeMode::Train) {
    CodeWeightInitFuncForTrain(cofs, ctx_);
    CodeExportWeightState(hofs, ctx_->GetCurModelIndex());
    CodeWeightExportFunc(cofs, ctx_, *config_);
  } else {
    CodeWeightInitFunc(cofs, ctx_, *config_);
  }
  hofs.close();
  cofs.close();
  return RET_OK;
}

void Generator::CodeCommonNetH(std::ofstream &ofs, const std::string &execute_param_name) {
  ofs << g_hwLicense;
  // Execute() prototypes use bool; a minimal graph (e.g. a lone Reshape) collects no nnacl
  // header that pulls stdbool in, so include it explicitly.
  ofs << "#include <stdbool.h>\n\n";
  ofs << kExternCpp;
  CodeInputState(ofs, ctx_->GetCurModelIndex());
  if (is_get_quant_args_) {
    CodeGraphQuantArgsState(ofs, ctx_->GetCurModelIndex());
  }
  CodeManageResourceState(ofs, ctx_->GetCurModelIndex(), *config_);
  CodeExecuteState(ofs, ctx_->GetCurModelIndex(), *config_, execute_param_name);
}

void Generator::CodeCommonNetC(std::ofstream &ofs) {
  ofs << g_hwLicense << "\n";
  if (config_->allocate_memory_type() == AllocateMemoryType::kStatic) {
    ofs << "#include \"" << net_weight_hfile_ << "\"\n"
        << "#include \"" << net_inc_hfile_ << "\"\n\n";
  } else {
    ofs << "#include \"src/model" << ctx_->GetCurModelIndex() << "/" << net_weight_hfile_ << "\"\n"
        << "#include \"src/model" << ctx_->GetCurModelIndex() << "/" << net_inc_hfile_ << "\"\n\n";
  }
  if ((config_->target() == kRiscV) && config_->debug_mode()) {
    ofs << "#include \"src/log.h"
        << "\"\n\n";
  }
  if (config_->support_parallel()) {
    ofs << "#include \"" << kThreadWrapper << "\"\n\n";
  }
  if (config_->target() != kRiscV && config_->debug_mode()) {
    ofs << "#include \"src/" << kDebugUtils << "\"\n";
  }
  CodeGlobalCodeBlocks(ofs, ctx_);
  CodeInputImplement(ofs, ctx_);
  CodeInitResourceImplement(ofs, ctx_, *config_);
  CodeFreeResourceImplement(ofs, ctx_, *config_);
  if (is_get_quant_args_) {
    CodeGraphQuantArgsImplement(ofs, ctx_);
  }
}

int Generator::CodeRegKernelHFile() {
  if (!KernelRegistry::GetInstance()->HasKernelRegistered()) return RET_OK;
  if (!KernelRegistry::GetInstance()->CheckRegistered(schema::PrimitiveType_Custom)) {
    MS_LOG(ERROR) << "Only support custom kernel to register now!";
    return RET_ERROR;
  }

  std::string reg_kernel_header = net_src_file_path_ + "registered_kernel.h";
  std::ofstream cofs(reg_kernel_header);
  MS_CHECK_TRUE(!cofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << reg_kernel_header;
  cofs << g_hwLicense;
  cofs << "#include \"nnacl_c/tensor_c.h\"\n";
  cofs << "#include \"nnacl_c/custom_parameter.h\"\n\n";
  cofs << KernelRegistry::GetInstance()->GenKernelInterface(kCustomKernelName, kCustomKernelParam) << "\n";
  return RET_OK;
}

void Generator::CodeAllocatorSourceContent(std::ofstream &cofs, bool isStaticAlloc) {
  if (config_->target() != kRiscV) {
    if (isStaticAlloc) {
      cofs << "#include \"allocator.h\"\n"
           << "#include \"stdatomic.h\"\n"
           << "#include \"stdlib.h\"\n"
           << "#include \"string.h\"\n"
           << "#include <stdbool.h>\n\n";
      cofs << "static char g_memory_pool[sizeof(MemBlock) + " << ctx_->max_buffer_size() << "];\n";
      cofs << "static MemBlock g_mem_block;\n";
      cofs << "MemBlock *mem_block = NULL;\n"
           << "atomic_int kReferenceCount = 0;\n"
           << "#ifdef __clang__\n"
           << "  bool kLock = false;\n"
           << "#else\n"
           << "  bool kLock = false;\n"
           << "#endif\n";
    } else {
      cofs << "#include \"src/allocator.h\"\n"
           << "#include \"stdatomic.h\"\n"
           << "#include \"stdlib.h\"\n"
           << "#include <stdbool.h>\n\n";
      cofs << "MemBlock *mem_block = NULL;\n"
           << "atomic_int kReferenceCount = 0;\n"
           << "#ifdef __clang__\n"
           << "  atomic_bool kLock = false;\n"
           << "#else\n"
           << "  bool kLock = false;\n"
           << "#endif\n";
    }
  } else {
    if (isStaticAlloc) {
      cofs << "#include \"allocator.h\"\n"
           << "#include \"stdatomic.h\"\n"
           << "#include \"stdlib.h\"\n"
           << "#include \"string.h\"\n"
           << "#include <stdbool.h>\n\n";
      cofs << "static char g_memory_pool[sizeof(MemBlock) + " << ctx_->max_buffer_size() << "];\n";
      cofs << "static MemBlock g_mem_block;\n";
    } else {
      cofs << "#include \"src/allocator.h\"\n"
           << "#include \"stdatomic.h\"\n"
           << "#include \"stdlib.h\"\n"
           << "#include <stdbool.h>\n\n";
    }
    cofs << "MemBlock *mem_block = NULL;\n";
  }
}

int Generator::CodeAllocatorFile() {
  if (config_->target() == kCortex_M || config_->dynamic_shape()) {
    return RET_OK;
  }
  bool isStaticAlloc = (config_->allocate_memory_type() == AllocateMemoryType::kStatic);

  std::string hfile = net_src_file_path_ + "allocator.h";
  std::ofstream hofs(hfile);
  MS_CHECK_TRUE(!hofs.bad(), "failed to open file");
  MS_LOG(INFO) << "write " << hfile;
  CodeAllocatorFileHeader(hofs);

  std::string cfile = net_src_file_path_ + "allocator.c";
  std::ofstream cofs(cfile);
  MS_CHECK_TRUE(!cofs.bad(), "failed to open file");
  MS_LOG(INFO) << "write " << cfile;
  cofs << g_hwLicense;
  CodeAllocatorSourceContent(cofs, isStaticAlloc);
  CodeCalcRefCount(cofs);
  CodeGlobalMemory(cofs, ctx_->max_buffer_size());
  CodeMemoryOccupied(cofs);
  CodeLockOccupied(cofs, isStaticAlloc);
  hofs.close();
  cofs.close();
  return RET_OK;
}

int Generator::CreateCommonFiles() {
  MS_CHECK_RET_CODE(CodeStaticContent(), "code static content failed.");
  MS_CHECK_RET_CODE(CodeModelHandleHFile(), "code model_handle h file failed.");
  if (config_->target() == kRiscV) {
    MS_CHECK_RET_CODE(CodeLogRegFile(), "code log_reg h file failed.");
  }
  MS_CHECK_RET_CODE(CodeCommonModelFile(), "code common model file failed.");
  if (!config_->dynamic_shape()) {
    MS_CHECK_RET_CODE(CodeAllocatorFile(), "code allocator file failed.");
  }
  MS_CHECK_RET_CODE(CodeRegKernelHFile(), "code registered kernel header file failed.");
  MS_CHECK_RET_CODE(CodeSourceCMakeFile(), "code net cmake file failed.");
  return RET_OK;
}

int Generator::CreateModelFiles() {
  MS_CHECK_RET_CODE(CodeNetHFile(), "code net h file failed.");
  MS_CHECK_RET_CODE(CodeNetCFile(), "code net c file failed.");
  MS_CHECK_RET_CODE(CodeWeightFile(), "code weight file failed.");
  MS_CHECK_RET_CODE(CodeMSModelImplement(), "code session file failed.");
  return RET_OK;
}

int Generator::GenerateCode() {
  auto ret = CreateModelFiles();
  MS_CHECK_RET_CODE(ret, "Create model files failed.");
  MS_CHECK_PTR(ctx_);
  MS_CHECK_PTR(config_);
  if (ctx_->end_flag()) {
    ret = CreateCommonFiles();
    MS_CHECK_RET_CODE(ret, "Create common files failed.");
  }
  return RET_OK;
}
}  // namespace mindspore::lite::micro
