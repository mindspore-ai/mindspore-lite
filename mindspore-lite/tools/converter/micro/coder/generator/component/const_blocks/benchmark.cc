/**
 * Copyright 2022-2026 Huawei Technologies Co., Ltd
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

#include "coder/generator/component/const_blocks/benchmark.h"
#include "coder/generator/component/const_blocks/benchmark_common.h"

namespace mindspore::lite::micro {

// Inference benchmark main function
static std::string GetBenchmarkSourceStr() {
  return std::string(R"RAW(/**
 * Copyright 2022-2026 Huawei Technologies Co., Ltd
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

)RAW")
    .append(benchmark_common::GetBenchmarkIncludes())
    .append(R"RAW(

#define kMaxThreadNum 4
#define kBindDefault 1

void usage() {
  printf(
    "-- mindspore benchmark params usage:\n"
    "args[0]: executable file\n"
    "args[1]: inputs binary file\n"
    "args[2]: model weight binary file\n"
    "args[3]: loop count for performance test\n"
    "args[4]: calibration file\n"
    "args[5]: runtime thread num, default is 1\n"
    "args[6]: runtime thread bind mode, 0: No bind, 1: Bind high cpu, 2: Bind mid cpu, default is 1\n"
    "args[7]: warm up loop count, default is 3\n"
    "args[8]: cosine distance threshold, default is 0.9999\n\n");
}

)RAW")
    .append(benchmark_common::GetGetTimeUs())
    .append(R"RAW(
)RAW")
    .append(benchmark_common::GetPrintTensorHandle(true))
    .append(R"RAW(

int main(int argc, const char **argv) {
  if (argc < 2) {
    printf("input command is invalid\n");
    usage();
    return kMSStatusLiteError;
  }
  printf("=======run benchmark======\n");

)RAW")
    .append(benchmark_common::GetContextSetup())
    .append(R"RAW(
)RAW")
    .append(benchmark_common::GetModelLoadLogic("file"))
    .append(R"RAW(
)RAW")
    .append(benchmark_common::GetInputsLoadLogic())
    .append(R"RAW(  MSTensorHandleArray outputs_handle = MSModelGetOutputs(model_handle);
  if (!outputs_handle.handle_list) {
    printf("MSModelGetOutputs failed, ret: %d", ret);
    return ret;
  }
)RAW")
    .append(benchmark_common::GetWarmupAndBenchmark("MSModelPredict"))
    .append(R"RAW(  printf("========run success=======\n");
  printf("\noutputs: \n");
  for (size_t i = 0; i < outputs_handle.handle_num; i++) {
    MSTensorHandle output = outputs_handle.handle_list[i];
    PrintTensorHandle(output);
  }
)RAW")
    .append(benchmark_common::GetCalibrationLogic())
    .append(R"RAW(  printf("========run success=======\n");
  MSModelDestroy(&model_handle);
  return kMSStatusSuccess;
}
)RAW");
}

const char *benchmark_source = []() {
  static std::string s = GetBenchmarkSourceStr();
  return s.c_str();
}();

// Cortex-M benchmark function
static std::string GetBenchmarkSourceCortexStr() {
  return std::string(R"RAW(/**
 * Copyright 2022-2026 Huawei Technologies Co., Ltd
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

#include "benchmark.h"
#include "calib_output.h"
#include "load_input.h"
#include "data.h"
#include "c_api/types_c.h"
#include "c_api/model_c.h"
#include "c_api/context_c.h"
#include "src/tensor.h"
#include <time.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint8_t g_WorkSpace[WORK_SPACE_SIZE];

)RAW")
    .append(benchmark_common::GetPrintTensorHandle(false))
    .append(R"RAW(

int benchmark() {
  int ret;
  printf("========run benchmark======\n");
  printf("========Model build========\n");
  MSModelHandle model_handle = MSModelCreate();
  if (model_handle == NULL) {
    printf("MSModelCreate failed.\n");
    return kMSStatusLiteNullptr;
  }
  size_t workspace_size = MSModelCalcWorkspaceSize(model_handle);
  if (workspace_size > WORK_SPACE_SIZE) {
    printf("This Model inference requires %ul bytes of memory.\n", workspace_size);
    return kMSStatusLiteError;
  }
  MSModelSetWorkspace(model_handle, g_WorkSpace, WORK_SPACE_SIZE);
  ret = MSModelBuild(model_handle, NULL, 0, kMSModelTypeMindIR, NULL);
  if (ret != kMSStatusSuccess) {
    printf("MSModelBuildFromFile failed, ret : %d.\n", ret);
    MSModelDestroy(&model_handle);
    return ret;
  }

  printf("========Load inputs=======\n");
  MSTensorHandleArray inputs_handle = MSModelGetInputs(model_handle);
  if (inputs_handle.handle_list == NULL) {
    printf("MSModelGetInputs failed.");
    MSModelDestroy(&model_handle);
    return kMSStatusLiteError;
  }
  ret = SetDataToMSTensor(&inputs_handle, &g_inputs);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    return ret;
  }
  ret = LoadCalibInputs(&inputs_handle, &g_calib_inputs);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    return ret;
  }

  printf("========Set outputs data pointer=======\n");
  MSTensorHandleArray outputs_handle = MSModelGetOutputs(model_handle);
  if (outputs_handle.handle_list == NULL) {
    printf("MSModelGetOutputs failed.");
    MSModelDestroy(&model_handle);
    return kMSStatusLiteError;
  }
  ret = SetDataToMSTensor(&outputs_handle, &g_outputs);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    return ret;
  }

  printf("========Infer start=======\n");
  ret = MSModelPredict(model_handle, inputs_handle, &outputs_handle, NULL, NULL);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    return ret;
  }

  printf("========Compare outputs=======\n");
  for (size_t i = 0; i < outputs_handle.handle_num; i++) {
    MSTensorHandle output = outputs_handle.handle_list[i];
    PrintTensorHandle(output);
  }

  ret = CompareOutputs(&outputs_handle, &g_calib_outputs);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    return ret;
  }

  printf("========Calib success=======\n");
  MSModelDestroy(&model_handle);
  return kMSStatusSuccess;
}
)RAW");
}

const char *benchmark_source_cortex = []() {
  static std::string s = GetBenchmarkSourceCortexStr();
  return s.c_str();
}();

}  // namespace mindspore::lite::micro
