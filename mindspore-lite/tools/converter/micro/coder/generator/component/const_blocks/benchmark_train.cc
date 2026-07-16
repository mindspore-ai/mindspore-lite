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

#include "coder/generator/component/const_blocks/benchmark_train.h"
#include "coder/generator/component/const_blocks/benchmark_common.h"

namespace mindspore::lite::micro {

// Training benchmark main function
static std::string GetBenchmarkTrainSourceStr() {
  return std::string(R"RAW(/**
 * Copyright 2022-2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License)");
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

void usage() {
  printf(
    "-- mindspore benchmark_train params usage:\n"
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
    .append(benchmark_common::GetPrintTensorHandle(false))
    .append(R"RAW(

int main(int argc, const char **argv) {
  if (argc < 2) {
    printf("input command is invalid\n");
    usage();
    return kMSStatusLiteError;
  }
  printf("=======run benchmark_train======\n");

  MSContextHandle ms_context_handle = MSContextCreate();
  if (argc >= 6) {
    int thread_num = atoi(argv[5]);
    if (thread_num < 1 || thread_num > kMaxThreadNum) {
      printf("Thread number error! It should be greater than 0 and less than 5\n");
      return kMSStatusLiteParamInvalid;
    }
    int bind_mode = 1;
    if (argc >= 7) {
      bind_mode = atoi(argv[6]);
      if (bind_mode < 0 || bind_mode > 2) {
        printf("Thread bind mode error! 0: No bind, 1: Bind hign cpu, 2: Bind mid cpu.\n");
        return kMSStatusLiteParamInvalid;
      }
    }
    if (ms_context_handle) {
      MSContextSetThreadNum(ms_context_handle, thread_num);
      MSContextSetThreadAffinityMode(ms_context_handle, bind_mode);
    }
    printf("context: ThreadNum: %d, BindMode: %d\n", thread_num, bind_mode);
  }

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

  int warm_up_loop_count = 3;
  if (argc >= 8) {
      warm_up_loop_count = atoi(argv[7]);
      if (warm_up_loop_count < 0) {
        printf("The warm up loop count error! Cannot be less than 0.\n");
        return kMSStatusLiteParamInvalid;
      }
  }
  printf("Running warm up loops...\n");
  for (int i = 0; i < warm_up_loop_count; ++i) {
    ret = MSModelRunStep(model_handle, NULL, NULL);
    if (ret != kMSStatusSuccess) {
      MSModelDestroy(&model_handle);
      printf("MSModelRunStep failed, ret: %d", ret);
      return ret;
    }
  }

  if (argc >= 4) {
    int loop_count = atoi(argv[3]);
    printf("\nloop count: %d\n", loop_count);
    uint64_t start_time = GetTimeUs();
    for (int i = 0; i < loop_count; ++i) {
      ret = MSModelRunStep(model_handle, NULL, NULL);
      if (ret != kMSStatusSuccess) {
        MSModelDestroy(&model_handle);
        printf("MSModelRunStep failed, ret: %d", ret);
        return ret;
      }
    }
    uint64_t end_time = GetTimeUs();
    float total_time = (float)(end_time - start_time) / 1000.0f;
    printf("total time: %.5fms, per time: %.5fms\n", total_time, total_time / loop_count);
  }
  ret = MSModelRunStep(model_handle, NULL, NULL);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    printf("MSModelRunStep failed, ret: %d", ret);
    return ret;
  }
  printf("========run train mode success=======\n");
  printf("outputs: \n");
  for (size_t i = 0; i < outputs_handle.handle_num; i++) {
    MSTensorHandle output = outputs_handle.handle_list[i];
    PrintTensorHandle(output);
  }

  ret = MSModelSetTrainMode(model_handle, false);  // when change train mode, outputs handle needs to be refreshed
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    printf("MSModelSetTrainMode failed, ret: %d", ret);
    return ret;
  }
  outputs_handle = MSModelGetOutputs(model_handle);
  if (!outputs_handle.handle_list) {
    printf("MSModelGetOutputs failed, ret: %d", ret);
    return ret;
  }
  ret = MSModelRunStep(model_handle, NULL, NULL);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    printf("MSModelRunStep failed, ret: %d", ret);
    return ret;
  }
  printf("\n========run eval mode success=======\n");
  printf("outputs: \n");
  for (size_t i = 0; i < outputs_handle.handle_num; i++) {
    MSTensorHandle output = outputs_handle.handle_list[i];
    PrintTensorHandle(output);
  }
)RAW")
    .append(benchmark_common::GetCalibrationLogic())
    .append(R"RAW(
  ret = MSModelExportWeight(model_handle, "./export.bin");
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    printf("MSModelExportWeight failed, ret: %d", ret);
    return ret;
  }
  printf("========export weight success=======\n");

  printf("========run success=======\n");
  MSModelDestroy(&model_handle);
  return kMSStatusSuccess;
}
)RAW");
}

const char *benchmark_train_source = []() {
  static std::string s = GetBenchmarkTrainSourceStr();
  return s.c_str();
}();

}  // namespace mindspore::lite::micro
