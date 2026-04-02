---
name: lite-debug-test
description: Debugging, unit testing, benchmarking and performance analysis. Use when running gtest, benchmark tools, profiling latency or accuracy, diagnosing operator precision issues, delegate fallback, or memory leaks.
paths:
  - mindspore-lite/test/**
  - mindspore-lite/tools/benchmark/**
  - mindspore-lite/tools/benchmark_train/**
  - mindspore-lite/python/test/**
---

# MindSpore Lite Debugging, Testing and Performance Analysis

## Testing Framework

Based on Google Test (gtest/gmock), defined in `mindspore-lite/test/`.

### Test Directory Structure

```
mindspore-lite/test/
  ut/
    src/
      api/                    # C/C++ API tests
      registry/               # Custom operator registration tests
      runtime/
        kernel/arm/           # ARM kernel tests
        kernel/cuda/          # CUDA kernel tests
        kernel/dsp/           # DSP kernel tests
        kernel/opencl/        # OpenCL kernel tests
    nnacl/                    # NNACL kernel tests (fp32, int8, infer)
    python/                   # Python API tests
  st/
    java/                     # Java system tests (gradlew)
  config_level0/              # Level 0 baseline tests (~99 .cfg files)
    ascend/                   # Ascend-specific configs
    micro/                    # Micro-specific configs
  config_level1/              # Level 1 extended tests
  runtest.sh                  # Test runner script
```

## Running Unit Tests

```bash
# Build with tests
bash build.sh -I x86_64 -e cpu -t on -j8   # -t on = MSLITE_ENABLE_TESTCASES=ON

# Run all UT
cd output/bin && bash test/runtest.sh

# Run specific tests
./test/ut_linux_x86 --gtest_filter=TestConv2D.*
./test/ut_linux_x86 --gtest_filter=TestMatMulFp32.*

# Python tests
cd mindspore-lite/python && python -m pytest test/ut/python/ -v

# Java tests
cd mindspore-lite/test/st/java && ./gradlew test
```

## System Tests

System tests use `.cfg` configuration files containing model path, input data, expected output (golden data), and accuracy threshold.

```bash
cd test/config_level0   # Basic functionality verification
cd test/config_level1   # Extended functionality verification
```

## Benchmark Tool

`mindspore-lite/tools/benchmark/` provides performance benchmarking and accuracy verification.

### Basic Usage

```bash
# Measure latency
./benchmark --modelFile=model.ms --inputShapes=1,224,224,3

# Specify device and threads
./benchmark --modelFile=model.ms --device=CPU --numThreads=4

# Accuracy verification vs golden data
./benchmark --modelFile=model.ms \
  --inDataFile=input.bin \
  --benchmarkDataFile=golden.bin \
  --accuracyThreshold=0.001

# Warmup and repeat
./benchmark --modelFile=model.ms --warmUpLoopCount=3 --loopCount=100

# Per-kernel timing
./benchmark --modelFile=model.ms --timeProfiling=true

# NPU/DSP device
./benchmark --modelFile=model.ms --device=NPU
./benchmark --modelFile=model.ms --device=DSP
```

### Key Parameters

| Parameter | Default | Description |
|------|---------|-------------|
| `--modelFile` | Required | Path to .ms model |
| `--device` | CPU | CPU / GPU / Kirin NPU / DSP |
| `--numThreads` | 2 | Thread count |
| `--loopCount` | 10 | Test iterations |
| `--warmUpLoopCount` | 3 | Warmup iterations |
| `--accuracyThreshold` | 0.5 | Accuracy threshold (float) |
| `--inDataFile` | Null | Input data; multiple files separated by `,` |
| `--benchmarkDataFile` | Null | Golden output for accuracy comparison |
| `--inputShapes` | Null | NHWC format; dims by `,`; multiple inputs by `:` |
| `--timeProfiling` | false | Print per-kernel timing |
| `--perfProfiling` | false | CPU PMU data (aarch64 only) |
| `--enableFp16` | false | Prefer FP16 operators |
| `--cosineDistanceThreshold` | -1.1 | Cosine distance threshold |
| `--cpuBindMode` | 1 | 0=none, 1=big cores, 2=mid cores |

### Output Metrics

Avg/Max/Min inference time (ms), FPS, memory usage, accuracy vs golden.

### Training Benchmark

```bash
./benchmark_train --modelFile=train_model.ms --epochs=10 --timeProfiling=true
```

### Dump Configuration

```json
{
    "common_dump_settings": {
        "dump_mode": 1,
        "path": "/absolute_path",
        "net_name": "ResNet50",
        "input_output": 0,
        "kernels": ["Default/Conv-op12"]
    }
}
```

- `dump_mode`: 0=all kernels, 1=listed kernels only
- `input_output`: 0=input+output, 1=input only, 2=output only
- Set via: `export MINDSPORE_DUMP_CONFIG=/path/to/data_dump.json`

## Performance Analysis

### CPU Profiling

```bash
bash build.sh -I x86_64 -e cpu -p on -j8   # -p on = MSLITE_ENABLE_PROFILE=ON
export MSLITE_PROFILE=1
./benchmark --modelFile=model.ms
```

### Hotspot Identification

```bash
perf record -g ./benchmark --modelFile=model.ms && perf report
```

### Memory Analysis

```bash
valgrind --leak-check=full ./benchmark --modelFile=model.ms --loopCount=1
```

### GPU Timing (OpenCL)

```bash
export OPENCL_PROFILING=1
./benchmark --modelFile=model.ms --device=GPU
```

## Common Debugging Patterns

### 1. Operator Precision Issues

Use `MSKernelCallBack` for per-node comparison:

```cpp
auto after_call = [&](const std::vector<MSTensor> &inputs,
                      const std::vector<MSTensor> &outputs,
                      const MSCallBackParam &param) -> bool {
  MS_LOG(INFO) << param.node_name_ << " " << param.node_type_;
  return true;
};
model->Predict(inputs, &outputs, nullptr, after_call);
```

### 2. Delegate Fallback

Expected NPU but fell back to CPU:

1. Check delegate initialization success
2. Review unsupported operator list
3. Confirm all operators in delegate support range

### 3. Memory Leaks

1. Valgrind detection
2. Check `Malloc`/`Free` pairing
3. Check Tensor lifecycle management

## Test Build Configuration

Test compilation in `test/CMakeLists.txt` (~302 lines), controlled by `MSLITE_ENABLE_TESTCASES`. Disabled by default.
