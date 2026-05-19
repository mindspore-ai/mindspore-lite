---
name: lite-build
description: Build configuration, CMake options, cross-compilation and packaging. Use when building MindSpore Lite, configuring CMake, cross-compiling for ARM/iOS/MCU, packaging release archives, or troubleshooting build errors.
paths:
  - build.sh
  - build.bat
  - scripts/build/**
  - mindspore-lite/CMakeLists.txt
  - cmake/**
---

# MindSpore Lite Build Configuration and Compilation

## Build Entry Point

```
build.sh (Linux/macOS) / build.bat (Windows)
  -> scripts/build/build_lite.sh
       -> CMake (mindspore-lite/CMakeLists.txt)
            -> cmake/package_lite.cmake (packaging)
```

### Quick Build Commands

```bash
# End-side ARM64 (most common)
bash build.sh -I arm64 -j8

# End-side ARM32
bash build.sh -I arm32 -j8

# x86 debug build
bash build.sh -I x86_64 -j8

# iOS
bash build.sh -I arm64 -T ios -j8

# Micro Cortex-M (IoT bare metal)
bash build.sh -I cortex-m -j8
```

## CMake Options Reference

Options defined in `mindspore-lite/CMakeLists.txt`, set via `build.sh` flags or passed directly.

### Basic Options

| CMake Variable | Default | Description |
|-----------|--------|------|
| `MSLITE_ENABLE_SSE` | ON (x86) | Enable SSE instruction set |
| `MSLITE_ENABLE_AVX` | ON (x86) | Enable AVX instruction set |
| `MSLITE_ENABLE_AVX512` | OFF | Enable AVX-512 |
| `MSLITE_ENABLE_FP16` | ON (ARM) | Enable FP16 operators |
| `MSLITE_ENABLE_INT8` | ON | Enable INT8 quantization operators |
| `MSLITE_ENABLE_CONTROL_FLOW` | ON | Support control flow operators (If/While) |
| `MSLITE_ENABLE_DYNAMIC_THREAD` | OFF | Dynamic thread scheduling |
| `MSLITE_ENABLE_WEIGHT_DECODE` | ON | Quantized weight decoding |

### Device-side (LiteRT) Options

| CMake Variable | Default | Description |
|-----------|--------|------|
| `MSLITE_ENABLE_GPU_OPENCL` | OFF | OpenCL GPU backend |
| `MSLITE_ENABLE_GPU_VULKAN` | OFF | Vulkan GPU backend |
| `MSLITE_ENABLE_GPU_TENSORRT` | OFF | TensorRT GPU backend |
| `MSLITE_ENABLE_NPU` | OFF | HiSilicon NPU delegate |
| `MSLITE_ENABLE_COREML` | OFF | Apple CoreML delegate |
| `MSLITE_ENABLE_TRAIN` | OFF | Device-side training support |
| `MSLITE_ENABLE_MICRO` | OFF | Micro code generation |
| `MSLITE_ENABLE_RISCV` | OFF | RISC-V support |

### Debug Options

| CMake Variable | Default | Description |
|-----------|--------|------|
| `MSLITE_ENABLE_TESTCASES` | OFF | Compile UT test cases |
| `MSLITE_ENABLE_DEBUG` | OFF | Debug mode (-g, no optimization) |
| `MSLITE_ENABLE_PROFILE` | OFF | Performance profiling |

## Cross-compilation Configuration

### Android NDK

```bash
export ANDROID_NDK=/path/to/android-ndk-r21e
bash build.sh -I arm64 -j8   # ARM64
bash build.sh -I arm32 -j8   # ARM32
```

NDK r21e+ required, r25c recommended. Version mismatch is the most common build failure cause.

### iOS

```bash
bash build.sh -I arm64 -T ios -j8        # Device
bash build.sh -I x86_64 -T ios -j8       # Simulator
```

Uses `cmake/lite_ios.cmake` toolchain file.

### ARM Linux (aarch64)

```bash
bash build.sh -I arm64 -e cpu -a aarch64 -j8
```

## Incremental Build Acceleration

### ccache

```bash
export CCACHE_DIR=/path/to/ccache
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
```

### ninja (alternative to make)

```bash
cmake -G Ninja .. && ninja -j8
```

ninja's incremental build detection is more precise than make.

### Clearing CMake Cache

```bash
rm -rf build/CMakeCache.txt   # Reconfigure only
rm -rf build/                  # Full cleanup
```

## Packaging Output

Packaging logic in `cmake/package_lite.cmake` (~961 lines).

### Output Package Structure

```
mindspore-lite-{version}-{os}-{arch}.tar.gz
  runtime/
    include/api/          # C++ API headers
    include/c_api/        # C API headers
    lib/libmindspore-lite.so
    third_party/
  tools/
    converter/            # Model conversion tool
    benchmark/            # Performance benchmark tool
    cropper/              # Library cropping tool
  micro/                  # Micro code generation (if enabled)
```

### Package Naming

| Scenario | Package Name Template |
|------|---------|
| Android ARM64 | `mindspore-lite-{version}-android-aarch64.tar.gz` |
| Android ARM32 | `mindspore-lite-{version}-android-armv7.tar.gz` |
| Linux x86_64 | `mindspore-lite-{version}-linux-x64.tar.gz` |
| Windows x86_64 | `mindspore-lite-{version}-win-x64.zip` |

## Common Build Errors

| Error | Solution |
|-------|----------|
| `Could not find Protobuf` | Install protobuf 3.x or use auto-download in `cmake/external_libs/` |
| `protobuf version mismatch 3.x vs 4.x` | MindSpore Lite requires protobuf 3.x; specify path: `cmake -DPROTOBUF_INCLUDE_DIR=...` |
| `android-ndk-rXX does not support XXX` | Use NDK r21e+, check `ANDROID_NDK` env var |
| FlatBuffers version mismatch | Use `third_party/flatbuffers/` bundled version |

## Build Script Reference

| File | Purpose |
|------|------|
| `build.sh` | Main build entry point |
| `build.bat` | Windows build entry point |
| `scripts/build/build_lite.sh` | Lite build orchestration |
| `scripts/build/default_options.sh` | Default build configuration |
| `scripts/build/option_proc_lite.sh` | Lite option processing |
| `scripts/build/process_options.sh` | Command line argument parsing |
| `scripts/build/usage.sh` | Help information |
