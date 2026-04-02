# Environment

MindSpore Lite is a C++ project with Python/Java bindings. Build requires CMake 3.22.3+, GCC 7.3+ (C++17), and Android NDK r21e+ for cross-compilation. If any required tool is missing, stop and ask the user to set up the environment. Do NOT try to install tools or find alternatives on your own.

# Build

All build (end-side and cloud-side) is done via `build.sh` (Linux/macOS) or `build.bat` (Windows). You should NEVER run CMake directly unless the user explicitly asks.

```bash
# End-side ARM64 (most common)
bash build.sh -I arm64 -j8

# x86 debug build
bash build.sh -I x86_64 -j8

# Cloud-side inference
bash build.sh -I x86_64 -e cpu -a x64 -j8

# iOS
bash build.sh -I arm64 -T ios -j8

# With test cases enabled
bash build.sh -I x86_64 -t on -j8
```

Always confirm build configuration before running. Key CMake options are controlled via build.sh flags, not directly.

# Testing

Unit tests use Google Test (gtest/gmock). Build with `-t on` to enable test compilation.

```bash
# Run all UT
cd output/bin && bash test/runtest.sh

# Run specific test with gtest filter
./test/ut_linux_x86 --gtest_filter=TestConv2D.*

# Python tests
cd mindspore-lite/python && python -m pytest test/ut/python/ -v

# Java tests
cd mindspore-lite/test/st/java && ./gradlew test
```

For benchmark testing, use the `benchmark` tool in the output package:

```bash
./benchmark --modelFile=model.ms --inputShapes=1,224,224,3 --loopCount=100
```

# Linting and Formatting

Use the project's clang-format configuration:

```bash
# Check format (no changes)
bash scripts/check_clang_format.sh

# Apply formatting
bash scripts/format_source_code.sh

# Format single file
clang-format -i path/to/file.cc
```

Only use `clang-format` for formatting. Do NOT use other formatters.

# Code Style Guidelines

Follow these rules for all code changes in this repository:

- Match existing code style and architectural patterns in the file you are editing.
- Minimize unnecessary comments; code should be self-explanatory. Comments should provide non-obvious context.
- Use `MS_LOG` macros for logging, never `printf`, `cout`, or `std::cerr`.
- Use `mindspore::StatusCode` or `RET_OK`/`RET_ERROR` for error codes, never custom values.
- Use smart pointers (`std::shared_ptr`, `std::unique_ptr`) for heap allocations. Avoid raw `new`/`delete`.
- Validate all external inputs (model files, tensor shapes, user parameters) at system boundaries.
- Internal code can trust internal invariants — don't add redundant validation.
- Prefer snake_case for variables and functions, PascalCase for classes and public API methods.
- File naming: `operator_datatype.cc` (e.g., `conv2d_fp32.cc`, `matmul_int8.cc`).
- Assume the reader has familiarity with MindSpore Lite. Do not over-explain basic concepts.

# Dual Runtime Architecture

MindSpore Lite has two parallel runtimes:

- **LiteRT** (`src/litert/`): End-side inference for mobile/IoT/embedded devices. Uses FlatBuffers `.ms` model format.
- **ExtendRT** (`src/extendrt/`): Cloud-side inference for servers. Uses `.mindir` model format. Supports Ascend, TensorRT, and LLM serving.

They share common tensor/allocator infrastructure but target different deployment scenarios. When working on runtime code, always confirm which runtime (LiteRT or ExtendRT) is the target.

# Commit Messages

Don't commit unless the user explicitly asks you to.

When writing a commit message:

- Focus on the "why" rather than listing every change.
- For large PRs, explain the logical order of changes to review.
- For small PRs, a single concise sentence is sufficient.
- Use English for commit messages.
- Follow the repository's existing commit message style.

# Skills

This project provides the following Claude Code skills:

| Skill | Purpose |
|-------|---------|
| `lite-build` | Build configuration, CMake options, cross-compilation |
| `lite-kernel-dev` | Operator/kernel development, NNACL, delegate mechanism |
| `lite-converter` | Model conversion, parser, optimizer, quantization |
| `lite-debug-test` | Debugging, testing, benchmarking, profiling |
| `lite-cloud-side-infer` | Cloud-side inference (ExtendRT), Ascend, TensorRT, LLM |
| `lite-device-side-infer` | Device-side inference (LiteRT), Android/iOS, Micro, training |
| `lite-code-quality` | Code standards, security checks, CI |

Use these skills when the task matches their description. Each skill provides detailed context and code examples for its domain.
