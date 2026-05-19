---
name: lite-code-quality
description: Code formatting, naming conventions, security checks and CI verification. Use when running clang-format, checking code style, writing secure code for model parsing, reviewing code quality, or configuring CI/Jenkins pipelines.
paths:
  - .clang-format
  - scripts/check_clang_format.sh
  - scripts/format_source_code.sh
  - scripts/pre-push
  - .jenkins/**
---

# MindSpore Lite Code Standards and Security Checks

## Formatting Standards

### clang-format

The project uses `.clang-format` for C/C++ formatting. Core rules:

| Rule | Setting |
|------|------|
| Indent width | 2 spaces |
| Brace style | Allman (new line) |
| Line width limit | 120 characters |
| Pointer/reference alignment | Left (`int *p`) |
| include sorting | Grouped by angle brackets/project |

### Formatting Commands

```bash
bash scripts/format_source_code.sh       # Format all source files
bash scripts/check_clang_format.sh       # Check formatting (no modifications)
clang-format -i path/to/file.cc          # Format single file
```

## CI Checks

### Jenkins CI

`.jenkins/` defines CI pipelines:

```
.jenkins/
  check/        # Code check rules
  rules/        # Build rules
  task/         # Build task definitions
```

CI check items:

1. clang-format format check
2. Compilation warnings (`-Werror` enabled in CI)
3. Unit tests
4. Model benchmark regression tests

### Local pre-push Hook

`scripts/pre-push` runs checks before pushing:

```bash
cp scripts/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

Checks: clang-format, blocked branches, commit message format.

## Naming Conventions

### File Naming

| Type | Rule | Example |
|------|------|------|
| Kernel implementation | `operator_datatype.cc` | `conv2d_fp32.cc`, `matmul_int8.cc` |
| NNACL C files | `operator_datatype.c` | `conv_fp32.c`, `matmul_fp16.c` |
| Header files | Corresponds to implementation | `conv2d_fp32.h` |
| Test files | `test_module.cc` or `module_test.cc` | `test_conv2d.cc` |

### Class Naming

```cpp
// PascalCase
class LiteKernel;
class ModelImpl;
class AscendDeviceInfo;

// Kernel class: operator + architecture + data type
class Conv2DCPUKernel;
class MatMulOpenCLKernel;
```

### Function Naming

```cpp
// Public API: PascalCase
Model::Build();
Context::SetThreadNum();

// Internal: snake_case
int init_executor();
void free_buffer();

// NNACL pure C: snake_case + module prefix
int NnaclConvFp32();
```

### Variable Naming

```cpp
int thread_num;                      // local: snake_case
class LiteKernel { int thread_num_; }; // member: trailing underscore
const int kMaxThreadNum = 8;          // constant: k + PascalCase
enum class DataType : int {            // enum: k + PascalCase
  kFloat32 = 0,
  kFloat16 = 1,
};
```

## Security Checks

### Input Validation

```cpp
// FlatBuffers boundary check
const auto *model = flatbuffers::GetMutableRoot<Model>(buffer);
if (!model || !model->Verify(flatbuffers::Verifier(buffer, size))) {
  return RET_ERROR;
}

// Tensor shape validation
if (input_tensor.Shape().size() != expected_dims) {
  MS_LOG(ERROR) << "Invalid input shape";
  return RET_ERROR;
}
```

### Memory Safety

```cpp
auto tensor = std::make_shared<Tensor>();               // Smart pointers
auto buffer = std::unique_ptr<float[]>(new float[size]);

if (input_size > buffer_capacity) { return RET_ERROR; } // Buffer overflow
size_t total = num_elements * sizeof(float);
if (total / sizeof(float) != num_elements) { return RET_ERROR; } // Integer overflow
```

## Common Security Risks

1. **Model file parsing** -- Treat as untrusted input. FlatBuffers `Verify()` is first defense.
2. **Uninitialized memory** -- Always zero-initialize: `new float[size]()` or `std::vector<float>(size, 0.0f)`.
3. **Race conditions** -- Model objects must not be shared across threads. Context is read-only after Build.
4. **Operator security** -- Check divide-by-zero, numeric overflow, dimension validity.

## Dependency Security

```
third_party/
  securec/        # Security utility library
  eigen/          # Linear algebra (with patches)
  protobuf/       # Protobuf 3.x (with patches)
  ...
```

- Monitor CVE advisories for third-party dependencies
- Update patches when upgrading dependencies

## Code Review Checklist

- [ ] `clang-format` format check passes
- [ ] No compilation warnings (`-Werror`)
- [ ] New code has corresponding unit tests
- [ ] Error codes use `mindspore::StatusCode` not custom values
- [ ] Logging uses `MS_LOG` macros not `printf/cout`
- [ ] External inputs have validation checks
- [ ] Memory allocations have failure handling
- [ ] Thread safety: no shared mutable state or properly locked
