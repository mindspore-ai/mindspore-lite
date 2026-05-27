# Using MindSpore Lite Securely

## Security Model Overview

MindSpore Lite is a lightweight inference framework. Model files (`.ms` or `.mindir` format) are programs expressed as computation graphs that the framework executes, containing operator definitions and weight data. **Loading model files from untrusted sources is equivalent to running untrusted code**, which may lead to memory out-of-bounds access, segmentation faults, and other security issues.

MindSpore Lite does not provide built-in encryption or decryption capabilities. Securing model files is the user's responsibility. This document describes the security boundaries of MindSpore Lite and provides security recommendations for different usage scenarios.

## Model File Security

### Model File Risks

MindSpore Lite supports loading the following model file formats:

- `.ms` format: Based on FlatBuffers serialization, a lightweight model format for on-device inference
- `.mindir` format: Based on Protocol Buffers serialization, the native MindSpore model format for cloud-side inference

Model files contain computation graph structures and weight parameters, which introduce the following security risks:

- **Malicious construction**: Attackers may craft model files with invalid operator parameters (e.g., invalid offsets, out-of-bounds dimension values), leading to memory out-of-bounds access or segmentation faults
- **Man-in-the-middle tampering**: Model files may be tampered with during transmission or storage, injecting malicious computation graph nodes or modifying weight parameters
- **Model reverse engineering**: Unprotected model files may be reverse engineered, leaking model structure and intellectual property

### Model Source Verification

To ensure model file authenticity, the following measures are recommended:

1. **Trusted sources**: Only use model files from trusted sources and avoid loading models from unknown origins
2. **Integrity verification**: Perform integrity checks on model files (e.g., SHA256 hash comparison) to ensure files have not been tampered with
3. **Secure conversion**: Ensure the environment is secure during model conversion and prevent the conversion toolchain from being compromised

### Model Protection Recommendations

MindSpore Lite does not provide built-in encryption or decryption capabilities. Model file protection is the user's responsibility:

1. **Encryption**: Implement encryption protection during model distribution and deployment to prevent model files from being reverse engineered or tampered with
2. **Key management**: Do not hardcode keys in source code. Use secure key storage solutions (e.g., OS key management services, hardware security modules)
3. **Secure transmission**: Use secure channels (e.g., HTTPS) for model file distribution to prevent man-in-the-middle attacks

## Input Data Security

### Untrusted Input Risks

Input data during inference (images, text, audio, etc.) from untrusted sources introduces the following security risks:

- Maliciously crafted input data may trigger boundary check vulnerabilities within the framework
- Abnormal tensor shapes, data types, or value ranges may cause buffer overflows or illegal memory access
- Specifically crafted inputs may cause undefined behavior in certain operators

### Input Handling Recommendations

To mitigate input data security risks, the following measures are recommended:

1. **Format validation**: Strictly validate input data by checking tensor shapes, data types, and value ranges against model expectations
2. **Image inputs**: Validate resolution and channel count for image inputs, rejecting inputs that exceed reasonable ranges
3. **Text inputs**: Apply length limits and character filtering to text inputs, preventing exceptions caused by excessively long inputs
4. **Pre-processing validation layer**: Add a pre-processing validation layer before passing user inputs to the model, intercepting invalid inputs

### Safe Input Formats

Security risks vary across input formats:

- **Low-risk formats**: Decoding libraries for common image formats such as PNG, BMP, JPEG, and RAW have been extensively tested and patched, presenting lower risk
- **High-risk formats**: Custom formats or complex input parsing logic may contain unknown vulnerabilities and should be run in a sandboxed environment

## Memory and Runtime Security

### Memory Safety Risks

MindSpore Lite has the following memory safety risks at runtime:

- **Operator parameters**: Some parameters in operator implementations are read directly from model files and used for memory access (e.g., vector offsets). Invalid values may cause out-of-bounds read/write operations
- **Memory management**: Tensor memory allocation and deallocation must be correctly paired. Improper usage may lead to memory leaks or use-after-free issues

### Runtime Security Recommendations

1. **Sandbox isolation**: When running models from untrusted sources, execute inference in a sandboxed environment (e.g., containers, virtual machines) to limit the impact of abnormal behavior on the host system
2. **Exception handling**: Wrap MindSpore Lite API calls with C++ exception handling mechanisms to improve application robustness. Catch exceptions during critical operations such as model loading and inference to prevent application crashes
3. **Resource limits**: Set reasonable resource limits (e.g., maximum memory usage) in production environments to prevent resource exhaustion caused by abnormal inputs
4. **Lifecycle management**: Ensure correct model lifecycle management — the sequence of model loading, inference, and release must strictly follow API specifications to avoid accessing resources after model release
5. **Thread safety**: Avoid sharing the same model instance for concurrent inference in multi-threaded environments unless using thread-safe API interfaces

### Operator Parameter Security

For performance reasons, some operator implementations within the framework perform only essential boundary validation on model parameters in hot code paths. Therefore:

- **Model validation**: Users should thoroughly validate models before deployment to ensure operator parameter values are within reasonable ranges
- **Training-stage assurance**: Ensure that parameter values produced during the training stage are within reasonable ranges, preventing abnormal parameters from propagating to the inference stage
- **Custom operators**: For custom operators, users are responsible for ensuring the validity and verification of parameter inputs
