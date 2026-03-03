#!/bin/bash
set -euo pipefail  # Exit on error, undefined variables, or pipeline failures

#######################################
# Configuration & Initialization
#######################################

# Get absolute path of script directory
basepath=$(pwd)

echo ${basepath}

BASE_DIR=${basepath}/../../../../
VERSION_PATH="${BASE_DIR}/version.txt"

echo ${BASE_DIR}

if [ -f "$VERSION_PATH" ]; then
    cd ${BASE_DIR}
else
    echo "not the right project dir"
    exit 1
fi

UT_DIR="${BASE_DIR}/ut"
LIB_DIR="${UT_DIR}/riscv_libs/lib"

rm -rf ${UT_DIR}

# Verify RISC-V toolchain availability
if ! command -v riscv64-unknown-linux-gnu-gcc &>/dev/null; then
    echo "[FATAL] riscv64-unknown-linux-gnu-gcc not found. Please install RISC-V toolchain." >&2
    exit 1
fi

SYSROOT="$(riscv64-unknown-linux-gnu-gcc -print-sysroot)"

#######################################
# Compile
#######################################
#rm -rf build

#TARGET_FILE="mindspore-lite/providers/flatbuffer/native_flatbuffer.cfg"
#rm -f "${TARGET_FILE}"
#echo "/opt/flatbuffers" > "${TARGET_FILE}"

#export MSLITE_CROSS_RISCV=on
#./build.sh -I x86_64 -j$(nproc)

#rm -f "${TARGET_FILE}"

#######################################
# Helper Functions
#######################################

log_info()  { echo "[INFO]  $*" >&2; }
log_warn()  { echo "[WARN]  $*" >&2; }
log_error() { echo "[ERROR] $*" >&2; }
log_fatal() { echo "[FATAL] $*" >&2; exit 1; }

copy_files() {
    local src_pattern="$1" dest_dir="$2"
    if ! compgen -G "${src_pattern}" &>/dev/null; then
        log_warn "No files match pattern: ${src_pattern}"
        return 1
    fi
    for file in ${src_pattern}; do
        if ! cp -fpPR "${file}" "${dest_dir}/" 2>/dev/null; then
            log_fatal "Failed to copy: ${file} -> ${dest_dir}"
        fi
    done
}

#######################################
# Prepare Test Environment
#######################################

log_info "Setting up test environment in: ${UT_DIR}"
mkdir -p "${LIB_DIR}"

#######################################
# Copy RISC-V System Libraries
#######################################

log_info "Copying system libraries from sysroot: ${SYSROOT}"

declare -a REQUIRED_LIBS=(
    "ld-linux-riscv64-lp64d.so.1"
    "libc.so.6"
    "libm.so.6"
    "libgcc_s.so*"
    "libstdc++.so.6*"
    "libatomic.so*"
)

for lib_pattern in "${REQUIRED_LIBS[@]}"; do
    copy_files "${SYSROOT}/lib/${lib_pattern}" "${LIB_DIR}" || true
done

#######################################
# Locate and Copy MindSpore Lite Package
#######################################

log_info "Searching for build artifacts in output/ directory"

if ! compgen -G "output/*.tar.gz" &>/dev/null; then
    log_fatal "No *.tar.gz found in output/ directory. Build may have failed."
fi

TAR_PATH=$(compgen -G "output/*.tar.gz" | head -n1)
TAR_NAME=$(basename "${TAR_PATH}")
MS_DIR_NAME="${TAR_NAME%.tar.gz}"

log_info "Found package: ${TAR_NAME}"
log_info "Expected extraction directory: ${MS_DIR_NAME}"

# Copy package and test binary
copy_files "${TAR_PATH}" "${UT_DIR}"
copy_files "build/test/lite-test" "${UT_DIR}"

# Copy GoogleTest dependencies
GTEST_LIB_DIR="build/googletest/googlemock/gtest"
if [ -d "${GTEST_LIB_DIR}" ]; then
    log_info "Copying GoogleTest libraries"
    copy_files "${GTEST_LIB_DIR}/*.so" "${LIB_DIR}" || true
else
    log_warn "GoogleTest library directory not found: ${GTEST_LIB_DIR}"
fi

#######################################
# Extract and Integrate Runtime Libraries
#######################################

cd "${UT_DIR}"

log_info "Extracting package: ${TAR_NAME}"
if ! tar -zxf "${TAR_NAME}" &>/dev/null; then
    log_fatal "Failed to extract ${TAR_NAME}"
fi

if [ ! -d "${MS_DIR_NAME}" ]; then
    log_fatal "Extraction failed: directory '${MS_DIR_NAME}' not found"
fi

# Copy MindSpore Lite runtime libraries
if [ -d "${MS_DIR_NAME}/runtime/lib" ]; then
    log_info "Copying MindSpore Lite runtime libraries"
    copy_files "${MS_DIR_NAME}/runtime/lib/*" "${LIB_DIR}"
else
    log_warn "Runtime library directory not found: ${MS_DIR_NAME}/runtime/lib"
fi

# Cleanup temporary files
rm -f "${TAR_NAME}"
rm -rf "${MS_DIR_NAME}"

#######################################
# Execute Unit Tests in QEMU
#######################################

log_info "Preparing to execute RISC-V unit tests"

# Verify QEMU availability
if ! command -v qemu-riscv64 &>/dev/null; then
    log_fatal "qemu-riscv64 not found. Please install QEMU RISC-V emulator."
fi

# Verify test binary exists
if [ ! -x "./lite-test" ]; then
    log_fatal "Test binary not found or not executable: ./lite-test"
fi

# Set library path for QEMU (parent of lib_dir = riscv_libs)
QEMU_LIB_PATH="$(dirname "${LIB_DIR}")"

log_info "Launching tests with QEMU (sysroot: ${QEMU_LIB_PATH})"

declare -a TC_LIST=(
    "TestMatMulFp32*"
    "TestConvolutionFp32*"
)

tcs="$(IFS=:; echo "${TC_LIST[*]}")"

qemu-riscv64 -cpu rv64,v=true,vlen=128 -L "${QEMU_LIB_PATH}" -- ./lite-test --gtest_filter=${tcs}

log_info "All unit tests completed successfully"
cd "${basepath}"
exit 0
