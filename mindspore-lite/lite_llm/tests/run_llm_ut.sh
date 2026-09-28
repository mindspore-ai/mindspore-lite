#!/usr/bin/env bash
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eo pipefail

TESTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd -- "$TESTS_DIR/.." && pwd)"
OUTPUT_DIR="$PWD/ut_results"
JOBS=8

while (( $# )); do
  case "$1" in
    -h|--help)
      echo "Usage: bash run_llm_ut.sh [--output-dir DIR] [--jobs N]"
      echo "Runs all Python UT under tests/py and all C++ UT registered with CTest."
      echo "Output directory: ./ut_results by default; each run uses a new run_* directory."
      exit 0 ;;
    --output-dir|--jobs)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || {
        echo "Missing value: $1" >&2
        exit 1
      }
      case "$1" in
        --output-dir) OUTPUT_DIR="$2" ;;
        --jobs) JOBS="$2" ;;
      esac
      shift 2 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1 ;;
  esac
done

[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || {
  echo "[UT] ERROR: --jobs must be a positive integer" >&2
  exit 1
}

for command in python cmake ctest; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "[UT] ERROR: required command is unavailable: $command" >&2
    exit 1
  }
done

mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath -e -- "$OUTPUT_DIR")
RUN_DIR=$(mktemp -d "$OUTPUT_DIR/run_XXXXXXXX")
BUILD_DIR="$RUN_DIR/cpp_build"
echo "[UT] Results: $RUN_DIR"

on_exit()
{
  local status=$?
  printf '%s\n' "$status" > "$RUN_DIR/exit_code.txt"
  echo "[UT] Exit code: $status; results: $RUN_DIR"
}
trap on_exit EXIT

export PYTEST_ADDOPTS="" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$RUN_DIR/cache"

run_python_tests() (
  set -e
  python - <<'PY'
import importlib.util

modules = (
    "pytest", "torch", "transformers", "onnx", "onnxslim", "numpy",
    "gguf", "safetensors", "tqdm", "accelerate", "mslite_llm_ops",
)
missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python dependencies: " + ", ".join(missing))
print("[UT] Python dependencies checked")
PY

  echo "[UT] Running Python unit tests"
  python -m pytest -v -ra -o addopts= \
    -o "cache_dir=$RUN_DIR/cache/pytest" \
    "$TESTS_DIR/py" --basetemp="$RUN_DIR/pytest" \
    --junitxml="$RUN_DIR/python-results.xml"
)

run_cpp_tests() (
  set -e
  echo "[UT] Configuring C++ unit tests"
  cmake -S "$LLM_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMSLITE_LLM_ENABLE_NNRT=off \
    -DMSLITE_LLM_BUILD_TESTS=on

  echo "[UT] Building C++ unit tests"
  cmake --build "$BUILD_DIR" --parallel "$JOBS"

  local test_list test_count
  test_list=$(ctest --test-dir "$BUILD_DIR" -N)
  printf '%s\n' "$test_list"
  test_count=$(printf '%s\n' "$test_list" | sed -n \
    's/.*Total Tests: *\([0-9][0-9]*\).*/\1/p' | tail -n 1)
  [[ -n "$test_count" && "$test_count" -gt 0 ]] || {
    echo "[UT] ERROR: no C++ unit tests were registered" >&2
    exit 1
  }

  echo "[UT] Running $test_count C++ unit tests"
  ctest --test-dir "$BUILD_DIR" --output-on-failure
)

set +e
run_python_tests
python_status=$?
run_cpp_tests
cpp_status=$?
set -e

if (( python_status != 0 || cpp_status != 0 )); then
  echo "[UT] FAILED: Python status=$python_status, C++ status=$cpp_status" >&2
  exit 1
fi
echo "[UT] PASS: Python and C++ unit tests passed"
