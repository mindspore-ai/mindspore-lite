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
ST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE="" MODEL_DIR="" DEVICE="" BR_PORT="8888" OUTPUT_DIR="$PWD/st_results"
REQUIRED_KERNELS=(
  ms_add_rms_norm.o
  ms_add_softmax.o
  ms_float_cast_int.o
  ms_group_matmul.o
  ms_quant4_n0_group32.o
  ms_rms_norm.o
  ms_rotary_pos_emb.o
  ms_scatter_nd.o
)
while (( $# )); do
  case "$1" in
    -h|--help)
      echo "Usage: bash run_llm_st.sh --package FILE --model-dir DIR --device ID [--br-port PORT] [--output-dir DIR]"
      echo "Device: device serial number listed by br devices; passed unchanged to BinRunner"
      echo "BinRunner port: local hdc forwarding port; 8888 by default"
      echo "Output directory: ./st_results by default; DDK and Python must already be activated"
      exit 0 ;;
    --package|--model-dir|--device|--br-port|--output-dir)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Missing value: $1" >&2; exit 1; }
      case "$1" in
        --package) PACKAGE="$2" ;;
        --model-dir) MODEL_DIR="$2" ;;
        --device) DEVICE="$2" ;;
        --br-port) BR_PORT="$2" ;;
        --output-dir) OUTPUT_DIR="$2" ;;
      esac
      shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done
[[ -n "$PACKAGE" && -n "$MODEL_DIR" && -n "$DEVICE" ]] || {
  echo "Required: --package, --model-dir, --device (see --help)" >&2; exit 1;
}
[[ "$BR_PORT" =~ ^[0-9]+$ ]] && (( BR_PORT >= 1 && BR_PORT <= 65535 )) || {
  echo "Invalid --br-port: $BR_PORT (expected 1-65535)" >&2; exit 1;
}
: "${DDK_PATH:?Activate the DDK before running ST}"
PACKAGE=$(realpath -e -- "$PACKAGE")
MODEL_DIR=$(realpath -e -- "$MODEL_DIR")
[[ -d "$MODEL_DIR" ]] || { echo "Model directory does not exist: $MODEL_DIR" >&2; exit 1; }
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath -e -- "$OUTPUT_DIR")
# OMG rejects directory names such as run.123; use a conservative path alphabet.
[[ "$OUTPUT_DIR" =~ ^/[a-zA-Z0-9_/-]+$ ]] || {
  echo "Result path must use only letters, digits, /, _ and -" >&2; exit 1;
}
RUN_DIR=$(mktemp -d "$OUTPUT_DIR/run_XXXXXXXX")
mkdir -p "$RUN_DIR/tmp" "$RUN_DIR/work"
echo "[ST] Results: $RUN_DIR"

check_custom_ops() (
  set -e
  local ddk_root platform_root kernel
  local -a missing=()

  ddk_root=$(realpath -e -- "$DDK_PATH") || {
    echo "[ST] ERROR: DDK_PATH does not point to an existing directory: $DDK_PATH" >&2
    return 1
  }
  platform_root="$ddk_root/tools/platform/kirin9020"
  [[ -f "$platform_root/lib64/libcustom_op.so" ]] || \
    missing+=("$platform_root/lib64/libcustom_op.so")
  for kernel in "${REQUIRED_KERNELS[@]}"; do
    [[ -f "$platform_root/ops/impl/custom/$kernel" ]] || \
      missing+=("$platform_root/ops/impl/custom/$kernel")
  done
  if (( ${#missing[@]} )); then
    printf '[ST] ERROR: gate environment has not installed required custom operators:\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
    return 1
  fi
)

run_tests() (
  set -e
  export TMPDIR="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp"
  export XDG_CACHE_HOME="$RUN_DIR/cache" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
  # Keep model compilation reproducible; parallel TeFusion can produce an OMC
  # that NNRT rejects at load time.
  export TE_PARALLEL_COMPILER=1
  # Accept the device ID as listed by br devices (normally a serial number).
  export MSLITE_LLM_ST_DEVICE=1 BR_UDID="$DEVICE"
  export PYTEST_ADDOPTS="" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  cd "$RUN_DIR/work"

  python - <<'PY'
import ctypes
import importlib.util

modules = (
    "pytest", "PIL", "torch", "transformers", "onnx", "onnxslim",
    "numpy", "gguf", "safetensors", "tqdm", "accelerate",
)
missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python dependencies: " + ", ".join(missing))
try:
    import te_fusion.log_util  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit("DDK Python module te_fusion.log_util is unavailable") from exc
try:
    ctypes.CDLL("libregister.so")
except OSError as exc:
    raise SystemExit(
        "DDK libregister.so cannot be loaded; ensure the active libstdc++ "
        "provides the GLIBCXX version required by DDK 6.1.1: " + str(exc)
    ) from exc
import mslite_llm_ops

required = (
    "MsRmsNorm", "MsAddRmsNorm", "MsAddSoftmax", "MsGroupMatmul",
    "MsQuant4N0Group32", "MsRotaryPosEmb", "MsScatterND",
)
missing = [name for name in required if not hasattr(mslite_llm_ops, name)]
if missing:
    raise SystemExit("mslite_llm_ops is missing: " + ", ".join(missing))
from transformers import Qwen2ForCausalLM  # noqa: F401,E402
PY

  # Discover every ST in the directory, including tests added by subsequent PRs.
  python -m pytest -v -s -rA -o addopts= -o "cache_dir=$RUN_DIR/cache/pytest" \
    "$ST_DIR/" --package="$PACKAGE" --model-dir="$MODEL_DIR" --br-port="$BR_PORT" \
    --basetemp="$RUN_DIR/pytest" --junitxml="$RUN_DIR/results.xml"
  # pytest may return 0 when tests are skipped; the gate requires actual passes.
  python - "$RUN_DIR/results.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

cases = list(ET.parse(sys.argv[1]).getroot().iter("testcase"))
if not cases or any(case.find(tag) is not None for case in cases
                    for tag in ("failure", "error", "skipped")):
    sys.exit("ST failed: no executed tests, or failures/errors/skips in the report")
print(f"[ST] PASS: {len(cases)} test(s)")
PY
)

set +e
check_custom_ops
status=$?
if (( status != 0 )); then
  echo "[ST] FAILED; results: $RUN_DIR"
  exit "$status"
fi

run_tests
status=$?
if (( status != 0 )); then
  echo "[ST] FAILED; results: $RUN_DIR"
fi
exit "$status"
