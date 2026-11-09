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
PACKAGE="" MODEL_DIR="" DEVICE="" PLATFORM="" OUTPUT_DIR="$PWD/st_results"
BR_PORT=8888
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
      echo "Usage: bash run_llm_st.sh --package FILE --model-dir DIR --device ID --platform PLATFORM [--br-port PORT] [--output-dir DIR]"
      echo "Platform: kirin9020 or kirin9030; only tests marked for that platform are executed"
      echo "BinRunner port: 8888 by default"
      echo "Output directory: ./st_results by default; DDK and Python must already be activated"
      exit 0 ;;
    --package|--model-dir|--device|--platform|--br-port|--output-dir)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Missing value: $1" >&2; exit 1; }
      case "$1" in
        --package) PACKAGE="$2" ;;
        --model-dir) MODEL_DIR="$2" ;;
        --device) DEVICE="$2" ;;
        --platform) PLATFORM="$2" ;;
        --br-port) BR_PORT="$2" ;;
        --output-dir) OUTPUT_DIR="$2" ;;
      esac
      shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done
[[ -n "$PACKAGE" && -n "$MODEL_DIR" && -n "$DEVICE" && -n "$PLATFORM" ]] || {
  echo "Required: --package, --model-dir, --device, --platform (see --help)" >&2; exit 1;
}
case "$PLATFORM" in
  kirin9020|kirin9030) ;;
  *) echo "Invalid --platform: $PLATFORM (expected kirin9020 or kirin9030)" >&2; exit 1 ;;
esac
[[ "$DEVICE" != *$'\t'* && "$DEVICE" != *$'\r'* && "$DEVICE" != *$'\n'* ]] || {
  echo "Invalid --device: control characters are not allowed" >&2; exit 1;
}
[[ "$BR_PORT" =~ ^[0-9]{1,5}$ ]] || {
  echo "Invalid --br-port: expected 1-65535" >&2; exit 1;
}
BR_PORT=$((10#$BR_PORT))
(( BR_PORT >= 1 && BR_PORT <= 65535 )) || {
  echo "Invalid --br-port: expected 1-65535" >&2; exit 1;
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
echo "[ST] Results: $RUN_DIR"
export PYTEST_ADDOPTS="" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

collect_platform() {
  local platform="$1" collect_dir="$RUN_DIR/collect" status
  mkdir -p "$collect_dir"
  set +e
  python -m pytest --collect-only --strict-markers -q -o addopts= \
    -o "cache_dir=$collect_dir/cache/pytest" "$ST_DIR/" \
    -m "$platform" --platform="$platform" > "$collect_dir/collect.log" 2>&1
  status=$?
  set -e
  case "$status" in
    0) return 0 ;;
    5) return 1 ;;
    *)
      echo "[ST] ERROR: failed to collect $platform tests" >&2
      tail -100 "$collect_dir/collect.log" >&2
      return 2
      ;;
  esac
}

if collect_platform "$PLATFORM"; then
  status=0
else
  status=$?
fi
if (( status == 1 )); then
  echo "[ST] SKIP: no tests marked for $PLATFORM"
  exit 0
fi
if (( status != 0 )); then
  exit "$status"
fi

check_custom_ops() {
  local ddk_root="$1" platform="$2" platform_root kernel
  local -a missing=()
  platform_root="$ddk_root/tools/platform/$platform"
  [[ -f "$platform_root/lib64/libcustom_op.so" ]] || \
    missing+=("$platform_root/lib64/libcustom_op.so")
  for kernel in "${REQUIRED_KERNELS[@]}"; do
    [[ -f "$platform_root/ops/impl/custom/$kernel" ]] || \
      missing+=("$platform_root/ops/impl/custom/$kernel")
  done
  if (( ${#missing[@]} )); then
    printf '[ST] ERROR: gate environment has not installed required custom operators for %s:\n' \
      "$platform" >&2
    printf '  %s\n' "${missing[@]}" >&2
    return 1
  fi
}

run_platform() (
  set -e
  local platform="$1" device="$2" br_port="$3"
  local platform_dir="$RUN_DIR" ddk_root platform_root
  mkdir -p "$platform_dir/tmp" "$platform_dir/work"
  ddk_root=$(realpath -e -- "$DDK_PATH") || {
    echo "[ST] ERROR: DDK_PATH does not point to an existing directory: $DDK_PATH" >&2
    return 1
  }
  platform_root="$ddk_root/tools/platform/$platform"
  # DDK activation may default to one SoC. Put this run's platform first so
  # Python custom-op wrappers and runtime libraries cannot come from another SoC.
  export PYTHONPATH="$platform_root/ops/impl${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="$platform_root/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  check_custom_ops "$ddk_root" "$platform"

  export TMPDIR="$platform_dir/tmp" TMP="$platform_dir/tmp" TEMP="$platform_dir/tmp"
  export XDG_CACHE_HOME="$platform_dir/cache" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
  # Keep model compilation reproducible; parallel TeFusion can produce an OMC
  # that NNRT rejects at load time.
  export TE_PARALLEL_COMPILER=1
  export MSLITE_LLM_ST_DEVICE=1 BR_UDID="$device"
  cd "$platform_dir/work"

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

  echo "[ST] Running $platform tests"
  python -m pytest -v -s -rA -o addopts= -o "cache_dir=$platform_dir/cache/pytest" \
    "$ST_DIR/" -m "$platform" \
    --package="$PACKAGE" --model-dir="$MODEL_DIR" --platform="$platform" \
    --br-port="$br_port" \
    --basetemp="$platform_dir/pytest" --junitxml="$platform_dir/results.xml"
  # pytest may return 0 when tests are skipped; the gate requires actual passes.
  python - "$platform_dir/results.xml" "$platform" <<'PY'
import sys
import xml.etree.ElementTree as ET

cases = list(ET.parse(sys.argv[1]).getroot().iter("testcase"))
if not cases or any(case.find(tag) is not None for case in cases
                    for tag in ("failure", "error", "skipped")):
    sys.exit("ST failed: no executed tests, or failures/errors/skips in the report")
print(f"[ST] PASS ({sys.argv[2]}): {len(cases)} test(s)")
PY
)

set +e
run_platform "$PLATFORM" "$DEVICE" "$BR_PORT"
status=$?
set -e
if (( status == 0 )); then
  echo "[ST] PASS ($PLATFORM); results: $RUN_DIR"
  exit 0
fi
echo "[ST] FAILED ($PLATFORM); results: $RUN_DIR"
exit "$status"
