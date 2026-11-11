#!/bin/bash
# TC-TRC-09/10 + TC-REG-05 gate: the exported .ms must be byte-identical to the pre-change
# baseline for (a) plain conversion without any quant config (records exist, fusion runs, so this
# is the leak-prone path) and (b) FULL_QUANT without per-layer lists.
#
# Method (works offline, no second build tree): the MAIN build tree is temporarily reverted to the
# baseline commit, rebuilt, baseline artifacts are produced, then the feature sources are restored,
# rebuilt again and the same conversions are compared by md5.
#
# Usage:  bash run_ms_binary_compare.sh <baseline_commit> <model1.onnx> [model2.onnx ...]
#         CALIB_NAME=<input_name> CALIB_FLOATS=<n> may override the calib input name/size
#         (defaults: x / 64 — must match each model's actual input name and element count).
# Expect: identical md5 for every "<model>_plain.ms" and "<model>_quant.ms" pair.
set -euo pipefail
BASE_COMMIT=${1:?baseline commit required}
shift || true
MODELS=("$@")
[ ${#MODELS[@]} -ge 1 ] || { echo "usage: $0 <baseline_commit> model.onnx [...]"; exit 2; }
REPO=$(git rev-parse --show-toplevel)
BUILD=${BUILD:-$REPO/build}
TMP=$(mktemp -d)
CONV=$BUILD/tools/converter/converter_lite/converter_lite
export LD_LIBRARY_PATH=$BUILD/tools/converter:$BUILD/tools/converter/registry:$BUILD/tools/converter/mindspore_core:$BUILD/tools/converter/mindspore_ops:$BUILD/_deps/glog-src/_build:$BUILD/.mslib/opencv_4.11.0_aa53d4de18ef98381603cb4244f06a35ebef5a5847f771e50b20ac1e05fa49c1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Restore the feature sources on ANY exit (including Ctrl-C), and print the patch location
# so a manual recovery is always possible.
restore_tree() {
  if [ -f "$TMP/pr.patch" ] && ! git -C "$REPO" apply "$TMP/pr.patch" 2>/dev/null; then
    echo "WARNING: failed to re-apply the feature patch; it is kept at $TMP/pr.patch" >&2
  fi
}
trap 'restore_tree' EXIT

convert_all() {  # $1 = output dir
  mkdir -p "$1"
  for m in "${MODELS[@]}"; do
    local name calib_dir floats
    name=$(basename "$m" .onnx)
    calib_dir="$TMP/${name}_calib"
    floats=${CALIB_FLOATS:-64}
    "$CONV" --fmk=ONNX --modelFile="$m" --outputFile="$1/${name}_plain" --configFile=/dev/null
    cat > "$1/q.cfg" <<CFG
[common_quant_param]
quant_type=FULL_QUANT
bit_num=8

[data_preprocess_param]
calibrate_path=${CALIB_NAME:-x}:$calib_dir
calibrate_size=1
input_type=BIN

[full_quant_param]
activation_quant_method=MAX_MIN
bias_correction=false
CFG
    mkdir -p "$calib_dir"
    # deterministic calib blob; size must match the model input's element count
    python3 - "$calib_dir/x.bin" "$floats" <<'PY'
import sys, struct
path, n = sys.argv[1], int(sys.argv[2])
with open(path, 'wb') as f:
    f.write(b'\x00' * (4 * n))
PY
    "$CONV" --fmk=ONNX --modelFile="$m" --outputFile="$1/${name}_quant" --configFile="$1/q.cfg"
  done
}

# The whole diff (no path limiter) so the baseline build is truly pre-merge; untracked files
# would survive git apply -R, so fail fast on a dirty tree before touching anything.
if ! git -C "$REPO" diff-index --quiet HEAD -- || [[ -n $(git -C "$REPO" status --porcelain) ]]; then
    echo "ERROR: working tree has uncommitted/untracked changes; commit or stash first." >&2
    exit 1
fi
git -C "$REPO" diff "$BASE_COMMIT" > "$TMP/pr.patch"
echo "[1/3] reverting to baseline $BASE_COMMIT and rebuilding..."
git -C "$REPO" apply -R "$TMP/pr.patch"
(cd "$BUILD" && cmake . > /dev/null && make converter_lite -j"$(nproc)")
convert_all "$TMP/base"
echo "[2/3] restoring feature sources and rebuilding..."
git -C "$REPO" apply "$TMP/pr.patch"
trap - EXIT   # tree restored; drop the safety trap
(cd "$BUILD" && cmake . > /dev/null && make converter_lite -j"$(nproc)")
convert_all "$TMP/mine"
echo "[3/3] comparing md5..."
rc=0
for f in "$TMP"/base/*.ms; do
  name=$(basename "$f")
  if cmp -s "$f" "$TMP/mine/$name"; then
    echo "IDENTICAL  $name"
  else
    echo "DIFFERENT  $name"
    rc=1
  fi
done
rm -rf "$TMP"
exit $rc
