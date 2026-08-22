#!/bin/bash
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
# ============================================================================

# ============================================================================
# Qwen3-8B export + convert - unified, parameterized by parallelism (1p / 2p / 4p)
#
# All paths use online GE convert (optimize=none) so that provider=ge can be
# used at runtime. Runtime GE shapes come from ge.dynamicDims 6-bucket cfgs
# (one dynamic single graph per prefill/decode, 6 buckets each):
#   * 1p  -> configs/ge_{prefill,decode}.cfg     (KV dim3 = 8 heads)
#   * 2p  -> configs/tp2/ge_{prefill,decode}.cfg (KV dim3 = 4 heads per rank)
#   * 4p  -> configs/tp4/ge_{prefill,decode}.cfg (KV dim3 = 2 heads, unvalidated)
# 6 prefill buckets -> 6 decode buckets per cfg_dir:
#   prefill seq = {512, 1024, 1664, 2048, 2816, 3072}
#   decode kv   = prefill seq + 512 (each bucket supports up to 512 new tokens)
# 2p is the validated path on this box (300I Duo). Export uses --tp-dynamic so
# prefill gets a dynamic seq axis + decode a dynamic KV-len axis; a single MindIR
# per rank serves all 6 buckets (ge.dynamicNodeType=1 enables online re-specialize).
#
# Usage:
#   bash export_and_convert.sh 1p    # Single-chip: 1-rank ONNX + GE convert (optimize=none)
#   bash export_and_convert.sh 2p    # TP=2: 2-rank ONNX + GE convert (optimize=none)
#   bash export_and_convert.sh 4p    # TP=4: 4-rank ONNX + GE convert (optimize=none)
#
# Output paths (consumed by infer.sh):
#   1p -> qwen3_8b_onnx/{prefill,decode}/*_rank0_graph.mindir
#   2p -> qwen3_8b_tp_onnx/{prefill,decode}/*_rank{0,1}_graph.mindir
#   4p -> qwen3_8b_tp4_onnx/{prefill,decode}/*_rank{0..3}_graph.mindir
#
# Prerequisites (env vars, see README.md): $CONV (converter_lite path),
#   $ASCEND_CUSTOM_OPP_PATH (exported), $MODEL_ID (Qwen3-8B weights dir), $DTYPE.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Defaults (override via env or env.sh; see README.md "环境准备") ----
DTYPE="${DTYPE:-fp16}"
MODEL_ID="${MODEL_ID:-./Qwen3-8B}"
if [ -z "${CONV:-}" ]; then
  if [ -n "${MSLITE_HOME_PATH:-}" ] && [ -x "$MSLITE_HOME_PATH/tools/converter/converter/converter_lite" ]; then
    CONV="$MSLITE_HOME_PATH/tools/converter/converter/converter_lite"
  elif command -v converter_lite >/dev/null 2>&1; then
    CONV="$(command -v converter_lite)"
  else
    echo "Error: converter_lite not found. Set \$CONV or \$MSLITE_HOME_PATH (see README.md)." >&2
    exit 1
  fi
fi
if [ ! -x "$CONV" ]; then
  echo "Error: CONV='$CONV' is not an executable converter_lite (see README.md)." >&2
  exit 1
fi
if [ ! -e "$MODEL_ID" ]; then
  echo "Error: MODEL_ID='$MODEL_ID' does not exist. Set \$MODEL_ID to the Qwen3-8B weights dir." >&2
  exit 1
fi

# ---- Parse argument ----
if [ -z "${1:-}" ]; then
  echo "Error: missing argument."
  echo ""
  echo "Usage:"
  echo "  bash export_and_convert.sh 1p    # Single-chip export (GE: optimize=none, 6-bucket cfg at runtime"
  echo "  bash export_and_convert.sh 2p    # TP=2 export (GE: optimize=none, cfgs under configs/tp2)"
  echo "  bash export_and_convert.sh 4p    # TP=4 export (GE: optimize=none, cfgs under configs/tp4)"
  exit 1
fi
TP_ARG="$1"
TP_SIZE="${TP_ARG%p}"
if ! [[ "$TP_SIZE" =~ ^[0-9]+$ ]]; then
  echo "Error: expected 1p / 2p / 4p, got '$TP_ARG'"
  exit 1
fi

echo "=== TP_SIZE=$TP_SIZE  dtype=$DTYPE ==="

# ---- Step 1: Export ONNX (unified script for all TP sizes) ----
if [ "$TP_SIZE" -eq 1 ]; then
  OUT_DIR="./qwen3_8b_onnx"
elif [ "$TP_SIZE" -eq 4 ]; then
  OUT_DIR="./qwen3_8b_tp4_onnx"
else
  OUT_DIR="./qwen3_8b_tp_onnx"
fi

echo "=== [1] Exporting TP=$TP_SIZE prefill + decode ($DTYPE) -> $OUT_DIR ==="
# TP=2 uses ONE dynamic ONNX (dynamic prefill seq + dynamic decode KV len) served
# by the two dynamicDims cfgs configs/tp2/ge_{prefill,decode}.cfg at runtime.
# 1p/4p keep the multi-static-bucket flow.
EXTRA_ARGS=()
if [ "$TP_SIZE" -eq 1 ]; then
  echo "     cfgs=configs/ge_{prefill,decode}.cfg  (KV dim3=8 heads, ge.dynamicDims 6 buckets)."
  echo "     1p runs via infer_qwen3_8b_mslite_1p.py (single dynamic graph + resize per bucket)."
elif [ "$TP_SIZE" -eq 2 ]; then
  echo "     cfgs=configs/tp2/ge_{prefill,decode}.cfg  (KV dim3=4 heads per rank, ge.dynamicDims 6 buckets)."
  echo "     Prefill exported with dynamic seq axis + decode with dynamic KV-len axis (--tp-dynamic)."
  EXTRA_ARGS+=(--tp-dynamic)
else
  echo "     cfgs=configs/tp4/  (KV dim3=2 heads per rank)."
fi
python3 export_qwen3_8b_onnx.py \
  --model-id "$MODEL_ID" \
  --output-dir "$OUT_DIR" \
  --device cpu \
  --dtype "$DTYPE" \
  --tp-size "$TP_SIZE" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

# ---- Step 2: Convert ONNX -> MindIR (online GE: optimize=none for all TP sizes) ----
echo "=== [2] Converting (online GE: optimize=none) -> build_from_file picks bucket cfg at runtime ==="
for SUB in prefill decode; do
  for R in $(seq 0 $((TP_SIZE - 1))); do
    ONNX="$OUT_DIR/$SUB/qwen3_8b_llm_${SUB}_rank${R}.onnx"
    OUT="$OUT_DIR/$SUB/qwen3_8b_llm_${SUB}_rank${R}"
    echo "  $SUB rank$R ..."
    "$CONV" --fmk=ONNX \
      --modelFile="$ONNX" \
      --outputFile="$OUT" \
      --optimize=none --saveType=MINDIR
  done
done

echo "=== Done. Models at $OUT_DIR ==="
echo "=== Run (1p)    : bash infer.sh <device_id>        # auto uses configs/ 6-bucket cfgs"
echo "=== Run (2p)    : bash infer.sh <id0,id1>         # auto uses configs/tp2 cfgs"
echo "=== Run (4p)    : bash infer.sh <id0,id1,id2,id3>  # auto uses configs/tp4 cfgs"
echo "=== Override cfg dir: python3 infer_qwen3_8b_mslite_tp.py ... --bucket-cfg-dir configs/tp2"
