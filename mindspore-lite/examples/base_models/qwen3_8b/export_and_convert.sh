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
# Usage:
#   bash export_and_convert.sh 1p    # Single-chip: export 1-rank ONNX + offline ACL convert
#   bash export_and_convert.sh 2p    # TP=2: export 2-rank ONNX + online GE convert
#   bash export_and_convert.sh 4p    # TP=4: export 4-rank ONNX + online GE convert
#
# Model weights are auto-downloaded from ModelScope to ./Qwen3-8B if missing
#   (requires: pip install modelscope; or pre-download manually to skip).
#
# Output paths (consumed by infer.sh):
#   1p -> qwen3_8b_onnx/{prefill,decode}/*_rank0_graph.mindir
#   2p -> qwen3_8b_tp_onnx/{prefill,decode}/*_rank{0,1}_graph.mindir
#   4p -> qwen3_8b_tp4_onnx/{prefill,decode}/*_rank{0..3}_graph.mindir
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Parse argument ----
if [ -z "${1:-}" ]; then
  echo "Error: missing argument."
  echo ""
  echo "Usage:"
  echo "  bash export_and_convert.sh 1p    # Single-chip export (offline ACL: ascend_oriented + dynamic dims)"
  echo "  bash export_and_convert.sh 2p    # TP=2 export (online GE: optimize=none)"
  echo "  bash export_and_convert.sh 4p    # TP=4 export (online GE: optimize=none)"
  exit 1
fi
TP_ARG="$1"
TP_SIZE="${TP_ARG%p}"
if ! [[ "$TP_SIZE" =~ ^[0-9]+$ ]]; then
  echo "Error: expected 1p / 2p / 4p, got '$TP_ARG'"
  exit 1
fi

export ASCEND_CUSTOM_OPP_PATH=${ASCEND_OPP_PATH}/vendors/mslite_custom_ops
CONV="${MSLITE_HOME_PATH}/tools/converter/converter/converter_lite"
MODEL_ID="${MODEL_ID:-./Qwen3-8B}"
MODELSCOPE_ID="${MODELSCOPE_ID:-Qwen/Qwen3-8B}"
DTYPE="fp16"

echo "=== TP_SIZE=$TP_SIZE  dtype=$DTYPE ==="

# ---- Step 0: Download weights from ModelScope if missing ----
if [ ! -d "$MODEL_ID" ] || [ -z "$(ls -A "$MODEL_ID" 2>/dev/null)" ]; then
  echo "=== [0] Downloading $MODELSCOPE_ID from ModelScope -> $MODEL_ID ==="
  python3 -c "
from modelscope import snapshot_download
snapshot_download('$MODELSCOPE_ID', local_dir='$MODEL_ID')
print('download done')
" || {
    echo "Error: ModelScope download failed. Install with 'pip install modelscope' and retry,"
    echo "       or manually download weights to $MODEL_ID and rerun."
    exit 1
  }
fi

# ---- Step 0b: Remove CGDR (breaks hccl_graph_optimizer, TP>=2 only) ----
if [ "$TP_SIZE" -ge 2 ]; then
  echo "=== [0b] Removing CGDR ==="
  python3 -c "
import json, glob, os, shutil
V = os.environ['ASCEND_OPP_PATH'] + '/vendors'
for vendor in os.listdir(V):
    base = f'{V}/{vendor}'
    if not os.path.isdir(base): continue
    for jf in glob.glob(f'{base}/**/*ops-info*.json', recursive=True) + glob.glob(f'{base}/**/npu_supported_ops.json', recursive=True):
        try: d = json.load(open(jf))
        except: continue
        changed = False
        def strip(o):
            global changed
            if isinstance(o, dict):
                for k in list(o):
                    if 'ChunkGatedDeltaRule' in k: del o[k]; changed = True
                    else: strip(o[k])
            elif isinstance(o, list):
                for it in o: strip(it)
        strip(d)
        if changed: json.dump(d, open(jf, 'w'), indent=2)
for f in glob.glob(f'{V}/**/*chunk_gated_delta_rule*', recursive=True):
    if not f.endswith('.cgdrbak'): shutil.rmtree(f, ignore_errors=True)
print('CGDR removed')
"
fi

# ---- Step 1: Export ONNX (unified script for all TP sizes) ----
if [ "$TP_SIZE" -eq 1 ]; then
  OUT_DIR="./qwen3_8b_onnx"
elif [ "$TP_SIZE" -eq 4 ]; then
  OUT_DIR="./qwen3_8b_tp4_onnx"
else
  OUT_DIR="./qwen3_8b_tp_onnx"
fi

echo "=== [1] Exporting TP=$TP_SIZE prefill + decode ($DTYPE) -> $OUT_DIR ==="
python3 export_qwen3_8b_onnx.py \
  --model-id "$MODEL_ID" \
  --output-dir "$OUT_DIR" \
  --device cpu \
  --dtype "$DTYPE" \
  --tp-size "$TP_SIZE"

# ---- Step 2: Convert ONNX -> MindIR ----
if [ "$TP_SIZE" -eq 1 ]; then
  # 1p: offline ACL convert (ascend_oriented + dynamic dims config)
  echo "=== [2] Converting prefill (offline ACL: ascend_oriented + dynamic dims) ==="
  "$CONV" --fmk=ONNX \
    --modelFile="$OUT_DIR/prefill/qwen3_8b_llm_prefill_rank0.onnx" \
    --outputFile="$OUT_DIR/prefill/qwen3_8b_llm_prefill_rank0" \
    --optimize=ascend_oriented --saveType=MINDIR \
    --configFile=./configs/qwen3_8b_llm_prefill.config

  echo "=== [3] Converting decode (offline ACL: ascend_oriented + static shape) ==="
  "$CONV" --fmk=ONNX \
    --modelFile="$OUT_DIR/decode/qwen3_8b_llm_decode_rank0.onnx" \
    --outputFile="$OUT_DIR/decode/qwen3_8b_llm_decode_rank0" \
    --optimize=ascend_oriented --saveType=MINDIR \
    --configFile=./configs/qwen3_8b_llm_decode.config
else
  # 2p/4p: online GE convert (optimize=none for provider=ge)
  echo "=== [2] Converting (online GE: optimize=none) ==="
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
fi

echo "=== Done. Models at $OUT_DIR ==="
echo "=== Run: bash infer.sh <device_ids> ==="
