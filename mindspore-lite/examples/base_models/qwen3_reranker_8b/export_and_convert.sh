#!/bin/bash
# ============================================================
# Qwen3-Reranker-8B - CP2 (Ulysses) + shared prefix - export ONNX, convert to MindIR
#
#   Exports the prefix/suffix ONNX pair (both ranks), converts them with converter_lite,
#   then installs the pre-generated inference configs (static prefix / dynamic suffix).
#
# Usage:
#   bash export_and_convert.sh [model_id] [dtype] [prefix_len] [suffix_len] [out_dir]
#
#     [model_id]   weights/tokenizer dir                (default: 5 MODEL)
#     [dtype]      export precision fp16 / fp32         (default: fp16)
#     [prefix_len] shared-prefix bucket, 128-aligned    (default: 512; real prefix <= bucket)
#     [suffix_len] suffix dummy length; real length is a dynamic axis (default: 384)
#     [out_dir]    output dir                           (default: 6 OUT)
#
#   e.g. bash export_and_convert.sh /path/to/Qwen3-Reranker-8B fp16 512 384 /path/to/out
#
# Quantization switch (off by default = non-quantized fp16):
#   QUANT=<A8W8 pack dir> DSH_QUANT_MODE=dyn|static  (forwarded as --quant-mode)
#   DSH_QUANT_DOWN_PROJ=auto|1|0    DSH_FUSE_SWIGLU=0|1 (1: quantized path only)
#
# Output: $OUT/{export,mindir,configs}
# ============================================================
set -eo pipefail

# ============================================================================
# Absolute paths: this is the only block to edit when moving machine/dir (every item
# can be overridden by the same-named env var)
# ============================================================================

# (1) Example dir (this script's dir): configs/ and the export script are taken from here
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# (2) CANN root: needs set_env.sh and opp/vendors/mslite_custom_ops/ (vendor custom ops)
CANN=${CANN:-/path/to/ascend/9.1.0/cann-9.1.0}

# (3) MindSpore Lite package (aarch64): needs runtime/lib/ and
#     tools/converter/converter/converter_lite
MSLITE_HOME=${MSLITE_HOME:-/path/to/ms/ge/mindspore-lite-2.11.0-linux-aarch64}

# (4) Python interpreter (export only): torch / transformers / onnx, CPU is enough
PY=${PY:-/path/to/bin/python3.11}

# (5) Model weights + tokenizer dir (HF / ModelScope layout)
MODEL=${MODEL:-/path/to/workspace/qwen3_rerank_8B/Qwen3-Reranker-8B}

# (6) Output root: export/ (ONNX), mindir/ (MindIR) and configs/ are written here.
#     The script rm -rf's this dir at startup, so use a fresh dir for a clean run.
OUT=${OUT:-$SELF/output}

# (7) A8W8 quant pack dir: empty = non-quantized fp16 (default path)
QUANT=${QUANT:-}

# ---- Derived paths ----
CONV=$MSLITE_HOME/tools/converter/converter/converter_lite        # ONNX -> MindIR converter
VENDOR_ENV=$CANN/opp/vendors/mslite_custom_ops/bin/set_env.bash   # vendor custom-op env script
# ============================================================================

# ---- Positional args (defaults come from above) ----
MODEL=${1:-$MODEL}
DTYPE=${2:-fp16}
PREFIX_BUCKET=${3:-${PREFIX_BUCKET:-512}}
SUFFIX_LEN=${4:-384}
OUT=${5:-$OUT}
DEVICES=${DEVICES:-2,3}                      # unused by export/convert; the inference stage shares it
QUANT_MODE=${DSH_QUANT_MODE:-off}               # forwarded to --quant-mode
export DSH_FUSE_SWIGLU=${DSH_FUSE_SWIGLU:-0}   # SwigluDual fusion (0 = built-in SwiGlu; 1 = quantized path only)

# ---- Environment ----
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"   # the vendor script has no default and would trip "set -u"
[ -f "$CANN/set_env.sh" ] || { echo "✗ CANN 未找到: $CANN（可用 CANN=… 指定）"; exit 1; }
source "$CANN/set_env.sh" >/dev/null 2>&1
source "$VENDOR_ENV" >/dev/null 2>&1 || true
export LD_LIBRARY_PATH=$MSLITE_HOME/runtime/lib:$MSLITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
[ -x "$CONV" ] || { echo "✗ converter_lite 未找到: $CONV（可用 MSLITE_HOME=… 指定）"; exit 1; }

QARGS=(--quant-mode "$QUANT_MODE")
[ -n "$QUANT" ] && QARGS+=(--quant-dir "$QUANT")

echo "============================================================"
echo " CP2 + 公共前缀 导出/转换   $(date '+%F %T')"
echo " 模型=$MODEL  精度=$DTYPE  prefix桶=$PREFIX_BUCKET  suffix dummy=$SUFFIX_LEN"
if [ -n "$QUANT" ]; then
  echo " 量化: ON  --quant-mode=$QUANT_MODE  DSH_QUANT_DOWN_PROJ=${DSH_QUANT_DOWN_PROJ:-auto}"
  echo " 权重包: $QUANT"
else
  echo " 量化: OFF（非量化 fp16，默认路径）"
fi
echo " 产物: $OUT"
echo "============================================================"

rm -rf "$OUT"
mkdir -p "$OUT"

# ---------- 1) export ONNX ----------
echo ""; echo "=== [1/3] 导出 ONNX  $(date +%T) ==="
for MODE in prefix suffix; do
  echo "--- $MODE ---"
  $PY $SELF/export_qwen3_reranker_8b_onnx_cp.py \
    --model-id "$MODEL" "${QARGS[@]}" \
    --output-dir "$OUT/export" --dtype "$DTYPE" --cp-size 2 \
    --prefix-len $PREFIX_BUCKET --suffix-len $SUFFIX_LEN --mode $MODE 2>&1 | tail -2
done

# ---------- 2) convert to MindIR (prefix static / suffix dynamic) ----------
echo ""; echo "=== [2/3] 转换 mindir  $(date +%T) ==="
for R in 0 1; do mkdir -p "$OUT/mindir/rank$R/prefix" "$OUT/mindir/rank$R/suffix"; done
echo "--- prefix：静态转换（固定 $PREFIX_BUCKET，部署时绝不 resize） ---"
for R in 0 1; do
  "$CONV" --fmk=ONNX \
    --modelFile="$OUT/export/rank$R/prefix/qwen3_reranker_8b_prefix_rank$R.onnx" \
    --outputFile="$OUT/mindir/rank$R/prefix/qwen3_reranker_8b_prefix_rank$R" \
    --optimize=none --saveType=MINDIR \
    --inputShape="input_ids:1,$PREFIX_BUCKET;attention_mask:1,$PREFIX_BUCKET;position_ids:1,$PREFIX_BUCKET" \
    2>&1 | tail -1
done
echo "--- suffix：动态转换（不带 --inputShape，运行时按桶 resize） ---"
for R in 0 1; do
  "$CONV" --fmk=ONNX \
    --modelFile="$OUT/export/rank$R/suffix/qwen3_reranker_8b_suffix_rank$R.onnx" \
    --outputFile="$OUT/mindir/rank$R/suffix/qwen3_reranker_8b_suffix_rank$R" \
    --optimize=none --saveType=MINDIR 2>&1 | tail -1
done

# ---------- 3) install configs ----------
echo ""; echo "=== [3/3] 放置配置  $(date +%T) ==="
mkdir -p "$OUT/configs"
# Both configs are pre-generated and shipped with the example (see configs/):
#   * prefix: the statically converted mindir has fixed shapes, so ge.inputShape is not
#     needed there, and model.resize() must never be called (it would switch the graph
#     back to the GE dynamic path and change the numerics).
#   * suffix: 4 dynamic buckets (ONNX input names + dynamicDims + dynamicNodeType=1).
# Keep both files in sync when the model structure or the export contract changes
# (layer count / KV heads / head_dim / bucket sizes).
cp "$SELF/configs/qwen3_reranker_8b_prefix.config" "$OUT/configs/"
cp "$SELF/configs/qwen3_reranker_8b_suffix.config" "$OUT/configs/"
if [ "$PREFIX_BUCKET" != "512" ]; then
  echo "  ⚠ PREFIX_BUCKET=$PREFIX_BUCKET，但 configs/*.config 是按 prefix 桶 512 预生成的，"
  echo "    其 dynamicDims / KV 形状需同步修改后再用。"
fi
echo "  prefix config -> $OUT/configs/qwen3_reranker_8b_prefix.config"
echo "  suffix config -> $OUT/configs/qwen3_reranker_8b_suffix.config"

echo ""; echo "=== 导出/转换完成  $(date '+%F %T')  产物: $OUT ==="
echo "next: inference command, see README section 4"
