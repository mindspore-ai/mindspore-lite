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
"""Accuracy and performance tests for the Qwen3.5-9B LiteBoost adapter.

One module covers both halves of the suite; ``--mode`` picks which half runs so
the two never pay for the same checkpoint load back to back:

    pytest test_qwen3_5_9b.py --mode accuracy   # end-to-end logits
    pytest test_qwen3_5_9b.py --mode perf       # 128..2048 prefill/decode sweep
    pytest test_qwen3_5_9b.py --mode all        # default; runs both

Accuracy:
* ``test_end_to_end_logits_match_reference`` — the boosted pooled-prefill graph
  against the pre-LiteBoost baseline in ``accuracy_reference.npz``: logits of
  the untouched checkpoint in float32 on the host, at four positions of a fixed
  64-token prompt, plus a top-1 id for every position. The fixture is generated
  offline by ``accuracy_runner.py --mode reference`` (see its docstring), which
  runs the unmodified torch pipeline in float32 on the host; the case skips
  until that file exists. The test itself only runs the fused path, so a plain
  Python baseline never has to load the checkpoint.

Performance:
* one torchrun run of the shipped benchmark sweeps every prefill bucket; the
  per-shape cases then check that the reported numbers are sane. Bounds are
  loose by design (regression guard, not a perf gate) and can be adjusted with
  ``QWEN35_9B_PREFILL_CAP_MS`` / ``QWEN35_9B_DECODE_CAP_MS``.

The end-to-end and performance cases need a checkpoint plus NPUs, given through
``QWEN35_9B_MODEL_DIR`` / ``QWEN35_9B_TP``; without them they skip (see
``conftest.py``).
"""
import json
import os

import numpy as np
import pytest

from conftest import ACCURACY_RUNNER, run_torchrun, tp_size

_HERE = os.path.dirname(os.path.abspath(__file__))
PERF_RUNNER = os.path.join(_HERE, "perf_runner.py")
ACCURACY_REFERENCE = os.path.join(_HERE, "accuracy_reference.npz")

E2E_COSINE_MIN = 0.99
E2E_TOP1_AGREEMENT_MIN = 0.90

SHAPES = (128, 256, 512, 1024, 2048)
MAX_NEW_TOKENS = 32
PREFILL_CAP_MS = float(os.environ.get("QWEN35_9B_PREFILL_CAP_MS", "1000"))
DECODE_CAP_MS = float(os.environ.get("QWEN35_9B_DECODE_CAP_MS", "200"))
DECODE_SPREAD_MAX = 1.5


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.reshape(-1).astype(np.float64)
    b64 = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a64) * np.linalg.norm(b64)
    if denom == 0:
        return 1.0 if np.linalg.norm(a64) == np.linalg.norm(b64) else 0.0
    return float(np.dot(a64, b64) / denom)


@pytest.mark.L1
@pytest.mark.ascend_a2
@pytest.mark.accuracy
def test_end_to_end_logits_match_reference(model_dir, parallel_size,
                                           parallel_mode, tmp_path):
    """Boosted pooled prefill matches the pre-LiteBoost float32 baseline."""
    if not os.path.exists(ACCURACY_REFERENCE):
        pytest.skip(
            f"{os.path.basename(ACCURACY_REFERENCE)} not found; generate it once "
            "on the host (unmodified torch pipeline, float32 eager, several "
            f"minutes and tens of GiB of RAM) with:\n  python {ACCURACY_RUNNER} "
            f"--mode reference --model-dir <bf16 checkpoint dir> "
            f"--out {ACCURACY_REFERENCE}")
    with np.load(ACCURACY_REFERENCE) as ref:
        positions = ref["positions"].astype(np.int64)
        reference = ref["logits"].astype(np.float32)
        reference_top1 = ref["top1"].astype(np.int64)

    fused_path = str(tmp_path / "fused.npz")
    run_torchrun(ACCURACY_RUNNER,
                 ["--model-dir", model_dir, "--mode", "fused",
                  "--input-ids", ACCURACY_REFERENCE, "--out", fused_path,
                  "--parallel-mode", parallel_mode],
                 parallel_size)
    with np.load(fused_path) as fused_data:
        fused = fused_data["logits"].astype(np.float32)
        fused_top1 = fused_data["top1"].astype(np.int64)

    assert fused.shape == reference.shape, (
        f"fused logits {fused.shape} do not match the reference layout "
        f"{reference.shape}")
    assert fused_top1.shape == reference_top1.shape
    cosines = [_cosine(fused[index], reference[index])
               for index in range(reference.shape[0])]
    cos = float(np.mean(cosines))
    sampled_agreement = int((fused_top1[positions]
                             == reference_top1[positions]).sum())
    agreement = float((fused_top1 == reference_top1).mean())
    print(f"End-to-end logits vs pre-LiteBoost reference at positions "
          f"{positions.tolist()}: cosine={cos:.6f} "
          f"(per position {[round(value, 6) for value in cosines]}), "
          f"sampled top-1={sampled_agreement}/{len(positions)}, "
          f"all-position top-1 agreement={agreement:.4f}")
    assert cos >= E2E_COSINE_MIN
    # The logged positions decide the next token, so they must agree exactly;
    # the rest of the prompt is a looser sanity bound (bf16 near-ties flip).
    assert sampled_agreement == len(positions)
    assert agreement >= E2E_TOP1_AGREEMENT_MIN


@pytest.fixture(scope="session")
def perf_rows(model_dir, parallel_size, parallel_mode, tmp_path_factory):
    """Run the benchmark once and return ``{prefill_tokens: row}`` plus startup."""
    out = tmp_path_factory.mktemp("qwen3_5_9b_perf") / "perf.json"
    run_torchrun(
        PERF_RUNNER,
        ["--model-dir", model_dir,
         "--lengths", ",".join(str(size) for size in SHAPES),
         "--buckets", ",".join(str(size) for size in SHAPES),
         "--max-new-tokens", str(MAX_NEW_TOKENS),
         "--parallel-mode", parallel_mode,
         "--json-out", str(out)],
        parallel_size,
    )
    with open(out, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = {row["prefill_tokens"]: row for row in payload["results"]}
    missing = [size for size in SHAPES if size not in rows]
    assert not missing, f"benchmark did not report shapes {missing}"
    return {"rows": rows, "startup": payload["startup"],
            "parallel_mode": payload.get("parallel_mode", "tp")}


@pytest.mark.L1
@pytest.mark.ascend_a2
@pytest.mark.perf
@pytest.mark.parametrize("shape", SHAPES)
def test_prefill_decode_latency(perf_rows, shape):
    """Each bucket reports a positive, cap-bounded prefill and decode latency."""
    row = perf_rows["rows"][shape]
    print(f"prefill {shape}: {row['prefill_ms']:.2f} ms, "
          f"TTFT {row['first_token_ms']:.2f} ms, "
          f"decode {row['decode_ms_per_tok']:.2f} ms/tok "
          f"(bucket {row['prefill_bucket']})")
    assert row["decode_tokens"] == MAX_NEW_TOKENS
    assert 0.0 < row["prefill_ms"] < PREFILL_CAP_MS
    assert 0.0 < row["decode_ms_per_tok"] < DECODE_CAP_MS


@pytest.mark.L1
@pytest.mark.ascend_a2
@pytest.mark.perf
def test_prefill_scales_with_length(perf_rows):
    """Prefill latency grows with the token count (graph replay of each bucket)."""
    rows = perf_rows["rows"]
    small, mid, large = (rows[size]["prefill_ms"] for size in (128, 512, 2048))
    print(f"prefill scaling: 128→{small:.2f} ms, 512→{mid:.2f} ms, "
          f"2048→{large:.2f} ms")
    assert small < mid < large


@pytest.mark.L1
@pytest.mark.ascend_a2
@pytest.mark.perf
def test_decode_latency_is_stable(perf_rows):
    """Decode latency per token does not depend on the prefill bucket."""
    values = [row["decode_ms_per_tok"] for row in perf_rows["rows"].values()]
    spread = max(values) / min(values)
    print(f"decode spread across buckets: {spread:.3f}x "
          f"({min(values):.2f}..{max(values):.2f} ms/tok)")
    assert spread <= DECODE_SPREAD_MAX


@pytest.mark.L1
@pytest.mark.ascend_a2
@pytest.mark.perf
def test_graph_capture_completes(perf_rows, parallel_size, parallel_mode):
    """Every bucket captured its prefill graph (no silent fallback to eager)."""
    startup = perf_rows["startup"]
    captures = {key: value for key, value in startup.items()
                if key.startswith("init_capture_")}
    print(f"prefill graph capture ({perf_rows['parallel_mode']}, "
          f"world={parallel_size}): {captures} s")
    assert parallel_size == tp_size()
    # The benchmark must have run the scheme the suite asked for; otherwise a
    # QWEN35_9B_MODE=cp run would silently measure TP and pass.
    assert perf_rows["parallel_mode"] == parallel_mode
    assert len(captures) == len(SHAPES)
    assert all(value > 0.0 for value in captures.values())
    assert startup.get("decode_capture", 0.0) > 0.0
