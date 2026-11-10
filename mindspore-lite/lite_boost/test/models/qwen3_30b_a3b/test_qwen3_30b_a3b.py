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
"""Accuracy and performance tests for the Qwen3-30B-A3B LiteBoost adapter.

One module covers both halves of the suite; ``--mode`` picks which half runs so
the two never pay for the same 61 GB checkpoint load back to back:

    pytest test_qwen3_30b_a3b.py --mode accuracy   # end-to-end logits
    pytest test_qwen3_30b_a3b.py --mode perf       # 128..2048 prefill/decode sweep
    pytest test_qwen3_30b_a3b.py --mode all        # default; runs both

Accuracy:
* ``test_end_to_end_logits_match_reference`` — the fully boosted TP model
  against the pre-LiteBoost baseline in ``accuracy_reference.npz``: logits of
  the untouched ``Qwen3MoeForCausalLM`` in float32 on the host, at four
  positions of a fixed 256-token prompt, plus a top-1 id and a top1-top2 margin
  for every position. The fixture is generated offline by
  ``accuracy_runner.py --mode reference`` (see its docstring); the test itself
  only runs the fused path, so a plain Python baseline never has to load the
  checkpoint. The strict top-1 check counts only positions where the float32
  baseline has a margin large enough that a rounding difference cannot flip
  it, and the forward is repeated under ``HCCL_DETERMINISTIC=true`` so an
  unstable top-1 is reported as a defect instead of hiding inside a percentage.

Performance:
* one torchrun run of the shipped benchmark sweeps every prefill bucket; the
  per-shape cases then check that the reported numbers are sane. Bounds are
  loose by design (regression guard, not a perf gate) and can be adjusted with
  ``QWEN3_30B_A3B_PREFILL_CAP_MS`` / ``QWEN3_30B_A3B_DECODE_CAP_MS``.

The end-to-end and performance cases need a checkpoint plus several NPUs, given
through ``QWEN3_30B_A3B_MODEL_DIR`` / ``QWEN3_30B_A3B_TP``; without them they
skip (see ``conftest.py``).
"""
import json
import os

import numpy as np
import pytest

from conftest import ACCURACY_RUNNER, run_torchrun, tp_size

_HERE = os.path.dirname(os.path.abspath(__file__))
PERF_RUNNER = os.path.join(_HERE, "perf_runner.py")
ACCURACY_REFERENCE = os.path.join(_HERE, "accuracy_reference.npz")

ACCURACY_ROUNDS = 3
E2E_COSINE_MIN = 0.99
# Coarse sanity bound over every position; the strict check is the decisive one
# below, because roughly four in ten positions of ordinary prose are near ties.
E2E_TOP1_AGREEMENT_MIN = 0.95
# A position whose float32 top1-top2 margin is at least this large is a real
# decision: a bf16 rounding difference of the observed size cannot flip it.
DECISIVE_MARGIN = 1.0

SHAPES = (128, 256, 512, 1024, 2048)
MAX_NEW_TOKENS = 32
PREFILL_CAP_MS = float(os.environ.get("QWEN3_30B_A3B_PREFILL_CAP_MS", "2000"))
DECODE_CAP_MS = float(os.environ.get("QWEN3_30B_A3B_DECODE_CAP_MS", "200"))
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
def test_end_to_end_logits_match_reference(model_dir, parallel_size, tmp_path):
    """Boosted TP prefill logits match the pre-LiteBoost float32 baseline."""
    with np.load(ACCURACY_REFERENCE) as ref:
        positions = ref["positions"].astype(np.int64)
        reference = ref["logits"].astype(np.float32)
        reference_top1 = ref["top1"].astype(np.int64)
        reference_margin = ref["margin"].astype(np.float32)

    fused_path = str(tmp_path / "fused.npz")
    run_torchrun(ACCURACY_RUNNER,
                 ["--model-dir", model_dir, "--mode", "fused",
                  "--input-ids", ACCURACY_REFERENCE,
                  "--rounds", str(ACCURACY_ROUNDS), "--out", fused_path],
                 parallel_size,
                 env_overrides={"HCCL_DETERMINISTIC": "true"})
    with np.load(fused_path) as out:
        fused = out["logits"].astype(np.float32)
        round_top1 = out["round_top1"].astype(np.int64)

    seq_len, vocab = reference_top1.shape[0], reference.shape[1]
    assert fused.shape == (seq_len, vocab), (
        f"fused logits {fused.shape} do not match the reference layout "
        f"({seq_len}, {vocab})")
    assert round_top1.shape == (ACCURACY_ROUNDS, seq_len), (
        f"round top-1 {round_top1.shape} does not match "
        f"({ACCURACY_ROUNDS}, {seq_len})")

    cosines = [_cosine(fused[pos], reference[i]) for i, pos in enumerate(positions)]
    cos = float(np.mean(cosines))
    fused_top1 = fused.argmax(-1)
    agreement = float((fused_top1 == reference_top1).mean())
    decisive = reference_margin >= DECISIVE_MARGIN
    decisive_agreement = float((fused_top1[decisive] == reference_top1[decisive]).mean())
    unstable = np.flatnonzero((round_top1 != round_top1[0]).any(axis=0))
    print(f"End-to-end logits vs pre-LiteBoost reference: cosine={cos:.6f} "
          f"(per position {[round(c, 6) for c in cosines]}), "
          f"top-1 agreement={agreement:.4f}, "
          f"decisive agreement={decisive_agreement:.4f} "
          f"({int(decisive.sum())}/{seq_len} positions with margin >= "
          f"{DECISIVE_MARGIN}), unstable positions={unstable.tolist()}")

    assert cos >= E2E_COSINE_MIN
    assert agreement >= E2E_TOP1_AGREEMENT_MIN
    assert decisive_agreement == 1.0
    assert unstable.size == 0, (
        f"top-1 changed between rounds at positions {unstable.tolist()}; "
        f"the fused path is expected to be bitwise reproducible under "
        f"HCCL_DETERMINISTIC=true")


@pytest.fixture(scope="session")
def perf_rows(model_dir, parallel_size, tmp_path_factory):
    """Run the benchmark once and return ``{prefill_tokens: row}``."""
    out = tmp_path_factory.mktemp("qwen3_30b_a3b_perf") / "perf.json"
    run_torchrun(
        PERF_RUNNER,
        ["--model-dir", model_dir, "--graph",
         "--lengths", ",".join(str(s) for s in SHAPES),
         "--max-new-tokens", str(MAX_NEW_TOKENS),
         "--json-out", str(out)],
        parallel_size,
    )
    with open(out, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = {row["prefill_tokens"]: row for row in payload["results"]}
    missing = [s for s in SHAPES if s not in rows]
    assert not missing, f"benchmark did not report shapes {missing}"
    return rows


@pytest.mark.L1
@pytest.mark.ascend_a2
@pytest.mark.perf
@pytest.mark.parametrize("shape", SHAPES)
def test_prefill_decode_latency(perf_rows, shape):
    """Each bucket reports a positive, cap-bounded prefill and decode latency."""
    row = perf_rows[shape]
    print(f"prefill {shape}: {row['prefill_ms']:.2f} ms, "
          f"decode {row['decode_ms_per_tok']:.2f} ms/tok "
          f"(bucket {row['prefill_bucket']}, capture {row['prefill_capture_s']:.2f}s)")
    assert row["decode_tokens"] == MAX_NEW_TOKENS
    assert 0.0 < row["prefill_ms"] < PREFILL_CAP_MS
    assert 0.0 < row["decode_ms_per_tok"] < DECODE_CAP_MS


@pytest.mark.L1
@pytest.mark.ascend_a2
@pytest.mark.perf
def test_prefill_scales_with_length(perf_rows):
    """Prefill latency grows with the token count (graph replay of each bucket)."""
    small, mid, large = (perf_rows[s]["prefill_ms"] for s in (128, 512, 2048))
    print(f"prefill scaling: 128→{small:.2f} ms, 512→{mid:.2f} ms, "
          f"2048→{large:.2f} ms")
    assert small < mid < large


@pytest.mark.L1
@pytest.mark.ascend_a2
@pytest.mark.perf
def test_decode_latency_is_stable(perf_rows):
    """Decode latency per token does not depend on the prefill bucket."""
    values = [row["decode_ms_per_tok"] for row in perf_rows.values()]
    spread = max(values) / min(values)
    print(f"decode spread across buckets: {spread:.3f}x "
          f"({min(values):.2f}..{max(values):.2f} ms/tok)")
    assert spread <= DECODE_SPREAD_MAX


@pytest.mark.L1
@pytest.mark.ascend_a2
@pytest.mark.perf
def test_graph_capture_completes(perf_rows, parallel_size):
    """Every bucket captured its prefill graph (no silent fallback to eager)."""
    captures = [row["prefill_capture_s"] for row in perf_rows.values()]
    print(f"prefill graph capture: {[round(c, 2) for c in captures]} s")
    assert parallel_size == tp_size()
    assert all(c > 0.0 for c in captures)
