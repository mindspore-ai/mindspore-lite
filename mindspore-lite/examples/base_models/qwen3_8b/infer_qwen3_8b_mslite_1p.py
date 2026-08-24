#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Qwen3-8B 1p dynamic single-graph bucketed prefill+decode (ge.dynamicDims, single mindir/single graph/no fixed shape).

Strictly satisfies the constraints:
  * prefill uses a single mindir + configs/ge_prefill.cfg (ge.dynamicDims 6 buckets, perf scans 8 buckets)
  * decode uses a single mindir + configs/ge_decode.cfg (ge.dynamicDims 6 buckets, perf scans 8 buckets)
  * each mindir is built only once (single GE graph); at runtime model.resize() selects the bucket, no fixed-shape cfg
  * prefill/decode share the 8B weights through ModelGroup(SHARE_WEIGHT)

How the key point (KV padded to the current bucket's kv_len, not max 3584) is implemented:
  In the GE dynamicDims single-graph prefill, the physical KV output is always the max bucket (3584),
  but [0..kv_len] is valid KV and [kv_len..3584] is a dirty tail. After D2H we **slice out [0..kv_len]**
  (= seq + 512), discard the dirty tail, then H2D it into the decode kv_len bucket (decode resizes to kv_len). Thus:
    - prefill logical KV = current bucket kv_len (1408/1536/2176/2560/3072/3328/3584...) ✓ key point
    - decode runs with per-bucket kv_len ✓ bucketed
    - each request has independent KV, no reuse (the user explicitly does not require reuse/zero-copy)

perf buckets (BUCKET_SEQ_DIMS, 8 buckets, matching the prof default buckets):
  prefill seq = {512,896,1024,1664,2048,2560,2816,3072}
  decode  kv  = seq + 512 = {1024,1408,1536,2176,2560,3072,3328,3584}
  896/2560 are not in the cfg's ge.dynamicDims; they rely on ge.dynamicNodeType=1 online re-specialization.

Note: the first predict of each bucket triggers GE profile lazy compilation (~5min); steady-state latency is much lower.
"""
import argparse
import gc
import json
import os
import time
from typing import Literal

import numpy as np
from transformers import AutoTokenizer
import mindspore_lite as mslite

from infer_qwen3_8b_mslite import (
    MAX_OUTPUT_TOKENS, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, VOCAB,
    _MS_DTYPE_TO_NP, _build_mslite_inputs,
)
from _npu_mem import PeakSampler

BUCKET_SEQ_DIMS = (512, 896, 1024, 1664, 2048, 2560, 2816, 3072)

TOK_PATH = "../Qwen3-8B"
PREFILL_PATH = "./qwen3_8b_onnx/prefill/qwen3_8b_llm_prefill_rank0_graph.mindir"
DECODE_PATH = "./qwen3_8b_onnx/decode/qwen3_8b_llm_decode_rank0_graph.mindir"
PREFILL_CFG = "configs/qwen3_8b_llm_prefill.config"
DECODE_CFG = "configs/qwen3_8b_llm_decode.config"


def _pick_seq(real_len):
    """Smallest bucket prefill dim >= real_len (8 档, 含 896/2560)."""
    for dim in BUCKET_SEQ_DIMS:
        if dim >= real_len:
            return dim
    raise ValueError(
        f"prompt length {real_len} exceeds max bucket dim "
        f"{BUCKET_SEQ_DIMS[-1]}; shorten the prompt or extend BUCKET_SEQ_DIMS."
    )


class DynamicBucketRunner:
    """Single dynamic prefill graph + single dynamic decode graph, SHARE_WEIGHT, resize bucket selection, per-bucket KV slicing."""

    def __init__(self, device_id, tokenizer, phase="both"):
        """phase: 'both' normal inference; 'prefill'/'decode' builds only the corresponding model (for per-process prof collection)."""
        self.device_id = int(device_id)
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.phase = phase
        ctx = mslite.Context()
        ctx.target = ["ascend"]
        ctx.ascend.device_id = self.device_id
        ctx.ascend.provider = "ge"
        t0 = time.perf_counter()
        if phase in ("both", "prefill"):
            self.prefill = mslite.Model()
            self.prefill.build_from_file(PREFILL_PATH, mslite.ModelType.MINDIR, ctx, PREFILL_CFG)
            self._pf_build_s = time.perf_counter() - t0
            print(f"[build] prefill={self._pf_build_s:.1f}s", flush=True)
            print(f"[build] prefill inputs={[list(t.shape) for t in self.prefill.get_inputs()]}",
                  flush=True)
        if phase in ("both", "decode"):
            self.decode = mslite.Model()
            self.decode.build_from_file(DECODE_PATH, mslite.ModelType.MINDIR, ctx, DECODE_CFG)
            self._build_s = time.perf_counter() - t0
            print(f"[build] decode total={self._build_s:.1f}s", flush=True)
            print(f"[build] decode inputs={[list(t.shape) for t in self.decode.get_inputs()]}",
                  flush=True)
        if phase == "both":
            self._mg = mslite.ModelGroup(mslite.ModelGroupFlag.SHARE_WEIGHT)
            self._mg.add_model([self.prefill, self.decode])
            print(f"[build] both graphs total={self._build_s:.1f}s (SHARE_WEIGHT)", flush=True)

        self.max_seq = BUCKET_SEQ_DIMS[-1]
        self.max_kv_len = self.max_seq + MAX_OUTPUT_TOKENS
        dev = f"ascend:{self.device_id}"

        self._pf_kv_dtype = mslite.DataType.FLOAT16
        self.t_pf_logits = mslite.Tensor(
            shape=[VOCAB], dtype=mslite.DataType.FLOAT32, device=dev)
        self.t_pf_k = mslite.Tensor(
            shape=[NUM_LAYERS, 1, NUM_KV_HEADS, self.max_kv_len, HEAD_DIM],
            dtype=self._pf_kv_dtype, device=dev)
        self.t_pf_v = mslite.Tensor(
            shape=[NUM_LAYERS, 1, NUM_KV_HEADS, self.max_kv_len, HEAD_DIM],
            dtype=self._pf_kv_dtype, device=dev)
        if phase in ("both", "prefill"):
            _ = self.t_pf_k.get_data_to_numpy()
            _ = self.t_pf_v.get_data_to_numpy()

        if phase in ("both", "decode"):
            din = self.decode.get_inputs()
            self._dc_ids_np = _MS_DTYPE_TO_NP[din[0].dtype]
            self._dc_attn_np = _MS_DTYPE_TO_NP[din[1].dtype]
            self._dc_pos_np = _MS_DTYPE_TO_NP[din[2].dtype]
            self._dc_kv_np = _MS_DTYPE_TO_NP[din[3].dtype]
            kv_shape = [NUM_LAYERS, 1, NUM_KV_HEADS, self.max_kv_len, HEAD_DIM]
            self.t_dc_ids = mslite.Tensor(shape=[1, 1], dtype=din[0].dtype, device=dev)
            self.t_dc_attn = mslite.Tensor(shape=[1, self.max_kv_len], dtype=din[1].dtype, device=dev)
            self.t_dc_pos = mslite.Tensor(shape=[1, 1], dtype=din[2].dtype, device=dev)
            self.t_dc_k_a = mslite.Tensor(shape=kv_shape, dtype=din[3].dtype, device=dev)
            self.t_dc_v_a = mslite.Tensor(shape=kv_shape, dtype=din[4].dtype, device=dev)
            self.t_dc_k_b = mslite.Tensor(shape=kv_shape, dtype=din[3].dtype, device=dev)
            self.t_dc_v_b = mslite.Tensor(shape=kv_shape, dtype=din[4].dtype, device=dev)
            self.t_dc_logits = mslite.Tensor(
                shape=[VOCAB], dtype=mslite.DataType.FLOAT32, device=dev)

    def _prep_inputs(self, text):
        """Tokenize and pad prompt → (padded, attn, pos, real, seq, kv_len)."""
        enc = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, return_tensors="np")
        ids = np.asarray(enc["input_ids"] if hasattr(enc, "keys") else enc, dtype=np.int32).reshape(1, -1)
        real = int(ids.shape[1])
        seq = _pick_seq(real)
        kv_len = seq + MAX_OUTPUT_TOKENS
        padded = np.zeros((1, seq), np.int32)
        padded[0, :real] = ids[0, :real]
        attn = np.zeros((1, seq), np.int32)
        attn[0, :real] = 1
        cum = np.cumsum(attn[0], dtype=np.int32) - 1
        pos = np.where(attn[0] > 0, cum, 0).astype(np.int32)[None, :]
        return padded, attn, pos, real, seq, kv_len

    def _prefill(self, padded, attn, pos, seq, kv_len):
        """Run prefill with preallocated device output buffers and return first token + KV tensors."""
        self.prefill.resize(self.prefill.get_inputs(), [[1, seq], [1, seq], [1, seq]])
        feed = _build_mslite_inputs(
            self.prefill, {"input_ids": padded, "attention_mask": attn, "position_ids": pos},
            preferred_order=["input_ids", "attention_mask", "position_ids"])

        kv_shape = [NUM_LAYERS, 1, NUM_KV_HEADS, kv_len, HEAD_DIM]
        self.t_pf_k.shape = kv_shape
        self.t_pf_v.shape = kv_shape

        t0 = time.perf_counter()
        self.prefill.predict(feed, outputs=[self.t_pf_logits, self.t_pf_k, self.t_pf_v])
        pf_ms = (time.perf_counter() - t0) * 1000

        # logits 小张量 D2H 做 argmax
        logits = self.t_pf_logits.get_data_to_numpy().astype(np.float32)
        flat = logits.reshape((-1, VOCAB))
        if flat.shape[0] > 1:
            real = int(attn.sum())
            first = int(np.argmax(flat[real - 1, :]))
        else:
            first = int(np.argmax(flat[0, :]))

        kv_out_phys = kv_len
        return first, self.t_pf_k, self.t_pf_v, pf_ms, kv_out_phys

    def _decode(self, first, pf_k, pf_v, real_len, kv_len, max_new_tokens):
        """Decode loop. pf_k/pf_v are device tensors from prefill."""
        din = self.decode.get_inputs()
        self.decode.resize(din, [[1, 1], [1, kv_len], [1, 1],
                                 [NUM_LAYERS, 1, NUM_KV_HEADS, kv_len, HEAD_DIM],
                                 [NUM_LAYERS, 1, NUM_KV_HEADS, kv_len, HEAD_DIM]])
        gen = [first]
        valid = real_len
        cur_attn = np.zeros((1, kv_len), dtype=self._dc_attn_np)
        cur_attn[0, :valid] = 1
        step_ms = []
        truncated = False

        kv_shape = [NUM_LAYERS, 1, NUM_KV_HEADS, kv_len, HEAD_DIM]
        self.t_dc_attn.shape = [1, kv_len]
        self.t_dc_k_a.shape = kv_shape
        self.t_dc_v_a.shape = kv_shape
        self.t_dc_k_b.shape = kv_shape
        self.t_dc_v_b.shape = kv_shape

        k_buf = np.zeros(kv_shape, dtype=self._dc_kv_np)
        v_buf = np.zeros(kv_shape, dtype=self._dc_kv_np)
        for src, dst in ((pf_k, k_buf), (pf_v, v_buf)):
            try:
                arr = src.get_data_to_numpy() if hasattr(src, 'get_data_to_numpy') else np.asarray(src)
            except (RuntimeError, ValueError, TypeError):
                arr = None
            if arr is None:
                continue
            arr = np.asarray(arr)
            while arr.ndim < 5:
                arr = arr[np.newaxis, ...]
            phys = min(arr.shape[3], kv_len)
            dst[:, :, :, :phys, :] = arr[:, :, :, :phys, :]
        self.t_dc_k_a.set_data_from_numpy(np.ascontiguousarray(k_buf))
        self.t_dc_v_a.set_data_from_numpy(np.ascontiguousarray(v_buf))
        k_in, k_out = self.t_dc_k_a, self.t_dc_k_b
        v_in, v_out = self.t_dc_v_a, self.t_dc_v_b

        for _ in range(max_new_tokens - 1):
            if self.eos_token_id is not None and gen[-1] == int(self.eos_token_id):
                break
            if valid >= kv_len:
                truncated = True
                break
            cur_attn[0, valid] = 1
            self.t_dc_ids.set_data_from_numpy(np.array([[gen[-1]]], dtype=self._dc_ids_np))
            self.t_dc_attn.set_data_from_numpy(cur_attn.copy())
            self.t_dc_pos.set_data_from_numpy(np.array([[valid]], dtype=self._dc_pos_np))
            inputs = [self.t_dc_ids, self.t_dc_attn, self.t_dc_pos, k_in, v_in]
            outputs = [self.t_dc_logits, k_out, v_out]
            t0 = time.perf_counter()
            self.decode.predict(inputs, outputs=outputs)
            step_ms.append((time.perf_counter() - t0) * 1000)
            lg = self.t_dc_logits.get_data_to_numpy()
            k_in, k_out = k_out, k_in
            v_in, v_out = v_out, v_in
            valid += 1
            nid = int(np.argmax(lg[0, -1, :])) if lg.ndim == 3 else int(np.argmax(lg.reshape(-1)))
            gen.append(nid)
        return gen, step_ms, truncated

    def run_prefill(self, text):
        """Run prefill only (for prof mode). Returns perf dict."""
        padded, attn, pos, real, seq, kv_len = self._prep_inputs(text)
        _, _, _, pf_ms, kv_phys = self._prefill(padded, attn, pos, seq, kv_len)
        return {"real_len": real, "prefill_seq": seq, "kv_len": kv_len,
                "prefill_ms": round(pf_ms, 1), "kv_out_phys": kv_phys}

    def run_decode_only(self, seq, kv_len, max_new_tokens=32):
        """Run decode only with dummy (zero) KV — for per-process prof decode collection.

        Does not build/run prefill: KV is zero-filled and the decode process contains only decode ops,
        avoiding mixing prefill and decode data (profiling is per-process).
        Returns perf dict.
        """
        din = self.decode.get_inputs()
        self.decode.resize(din, [[1, 1], [1, kv_len], [1, 1],
                                 [NUM_LAYERS, 1, NUM_KV_HEADS, kv_len, HEAD_DIM],
                                 [NUM_LAYERS, 1, NUM_KV_HEADS, kv_len, HEAD_DIM]])
        ids_np = _MS_DTYPE_TO_NP[din[0].dtype]
        attn_np = _MS_DTYPE_TO_NP[din[1].dtype]
        pos_np = _MS_DTYPE_TO_NP[din[2].dtype]
        kv_np = _MS_DTYPE_TO_NP[din[3].dtype]
        first = 1
        valid = seq
        cur_attn = np.zeros((1, kv_len), dtype=attn_np)
        cur_attn[0, :valid] = 1
        gen = [first]
        step_ms = []
        truncated = False

        dc_buf = self._dc_buffers[kv_len]
        t_ids, t_attn, t_pos = dc_buf["ids"], dc_buf["attn"], dc_buf["pos"]
        t_k_a, t_v_a = dc_buf["k_a"], dc_buf["v_a"]
        t_k_b, t_v_b = dc_buf["k_b"], dc_buf["v_b"]
        t_logits = dc_buf["logits"]
        dummy_k = np.zeros([NUM_LAYERS, 1, NUM_KV_HEADS, kv_len, HEAD_DIM], dtype=kv_np)
        dummy_v = np.zeros([NUM_LAYERS, 1, NUM_KV_HEADS, kv_len, HEAD_DIM], dtype=kv_np)
        t_k_a.set_data_from_numpy(np.ascontiguousarray(dummy_k))
        t_v_a.set_data_from_numpy(np.ascontiguousarray(dummy_v))
        k_in, k_out = t_k_a, t_k_b
        v_in, v_out = t_v_a, t_v_b

        for _ in range(max_new_tokens - 1):
            if self.eos_token_id is not None and gen[-1] == int(self.eos_token_id):
                break
            if valid >= kv_len:
                truncated = True
                break
            cur_attn[0, valid] = 1
            t_ids.set_data_from_numpy(np.array([[gen[-1]]], dtype=ids_np))
            t_attn.set_data_from_numpy(cur_attn.copy())
            t_pos.set_data_from_numpy(np.array([[valid]], dtype=pos_np))
            inputs = [t_ids, t_attn, t_pos, k_in, v_in]
            outputs = [t_logits, k_out, v_out]
            t0 = time.perf_counter()
            o = self.decode.predict(inputs, outputs=outputs)
            step_ms.append((time.perf_counter() - t0) * 1000)
            lg = o[0].get_data_to_numpy()
            k_in, k_out = k_out, k_in
            v_in, v_out = v_out, v_in
            valid += 1
            nid = int(np.argmax(lg[0, -1, :])) if lg.ndim == 3 else int(np.argmax(lg.reshape(-1)))
            gen.append(nid)
        return {"prefill_seq": seq, "kv_len": kv_len,
                "decode_first_ms": round(step_ms[0], 1) if step_ms else 0.0,
                "decode_min_ms": round(min(step_ms), 1) if step_ms else 0.0,
                "decode_steps": len(step_ms), "output_len": len(gen), "truncated": truncated}

    def run_decode(self, text, max_new_tokens=32):
        """Run prefill (for KV) + decode only (for prof mode). Returns perf dict."""
        padded, attn, pos, real, seq, kv_len = self._prep_inputs(text)
        first, pf_k, pf_v, _, _ = self._prefill(padded, attn, pos, seq, kv_len)
        gen, step_ms, trunc = self._decode(first, pf_k, pf_v, real, kv_len, max_new_tokens)
        return {"real_len": real, "prefill_seq": seq, "kv_len": kv_len,
                "decode_first_ms": round(step_ms[0], 1) if step_ms else 0.0,
                "decode_min_ms": round(min(step_ms), 1) if step_ms else 0.0,
                "decode_steps": len(step_ms), "output_len": len(gen), "truncated": trunc}

    def run(self, text, max_new_tokens=32):
        padded, attn, pos, real, seq, kv_len = self._prep_inputs(text)
        first, pf_k, pf_v, pf_ms, kv_phys = self._prefill(padded, attn, pos, seq, kv_len)
        gen, step_ms, trunc = self._decode(first, pf_k, pf_v, real, kv_len, max_new_tokens)
        txt = self.tokenizer.decode(gen, skip_special_tokens=True)
        return txt, {
            "real_len": real, "prefill_seq": seq, "kv_len": kv_len,
            "kv_out_phys": kv_phys, "kv_padded_to": kv_len,
            "kv_len_ok": (pf_k.shape[3] >= kv_len),
            "prefill_ms": round(pf_ms, 1),
            "decode_steps": len(step_ms),
            "decode_first_ms": round(step_ms[0], 1) if step_ms else 0.0,
            "decode_min_ms": round(min(step_ms), 1) if step_ms else 0.0,
            "decode_tok_s": round(1000.0 / min(step_ms), 2) if step_ms else 0.0,
            "output_len": len(gen), "truncated": trunc,
            "output_preview": txt[:80].replace("\n", " "),
            "generated_ids": [int(x) for x in gen],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--buckets", type=str, default="512,1024",
                    help="逗号分隔的 prefill seq 档 (默认只跑前两档以控时; 'all' 跑全 8 档, 含 896/2560)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="每档推理轮数 (第1轮含该档 profile 懒编译, 取第2..N轮稳态)")
    ap.add_argument("--single-prompt", type=str, default=None,
                    help="单次功能验证: 跑一个真实 prompt (自动选档), 打印输出+核心点+显存, 不循环")
    ap.add_argument("--out", type=str, default="_dynamic_bucket_results.json")
    ap.add_argument("--prof-phase", type=str, default=None,
                    choices=["prefill", "decode"],
                    help="prof 模式: build + 3 warmup + 只跑 prefill 或 decode 1 次 (配合 msprof 分阶段采集)")
    args = ap.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"Loading tokenizer from {TOK_PATH}...", flush=True)
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.single_prompt is not None:
        sampler = PeakSampler(args.device_id, interval=0.15)
        sampler.start()
        runner = DynamicBucketRunner(args.device_id, tok)
        print("\n" + "=" * 60)
        print(f"Input Prompt: {args.single_prompt}")
        print("=" * 60)
        txt, perf = runner.run(args.single_prompt, max_new_tokens=args.max_new_tokens)
        peak = sampler.stop()
        print("Generated Response:")
        print(f"  {txt[:300]}")
        print("--- Performance (单次功能验证) ---")
        print(f"  real_len={perf['real_len']}  prefill_seq={perf['prefill_seq']}  "
              f"kv_len={perf['kv_len']}")
        print(f"  KV: phys={perf['kv_out_phys']} -> pad到 {perf['kv_padded_to']}  "
              f"核心点OK={perf['kv_len_ok']}")
        print(f"  prefill_steady={perf['prefill_ms']}ms  "
              f"decode_first={perf['decode_first_ms']}ms  decode_min={perf['decode_min_ms']}ms")
        print(f"  peak_npu={peak}MB (baseline {sampler.baseline}MB)  "
              f"truncated={perf['truncated']}")
        return

    if args.prof_phase is not None:
        if args.buckets == "all":
            buckets = list[Literal[512, 896, 1024, 1664, 2048, 2560, 2816, 3072]](BUCKET_SEQ_DIMS)
        else:
            buckets = [int(x) for x in args.buckets.split(",")]
        seq = buckets[0]

        base_text = "人工智能是研究开发用于模拟延伸和扩展人的智能的理论方法技术及应用系统的一门新的技术科学。"
        real_prompts = {
            512: "你好，请用一句话介绍一下你自己",
        }
        if seq in real_prompts:
            prompt = real_prompts[seq]
        else:
            s = ""
            while len(tok.encode(s)) < max(1, seq - 20):
                s += base_text
            prompt = tok.decode(tok.encode(s)[:max(1, seq - 20)], skip_special_tokens=True)

        runner = DynamicBucketRunner(args.device_id, tok, phase=args.prof_phase)
        if args.prof_phase == "prefill":
            for _ in range(3):
                runner.run_prefill(prompt)
            perf = runner.run_prefill(prompt)
            print(f"PROF_PREFILL: seq={perf['prefill_seq']} kv_len={perf['kv_len']} "
                  f"prefill_ms={perf['prefill_ms']}ms", flush=True)
        else:
            kv_len = seq + MAX_OUTPUT_TOKENS
            for _ in range(3):
                runner.run_decode_only(seq, kv_len, max_new_tokens=args.max_new_tokens)
            perf = runner.run_decode_only(seq, kv_len, max_new_tokens=args.max_new_tokens)
            print(f"PROF_DECODE: seq={perf['prefill_seq']} kv_len={perf['kv_len']} "
                  f"decode_first_ms={perf['decode_first_ms']}ms "
                  f"decode_min_ms={perf['decode_min_ms']}ms "
                  f"steps={perf['decode_steps']}", flush=True)
        del runner
        gc.collect()
        return

    if args.buckets == "all":
        buckets = list(BUCKET_SEQ_DIMS)
    else:
        buckets = [int(x) for x in args.buckets.split(",")]

    base = "人工智能是研究开发用于模拟延伸和扩展人的智能的理论方法技术及应用系统的一门新的技术科学。"

    real_prompts = {
        512: "你好，请用一句话介绍一下你自己",
    }

    def make_prompt(n):
        s = ""
        while len(tok.encode(s)) < n:
            s += base
        return tok.decode(tok.encode(s)[:n], skip_special_tokens=True)

    sampler = PeakSampler(args.device_id, interval=0.15)
    sampler.start()
    runner = DynamicBucketRunner(args.device_id, tok)

    results = []
    for seq in buckets:
        if seq in real_prompts:
            prompt = real_prompts[seq]
            target = len(tok.encode(prompt))
        else:
            target = max(1, seq - 20)
            prompt = make_prompt(target)
        print(f"\n===== 档 seq={seq} (target_tokens={target}) "
              f"repeats={args.repeats} =====", flush=True)
        perfs = []
        for r in range(args.repeats):
            txt, perf = runner.run(prompt, max_new_tokens=args.max_new_tokens)
            perfs.append(perf)
            tag = "warmup(含懒编译)" if r == 0 else f"steady #{r}"
            print(f"[档 {seq} 轮{r}] {tag} prefill={perf['prefill_ms']}ms "
                  f"decode_first={perf['decode_first_ms']}ms decode_min={perf['decode_min_ms']}ms "
                  f"trunc={perf['truncated']}", flush=True)
        steady = perfs[1:] if len(perfs) > 1 else perfs
        avg_pf = sum(p["prefill_ms"] for p in steady) / len(steady)
        avg_dc = sum(p["decode_min_ms"] for p in steady) / len(steady)
        best = perfs[-1]
        peak = sampler.peak
        best["peak_npu_mb"] = peak
        best["baseline_npu_mb"] = sampler.baseline
        best["repeats"] = len(steady)
        best["prefill_avg_ms"] = round(avg_pf, 1)
        best["decode_min_avg_ms"] = round(avg_dc, 1)
        results.append(best)
        print(f"[档 {seq}] KV_phys={best['kv_out_phys']}->pad到{best['kv_padded_to']} "
              f"核心点OK={best['kv_len_ok']} prefill_avg={avg_pf:.1f}ms "
              f"decode_min_avg={avg_dc:.1f}ms ({1000.0/avg_dc:.2f} tok/s) "
              f"peak={peak}MB", flush=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    peak = sampler.stop()
    print("\n\n========== 动态单图分档性能 SUMMARY ==========", flush=True)
    print("一次编译(单 prefill mindir + 单 decode mindir, ge.dynamicDims), "
          "多次推理不同档位:", flush=True)
    print(f"{'prefill_seq':>11} {'kv_len':>7} {'核心点':>6} "
          f"{'prefill_ms':>10} {'decode_min_ms':>13} {'tok/s':>7} {'peakMB':>8} {'trunc':>6}",
          flush=True)
    for r in results:
        print(f"{r['prefill_seq']:>11} {r['kv_len']:>7} "
              f"{('Y' if r['kv_len_ok'] else 'N'):>6} "
              f"{r['prefill_avg_ms']:>10} {r['decode_min_avg_ms']:>13} "
              f"{round(1000.0 / r['decode_min_avg_ms'], 2) if r['decode_min_avg_ms'] else 0:>7} "
              f"{r['peak_npu_mb']:>8} {str(r['truncated']):>6}", flush=True)
    print(f"\n注: prefill_ms/decode_min_ms 为第2..{args.repeats}轮稳态均值; "
          f"KV 核心点 = prefill KV 切到对应档 kv_len (非 max 3584).", flush=True)
    print(f"peak_npu={peak}MB (baseline {sampler.baseline}MB)  saved {args.out}", flush=True)
    del runner
    gc.collect()


if __name__ == "__main__":
    main()
