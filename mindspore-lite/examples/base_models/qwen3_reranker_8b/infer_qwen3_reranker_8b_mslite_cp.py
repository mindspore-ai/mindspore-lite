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
"""Qwen3-Reranker-8B CP2 + shared-prefix inference (T2Reranking scoring, zero-copy KV).

Single inference entry point: each rank worker builds the prefix and suffix graphs in the
same process; the prefix K/V output is written into pre-allocated device tensors
(``mslite.Tensor(device="ascend:N")``) and then consumed in place by the suffix graph, so
there is no H2D/D2H copy and no intermediate npy file.

Contract:
  * prefix graph: FIXED-shape build (``ge.inputShape`` materialized, no dynamicDims/
    dynamicNodeType). Never call ``model.resize()`` on it: mslite still reports ``[-1]``
    for a static build, and resizing pulls the graph back onto the GE dynamic path, which
    corrupts K/V from layer 1 on. Predict without resize.
  * suffix graph: 4 dynamic buckets (dynamicDims + dynamicNodeType=1), resized per bucket;
    its K/V inputs are the prefix device tensors, read in place.
  * suffix ``predict()`` must pass explicit ``outputs=[t_logits]``: GE pads vocab 151669 to
    151680 and the default output path fails in ``ConvertGeTensorNoCopy``.
  * score = softmax([logit("no"), logit("yes")])[1] of the owner rank (CP_SIZE-1).

Usage:
  python infer_qwen3_reranker_8b_mslite_cp.py \\
    --device-ids 2,3 \\
    --model-id /path/to/Qwen3-Reranker-8B \\
    --mindir-root <dir with rank{0,1}/{prefix,suffix}/..._graph.mindir> \\
    --cfg-dir <dir with qwen3_reranker_8b_{prefix,suffix}.config> \\
    --prefix-bucket 512 --prefix-text-file <prefix.txt> \\
    --suffix-buckets 512,1024,1536,2048 \\
    --queries <tsv: query<tab>doc, newlines inside a field escaped> --json-out scores.json \\
    [--warmup 1 --work-dir ./run]

Note: pass the shared prefix as a file (--prefix-text-file, read byte-for-byte).
    --prefix-text with a shell "$(cat f)" drops the trailing newline and misaligns the
    reference.
    tsv fields support \\n / \\t / \\\\ escapes, so multi-line templates fit on one line.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import mindspore_lite as mslite
import numpy as np

os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "600")
os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "21500-21600")

# Rank stagger / settle time (seconds): keeps the two processes from initializing the
# device and HCCL at the same moment. Tunable with DSH_RANK_STAGGER_S / DSH_SETTLE_S.
RANK_STAGGER_S = float(os.environ.get("DSH_RANK_STAGGER_S", "6"))
SETTLE_S = float(os.environ.get("DSH_SETTLE_S", "6"))


def _read_ge_config(ge_cfg):
    """Read the whole .config and return [(section, [line, ...]), ...].

    Section and key order are preserved; comment and blank lines are dropped (the
    original .config still keeps them).
    """
    sections = []
    with open(ge_cfg, "r", encoding="utf-8") as f:
        for raw in f:
            st = raw.strip()
            if not st or st.startswith("#"):
                continue
            if st.startswith("[") and st.endswith("]"):
                sections.append((st[1:-1], []))
            elif sections:
                sections[-1][1].append(st)
    return sections


def _write_hccl_config(device_ids, run_dir, ge_cfg, tag):
    """Generate rank_table.json and the per-graph mini ini.

    Every key of ``ge_cfg`` is carried over first; the items rendered below then override
    same-named keys, so nothing from the .config is silently dropped:

    * ``[ascend_context]``: ``rank_table_file``, ``plugin_custom_ops=All``

    GE compile/model caching is not injected here: mslite picks it up from the
    ``model_cache_mode`` / ``model_cache_dir`` keys of ``[ascend_context]``, which are
    passed through verbatim.
    """
    os.makedirs(run_dir, exist_ok=True)
    devs = ",".join(f'{{"device_id":"{d}","rank_id":"{i}"}}'
                    for i, d in enumerate(device_ids))
    rt = ('{"version":"1.0","server_count":"1","server_list":['
          '{"server_id":"127.0.0.1","device":[' + devs + '],'
          '"host_nic_ip":"reserve"}],"status":"completed"}')
    rt_path = os.path.join(run_dir, "rank_table.json")
    with open(rt_path, "w", encoding="utf-8") as f:
        f.write(rt)

    # Items rendered here (same-named keys win); values are complete "key=value" lines
    rendered = {
        "ascend_context": {
            "rank_table_file": f"rank_table_file={rt_path}",
            "plugin_custom_ops": "plugin_custom_ops=All",
        },
    }

    sections = _read_ge_config(ge_cfg)
    have = {name for name, _ in sections}
    for name in rendered:                       # add sections missing in .config
        if name not in have:
            sections.append((name, []))

    cfg_path = os.path.join(run_dir, f"config_{tag}.ini")
    with open(cfg_path, "w", encoding="utf-8") as f:
        for name, lines in sections:
            over = rendered.get(name, {})
            f.write(f"[{name}]\n")
            written = set()
            for ln in lines:
                key = ln.split("=", 1)[0].strip()
                if key in over:                 # override the key from the .config
                    if key not in written:
                        f.write(over[key] + "\n")
                        written.add(key)
                    continue
                f.write(ln + "\n")
            for key, val in over.items():
                if key not in written:
                    f.write(val + "\n")
            f.write("\n")
    return cfg_path


def _build(path, cfg, tag=""):
    """Build one MindIR graph on the Ascend GE backend using cfg (device/rank/ini)."""
    ctx = mslite.Context()
    ctx.target = ["ascend"]
    ctx.ascend.device_id = int(cfg["device"])
    ctx.ascend.rank_id = int(cfg["rank"])
    ctx.ascend.provider = "ge"
    ini = cfg.get("ini_" + tag) or cfg.get("ini")
    if ini is None:
        raise KeyError(f"no ini for tag={tag!r} in cfg keys={list(cfg)}")
    model = mslite.Model()
    model.build_from_file(path, mslite.ModelType.MINDIR, ctx, ini)
    return model


def _cast_feed(model, arrays):
    """Cast numpy arrays to the declared input dtype of the model."""
    np_map = {mslite.DataType.INT32: np.int32, mslite.DataType.INT64: np.int64,
              mslite.DataType.FLOAT16: np.float16, mslite.DataType.FLOAT32: np.float32}
    out = []
    for t, v in zip(model.get_inputs(), arrays):
        want = np_map.get(t.dtype)
        if want is not None and v.dtype != want:
            v = v.astype(want)
        out.append(mslite.Tensor(np.ascontiguousarray(v)))
    return out


def _maybe_resize(model, shapes):
    """Resize only dynamic graphs (static builds have no -1 dims)."""
    ins = model.get_inputs()
    if any(-1 in t.shape for t in ins):
        model.resize(ins, shapes)

KV_HEADS = 8
KV_LAYERS = 36
HEAD_DIM = 128


def _unescape(s):
    """Restore the \\n / \\t / \\\\ escapes used in tsv fields.

    A line-based tsv cannot hold newlines, so template newlines are written as \\n;
    fields without a backslash are returned unchanged.
    """
    if "\\" not in s:
        return s
    return (s.replace("\\\\", "\x00").replace("\\n", "\n")
             .replace("\\t", "\t").replace("\x00", "\\"))


# --------------------------------------------------------------------------
# Combined worker: build prefix+suffix, zero-copy KV on device
# --------------------------------------------------------------------------
def _combined_worker(cfg, job_q, out_q, ready_q, p_bucket, pids, pmask, ppos,
                     warmup, vocab_size):
    """Per-rank worker: build prefix+suffix, run the prefix pass, serve scoring requests."""
    dev = f"ascend:{cfg['device']}"
    # ---- build both graphs ----
    try:
        pre = _build(cfg["prefix"], cfg, "prefix")
        suf = _build(cfg["suffix"], cfg, "suffix")
        ready_q.put(1)
    except Exception:
        import traceback as _tb
        _tb.print_exc()
        ready_q.put(-1)
        return
    # ---- prefix pass (fixed shape; NO resize) ----
    # The prefix graph has no collective communication, so no explicit rendezvous is
    # needed; each worker can run the prefix pass immediately.
    t_k = mslite.Tensor(shape=[KV_LAYERS, 1, p_bucket, KV_HEADS, HEAD_DIM],
                        dtype=mslite.DataType.FLOAT16, device=dev)
    t_v = mslite.Tensor(shape=[KV_LAYERS, 1, p_bucket, KV_HEADS, HEAD_DIM],
                        dtype=mslite.DataType.FLOAT16, device=dev)
    feeds_p = _cast_feed(pre, [pids, pmask, ppos])
    pre.predict(feeds_p, outputs=[t_k, t_v])   # warmup (fixed shape, fast)
    pre.predict(feeds_p, outputs=[t_k, t_v])   # real pass
    print(f"[rank{cfg['rank']}] prefix KV on device {t_k.shape} {t_k.dtype}",
          flush=True)
    # Pre-allocate logits output buffer padded to 32-aligned size.
    # Reason: GE pads the vocab dim to 32-align (reranker vocab=151669 -> 151680);
    # explicit outputs= avoids the ConvertGeTensorNoCopy mismatch on that padded dim.
    vocab_pad = (vocab_size + 31) // 32 * 32
    t_logits = mslite.Tensor(shape=[1, 1, vocab_pad],
                             dtype=mslite.DataType.FLOAT32, device=dev)

    # ---- suffix service (dynamic buckets, device KV inputs) ----
    warm_done = set()
    while True:
        req = job_q.get()
        if req is None:
            break
        s_bucket, sids, smask, spos = req
        if s_bucket not in warm_done:
            _maybe_resize(
                suf, [[1, s_bucket], [1, p_bucket + s_bucket],
                      [1, s_bucket],
                      [KV_LAYERS, 1, p_bucket, KV_HEADS, HEAD_DIM],
                      [KV_LAYERS, 1, p_bucket, KV_HEADS, HEAD_DIM]])
            for _ in range(int(warmup)):
                suf.predict(_zc_feed(suf, [sids, smask, spos], t_k, t_v),
                            outputs=[t_logits])
            warm_done.add(s_bucket)
        t0 = time.perf_counter()
        suf.predict(_zc_feed(suf, [sids, smask, spos], t_k, t_v),
                    outputs=[t_logits])
        el = (time.perf_counter() - t0) * 1000.0
        lg = np.ascontiguousarray(t_logits.get_data_to_numpy()).reshape(-1)[
             :vocab_size]
        out_q.put(("ok", cfg["rank"], lg, el))


def _zc_feed(suf, arrays, t_k, t_v):
    """Build suffix inputs: small host tensors for ids/mask/pos; the two KV
    slots are the prefix-output DEVICE tensors passed through (zero-copy).
    Suffix predict must receive explicit outputs= to avoid the zero-copy
    output path that fails on 32-padded GE logits."""
    np_map = {mslite.DataType.INT32: np.int32, mslite.DataType.INT64: np.int64,
              mslite.DataType.FLOAT16: np.float16, mslite.DataType.FLOAT32: np.float32}
    ins = suf.get_inputs()
    feeds = []
    for i, t in enumerate(ins):
        if i >= 3:                      # past_key_cache / past_value_cache
            feeds.append(t_k if i == 3 else t_v)
            continue
        v = arrays[i]
        want = np_map.get(t.dtype)
        if want is not None and v.dtype != want:
            v = v.astype(want)
        feeds.append(mslite.Tensor(np.ascontiguousarray(v)))
    return feeds


# --------------------------------------------------------------------------
# Driver (same rendezvous pattern as the prefix/suffix pair)
# --------------------------------------------------------------------------
def _run_pair(worker, cfgs, devices, tag, kwargs, n_put=1):
    """Start both rank workers, wait for the build rendezvous, return process/queue handles."""
    out_q = mp.Queue()
    ready_q = mp.Queue()
    job_qs = [mp.Queue() for _ in devices]
    procs = []
    for i, r in enumerate([1, 0]):
        p = mp.Process(target=worker,
                       args=(cfgs[r], job_qs[r], out_q, ready_q) + kwargs)
        p.start()
        procs.append(p)
        if i == 0:
            v = ready_q.get()
            if v != 1:
                raise RuntimeError(f"{tag} rank{r} build failed")
            time.sleep(RANK_STAGGER_S)      # stagger the second rank (DSH_RANK_STAGGER_S)
    v = ready_q.get()
    if v != 1:
        raise RuntimeError(f"{tag} last rank build failed")
    time.sleep(SETTLE_S)                    # short settle before releasing the gate
    for _ in range(n_put):
        for q in job_qs:
            q.put(1)
    return procs, job_qs, out_q


def _parse_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-ids", default="2,3")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--mindir-root", required=True)
    ap.add_argument("--cfg-dir", required=True,
                    help="dir with qwen3_reranker_8b_prefix.config + _suffix.config")
    ap.add_argument("--work-dir", default="zc_run")
    ap.add_argument("--prefix-bucket", type=int, default=512)
    ap.add_argument("--prefix-text", default=None,
                    help="共享前缀文本；与 --prefix-text-file 二选一")
    ap.add_argument("--prefix-text-file", default=None,
                    help="共享前缀文本文件（**逐字节读入**，推荐）——"
                         "shell 的 \"$(cat file)\" 会吃掉文件尾随换行，"
                         "多行/带尾换行的模板请用本参数")
    ap.add_argument("--suffix-buckets", default="512,1024,1536,2048")
    ap.add_argument("--queries", required=True,
                    help="打分输入 tsv：每行 query<tab>doc；字段内换行写成转义形式")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--json-out", required=True)
    return ap.parse_args()


def _load_tokenizer(model_id):
    """Load the tokenizer (fill in pad_token) and the vocab size."""
    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vocab_size = int(AutoConfig.from_pretrained(
        model_id, trust_remote_code=True).vocab_size)
    return tok, vocab_size


def _read_prefix_text(args):
    """Read the shared-prefix text; a file is read byte-for-byte to keep the trailing newline."""
    if args.prefix_text_file:
        with open(args.prefix_text_file, encoding="utf-8") as f:
            return f.read()
    if args.prefix_text is not None:
        return args.prefix_text
    raise SystemExit("需要 --prefix-text 或 --prefix-text-file")


def _make_prefix_feeds(tok, prefix_text, p_bucket, pad_id):
    """Left-align and pad the shared prefix to the fixed bucket; return (ids, mask, pos, real_len)."""
    p_ids_all = tok(prefix_text, add_special_tokens=False)["input_ids"]
    p_real = len(p_ids_all)
    if p_real > p_bucket:
        raise SystemExit(f"prefix {p_real} tokens > bucket {p_bucket}")
    pids = np.full((1, p_bucket), pad_id, np.int64)
    pids[:, :p_real] = p_ids_all
    pmask = np.zeros((1, p_bucket), np.int64)
    pmask[:, :p_real] = 1
    ppos = np.zeros((1, p_bucket), np.int64)
    ppos[:, :p_real] = np.arange(p_real, dtype=np.int64)
    return pids, pmask, ppos, p_real


def _build_cfgs(args, devices):
    """Write the HCCL/GE ini for each rank and collect the graph paths needed to build."""
    os.makedirs(args.work_dir, exist_ok=True)
    run_dir = os.path.join(args.work_dir, "run")
    cfgs = []
    for r, dev in enumerate(devices):
        mdir_p = (f"{args.mindir_root}/rank{r}/prefix/"
                  f"qwen3_reranker_8b_prefix_rank{r}_graph.mindir")
        mdir_s = (f"{args.mindir_root}/rank{r}/suffix/"
                  f"qwen3_reranker_8b_suffix_rank{r}_graph.mindir")
        ini_p = _write_hccl_config(devices, run_dir,
                                   args.cfg_dir + "/qwen3_reranker_8b_prefix.config",
                                   f"zc_r{r}")
        ini_s = _write_hccl_config(devices, run_dir,
                                   args.cfg_dir + "/qwen3_reranker_8b_suffix.config",
                                   f"zc_s{r}")
        cfgs.append({
            "device": dev, "rank": r, "ini": ini_p,
            "ini_prefix": ini_p, "ini_suffix": ini_s,
            "prefix": mdir_p, "suffix": mdir_s,
        })
    return cfgs


def _read_pairs(path):
    """Read the scoring tsv (query<tab>doc per line); \\n / \\t escapes are restored here."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            q, d = line.split("\t", 1)
            pairs.append((_unescape(q), _unescape(d)))
    return pairs


def _make_suffix_feeds(tok, pairs, p_bucket, p_real, s_buckets, pad_id):
    """Encode each (query, doc) pair into suffix bucket inputs (bucket, ids, mask, pos)."""
    max_bucket = max(s_buckets)
    for query, doc in pairs:
        full = tok(query + doc, add_special_tokens=False)["input_ids"]
        s_real = len(full)
        if s_real > max_bucket:
            raise SystemExit(f"suffix {s_real} tokens > max bucket {max_bucket}")
        s_bucket = next((b for b in s_buckets if s_real <= b), max_bucket)
        pad = s_bucket - s_real
        sids = np.full((1, s_bucket), pad_id, np.int64)
        sids[:, pad:] = full
        smask = np.zeros((1, p_bucket + s_bucket), np.int64)
        smask[:, :p_real] = 1
        smask[:, p_bucket + pad:] = 1
        spos = np.zeros((1, s_bucket), np.int64)
        spos[:, pad:] = np.arange(p_real, p_real + s_real, dtype=np.int64)
        yield s_bucket, sids, smask, spos


def _collect_scores(out_q, n_req, owner, false_id, true_id):
    """Gather all suffix replies; return P(yes), latencies and raw logits of the owner rank."""
    results = {}
    for _ in range(2 * n_req):
        msg = out_q.get()
        if msg[0] != "ok":
            raise RuntimeError("suffix worker failed")
        results.setdefault(msg[1], []).append(msg)
    scores, lat, ge_logits = [], [], []
    for msg in results.get(owner, []):
        lg = msg[2].astype(np.float64)
        e = np.exp([lg[false_id], lg[true_id]])
        scores.append(float(e[1] / e.sum()))
        lat.append(msg[3])
        ge_logits.append(lg)
    if len(scores) != n_req:
        raise RuntimeError(f"expected {n_req} owner answers, got {len(scores)}")
    return scores, lat, ge_logits


def main():
    """Entry point: build both graphs, feed the shared prefix, score pair by pair, write JSON."""
    args = _parse_args()
    tok, vocab_size = _load_tokenizer(args.model_id)
    true_id = int(tok.convert_tokens_to_ids("yes"))
    false_id = int(tok.convert_tokens_to_ids("no"))
    devices = [int(x) for x in args.device_ids.split(",")]
    assert len(devices) == 2
    s_buckets = [int(x) for x in args.suffix_buckets.split(",")]
    pad_id = int(tok.pad_token_id)

    # shared prefix feed (fixed bucket, real left-justified)
    prefix_text = _read_prefix_text(args)
    pids, pmask, ppos, p_real = _make_prefix_feeds(
        tok, prefix_text, args.prefix_bucket, pad_id)
    cfgs = _build_cfgs(args, devices)

    # score: feed each suffix pair into the suffix graph
    pairs = _read_pairs(args.queries)
    owner = len(devices) - 1
    procs, job_qs, out_q = _run_pair(
        _combined_worker, cfgs, devices, "zc",
        (args.prefix_bucket, pids, pmask, ppos, args.warmup, vocab_size),
        n_put=0)
    n_req = len(pairs)
    for req in _make_suffix_feeds(tok, pairs, args.prefix_bucket, p_real,
                                  s_buckets, pad_id):
        for q in job_qs:
            q.put(req)
    for q in job_qs:
        q.put(None)
    scores, lat, ge_logits = _collect_scores(out_q, n_req, owner,
                                             false_id, true_id)
    if ge_logits:
        np.save(os.path.join(args.work_dir, "logits_ge.npy"),
                np.asarray(ge_logits))
    for p in procs:
        p.join(timeout=20)
        if p.is_alive():
            p.terminate()
    out = {"scores": scores, "suffix_latency_ms": lat,
           "prefix_real_len": p_real, "prefix_bucket": args.prefix_bucket}
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
