"""Functional end-to-end BLIP2-OPT-2.7B VQA on MindSpore Lite (re-feed, no KV cache).

Basic-functionality path (per "functionality before fusion"): runs vision -> qformer
-> OPT full-forward, re-feeding the growing prefix each greedy step (BLIP-VQA pattern).
The OPT model is the dynamic-seq `blip2_opt_full` MindIR (no KV cache), so each step
runs the full prefix — slower than a fixed KV cache but avoids the fixed-shape cache
trace pitfalls. All model compute is pure mslite + numpy.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
import mindspore_lite as mslite
from transformers import AutoTokenizer, Blip2Processor

_MS = {"FLOAT16": np.float16, "FLOAT32": np.float32, "INT64": np.int64, "INT32": np.int32}


def build(path, device_id):
    """Build an mslite Model on a given Ascend device."""
    ctx = mslite.Context()
    ctx.target = ["ascend"]
    ctx.ascend.device_id = int(device_id)
    m = mslite.Model()
    m.build_from_file(str(path), mslite.ModelType.MINDIR, ctx)
    return m


def run(model, feed):
    """Run a model, casting each feed to the input tensor's dtype (name-matched)."""
    tensors = []
    for t in model.get_inputs():
        a = feed[t.name].astype(_MS.get(getattr(t.dtype, "name", str(t.dtype)), np.float32))
        tensors.append(mslite.Tensor(a))
    return [o.get_data_to_numpy() for o in model.predict(tensors)]


def main():
    """Run BLIP2-OPT VQA: vision -> qformer -> OPT greedy re-feed on Ascend."""
    p = argparse.ArgumentParser()
    p.add_argument("--vision-model", default="./blip2_onnx/blip2_vision_graph.mindir")
    p.add_argument("--qformer-model", default="./blip2_onnx/blip2_qformer.mindir")
    p.add_argument("--opt-model", default="./blip2_onnx/blip2_opt_full_graph.mindir")
    p.add_argument("--opt-embeddings", default="./opt_embed_tokens.npy")
    p.add_argument("--tokenizer", default="./blip2-opt-2.7b")
    p.add_argument("--image", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--max-seq", type=int, default=75)
    p.add_argument("--vision-device", type=int, default=1)
    p.add_argument("--llm-device", type=int, default=0)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    proc = Blip2Processor.from_pretrained(args.tokenizer)
    emb = np.load(args.opt_embeddings)  # [vocab, 2560]
    vision = build(args.vision_model, args.vision_device)
    qformer = build(args.qformer_model, args.vision_device)
    opt = build(args.opt_model, args.llm_device)

    img = Image.open(args.image).convert("RGB")
    t0 = time.perf_counter()
    pv = proc(images=img, return_tensors="np")["pixel_values"].astype(np.float32)
    image_embeds = run(vision, {"pixel_values": pv})[0]
    query_lm = run(qformer, {"image_embeds": image_embeds})[1]  # language_model_inputs [1,32,2560]
    t_vq = (time.perf_counter() - t0) * 1000

    q_ids = tok(args.question, return_tensors="np")["input_ids"][0].astype(np.int64)
    prefix = np.concatenate([query_lm[0], emb[q_ids]], axis=0).astype(np.float32)  # [32+qlen, 2560]

    eos = tok.eos_token_id
    t0 = time.perf_counter()
    out_ids = []
    n_opt = 0
    for _ in range(args.max_new_tokens):
        L = prefix.shape[0]
        if L > args.max_seq:
            break
        ie = prefix[None, :, :]
        pos = np.arange(L, dtype=np.int64)[None, :]
        logits = run(opt, {"inputs_embeds": ie, "position_ids": pos})[0]
        n_opt += 1
        nxt = int(np.argmax(logits[0, -1]))
        if nxt == eos:
            break
        out_ids.append(nxt)
        prefix = np.concatenate([prefix, emb[nxt:nxt + 1]], axis=0)
    t_opt = (time.perf_counter() - t0) * 1000

    answer = tok.decode(out_ids, skip_special_tokens=True)
    print(f"\nQuestion: {args.question}")
    print(f"Answer:   {answer}")
    print(f"\n--- Performance ---")
    print(f"  vision+qformer: {t_vq:.1f} ms")
    print(f"  OPT re-feed total: {t_opt:.1f} ms ({n_opt} steps, avg {t_opt/max(n_opt,1):.1f} ms/step)")
    print(f"  end-to-end:      {t_vq + t_opt:.1f} ms")


if __name__ == "__main__":
    main()
