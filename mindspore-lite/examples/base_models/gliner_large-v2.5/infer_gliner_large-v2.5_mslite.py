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

"""MindSpore Lite inference script for gliner_large-v2.5.

Same end-to-end NER pipeline for MindSpore Lite running on Ascend. Static-shape MindIR is used
because dynamic seq_len + native nn.LSTM is rejected by the Ascend
multi-batch compiler. The static shape is:

    input_ids/attention_mask/words_mask : (1, 128)
    text_lengths                        : (1, 1)
    span_idx                            : (1, 288, 2)   # 24 words * 12 widths
    span_mask                           : (1, 288)
    logits                              : (1, 24, 12, 3)

The label set is baked into the MindIR (the ONNX export dummy batch used
``[person, organization, country]``). To use a different label set, re-export
the ONNX with ``--labels`` (in the upstream GLiNER exporter) and re-convert.

No ``torch`` import: pure numpy + transformers + mindspore_lite.
"""

import argparse
import json
import re
import time as _time
from pathlib import Path

import numpy as np
import mindspore_lite as mslite
from transformers import AutoTokenizer

# Fixed shapes baked into the MindIR.
SEQ_LEN = 128
NUM_WORDS = 24
MAX_WIDTH = 12
NUM_CLASSES = 3

# Label set must match the labels used at ONNX export time.
DEFAULT_LABELS = ["person", "organization", "country"]

_WHITESPACE_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S")


def _split_words(text):
    """Split text into (token, char_start, char_end) using GLiNER's whitespace rule."""
    return [(m.group(), m.start(), m.end()) for m in _WHITESPACE_PATTERN.finditer(text)]


def _build_prompt(labels, ent_token, sep_token):
    """Build the GLiNER prompt: [ENT, label1, ENT, label2, ..., SEP]."""
    prompt = []
    for label in labels:
        prompt.append(ent_token)
        prompt.append(label)
    prompt.append(sep_token)
    return prompt


def _tokenize(tokenizer, prompt, words):
    """Tokenize prompt+words and return input_ids, attention_mask, words_mask.

    Mirrors GLiNER's tokenize_inputs + prepare_word_mask.
    """
    tokens = prompt + [w for w, _, _ in words]
    enc = tokenizer(tokens, is_split_into_words=True, add_special_tokens=True,
                    truncation=True, max_length=SEQ_LEN)
    input_ids = np.asarray(enc["input_ids"], dtype=np.int32)
    attention_mask = np.asarray(enc["attention_mask"], dtype=np.int32)
    word_ids = enc.word_ids()

    words_mask = np.zeros_like(input_ids, dtype=np.int32)
    prev_wid = None
    seen_words = 0
    skip_n = len(prompt)
    for i, wid in enumerate(word_ids):
        if wid is None:
            prev_wid = wid
            continue
        if wid != prev_wid:
            seen_words += 1
            prev_wid = wid
        if seen_words > skip_n and words_mask[i] == 0 and (i == 0 or word_ids[i - 1] != wid):
            words_mask[i] = seen_words - skip_n
    return input_ids, attention_mask, words_mask


def _prepare_span_idx():
    """Pre-compute the fixed (NUM_WORDS * MAX_WIDTH, 2) span index grid."""
    starts = np.arange(NUM_WORDS, dtype=np.int32).reshape(-1, 1)
    offsets = np.arange(MAX_WIDTH, dtype=np.int32).reshape(1, -1)
    grid = np.stack([
        np.broadcast_to(starts, (NUM_WORDS, MAX_WIDTH)),
        np.broadcast_to(starts + offsets, (NUM_WORDS, MAX_WIDTH)),
    ], axis=-1)
    return grid.reshape(-1, 2)


def _prepare_inputs(text, labels, tokenizer, ent_token, sep_token):
    """Build the 6 MindSpore Lite inputs for a single (text, labels) pair."""
    all_words = _split_words(text)
    prompt = _build_prompt(labels, ent_token, sep_token)

    # Truncate body words to fit both SEQ_LEN (tokens) and NUM_WORDS (word slots).
    words = all_words[:NUM_WORDS]
    while words:
        input_ids, attention_mask, words_mask = _tokenize(tokenizer, prompt, words)
        if input_ids.shape[0] <= SEQ_LEN:
            break
        words = words[:-1]
    num_body_words = len(words)

    # Pad seq dim up to SEQ_LEN.
    pad_len = SEQ_LEN - input_ids.shape[0]
    if pad_len > 0:
        input_ids = np.concatenate([input_ids, np.zeros(pad_len, dtype=np.int32)])
        attention_mask = np.concatenate([attention_mask, np.zeros(pad_len, dtype=np.int32)])
        words_mask = np.concatenate([words_mask, np.zeros(pad_len, dtype=np.int32)])

    text_lengths = np.asarray([[num_body_words]], dtype=np.int32)

    span_idx = _prepare_span_idx()
    span_mask = (span_idx[:, 1] < num_body_words).reshape(1, -1).astype(bool)

    feeds = [
        input_ids.reshape(1, -1),
        attention_mask.reshape(1, -1),
        words_mask.reshape(1, -1),
        text_lengths,
        span_idx.reshape(1, -1, 2),
        span_mask,
    ]
    return feeds, words, num_body_words


def _greedy_search(spans, flat_ner):
    """Greedy non-overlap selection. spans: list of (start, end, label, score)."""
    spans_sorted = sorted(spans, key=lambda s: s[3], reverse=True)
    picked = []
    for span in spans_sorted:
        start, end, _, _ = span
        if flat_ner:
            conflict = any(not (end < p_start or start > p_end)
                           for p_start, p_end, _, _ in picked)
            if conflict:
                continue
        picked.append(span)
    return picked


def _decode(logits, num_body_words, labels, threshold, flat_ner):
    """Decode (1, NUM_WORDS, MAX_WIDTH, NUM_CLASSES) logits to char spans."""
    probs = 1.0 / (1.0 + np.exp(-logits[0]))
    candidates = []
    for s in range(num_body_words):
        for k in range(MAX_WIDTH):
            if s + k >= num_body_words:
                continue
            for c, label in enumerate(labels):
                score = float(probs[s, k, c])
                if score >= threshold:
                    candidates.append((s, s + k, label, score))
    return _greedy_search(candidates, flat_ner)


def _map_to_chars(spans, words):
    """Map word-index spans (start, end, label, score) -> char dict list."""
    out = []
    for s, e, label, score in spans:
        char_start = words[s][1]
        char_end = words[e][2]
        out.append({
            "text": words[s][0] if s == e else " ".join(w[0] for w in words[s:e + 1]),
            "label": label,
            "score": score,
            "start": char_start,
            "end": char_end,
        })
    return out


def load_config(model_dir):
    """Load ent_token, sep_token, max_width from gliner_config.json."""
    cfg_path = Path(model_dir) / "gliner_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="gliner_large-v2.5 MindSpore Lite inference")
    parser.add_argument("--model-dir", default="gliner_large-v2.5",
                        help="Path to the original gliner checkpoint (for tokenizer + config)")
    parser.add_argument("--mindir-path", default="./onnx/model.mindir",
                        help="Path to the converted MindIR model")
    parser.add_argument("--text", default=None, help="Input text (overrides --text-file)")
    parser.add_argument("--text-file", default=None, help="File with one text per line")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS),
                        help="Comma-separated entity labels (must match export-time labels)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold (sigmoid probability)")
    parser.add_argument("--flat-ner", action="store_true", default=True,
                        help="Disallow overlapping spans (default True)")
    parser.add_argument("--device-id", type=int, default=0, help="Ascend device id")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup runs before timed inference (default 3, set 0 to disable)")
    return parser.parse_args()


def main():
    """Run gliner_large-v2.5 MindSpore Lite inference end-to-end."""
    args = parse_args()
    cfg = load_config(args.model_dir)
    ent_token = cfg["ent_token"]
    sep_token = cfg["sep_token"]
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if len(labels) != NUM_CLASSES:
        raise ValueError(
            f"Static MindIR has {NUM_CLASSES} classes baked in, but got {len(labels)} labels: {labels}. "
            f"Re-export ONNX with --labels and re-convert to change the label set."
        )
    print(f"[infer] config: ent='{ent_token}', sep='{sep_token}', labels={labels}", flush=True)

    print(f"[infer] loading tokenizer from {args.model_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)

    print(f"[infer] loading MindIR from {args.mindir_path}", flush=True)
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = args.device_id
    model = mslite.Model()
    model.build_from_file(args.mindir_path, mslite.ModelType.MINDIR, context)

    if args.text is not None:
        texts = [args.text]
    elif args.text_file is not None:
        with open(args.text_file, "r", encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f if line.strip()]
    else:
        texts = [
            "Cristiano Ronaldo dos Santos Aveiro plays for Al-Nassr FC and captains Portugal.",
            "Linus Torvalds created Linux in 1991 while at the University of Helsinki.",
            "The Eiffel Tower is located in Paris, France and was built in 1889.",
        ]

    if args.warmup > 0:
        warmup_feeds, _, _ = _prepare_inputs(
            texts[0], labels, tokenizer, ent_token, sep_token)
        print(f"[infer] warmup: {args.warmup} runs...", flush=True)
        for _ in range(args.warmup):
            model.predict(warmup_feeds)
        print("[infer] warmup done", flush=True)

    for text in texts:
        t_e2e = _time.perf_counter()
        feeds, words, num_body_words = _prepare_inputs(
            text, labels, tokenizer, ent_token, sep_token)
        t_pre = _time.perf_counter()
        outputs = model.predict(feeds)
        t_pred = _time.perf_counter()
        logits = outputs[0].get_data_to_numpy()
        spans = _decode(logits, num_body_words, labels, args.threshold, args.flat_ner)
        ents = _map_to_chars(spans, words)
        t_e2e_end = _time.perf_counter()
        print(f"\n[infer] text: {text}", flush=True)
        print(f"[infer] words ({num_body_words}): {[w[0] for w in words]}", flush=True)
        print(f"[infer] seq_len: {feeds[0].shape[1]}, logits shape: {logits.shape}", flush=True)
        for ent in ents:
            print(f"  - {ent['text']!r} [{ent['label']}] score={ent['score']:.4f} "
                  f"chars=({ent['start']}, {ent['end']})", flush=True)
        print(f"[infer] 模型推理: {(t_pred - t_pre) * 1000:.2f} ms | "
              f"端到端: {(t_e2e_end - t_e2e) * 1000:.2f} ms", flush=True)

if __name__ == "__main__":
    main()
