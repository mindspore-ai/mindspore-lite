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
"""
Score a Kev record on Ascend with the two MindIR graphs exported by export_kev_4b_onnx.py.

Kev is not a generator: a state row is read once into the conv / recurrent / KV caches
(``kev_state``), then every question row is read on top of those caches (``kev_branch``) and the
pointer head scores its options with three host-side array operations. The caches stay on the
device, so a question costs exactly one branch pass.

  python -I infer_kev_4b_mslite.py --state-mindir DIR/kev_state_graph.mindir \\
         --branch-mindir DIR/kev_branch_graph.mindir --head-npz head.npz \\
         --tokenizer MERGED [--record-json record.json]

  # length sweep; needs only the two graphs
  python -I infer_kev_4b_mslite.py --state-mindir DIR/kev_state_graph.mindir \\
         --branch-mindir DIR/kev_branch_graph.mindir --perf \\
         --state-lengths 128 256 512 1024 2048 --branch-lengths 128 256 512 1024 2048
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mindspore_lite as mslite
except ImportError:
    print("Error: mindspore_lite is required, see README for the matching package download.")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers is required for the tokenizer.")
    sys.exit(1)


# Kev's delimiters: state row opener, question opener, option opener/closer and the readout token.
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
# Additive mask fill: finite in fp16 (-65504 is the smallest, and a masked pair must not sum to -inf).
MASK_FILL = -10000.0
NUMPY_OF_DTYPE = {mslite.DataType.FLOAT16: np.float16, mslite.DataType.FLOAT32: np.float32,
                  mslite.DataType.FLOAT64: np.float64, mslite.DataType.INT32: np.int32,
                  mslite.DataType.INT64: np.int64}
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

# Demo record used when the caller gives no --record-json.
DEFAULT_RECORD = {
    "state": "Is my currency okay to add money with?",
    "questions": [{
        "instr": "Which banking intent best describes this customer message?",
        "options": ["activate_my_card", "age_limit",
                    "apple_pay_or_google_pay: Issue concerning apple pay or google pay",
                    "supported_cards_and_currencies", "verify_source_of_funds"],
    }],
}
# Perf sweep defaults: the same five lengths on both axes.
PERF_STATE_LENGTHS = (128, 256, 512, 1024, 2048)
PERF_BRANCH_LENGTHS = (128, 256, 512, 1024, 2048)
PERF_VOCAB = 1000
PERF_SEED = 0


def user_tokens(tok, text):
    """Tokenize caller text so it can never forge a delimiter (Kev rewrites `<|x|>` the same way)."""
    return tok(_SPECIAL_RE.sub("<¦\\1¦>", text), add_special_tokens=False).input_ids


def encode(tok, rec, max_state, max_branch):
    """Kev's encode with option_isolation=False: one state row plus one causal row per question.

    The state keeps its first ``max_state`` tokens, delimiter included. A question row that no
    longer fits the context is refused instead of cut: a cut row would score a question the model
    never saw whole.
    """
    state_tokens = user_tokens(tok, rec["state"])[: max_state - 1]
    ids = [tok.convert_tokens_to_ids(SPECIAL[0])] + state_tokens
    state_len = len(ids)
    pos = list(range(state_len))
    q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(token) for token in SPECIAL[1:])
    decide_idx, opt_idx = [], []
    for question in rec["questions"]:
        instr = [q_id] + user_tokens(tok, question["instr"])
        spans = [[o_id] + user_tokens(tok, option) + [c_id] for option in question["options"]]
        branch = instr + [token for span in spans for token in span] + [d_id]
        if len(branch) > max_branch - state_len:
            raise ValueError(f"branch of {len(branch)} tokens overflows a {max_branch}-token row")
        base = len(ids)
        ends, cursor = [], len(instr)
        for span in spans:
            cursor += len(span)
            ends.append(cursor - 1)
        ids += branch
        pos += list(range(state_len, state_len + len(branch)))
        decide_idx.append(base + len(branch) - 1)
        opt_idx.append([base + end for end in ends])
    return {"ids": ids, "pos": pos, "state_len": state_len, "decide_idx": decide_idx,
            "opt_idx": opt_idx}


def rows_of(enc):
    """Split the packed encoding into the state row and the per-question rows (Kev's rows_of)."""
    state_len = enc["state_len"]
    rows, start = [], state_len
    for decide, opts in zip(enc["decide_idx"], enc["opt_idx"]):
        end = decide + 1
        rows.append({"ids": enc["ids"][start:end], "pos": enc["pos"][start:end],
                     "decide": decide - start, "opts": [opt - start for opt in opts]})
        start = end
    return enc["ids"][:state_len], enc["pos"][:state_len], rows


def additive_mask(rows, past):
    """[1, 1, rows, past + rows]: the cached prefix is visible, the new row is causal."""
    column = np.arange(past + rows)[None, :]
    row = np.arange(rows)[:, None]
    allowed = column <= (past + row)
    return np.where(allowed, 0.0, MASK_FILL)[None, None].astype(np.float16)


def load_head(path):
    """The pointer head as numpy: its q/k projections and the calibration temperature."""
    data = np.load(path)
    head = {name: data[name].astype(np.float32)
            for name in ("q_weight", "q_bias", "k_weight", "k_bias")}
    return head, float(data["temperature"])


def score_options(head, temperature, hidden, row):
    """Kev's PointerHead over one branch: softmax of the scaled q/k dot products over the options."""
    key = hidden[row["opts"]] @ head["k_weight"].T + head["k_bias"]
    query = hidden[row["decide"]] @ head["q_weight"].T + head["q_bias"]
    logits = (key * query).sum(-1) / (np.sqrt(head["k_weight"].shape[0]) * temperature)
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def _cast_for(model, name, array):
    """Cast a host array to the dtype the converted graph declares for that input."""
    for tensor in model.get_inputs():
        if getattr(tensor, "name", None) == name:
            return array.astype(NUMPY_OF_DTYPE.get(tensor.dtype, array.dtype))
    return array


def _ordered(model, feed):
    """The graph's input order; the feed's own order is the fallback when names are hidden."""
    names = [getattr(tensor, "name", None) for tensor in model.get_inputs()]
    if names and all(name in feed for name in names):
        return names
    return list(feed)


def _feed(model, feed):
    """Order the feed by the graph's input order, casting host arrays to the declared dtypes."""
    tensors = []
    for name in _ordered(model, feed):
        value = feed[name]
        if not isinstance(value, mslite.Tensor):
            value = mslite.Tensor(_cast_for(model, name, np.asarray(value)))
        tensors.append(value)
    return tensors


def _resize(model, feed):
    """Resize a dynamic graph to this call's shapes; device-resident caches already match."""
    inputs = model.get_inputs()
    target = [list(feed[name].shape) for name in _ordered(model, feed)]
    if [list(tensor.shape) for tensor in inputs] != target:
        model.resize(inputs, target)


class Kev4BInferencer:
    """The two graphs plus the pointer head, with the state caches kept on the device."""

    def __init__(self, args):
        """Build both graphs for the requested device."""
        self.context = mslite.Context()
        self.context.target = ["ascend"]
        self.context.ascend.device_id = args.device_id
        self.state_model = self._build(args.state_mindir)
        self.branch_model = self._build(args.branch_mindir)
        self.conv_width = self._input_width("past_conv_states")
        self.head, self.temperature = load_head(args.head_npz) if args.head_npz else (None, None)
        self.tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer)) if args.tokenizer else None
        self.max_state = args.max_state
        self.max_branch = args.max_branch

    def _build(self, path):
        """Load one MindIR graph."""
        model = mslite.Model()
        model.build_from_file(str(path), mslite.ModelType.MINDIR, self.context)
        return model

    def _input_width(self, name):
        """Last dim the branch graph declares for ``name``, or None when it is dynamic."""
        for tensor in self.branch_model.get_inputs():
            if getattr(tensor, "name", None) == name:
                width = list(tensor.shape)[-1]
                return width if width > 0 else None
        return None

    def _pad_conv(self, conv):
        """Left-pad the conv cache with zeros when the state row was shorter than the conv kernel.

        The state graph emits the last ``min(state_len, kernel)`` pre-conv columns. A state row
        with fewer tokens than the kernel therefore yields a narrower cache, while the branch
        graph takes a fixed ``conv_kernel_size``-wide one. Kev (transformers) zero-initialises
        that cache, so the missing leading columns are zeros rather than missing history.
        """
        width = list(conv.shape)[-1]
        if not self.conv_width or width >= self.conv_width:
            return conv
        array = np.pad(conv.get_data_to_numpy(), ((0, 0), (0, 0), (0, 0),
                                                  (self.conv_width - width, 0)))
        return mslite.Tensor(array)

    def state_pass(self, ids, pos):
        """Read the state row and leave its three caches on the device."""
        feed = {"input_ids": np.asarray(ids)[None], "position_ids": np.asarray(pos)[None],
                "attention_mask": additive_mask(len(ids), 0)}
        _resize(self.state_model, feed)
        return self.state_model.predict(_feed(self.state_model, feed))

    def branch_pass(self, row, caches, past_len):
        """Read one question row over the cached state and return its hidden states."""
        feed = {"input_ids": np.asarray(row["ids"])[None],
                "position_ids": np.asarray(row["pos"])[None],
                "attention_mask": additive_mask(len(row["ids"]), past_len),
                "past_conv_states": self._pad_conv(caches[0]),
                "past_recurrent_states": caches[1], "past_kv_cache": caches[2]}
        _resize(self.branch_model, feed)
        return self.branch_model.predict(_feed(self.branch_model, feed))[0].get_data_to_numpy()

    def _timed(self, model, feed, iters):
        """Mean predict time after two warm-up calls; the second one performs the real resize."""
        _resize(model, feed)
        tensors = _feed(model, feed)
        model.predict(tensors)
        model.predict(tensors)
        start = time.perf_counter()
        outputs = None
        for _ in range(iters):
            outputs = model.predict(tensors)
        return (time.perf_counter() - start) * 1e3 / iters, outputs

    def _state_feed(self, rng, length):
        """Synthetic state row at ``length`` tokens with its causal mask."""
        return {"input_ids": rng.integers(0, PERF_VOCAB, (1, length), dtype=np.int32),
                "position_ids": np.arange(length, dtype=np.int32)[None],
                "attention_mask": additive_mask(length, 0)}

    def _branch_feed(self, rng, past, length, caches):
        """Synthetic question row of ``length`` tokens continuing the cached state."""
        return {"input_ids": rng.integers(0, PERF_VOCAB, (1, length), dtype=np.int32),
                "position_ids": np.arange(past, past + length, dtype=np.int32)[None],
                "attention_mask": additive_mask(length, past),
                "past_conv_states": self._pad_conv(caches[0]),
                "past_recurrent_states": caches[1], "past_kv_cache": caches[2]}

    def perf(self, state_lengths, branch_lengths, iters):
        """Sweep synthetic state and branch lengths, reusing the state caches on the device."""
        rng = np.random.default_rng(PERF_SEED)
        for state_length in state_lengths:
            elapsed, caches = self._timed(self.state_model, self._state_feed(rng, state_length), iters)
            print(f"state  S={state_length:5d}                 {elapsed:8.1f} ms", flush=True)
            for branch_length in branch_lengths:
                feed = self._branch_feed(rng, state_length, branch_length, caches)
                elapsed, _ = self._timed(self.branch_model, feed, iters)
                print(f"branch S={state_length:5d} Lb={branch_length:4d}         "
                      f"{elapsed:8.1f} ms", flush=True)

    def answer(self, rec):
        """Score every question of a record, returning the option probabilities per question."""
        state_ids, state_pos, rows = rows_of(encode(self.tokenizer, rec, self.max_state,
                                                   self.max_branch))
        caches = self.state_pass(state_ids, state_pos)
        answers = []
        for row in rows:
            hidden = self.branch_pass(row, caches, len(state_ids))[0]
            answers.append(score_options(self.head, self.temperature, hidden, row))
        return answers


def record_from_args(args):
    """Load the caller's record, or fall back to the built-in demo."""
    if args.record_json is None:
        return DEFAULT_RECORD
    if args.record_json.suffix == ".jsonl":
        with open(args.record_json, encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index == args.record_index:
                    return json.loads(line)
        raise SystemExit(f"record index {args.record_index} is past the end of the file")
    with open(args.record_json, encoding="utf-8") as handle:
        return json.load(handle)


def print_answers(rec, answers):
    """Report one line per option, the way Kev's own tooling presents a decision."""
    for question, probs in zip(rec["questions"], answers):
        print(f"--- {question['instr'][:80]}")
        for option, prob in zip(question["options"], probs):
            mark = "*" if prob == probs.max() else " "
            print(f"  {mark} {prob:8.5f}  {option[:100]}")


def parse_args(argv=None):
    """Command line interface."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-mindir", type=Path, required=True, help="kev_state MindIR")
    parser.add_argument("--branch-mindir", type=Path, required=True, help="kev_branch MindIR")
    parser.add_argument("--head-npz", type=Path, default=None,
                        help="pointer head arrays (not needed with --perf)")
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="tokenizer directory (not needed with --perf)")
    parser.add_argument("--record-json", type=Path, default=None,
                        help="one Kev record (JSON, or JSONL with --record-index); "
                             "defaults to the built-in demo record")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--max-state", type=int, default=8192)
    parser.add_argument("--max-branch", type=int, default=16384)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--perf", action="store_true",
                        help="sweep synthetic lengths; only the two MindIR files are needed")
    parser.add_argument("--state-lengths", type=int, nargs="+", default=PERF_STATE_LENGTHS,
                        help="state lengths for --perf")
    parser.add_argument("--branch-lengths", type=int, nargs="+", default=PERF_BRANCH_LENGTHS,
                        help="branch lengths for --perf")
    parser.add_argument("--iters", type=int, default=5, help="timed calls per --perf point (>= 1)")
    args = parser.parse_args(argv)
    if args.iters < 1:
        parser.error("--iters must be at least 1")
    if any(length <= 0 for length in list(args.state_lengths) + list(args.branch_lengths)):
        parser.error("--state-lengths/--branch-lengths must be positive")
    if not args.perf and (args.head_npz is None or args.tokenizer is None):
        parser.error("--head-npz and --tokenizer are required unless --perf is given")
    return args


def main(argv=None):
    """Score a record (or the demo), or sweep synthetic lengths."""
    args = parse_args(argv)
    inferencer = Kev4BInferencer(args)
    if args.perf:
        inferencer.perf(args.state_lengths, args.branch_lengths, args.iters)
        return 0
    rec = record_from_args(args)
    print_answers(rec, inferencer.answer(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
