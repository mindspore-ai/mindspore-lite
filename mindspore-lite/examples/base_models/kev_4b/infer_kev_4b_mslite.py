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
         --tokenizer MERGED [--record-json record.json] \\
         --branch-gears 128 256 512 1024 1536 2048 --past-capacity 3072

  # length sweep; needs only the two graphs
  python -I infer_kev_4b_mslite.py --state-mindir DIR/kev_state_graph.mindir \\
         --branch-mindir DIR/kev_branch_graph.mindir --perf \\
         --branch-gears 128 256 512 1024 1536 2048 --past-capacity 3072 \\
         --state-lengths 128 256 512 1024 2048 --branch-lengths 128 256 512 1024 2048

Every request is one branch pass on top of a state the script already holds. Two deployment
optimizations keep that pass cheap and are on unless turned off:

* the state pass is cached by its token ids (``--score-all`` streams a JSONL and reuses the state
  whenever the next record repeats it; ``--no-prefix-cache`` recomputes it, for an A/B) and the
  branch inputs / mask / output buffers stay on the device between passes (``--no-zero-copy``
  rebuilds them per call, for the same reason);
* a gear-converted branch graph (``--branch-gears``) only takes the row lengths it was compiled
  for, so a row is right-padded to its bucket and the output sliced back; the same graph pins
  the KV length to ``--past-capacity``, which makes the branch static in the state length. Both
  flags describe one converted graph and are required together.
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


def additive_mask(rows, past, capacity=None):
    """[1, 1, rows, width + rows]: cached prefix visible, new row causal, pad slots hidden.

    ``width`` is ``past`` when every cached column holds a real token. When the KV cache is
    zero-padded to a fixed ``capacity`` (see ``--past-capacity``) the columns between the real
    prefix and the new row are fillers that must never be attended.
    """
    width = past if capacity is None else capacity
    column = np.arange(width + rows)[None, :]
    row = np.arange(rows)[:, None]
    allowed = column <= (width + row)
    if width > past:
        allowed &= (column < past) | (column >= width)
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


def _input_dtype(model, name):
    """The dtype the converted graph declares for an input (int64 is rewritten to int32)."""
    for tensor in model.get_inputs():
        if getattr(tensor, "name", None) == name:
            return tensor.dtype
    return None


class BranchBuffers:
    """Device input / output buffers of one branch row shape, rewritten in place every call.

    Zero-copy path: the mask never changes for a shape, and the ids / positions are written into
    the same device tensors, so a request pays one ``set_data_from_numpy`` per input and one
    ``predict`` instead of allocating five tensors and a fresh mask each time. The output tensor is
    preallocated too, which leaves the hidden states as the only array that comes back to the host
    (the pointer head reads them in numpy).
    """

    def __init__(self, model, device, rows, past, capacity):
        width = past if capacity is None else capacity
        self.ids = self._tensor(model, "input_ids", [1, rows], device, np.int32)
        self.pos = self._tensor(model, "position_ids", [1, rows], device, np.int32)
        self.mask = self._tensor(model, "attention_mask", [1, 1, rows, width + rows], device,
                                 np.float16)
        self.mask.set_data_from_numpy(additive_mask(rows, past, capacity))
        hidden = int(list(model.get_outputs()[0].shape)[-1])
        if hidden <= 0:
            raise RuntimeError("branch graph does not declare a static hidden size, so the "
                               "preallocated output buffer cannot be built")
        self.out = mslite.Tensor(shape=[1, rows, hidden],
                                 dtype=model.get_outputs()[0].dtype, device=device)

    @staticmethod
    def _tensor(model, name, shape, device, default_dtype):
        """A device tensor in the dtype the graph declares for ``name``."""
        dtype = _input_dtype(model, name) or default_dtype
        return mslite.Tensor(shape=shape, dtype=dtype, device=device)

    def write(self, model, ids, pos):
        """Overwrite the padded ids / positions with this request's row."""
        self.ids.set_data_from_numpy(_cast_for(model, "input_ids", ids[None]))
        self.pos.set_data_from_numpy(_cast_for(model, "position_ids", pos[None]))


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
        # A gear-mode branch graph only accepts the row lengths it was compiled for; the row is
        # right-padded to the smallest one that fits and the output sliced back to the real row.
        self.branch_gears = sorted(set(getattr(args, "branch_gears", None) or []))
        self.pad_id = self._pad_id()
        # Capacity mode pins the KV length in the branch graph: the state KV is zero-padded once
        # per state pass and the mask hides the tail, so the graph is static in the state length.
        self.past_capacity = getattr(args, "past_capacity", None)
        self.zero_copy = getattr(args, "zero_copy", True)
        self.use_prefix = getattr(args, "prefix_cache", True)
        self.device = f"ascend:{args.device_id}"
        self._buffers = {}
        self._prefix_key = None
        self._caches = None

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

    def _pad_kv(self, kv):
        """Zero-pad the state KV to the branch graph's fixed capacity, once per state pass.

        The padded columns are never attended (the branch mask hides them), so this only changes
        the tensor's shape. Only the KV is padded: the conv / recurrent caches are recurrent end
        states, and appending padding tokens to the state row would change their values.
        """
        if not self.past_capacity:
            return kv
        past_len = list(kv.shape)[3]
        if past_len > self.past_capacity:
            raise ValueError(f"state of {past_len} tokens overflows the branch graph's KV "
                             f"capacity {self.past_capacity}")
        if past_len == self.past_capacity:
            return kv
        array = np.pad(kv.get_data_to_numpy(),
                       ((0, 0), (0, 0), (0, 0), (0, self.past_capacity - past_len), (0, 0)))
        buffer = mslite.Tensor(shape=list(array.shape), dtype=kv.dtype, device=self.device)
        buffer.set_data_from_numpy(array)
        return buffer

    def state_pass(self, ids, pos):
        """Read the state row and leave its three caches on the device."""
        feed = {"input_ids": np.asarray(ids)[None], "position_ids": np.asarray(pos)[None],
                "attention_mask": additive_mask(len(ids), 0)}
        _resize(self.state_model, feed)
        caches = self.state_model.predict(_feed(self.state_model, feed))
        caches[2] = self._pad_kv(caches[2])
        return caches

    def prefix_caches(self, ids, pos):
        """The state caches for this exact state row, computed again only when the state changes.

        Every question of a record runs on the same caches; a JSONL stream usually repeats one
        public state as well, and then only the first record pays the state pass. The branch
        buffers are tied to a state's length, so they are dropped whenever the state changes
        (not when ``--no-prefix-cache`` merely recomputes the same row).
        """
        key = tuple(ids)
        changed = self._caches is None or key != self._prefix_key
        if changed:
            self._buffers.clear()
        if changed or not self.use_prefix:
            self._caches = self.state_pass(ids, pos)
            self._prefix_key = key
        return self._caches

    def _pad_id(self):
        """Token id used to right-pad a gear-mode row: the eos id when defined, else 0.

        The padding columns are hidden by the causal mask and their outputs are dropped, so
        their value cannot change the result; eos only keeps the padded row legitimate text.
        Tokenizers may leave ``eos_token_id`` unset or hold a list of them, and ``int(None)``
        would fail where the row is built rather than where the tokenizer was chosen.
        """
        if not self.branch_gears:
            return 0
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos, (list, tuple)):
            eos = eos[0] if eos else None
        return 0 if eos is None else int(eos)

    def _gear_rows(self, rows):
        """The row length the branch graph is compiled for: the smallest gear that fits ``rows``."""
        if not self.branch_gears:
            return rows
        for gear in self.branch_gears:
            if rows <= gear:
                return gear
        raise ValueError(f"branch row of {rows} tokens overflows the largest gear "
                         f"{self.branch_gears[-1]}")

    def _device_feed(self, buffers, caches, conv):
        """The feed of one branch call, with the device buffers standing in for host arrays."""
        return {"input_ids": buffers.ids, "position_ids": buffers.pos,
                "attention_mask": buffers.mask, "past_conv_states": conv,
                "past_recurrent_states": caches[1], "past_kv_cache": caches[2]}

    def _run_branch(self, ids, pos, caches, conv, rows, past_len):
        """One branch pass: reuse the device buffers of this shape, or build a fresh feed."""
        if not self.zero_copy:
            feed = {"input_ids": ids[None], "position_ids": pos[None],
                    "attention_mask": additive_mask(rows, past_len, self.past_capacity),
                    "past_conv_states": conv, "past_recurrent_states": caches[1],
                    "past_kv_cache": caches[2]}
            _resize(self.branch_model, feed)
            return self.branch_model.predict(_feed(self.branch_model, feed))[0].get_data_to_numpy()
        buffers = self._buffers.get((rows, past_len))
        if buffers is None:
            buffers = BranchBuffers(self.branch_model, self.device, rows, past_len,
                                    self.past_capacity)
            self._buffers[(rows, past_len)] = buffers
        buffers.write(self.branch_model, ids, pos)
        feed = self._device_feed(buffers, caches, conv)
        # Alternate bucket sizes must resize back and forth: the check is a shape-list compare,
        # and the actual resize only runs when the graph is not already at this shape.
        _resize(self.branch_model, feed)
        tensors = [feed[name] for name in _ordered(self.branch_model, feed)]
        return self.branch_model.predict(tensors, outputs=[buffers.out])[0].get_data_to_numpy()

    def branch_pass(self, row, caches, past_len):
        """Read one question row over the cached state and return its real hidden states.

        With ``--branch-gears`` the row is right-padded to its bucket: padding ids are the eos
        token, padding positions continue the real ones, and the causal mask already hides the
        padding from every real row. Only the real rows of the output are returned.
        """
        rows = self._gear_rows(len(row["ids"]))
        ids = np.full(rows, self.pad_id, dtype=np.int32)
        ids[:len(row["ids"])] = row["ids"]
        pos = np.arange(past_len, past_len + rows, dtype=np.int32)
        pos[:len(row["pos"])] = row["pos"]
        hidden = self._run_branch(ids, pos, caches, self._pad_conv(caches[0]), rows, past_len)
        return hidden[:, :len(row["ids"])]

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
        """Synthetic question row of ``length`` tokens continuing the cached state.

        With gears the graph only takes the compiled lengths, so the sweep measures the bucket
        the row actually lands in rather than an unattainable exact shape.
        """
        length = self._gear_rows(length)
        return {"input_ids": rng.integers(0, PERF_VOCAB, (1, length), dtype=np.int32),
                "position_ids": np.arange(past, past + length, dtype=np.int32)[None],
                "attention_mask": additive_mask(length, past, self.past_capacity),
                "past_conv_states": self._pad_conv(caches[0]),
                "past_recurrent_states": caches[1], "past_kv_cache": caches[2]}

    def perf(self, state_lengths, branch_lengths, iters):
        """Sweep synthetic state and branch lengths, reusing the state caches on the device."""
        rng = np.random.default_rng(PERF_SEED)
        for state_length in state_lengths:
            elapsed, caches = self._timed(self.state_model, self._state_feed(rng, state_length), iters)
            caches[2] = self._pad_kv(caches[2])
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
        caches = self.prefix_caches(state_ids, state_pos)
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


def records_from_jsonl(path):
    """Every record of a JSONL file, in order, for the streaming mode."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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
    parser.add_argument("--score-all", action="store_true",
                        help="score every line of a JSONL --record-json, reusing the state pass "
                             "when consecutive records share the same state")
    parser.add_argument("--no-prefix-cache", dest="prefix_cache", action="store_false",
                        help="recompute the state pass for every record instead of keeping it "
                             "when the state row repeats")
    parser.add_argument("--max-state", type=int, default=8192)
    parser.add_argument("--max-branch", type=int, default=16384)
    parser.add_argument("--branch-gears", type=int, nargs="+", default=None,
                        help="bucket lengths a gear-mode branch MindIR was converted for; rows are "
                             "padded to the smallest one and the output sliced back to the real "
                             "row (needs --past-capacity: the same graph pins the KV length)")
    parser.add_argument("--past-capacity", type=int, default=None,
                        help="KV capacity the gear-mode branch MindIR was converted for; the state "
                             "KV is zero-padded to it once and the mask hides the padding "
                             "(needs --branch-gears)")
    parser.add_argument("--no-zero-copy", dest="zero_copy", action="store_false",
                        help="rebuild the mask and the input tensors on every branch pass instead "
                             "of rewriting the device buffers")
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
    if (args.branch_gears is None) != (args.past_capacity is None):
        parser.error("--branch-gears and --past-capacity go together: the gear-mode graph compiles "
                     "both the row buckets and the KV capacity in one ge.dynamicDims list, so "
                     "padding rows without pinning the cache feeds lengths it never compiled")
    if args.branch_gears is not None:
        if any(gear < 1 for gear in args.branch_gears):
            parser.error("--branch-gears must be positive")
        if args.past_capacity < 1:
            parser.error("--past-capacity must be at least 1")
    if args.score_all and (args.record_json is None or args.record_json.suffix != ".jsonl"):
        parser.error("--score-all needs a .jsonl --record-json")
    return args


def main(argv=None):
    """Score a record (or the demo), or sweep synthetic lengths."""
    args = parse_args(argv)
    inferencer = Kev4BInferencer(args)
    if args.perf:
        inferencer.perf(args.state_lengths, args.branch_lengths, args.iters)
        return 0
    if args.score_all:
        for index, record in enumerate(records_from_jsonl(args.record_json)):
            print(f"=== record {index}")
            print_answers(record, inferencer.answer(record))
        return 0
    rec = record_from_args(args)
    print_answers(rec, inferencer.answer(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
