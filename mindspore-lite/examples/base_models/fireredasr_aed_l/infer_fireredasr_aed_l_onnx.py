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
# WITHOUT WARRANTIES OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Inference script for FireRedASR-AED-L using ONNX models.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _add_repo_to_sys_path(repo_dir: str) -> None:
    if not repo_dir:
        return
    repo_dir = os.path.abspath(os.path.expanduser(repo_dir))
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


def load_feat(wav_path, cmvn_file):
    from fireredasr.data.asr_feat import ASRFeatExtractor

    feat_extractor = ASRFeatExtractor(kaldi_cmvn_file=cmvn_file)
    feats_pad, lengths, _ = feat_extractor([wav_path])
    return feats_pad, lengths


def greedy_decode(encoder_sess, decoder_sess, feat, feat_len, sos_id, eos_id, max_len=200):
    """Greedy decode with encoder/decoder_step ONNX sessions."""

    _, enc_mask, cross_k, cross_v = encoder_sess.run(
        None,
        {"padded_input": feat, "input_lengths": feat_len},
    )

    src_mask = enc_mask[:, 0:1, :].astype(np.uint8)
    n_layers = cross_k.shape[1]
    n_head = cross_k.shape[2]
    d_k = cross_k.shape[4]

    ys = np.array([[sos_id]], dtype=np.int64)
    cache_k_self = np.zeros((1, n_layers, n_head, 1, d_k), dtype=np.float32)
    cache_v_self = np.zeros((1, n_layers, n_head, 1, d_k), dtype=np.float32)

    token_ids = []
    for _ in range(max_len):
        log_probs, new_cache_k_self, new_cache_v_self = decoder_sess.run(
            None,
            {
                "ys": ys,
                "src_mask": src_mask,
                "cache_k_self": cache_k_self,
                "cache_v_self": cache_v_self,
                "cross_k": cross_k,
                "cross_v": cross_v,
            },
        )

        next_token = int(np.argmax(log_probs[0, -1], axis=-1))
        if next_token == eos_id:
            break
        token_ids.append(next_token)

        ys = np.array([[next_token]], dtype=np.int64)
        cache_k_self = new_cache_k_self
        cache_v_self = new_cache_v_self

    return token_ids


def tokens_to_text(token_ids, dict_file):
    """Converts token ids into final text by using dict.txt."""

    token_dict = {}
    with open(dict_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                token_dict[int(parts[1])] = parts[0]

    tokens = []
    for tid in token_ids:
        token = token_dict.get(tid, "")
        if token.endswith("▁"):
            tokens.append(token[:-1] + " ")
        else:
            tokens.append(token)
    return "".join(tokens).replace(" ", " ").strip()


def main():
    parser = argparse.ArgumentParser(description="Inference FireRedASR-AED-L with ONNX")
    parser.add_argument("--fireredasr-repo", type=str, default="", help="Path to cloned FireRedASR repo")
    parser.add_argument("--onnx-dir", type=str, default="", help="Directory containing ONNX models")
    parser.add_argument("--model-dir", type=str, default="", help="Directory containing cmvn.ark and dict.txt")
    parser.add_argument("--wav-path", type=str, required=True, help="Path to input WAV file")
    parser.add_argument("--cmvn-file", type=str, default="", help="Path to CMVN file (cmvn.ark)")
    parser.add_argument("--dict-file", type=str, default="", help="Path to dict file (dict.txt)")
    parser.add_argument("--max-len", type=int, default=200, help="Maximum decoding length")
    parser.add_argument("--sos-id", type=int, default=3, help="SOS token ID")
    parser.add_argument("--eos-id", type=int, default=4, help="EOS token ID")
    parser.add_argument("--provider", type=str, default="CPUExecutionProvider")

    args = parser.parse_args()
    _add_repo_to_sys_path(args.fireredasr_repo)

    onnx_dir = Path(os.path.abspath(os.path.expanduser(args.onnx_dir or ".")))
    if args.model_dir:
        model_dir = os.path.abspath(os.path.expanduser(args.model_dir))
    else:
        model_dir = ""

    cmvn_file = args.cmvn_file or (os.path.join(model_dir, "cmvn.ark") if model_dir else "")
    dict_file = args.dict_file or (os.path.join(model_dir, "dict.txt") if model_dir else "")
    if not cmvn_file or not dict_file:
        raise ValueError("cmvn-file/dict-file not set and model-dir is empty")

    encoder_path = str(onnx_dir / "onnx_encoder" / "fireredasr_aed_encoder.onnx")
    decoder_path = str(onnx_dir / "onnx_decoder" / "fireredasr_aed_decoder_step.onnx")

    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 1
    sess_opts.intra_op_num_threads = 1
    encoder_sess = ort.InferenceSession(encoder_path, sess_options=sess_opts, providers=[args.provider])
    decoder_sess = ort.InferenceSession(decoder_path, sess_options=sess_opts, providers=[args.provider])

    feat, feat_len = load_feat(args.wav_path, cmvn_file)
    feat_np = feat.numpy().astype(np.float32)
    feat_len_np = feat_len.numpy().astype(np.int64)

    t0 = time.time()
    token_ids = greedy_decode(
        encoder_sess,
        decoder_sess,
        feat_np,
        feat_len_np,
        sos_id=args.sos_id,
        eos_id=args.eos_id,
        max_len=args.max_len,
    )
    dt = time.time() - t0

    text = tokens_to_text(token_ids, dict_file)
    print({"text": text, "elapsed_sec": float(dt), "num_tokens": int(len(token_ids))})


if __name__ == "__main__":
    main()
