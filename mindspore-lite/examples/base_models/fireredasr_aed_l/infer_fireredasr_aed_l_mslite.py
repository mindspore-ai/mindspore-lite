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
Inference script for FireRedASR-AED-L using MindIR models (MindSpore Lite).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import mindspore_lite as mslite

def _warmup_predict(model: mslite.Model, inputs: list[mslite.Tensor], loops: int) -> None:
    """Runs model.predict() for warmup without reporting timing."""

    for _ in range(int(loops)):
        outputs = model.predict(inputs)
        for out in outputs:
            out.get_data_to_numpy()


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


def greedy_decode(encoder_model, decoder_model, feat, feat_len, sos_id, eos_id, max_len=200):
    """Greedy decode with MindSpore Lite encoder/decoder_step models."""

    feat = np.asarray(feat, dtype=np.float32)
    feat_len = np.asarray(feat_len, dtype=np.int32)
    encoder_inputs = [
        mslite.Tensor(feat),
        mslite.Tensor(feat_len),
    ]
    _warmup_predict(encoder_model, encoder_inputs, loops=3)

    t0 = time.time()
    encoder_outputs = encoder_model.predict(encoder_inputs)
    encoder_ms = (time.time() - t0) * 1000

    encoder_outputs[0].get_data_to_numpy()
    enc_mask = encoder_outputs[1].get_data_to_numpy()
    cross_k = encoder_outputs[2].get_data_to_numpy()
    cross_v = encoder_outputs[3].get_data_to_numpy()

    src_mask = enc_mask[:, 0:1, :].astype(np.uint8)
    src_len = int(src_mask.shape[-1])
    n_layers = cross_k.shape[1]
    n_head = cross_k.shape[2]
    d_k = cross_k.shape[4]

    ys = np.array([[sos_id]], dtype=np.int32)
    cache_k_self = np.zeros((1, n_layers, n_head, 1, d_k), dtype=np.float32)
    cache_v_self = np.zeros((1, n_layers, n_head, 1, d_k), dtype=np.float32)

    warmup_decoder_inputs = [
        mslite.Tensor(ys),
        mslite.Tensor(src_mask),
        mslite.Tensor(cache_k_self),
        mslite.Tensor(cache_v_self),
        mslite.Tensor(cross_k),
        mslite.Tensor(cross_v),
    ]
    _warmup_predict(decoder_model, warmup_decoder_inputs, loops=3)

    token_ids = []
    decode_times_ms = []
    decode_steps = 0
    for _ in range(max_len):
        decoder_inputs = [
            mslite.Tensor(ys),
            mslite.Tensor(src_mask),
            mslite.Tensor(cache_k_self),
            mslite.Tensor(cache_v_self),
            mslite.Tensor(cross_k),
            mslite.Tensor(cross_v),
        ]
        t0 = time.time()
        decoder_outputs = decoder_model.predict(decoder_inputs)
        decode_times_ms.append((time.time() - t0) * 1000)
        decode_steps += 1

        log_probs = decoder_outputs[0].get_data_to_numpy()
        new_cache_k_self = decoder_outputs[1].get_data_to_numpy()
        new_cache_v_self = decoder_outputs[2].get_data_to_numpy()

        next_token = int(np.argmax(log_probs[0, -1], axis=-1))
        if next_token == eos_id:
            break
        token_ids.append(next_token)

        ys = np.array([[next_token]], dtype=np.int32)
        cache_k_self = new_cache_k_self
        cache_v_self = new_cache_v_self

    total_decode_ms = float(sum(decode_times_ms))
    avg_decode_ms = float(total_decode_ms / len(decode_times_ms)) if decode_times_ms else 0.0
    total_ms = float(encoder_ms + total_decode_ms)
    throughput = float(len(token_ids) / (total_ms / 1000)) if total_ms > 0 else 0.0
    stats = {
        "feat_frames": int(feat.shape[1]),
        "feat_len": int(feat_len[0]) if feat_len.size > 0 else 0,
        "src_len": src_len,
        "num_tokens": int(len(token_ids)),
        "decode_steps": int(decode_steps),
        "encoder_ms": float(encoder_ms),
        "total_decode_ms": float(total_decode_ms),
        "avg_decode_step_ms": float(avg_decode_ms),
        "total_ms": float(total_ms),
        "throughput_tok_s": float(throughput),
    }
    return token_ids, stats


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
    parser = argparse.ArgumentParser(description="Inference FireRedASR-AED-L with MindIR")
    parser.add_argument("--fireredasr-repo", type=str, default="", help="Path to cloned FireRedASR repo")
    parser.add_argument("--mindir-dir", type=str, required=True, help="Directory containing MindIR models")
    parser.add_argument(
        "--encoder-mindir",
        type=str,
        default="mindir_encoder_graph.mindir",
        help="Encoder MindIR filename inside mindir-dir",
    )
    parser.add_argument(
        "--decoder-mindir",
        type=str,
        default="mindir_decoder_step_graph.mindir",
        help="DecoderStep MindIR filename inside mindir-dir",
    )
    parser.add_argument("--model-dir", type=str, default="", help="Directory containing cmvn.ark and dict.txt")
    parser.add_argument("--wav-path", type=str, required=True, help="Path to input WAV file")
    parser.add_argument("--cmvn-file", type=str, default="", help="Path to CMVN file")
    parser.add_argument("--dict-file", type=str, default="", help="Path to dict file")
    parser.add_argument("--max-len", type=int, default=200, help="Maximum decoding length")
    parser.add_argument("--sos-id", type=int, default=3, help="SOS token ID")
    parser.add_argument("--eos-id", type=int, default=4, help="EOS token ID")
    parser.add_argument(
        "--device",
        type=str,
        default="npu",
        help="Device target (only npu/Ascend is verified for now; cpu is not verified)",
    )

    args = parser.parse_args()
    args.device = str(args.device).lower()
    _add_repo_to_sys_path(args.fireredasr_repo)
    mindir_dir = Path(args.mindir_dir)

    model_dir = os.path.abspath(os.path.expanduser(args.model_dir)) if args.model_dir else ""
    cmvn_file = args.cmvn_file or (os.path.join(model_dir, "cmvn.ark") if model_dir else "")
    dict_file = args.dict_file or (os.path.join(model_dir, "dict.txt") if model_dir else "")
    if not cmvn_file or not dict_file:
        raise ValueError("cmvn-file/dict-file not set and model-dir is empty")

    if args.device != "npu":
        parser.error(
            "Only --device npu (Ascend) is supported for now; --device cpu is not verified. "
        )

    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = 0

    t0 = time.time()
    encoder_model = mslite.Model()
    encoder_model.build_from_file(
        str(mindir_dir / args.encoder_mindir),
        mslite.ModelType.MINDIR,
        context,
    )

    decoder_model = mslite.Model()
    decoder_model.build_from_file(
        str(mindir_dir / args.decoder_mindir),
        mslite.ModelType.MINDIR,
        context,
    )
    build_ms = (time.time() - t0) * 1000
    print(f"Model build time: {build_ms:.2f} ms")

    t0 = time.time()
    feat, feat_len = load_feat(args.wav_path, cmvn_file)
    feat_np = feat.numpy().astype(np.float32)
    feat_len_np = feat_len.numpy().astype(np.int64)
    feat_ms = (time.time() - t0) * 1000
    feat_frames = int(feat_np.shape[1])
    feat_len_value = int(feat_len_np[0]) if feat_len_np.size > 0 else 0
    print(f"Feature time: {feat_ms:.2f} ms, feat_frames: {feat_frames}, feat_len: {feat_len_value}")

    token_ids, stats = greedy_decode(
        encoder_model,
        decoder_model,
        feat_np,
        feat_len_np.astype(np.int32),
        sos_id=args.sos_id,
        eos_id=args.eos_id,
        max_len=args.max_len,
    )

    print(
        f"Encoder time: {stats['encoder_ms']:.2f} ms, src_len: {stats['src_len']}, "
        f"feat_len: {stats['feat_len']}"
    )
    print(
        f"Total decode time: {stats['total_decode_ms']:.2f} ms, "
        f"avg decode step: {stats['avg_decode_step_ms']:.2f} ms, "
        f"steps: {stats['decode_steps']}"
    )
    print(
        f"Total time: {stats['total_ms']:.2f} ms, "
        f"throughput: {stats['throughput_tok_s']:.2f} tok/s, "
        f"num_tokens: {stats['num_tokens']}"
    )
    text = tokens_to_text(token_ids, dict_file)
    print(f"Recognition result: {text}")


if __name__ == "__main__":
    main()
