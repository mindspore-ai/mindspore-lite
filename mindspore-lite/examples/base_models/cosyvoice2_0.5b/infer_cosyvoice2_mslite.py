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
CosyVoice2-0.5B MindSpore Lite inference script.

Pipeline:
  1. LLM Prefill  : text_ids + speech_ids -> logits + kv_cache
  2. LLM Decode   : speech_id + kv_cache  -> logits + kv_cache  (autoregressive)
  3. Flow Encoder : speech_tokens + embedding + prompt_feat -> mu, spks, cond, mask
  4. Flow Estimator (CFM) : mu + spks + cond + mask -> mel
  5. HiFT (PyTorch CPU) : mel -> waveform

Usage:
  python infer_cosyvoice2_mslite.py \
    --mindir-dir ./cosyvoice2_mindir \
    --model-dir /path/to/CosyVoice2-0.5B \
    --model-code-dir /path/to/CosyVoice \
    --text "你好，很高兴认识你" \
    --output output.wav
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mindspore_lite as mslite
except ImportError:
    print("Error: mindspore_lite not found. Please install MindSpore Lite first.")
    sys.exit(1)

try:
    import torch
    import torchaudio
except ImportError:
    print("Error: torch / torchaudio not found.")
    sys.exit(1)


SPEECH_TOKEN_SIZE = 6561
EOS_TOKEN = SPEECH_TOKEN_SIZE
STOP_TOKENS = (EOS_TOKEN, EOS_TOKEN + 1, EOS_TOKEN + 2)


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    """Stable softmax for 1D logits."""
    x = x.astype(np.float64, copy=False)
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    if not np.isfinite(s) or s <= 0:
        return np.full_like(x, 1.0 / x.size, dtype=np.float64)
    return e / s


def _nucleus_sample(rng: np.random.Generator, logits: np.ndarray, top_p: float, top_k: int) -> int:
    """Top-p/top-k sampling from 1D logits."""
    probs = _softmax_1d(logits)
    sorted_idx = np.argsort(probs)[::-1]
    if top_k is not None and top_k > 0:
        sorted_idx = sorted_idx[:top_k]
    cum = np.cumsum(probs[sorted_idx])
    cut = int(np.searchsorted(cum, top_p, side="left")) + 1
    cut = max(1, min(cut, sorted_idx.size))
    cand_idx = sorted_idx[:cut]
    cand_p = probs[cand_idx]
    cand_p = cand_p / np.sum(cand_p)
    return int(rng.choice(cand_idx, p=cand_p))


def _ras_sample(
    rng: np.random.Generator,
    logits: np.ndarray,
    decoded_tokens: list[int],
    top_p: float = 0.8,
    top_k: int = 25,
    win_size: int = 10,
    tau_r: float = 0.1,
) -> int:
    """
    Repetition-Aware Sampling (RAS) used by CosyVoice2.
    - Nucleus sampling first; if too repetitive in recent window, fallback to random sampling.
    """
    top_id = _nucleus_sample(rng, logits, top_p=top_p, top_k=top_k)
    if win_size > 0 and decoded_tokens:
        win = decoded_tokens[-win_size:]
        rep_num = sum(int(t == top_id) for t in win)
        if rep_num >= win_size * tau_r:
            probs = _softmax_1d(logits)
            top_id = int(rng.choice(np.arange(probs.size), p=probs))
    return top_id


def _mslite_tensor(np_array):
    """Create MSLite Tensor from numpy array."""
    return mslite.Tensor(np_array)


def _build_mslite_inputs(model, feed_dict, preferred_order=None):
    """Build MindSpore Lite input tensor list from a feed dict."""
    inputs = model.get_inputs()
    if not inputs:
        if preferred_order:
            return [_mslite_tensor(feed_dict[k]) for k in preferred_order]
        return [_mslite_tensor(v) for v in feed_dict.values()]
    tensors = []
    ok_by_name = True
    for t in inputs:
        name = getattr(t, "name", None)
        if name is None or name not in feed_dict:
            ok_by_name = False
            break
    if ok_by_name:
        for t in inputs:
            tensors.append(_mslite_tensor(feed_dict[t.name]))
        return tensors
    if preferred_order:
        return [_mslite_tensor(feed_dict[k]) for k in preferred_order]
    raise RuntimeError(
        f"input mismatch. model inputs={[getattr(x, 'name', '') for x in inputs]} "
        f"feed keys={list(feed_dict.keys())}"
    )


def _describe_model_io(model, label=""):
    """Print model input/output info for debugging."""
    if label:
        print(f"  [{label}]")
    for inp in model.get_inputs():
        print(f"    IN  {getattr(inp, 'name', '?'):30s} shape={list(inp.shape)} dtype={inp.dtype}")
    for out in model.get_outputs():
        print(f"    OUT {getattr(out, 'name', '?'):30s} shape={list(out.shape)} dtype={out.dtype}")


def _resolve_mindir_path(mindir_dir: Path, stem: str) -> str:
    """Prefer *_graph.mindir (for >2GB models), fallback to *.mindir."""
    graph = mindir_dir / f"{stem}_graph.mindir"
    if graph.exists():
        return str(graph)
    return str(mindir_dir / f"{stem}.mindir")


def _build_session(path, providers):
    """Create an ONNX Runtime session for speech_tokenizer/campplus (optional)."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


class CosyVoice2MsliteInferencer:
    """Run CosyVoice2-0.5B with MindSpore Lite + PyTorch HiFT vocoder."""

    def __init__(
        self,
        mindir_dir,
        model_dir,
        model_code_dir,
        device="ascend",
        device_id=0,
        seed: int = 0,
        flow_cfg_rate: float = 0.7,
        flow_steps: int = 10,
        decode_mode: str = "ras",
    ):
        self.model_dir = model_dir
        self.model_code_dir = model_code_dir
        self.device = device
        self.sample_rate = 24000
        self.flow_cfg_rate = float(flow_cfg_rate)
        self.flow_steps = int(flow_steps)
        self.decode_mode = str(decode_mode)
        self.rng = np.random.default_rng(int(seed))

        self._load_models(mindir_dir, device, device_id)
        self._load_hift(model_dir, model_code_dir)
        self._load_tokenizer(model_dir)
        self._load_speech_tokenizer(model_dir)

    def _load_models(self, mindir_dir, device, device_id):
        """Load split MindIR models (prefill/decode/flow encoder/flow estimator)."""
        mindir_dir = Path(mindir_dir)

        context = mslite.Context()
        context.target = [device]
        if device == "ascend":
            context.ascend.device_id = device_id

        prefill_path = _resolve_mindir_path(mindir_dir, "cosyvoice2_llm_prefill")
        decode_path = _resolve_mindir_path(mindir_dir, "cosyvoice2_llm_decode")
        flow_enc_path = _resolve_mindir_path(mindir_dir, "cosyvoice2_flow_encoder")
        flow_est_path = _resolve_mindir_path(mindir_dir, "cosyvoice2_flow_estimator")

        print(f"Loading LLM prefill from {prefill_path}...")
        self.prefill_model = mslite.Model()
        self.prefill_model.build_from_file(prefill_path, mslite.ModelType.MINDIR, context)
        _describe_model_io(self.prefill_model, "LLM Prefill")

        print(f"Loading LLM decode from {decode_path}...")
        self.decode_model = mslite.Model()
        self.decode_model.build_from_file(decode_path, mslite.ModelType.MINDIR, context)
        _describe_model_io(self.decode_model, "LLM Decode")

        print(f"Loading Flow encoder from {flow_enc_path}...")
        self.flow_enc_model = mslite.Model()
        self.flow_enc_model.build_from_file(flow_enc_path, mslite.ModelType.MINDIR, context)
        _describe_model_io(self.flow_enc_model, "Flow Encoder")

        print(f"Loading Flow estimator from {flow_est_path}...")
        self.flow_est_model = mslite.Model()
        self.flow_est_model.build_from_file(flow_est_path, mslite.ModelType.MINDIR, context)
        _describe_model_io(self.flow_est_model, "Flow Estimator")

    def _load_hift(self, model_dir, model_code_dir):
        """Load PyTorch HiFT vocoder from CosyVoice source + weights."""
        sys.path.insert(0, str(model_code_dir))
        sys.path.insert(0, str(Path(model_code_dir) / "third_party" / "Matcha-TTS"))
        from cosyvoice.hifigan.generator import HiFTGenerator
        from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor

        f0_predictor = ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
        self.hift = HiFTGenerator(
            in_channels=80, base_channels=512, nb_harmonics=8,
            sampling_rate=24000, nsf_alpha=0.1, nsf_sigma=0.003,
            nsf_voiced_threshold=10, upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            istft_params={"n_fft": 16, "hop_len": 4},
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            lrelu_slope=0.1, audio_limit=0.99, f0_predictor=f0_predictor,
        )
        hift_path = Path(model_dir) / "hift.pt"
        hift_sd = {k.replace("generator.", ""): v for k, v in
                   torch.load(hift_path, map_location="cpu", weights_only=True).items()}
        self.hift.load_state_dict(hift_sd, strict=True)
        self.hift.float().eval()

    def _load_tokenizer(self, model_dir):
        """Load HuggingFace tokenizer from weights directory."""
        from transformers import AutoTokenizer

        qwen_path = str(Path(model_dir) / "CosyVoice-BlankEN")
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)

    def _load_speech_tokenizer(self, model_dir):
        """Load speech_tokenizer_v1.onnx for prompt speech token extraction (optional)."""
        speech_tokenizer_path = Path(model_dir) / "speech_tokenizer_v1.onnx"
        if speech_tokenizer_path.exists():
            self.speech_tokenizer_sess = _build_session(speech_tokenizer_path, ["CPUExecutionProvider"])
        else:
            self.speech_tokenizer_sess = None

    def _tokenize_text(self, text: str) -> np.ndarray:
        """Tokenize input text to Qwen2 text_ids (np.int64)."""
        enc = self.tokenizer(text, return_tensors="np")
        return enc["input_ids"].astype(np.int64)

    def _compute_mel(self, wav_tensor):
        """Compute log-mel spectrogram (B, T, 80) for CosyVoice prompt."""
        import torchaudio.transforms as T
        if wav_tensor.dim() == 2 and wav_tensor.size(0) > 1:
            wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
        mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=1920,
            win_length=1920, hop_length=480, n_mels=80,
            f_min=0, f_max=8000, center=False,
        )
        mel = mel_transform(wav_tensor)
        mel = torch.log(torch.clamp(mel, min=1e-10))
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        # Convert to (B, T, 80) for prompt_feat / campplus / speech_tokenizer.
        mel = mel.transpose(1, 2).contiguous()
        return mel.numpy()

    def _extract_speech_tokens(self, wav_np: np.ndarray) -> np.ndarray:
        """Extract prompt speech tokens with speech_tokenizer_v1.onnx (optional)."""
        if self.speech_tokenizer_sess is None:
            return np.zeros((1, 0), dtype=np.int64)
        from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail
        feat_np = self._compute_mel(torch.from_numpy(wav_np).float()).astype(np.float32)
        input_name = self.speech_tokenizer_sess.get_inputs()[0].name
        try:
            outputs = self.speech_tokenizer_sess.run(None, {input_name: feat_np})
        except OrtFail:
            outputs = self.speech_tokenizer_sess.run(
                None, {input_name: np.transpose(feat_np, (0, 2, 1))}
            )
        return outputs[0].astype(np.int64)

    def _extract_embedding(self, wav_np: np.ndarray) -> np.ndarray:
        """Extract speaker embedding with campplus.onnx (optional)."""
        campplus_path = Path(self.model_dir) / "campplus.onnx"
        if not campplus_path.exists():
            return np.zeros((1, 192), dtype=np.float32)
        from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail
        sess = _build_session(campplus_path, ["CPUExecutionProvider"])
        if sess is None:
            return np.zeros((1, 192), dtype=np.float32)
        feat_np = self._compute_mel(torch.from_numpy(wav_np).float()).astype(np.float32)
        input_name = sess.get_inputs()[0].name
        try:
            out = sess.run(None, {input_name: feat_np})
        except OrtFail:
            out = sess.run(None, {input_name: np.transpose(feat_np, (0, 2, 1))})
        return out[0].astype(np.float32)

    def _prepare_llm_inputs(self, text_ids, speech_ids):
        """Prepare and validate LLM inputs, returning prefill inputs and metadata."""
        text_ids_np = text_ids.astype(np.int64, copy=False)
        speech_ids_np = speech_ids.astype(np.int64, copy=False)

        text_len = text_ids_np.shape[1]
        speech_len = speech_ids_np.shape[1]

        pad_empty_speech = speech_len == 0
        if pad_empty_speech:
            speech_ids_np = np.array([[0]], dtype=np.int64)
            speech_len = 1

        total_len = 2 + text_len + speech_len
        max_len = int(text_len * 20)
        min_len = int(text_len * 2)

        attention_mask = np.ones((1, total_len), dtype=np.int32)
        position_ids = np.arange(total_len, dtype=np.int32).reshape(1, -1)

        text_ids_input = text_ids_np.astype(np.int32, copy=False)
        speech_ids_input = speech_ids_np.astype(np.int32, copy=False)
        if text_ids_input.ndim == 1:
            text_ids_input = text_ids_input.reshape(1, -1)
        if speech_ids_input.ndim == 1:
            speech_ids_input = speech_ids_input.reshape(1, -1)

        prefill_inputs = _build_mslite_inputs(self.prefill_model, {
            "text_ids": text_ids_input,
            "speech_ids": speech_ids_input,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, preferred_order=["text_ids", "speech_ids", "attention_mask", "position_ids"])

        return prefill_inputs, total_len, max_len, min_len, pad_empty_speech

    def _run_llm_prefill(self, prefill_inputs):
        """Run LLM prefill stage and return logits, past_kv, and elapsed ms."""
        t0 = time.perf_counter()
        prefill_out = self.prefill_model.predict(prefill_inputs)
        prefill_ms = (time.perf_counter() - t0) * 1000.0
        logits = prefill_out[0].get_data_to_numpy()
        past_kv = prefill_out[1].get_data_to_numpy()
        return logits, past_kv, prefill_ms

    def _run_llm_decode_step(self, next_token, cur_pos, past_kv, step, min_len):
        """Run a single LLM decode step and return updated logits, past_kv, and timing."""
        speech_id_np = np.array([[next_token]], dtype=np.int32)
        cur_attention_mask = np.ones((1, cur_pos + 1), dtype=np.int32)
        cur_pos_ids = np.array([[cur_pos]], dtype=np.int32)

        decode_inputs = _build_mslite_inputs(self.decode_model, {
            "speech_id": speech_id_np,
            "attention_mask": cur_attention_mask,
            "position_ids": cur_pos_ids,
            "past_key_values": past_kv,
        }, preferred_order=["speech_id", "attention_mask", "position_ids", "past_key_values"])

        t1 = time.perf_counter()
        decode_out = self.decode_model.predict(decode_inputs)
        decode_ms = (time.perf_counter() - t1) * 1000.0
        logits = decode_out[0].get_data_to_numpy()
        past_kv = decode_out[1].get_data_to_numpy()

        step_logits = logits[0, -1].astype(np.float32, copy=False)
        if step < min_len:
            step_logits = step_logits.copy()
            step_logits[EOS_TOKEN:EOS_TOKEN + 3] = -1.0e10
        return step_logits, past_kv, decode_ms

    def _run_llm(self, text_ids, speech_ids):
        """LLM prefill + decode loop to generate speech tokens."""
        prefill_inputs, total_len, max_len, min_len, pad_empty_speech = self._prepare_llm_inputs(
            text_ids, speech_ids)

        logits, past_kv, prefill_ms = self._run_llm_prefill(prefill_inputs)

        if pad_empty_speech:
            logits = logits[:, :-1, :]
            past_kv = past_kv[:, :, :, :-1, :]
            total_len = total_len - 1

        generated: list[int] = []
        first_logits = logits[0, -1].astype(np.float32, copy=False)
        if min_len > 0:
            first_logits = first_logits.copy()
            first_logits[EOS_TOKEN:EOS_TOKEN + 3] = -1.0e10
        next_token = self._next_token(first_logits, generated)
        if next_token not in STOP_TOKENS:
            generated.append(next_token)

        cur_pos = total_len
        decode_ms_total = 0.0
        decode_steps = 0

        for step in range(max_len - 1):
            if step >= min_len and next_token in STOP_TOKENS:
                break

            step_logits, past_kv, step_ms = self._run_llm_decode_step(
                next_token, cur_pos, past_kv, step, min_len)
            decode_ms_total += step_ms
            decode_steps += 1

            next_token = self._next_token(step_logits, generated)
            if next_token in STOP_TOKENS:
                break
            generated.append(next_token)
            cur_pos += 1

            if len(generated) % 50 == 0:
                print(f"  LLM generated {len(generated)} tokens...")

        print(f"  LLM finished: {len(generated)} speech tokens")
        perf = {
            "prefill_ms": prefill_ms,
            "decode_ms_total": decode_ms_total,
            "decode_steps": decode_steps,
            "avg_decode_step_ms": (decode_ms_total / max(1, decode_steps)),
        }
        return np.array(generated, dtype=np.int64), perf

    def _next_token(self, logits_1d: np.ndarray, decoded: list[int]) -> int:
        if self.decode_mode == "greedy":
            return int(np.argmax(logits_1d))
        return _ras_sample(self.rng, logits_1d.astype(np.float32, copy=False), decoded_tokens=decoded)

    def _run_flow_encoder(self, speech_tokens, embedding, prompt_feat):
        """Flow Encoder inference (speech tokens -> mu/spks/cond/mask)."""
        token_np = speech_tokens.astype(np.int64).reshape(1, -1)
        token_len_np = np.array([token_np.shape[1]], dtype=np.int64)
        embedding_np = embedding.astype(np.float32)
        prompt_feat_np = prompt_feat.astype(np.float32)
        if prompt_feat_np.ndim == 2:
            prompt_feat_np = prompt_feat_np[np.newaxis, :]

        # MSLite Ascend runtime cannot handle size=0 tensors.
        # When prompt_feat is empty (no prompt audio), pad with 1 frame of zeros.
        if prompt_feat_np.shape[1] == 0:
            prompt_feat_np = np.zeros((1, 1, 80), dtype=np.float32)

        # Preserve 2D shape for token input
        token_input = token_np.astype(np.int32, copy=False)
        if token_input.ndim == 1:
            token_input = token_input.reshape(1, -1)
        token_len_input = token_len_np.astype(np.int32, copy=False)

        enc_inputs = _build_mslite_inputs(self.flow_enc_model, {
            "token": token_input,
            "token_len": token_len_input,
            "embedding": embedding_np,
            "prompt_feat": prompt_feat_np,
        }, preferred_order=["token", "token_len", "embedding", "prompt_feat"])

        t0 = time.perf_counter()
        enc_out = self.flow_enc_model.predict(enc_inputs)
        enc_ms = (time.perf_counter() - t0) * 1000.0
        mu = enc_out[0].get_data_to_numpy()
        spks = enc_out[1].get_data_to_numpy()
        cond = enc_out[2].get_data_to_numpy()
        mask = enc_out[3].get_data_to_numpy()
        return mu, spks, cond, mask, enc_ms

    def _run_flow_estimator(self, mu, spks, cond, mask, n_timesteps: int):
        """Flow Estimator (CFM) sampling with CFG. Returns mel and elapsed ms."""
        t_span = np.linspace(0, 1, n_timesteps + 1, dtype=np.float32)
        t_span = 1.0 - np.cos(t_span * 0.5 * np.pi)

        mu = mu.astype(np.float32, copy=False)
        spks = spks.astype(np.float32, copy=False)
        cond = cond.astype(np.float32, copy=False)
        mask = mask.astype(np.float32, copy=False)

        mel_len = int(mu.shape[2])
        z = self.rng.standard_normal((1, 80, mel_len), dtype=np.float32)
        cfg = float(self.flow_cfg_rate)

        t0 = time.perf_counter()

        for i in range(n_timesteps):
            t_cur = float(t_span[i])
            dt = float(t_span[i + 1] - t_span[i])

            if cfg == 0.0:
                est_inputs = _build_mslite_inputs(self.flow_est_model, {
                    "x": z,
                    "mask": mask,
                    "mu": mu,
                    "t": np.array([t_cur], dtype=np.float32),
                    "spks": spks,
                    "cond": cond,
                }, preferred_order=["x", "mask", "mu", "t", "spks", "cond"])
                est_out = self.flow_est_model.predict(est_inputs)
                dphi_dt = est_out[0].get_data_to_numpy()
            else:
                # Batch=2: [conditional, unconditional]
                x_in = np.concatenate([z, z], axis=0)
                mask_in = np.concatenate([mask, mask], axis=0)
                mu_in = np.zeros((2, 80, mel_len), dtype=np.float32)
                mu_in[0:1] = mu
                spks_in = np.zeros((2, spks.shape[1]), dtype=np.float32)
                spks_in[0:1] = spks
                cond_in = np.zeros((2, 80, mel_len), dtype=np.float32)
                cond_in[0:1] = cond
                t_in = np.array([t_cur, t_cur], dtype=np.float32)

                est_inputs = _build_mslite_inputs(self.flow_est_model, {
                    "x": x_in,
                    "mask": mask_in,
                    "mu": mu_in,
                    "t": t_in,
                    "spks": spks_in,
                    "cond": cond_in,
                }, preferred_order=["x", "mask", "mu", "t", "spks", "cond"])
                est_out = self.flow_est_model.predict(est_inputs)
                dphi_dt_all = est_out[0].get_data_to_numpy()
                dphi_dt = (1.0 + cfg) * dphi_dt_all[0:1] - cfg * dphi_dt_all[1:2]

            z = z + dphi_dt * dt

        est_ms = (time.perf_counter() - t0) * 1000.0
        return z.astype(np.float32), est_ms

    def _run_hift(self, mel_np):
        """HiFT vocoder inference (mel -> waveform) using PyTorch on CPU."""
        mel_tensor = torch.from_numpy(mel_np).float()
        with torch.no_grad():
            speech, _ = self.hift.inference(mel_tensor)
        return speech.cpu().numpy()

    def _prepare_prompt_features(self, prompt_wav_path):
        """Extract prompt features from wav file for voice cloning."""
        wav, sr = torchaudio.load(prompt_wav_path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav_np = wav.mean(dim=0).numpy()
        prompt_speech_token = self._extract_speech_tokens(wav_np)
        prompt_feat_np = self._compute_mel(wav.float()).astype(np.float32)
        if prompt_feat_np.ndim == 2:
            prompt_feat_np = prompt_feat_np[np.newaxis, :]
        embedding = self._extract_embedding(wav_np)
        return prompt_speech_token, prompt_feat_np, embedding

    def _save_output_and_print_perf(self, speech_np, output_path, t_start, llm_perf, flow_enc_ms,
                                     flow_est_ms, hift_ms):
        """Save waveform to file and print performance summary."""
        speech_tensor = torch.from_numpy(speech_np).float()
        if speech_tensor.dim() == 1:
            speech_tensor = speech_tensor.unsqueeze(0)
        try:
            import soundfile as sf
            sf.write(output_path, speech_tensor.squeeze(0).numpy(), self.sample_rate)
        except ImportError:
            torchaudio.save(output_path, speech_tensor, self.sample_rate)

        duration = speech_tensor.shape[-1] / self.sample_rate
        total_ms = (time.perf_counter() - t_start) * 1000.0
        rtf = (total_ms / 1000.0) / max(1e-6, duration)

        print(f"\nSaved to {output_path} ({duration:.2f}s)")
        print(f"Total time: {total_ms:.2f} ms (RTF: {rtf:.3f})")

        print("\n[Performance Markdown]")
        print(
            "| Prefill (ms) | Total Decode (ms) | Avg decode step (ms) | "
            "Flow Encoder (ms) | Flow Estimator (ms) | HiFT (ms) | Total (ms) |"
        )
        print("|---:|---:|---:|---:|---:|---:|---:|")
        print(
            f"| {llm_perf['prefill_ms']:.2f} | {llm_perf['decode_ms_total']:.2f} | "
            f"{llm_perf['avg_decode_step_ms']:.2f} | {flow_enc_ms:.2f} | "
            f"{flow_est_ms:.2f} | {hift_ms:.2f} | {total_ms:.2f} |"
        )

    def synthesize(self, text, prompt_wav_path=None, output_path="output.wav"):
        """Run full TTS pipeline and save waveform to output_path."""
        t_start = time.perf_counter()
        print(f"Input text: {text}")

        text_ids = self._tokenize_text(text)
        llm_speech_token = np.zeros((1, 0), dtype=np.int64)
        flow_speech_token = np.zeros((1, 0), dtype=np.int64)
        prompt_feat = np.zeros((1, 0, 80), dtype=np.float32)
        flow_embedding = np.zeros((1, 192), dtype=np.float32)

        if prompt_wav_path is not None:
            prompt_speech_token, prompt_feat_np, embedding = self._prepare_prompt_features(prompt_wav_path)
            prompt_feat = prompt_feat_np
            llm_speech_token = prompt_speech_token
            flow_speech_token = prompt_speech_token
            flow_embedding = embedding

        # Step 1: LLM -> speech tokens
        print("\n[1/4] Running LLM (Prefill + Decode)...")
        if text_ids.shape[1] == 0:
            text_ids = np.array([[1]], dtype=np.int64)
        if llm_speech_token.shape[1] == 0:
            llm_speech_token = np.zeros((1, 0), dtype=np.int64)

        generated_tokens, llm_perf = self._run_llm(text_ids, llm_speech_token)
        llm_ms = llm_perf["prefill_ms"] + llm_perf["decode_ms_total"]
        print(f"  LLM prefill: {llm_perf['prefill_ms']:.2f} ms")
        print(
            f"  LLM decode : {llm_perf['decode_ms_total']:.2f} ms "
            f"(steps={llm_perf['decode_steps']}, avg_step={llm_perf['avg_decode_step_ms']:.2f} ms)"
        )
        print(f"  LLM total  : {llm_ms:.2f} ms")

        if generated_tokens.size == 0:
            print("No speech tokens generated!")
            return

        # Step 2: Flow Encoder -> mu/spks/cond/mask
        print("\n[2/4] Running Flow Encoder...")
        all_tokens = np.concatenate([flow_speech_token.flatten(), generated_tokens], axis=0)
        all_tokens_np = all_tokens.reshape(1, -1).astype(np.int64)
        if prompt_feat.ndim == 2:
            prompt_feat = prompt_feat[np.newaxis, :]

        mu, spks, cond, mask, flow_enc_ms = self._run_flow_encoder(all_tokens_np, flow_embedding, prompt_feat)
        if float(mask.mean()) < 0.999:
            raise RuntimeError(
                "Flow encoder output mask is not all-ones in batch=1 inference. "
                "This will cause tail frames to remain noisy. "
                "Please re-export Flow Encoder and reconvert to MindIR."
            )
        print(f"  Flow Encoder: {flow_enc_ms:.2f} ms")

        # Step 3: Flow Estimator (CFM) -> mel
        print("\n[3/4] Running Flow Estimator (CFM)...")
        mel_np, flow_est_ms = self._run_flow_estimator(mu, spks, cond, mask, n_timesteps=self.flow_steps)
        mel_len1 = prompt_feat.shape[1]
        mel_np = mel_np[:, :, mel_len1:]
        print(f"  Flow Estimator ({self.flow_steps} steps, cfg={self.flow_cfg_rate}): {flow_est_ms:.2f} ms")

        # Step 4: HiFT -> waveform
        print("\n[4/4] Running HiFT vocoder...")
        hift_t0 = time.perf_counter()
        speech_np = self._run_hift(mel_np)
        hift_ms = (time.perf_counter() - hift_t0) * 1000.0
        print(f"  HiFT vocoder: {hift_ms:.2f} ms")

        self._save_output_and_print_perf(speech_np, output_path, t_start, llm_perf,
                                         flow_enc_ms, flow_est_ms, hift_ms)


def main():
    parser = argparse.ArgumentParser(description="CosyVoice2-0.5B MindSpore Lite inference")
    parser.add_argument("--mindir-dir", type=str, default="./cosyvoice2_mindir",
                        help="Directory with MindIR model files")
    parser.add_argument("--model-dir", type=str,
                        default="/Users/apple/git/models/models_weights/CosyVoice2-0.5B",
                        help="Path to CosyVoice2-0.5B weights")
    parser.add_argument("--model-code-dir", type=str,
                        default="/Users/apple/git/models/models_code/CosyVoice",
                        help="Path to CosyVoice source code")
    parser.add_argument("--device", type=str, default="ascend",
                        choices=["cpu", "ascend"],
                        help="Device for inference")
    parser.add_argument("--device-id", type=int, default=0,
                        help="Ascend device ID")
    parser.add_argument("--text", type=str, default="你好，很高兴认识你。",
                        help="Text to synthesize")
    parser.add_argument("--prompt-wav", type=str, default=None,
                        help="Prompt wav for voice cloning")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output wav path")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (affects LLM sampling and flow noise)")
    parser.add_argument("--decode-mode", type=str, default="ras",
                        choices=["greedy", "ras"],
                        help="LLM decode mode. greedy gives deterministic parity; ras follows CosyVoice2 sampling.")
    parser.add_argument("--flow-cfg", type=float, default=0.7,
                        help="Flow classifier-free guidance rate")
    parser.add_argument("--flow-steps", type=int, default=10,
                        help="Flow Euler steps (n_timesteps)")
    args = parser.parse_args()

    inferencer = CosyVoice2MsliteInferencer(
        mindir_dir=args.mindir_dir,
        model_dir=args.model_dir,
        model_code_dir=args.model_code_dir,
        device=args.device,
        device_id=args.device_id,
        seed=args.seed,
        flow_cfg_rate=args.flow_cfg,
        flow_steps=args.flow_steps,
        decode_mode=args.decode_mode,
    )
    inferencer.synthesize(
        text=args.text,
        prompt_wav_path=args.prompt_wav,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
