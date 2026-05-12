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
CosyVoice2-0.5B ONNX Runtime inference script.

Pipeline:
  1. LLM Prefill  : text_ids + speech_ids → logits + kv_cache
  2. LLM Decode   : speech_id + kv_cache  → logits + kv_cache  (autoregressive)
  3. Flow Encoder : speech_tokens + embedding + prompt_feat → mu, spks, cond, mask
  4. Flow Estimator (CFM) : mu + spks + cond + mask → mel
  5. HiFT (PyTorch) : mel → waveform

Usage:
  python infer_cosyvoice2_onnx.py \
    --onnx-dir ./cosyvoice2_onnx \
    --model-dir /path/to/CosyVoice2-0.5B \
    --model-code-dir /path/to/CosyVoice \
    --text "你好，很高兴认识你" \
    --output output.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not found. pip install onnxruntime")
    sys.exit(1)

from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail

try:
    import torch
    import torchaudio
except ImportError:
    print("Error: torch / torchaudio not found.")
    sys.exit(1)


SPEECH_TOKEN_SIZE = 6561
EOS_TOKEN = SPEECH_TOKEN_SIZE


def _build_session(path, providers):
    """Create an ONNX Runtime session with basic graph optimizations."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    if not np.isfinite(s) or s <= 0:
        # fallback to uniform distribution
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
    """Repetition-Aware Sampling (RAS) used by CosyVoice2."""
    top_id = _nucleus_sample(rng, logits, top_p=top_p, top_k=top_k)
    if win_size > 0 and decoded_tokens:
        win = decoded_tokens[-win_size:]
        rep_num = sum(int(t == top_id) for t in win)
        if rep_num >= win_size * tau_r:
            # fallback: random sampling from full distribution
            probs = _softmax_1d(logits)
            top_id = int(rng.choice(np.arange(probs.size), p=probs))
    return top_id


class CosyVoice2OnnxInferencer:
    """End-to-end CosyVoice2-0.5B inference via ONNX Runtime + PyTorch HiFT."""
    def __init__(
        self,
        onnx_dir,
        model_dir,
        model_code_dir,
        device="cpu",
        seed: int = 0,
        flow_cfg_rate: float = 0.7,
        flow_steps: int = 10,
        decode_mode: str = "ras",
    ):
        self.model_dir = model_dir
        self.model_code_dir = model_code_dir
        self.device = device
        self.flow_cfg_rate = float(flow_cfg_rate)
        self.flow_steps = int(flow_steps)
        self.decode_mode = str(decode_mode)

        self.rng = np.random.default_rng(int(seed))

        providers = ["CPUExecutionProvider"]

        onnx_dir = Path(onnx_dir)
        print(f"Loading LLM prefill from {onnx_dir / 'cosyvoice2_llm_prefill.onnx'}")
        self.prefill_sess = _build_session(onnx_dir / "cosyvoice2_llm_prefill.onnx", providers)
        print(f"Loading LLM decode from {onnx_dir / 'cosyvoice2_llm_decode.onnx'}")
        self.decode_sess = _build_session(onnx_dir / "cosyvoice2_llm_decode.onnx", providers)
        print(f"Loading Flow encoder from {onnx_dir / 'cosyvoice2_flow_encoder.onnx'}")
        self.flow_enc_sess = _build_session(onnx_dir / "cosyvoice2_flow_encoder.onnx", providers)
        print(f"Loading Flow estimator from {onnx_dir / 'cosyvoice2_flow_estimator.onnx'}")
        self.flow_est_sess = _build_session(onnx_dir / "cosyvoice2_flow_estimator.onnx", providers)

        self._load_hift(model_dir, model_code_dir)
        self._load_tokenizer(model_dir)
        self._load_speech_tokenizer(model_dir)

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
        self.sample_rate = 24000

    def _load_speech_tokenizer(self, model_dir):
        speech_tokenizer_path = Path(model_dir) / "speech_tokenizer_v1.onnx"
        if speech_tokenizer_path.exists():
            providers = ["CPUExecutionProvider"]
            self.speech_tokenizer_sess = _build_session(speech_tokenizer_path, providers)
        else:
            self.speech_tokenizer_sess = None

    def _tokenize_text(self, text: str) -> np.ndarray:
        """Tokenize input text to Qwen2 text_ids (np.int64)."""
        enc = self.tokenizer(text, return_tensors="np")
        return enc["input_ids"].astype(np.int64)

    def _extract_speech_tokens(self, wav_np: np.ndarray) -> np.ndarray:
        """Extract prompt speech tokens with speech_tokenizer_v1.onnx (optional)."""
        if self.speech_tokenizer_sess is None:
            return np.zeros((1, 0), dtype=np.int64)
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
        sess = _build_session(campplus_path, ["CPUExecutionProvider"])
        feat_np = self._compute_mel(torch.from_numpy(wav_np).float()).astype(np.float32)
        input_name = sess.get_inputs()[0].name
        try:
            out = sess.run(None, {input_name: feat_np})
        except OrtFail:
            out = sess.run(None, {input_name: np.transpose(feat_np, (0, 2, 1))})
        return out[0].astype(np.float32)

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

    # ------------------------------------------------------------------
    # LLM inference (prefill + decode)
    # ------------------------------------------------------------------
    def _next_token(self, logits_1d: np.ndarray, decoded: list[int]) -> int:
        if self.decode_mode == "greedy":
            return int(np.argmax(logits_1d))
        return _ras_sample(
            self.rng,
            logits_1d.astype(np.float32, copy=False),
            decoded_tokens=decoded,
            top_p=0.8,
            top_k=25,
            win_size=10,
            tau_r=0.1,
        )

    def _run_llm(self, text_ids, speech_ids):
        """LLM prefill + decode loop to generate speech tokens."""
        text_ids_np = text_ids.astype(np.int64)
        speech_ids_np = speech_ids.astype(np.int64)

        text_len = text_ids_np.shape[1]
        speech_len = speech_ids_np.shape[1]
        pad_empty_speech = speech_len == 0
        if pad_empty_speech:
            speech_ids_np = np.array([[0]], dtype=np.int64)
            speech_len = 1

        total_len = 2 + text_len + speech_len

        attention_mask = np.ones((1, total_len), dtype=np.int64)
        position_ids = np.arange(total_len, dtype=np.int64).reshape(1, -1)

        logits, past_kv = self.prefill_sess.run(None, {
            "text_ids": text_ids_np,
            "speech_ids": speech_ids_np,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        })
        if pad_empty_speech:
            logits = logits[:, :-1, :]
            past_kv = past_kv[:, :, :, :-1, :]
            total_len = total_len - 1

        generated: list[int] = []
        # Before min_len, forbid stop tokens (6561/6562/6563).
        first_logits = logits[0, -1].astype(np.float32, copy=False)
        first_logits[EOS_TOKEN:EOS_TOKEN + 3] = -1.0e10
        next_token = self._next_token(first_logits, generated)
        if next_token not in (EOS_TOKEN, EOS_TOKEN + 1, EOS_TOKEN + 2):
            generated.append(next_token)

        cur_pos = total_len
        max_len = int(text_len * 20)
        min_len = int(text_len * 2)

        for step in range(max_len - 1):
            if step >= min_len and next_token in (EOS_TOKEN, EOS_TOKEN + 1, EOS_TOKEN + 2):
                break

            speech_id_np = np.array([[next_token]], dtype=np.int64)
            cur_attention_mask = np.ones((1, cur_pos + 1), dtype=np.int64)
            cur_pos_ids = np.array([[cur_pos]], dtype=np.int64)

            logits, past_kv = self.decode_sess.run(None, {
                "speech_id": speech_id_np,
                "attention_mask": cur_attention_mask,
                "position_ids": cur_pos_ids,
                "past_key_values": past_kv,
            })

            step_logits = logits[0, -1].astype(np.float32, copy=False)
            if step < min_len:
                step_logits = step_logits.copy()
                step_logits[EOS_TOKEN:EOS_TOKEN + 3] = -1.0e10
            next_token = self._next_token(step_logits, generated)
            if next_token in (EOS_TOKEN, EOS_TOKEN + 1, EOS_TOKEN + 2):
                break
            generated.append(next_token)
            cur_pos += 1

            if len(generated) % 50 == 0:
                print(f"  LLM generated {len(generated)} tokens...")

        print(f"  LLM finished: {len(generated)} speech tokens")
        return np.array(generated, dtype=np.int64)

    def _run_flow_encoder(self, all_tokens_np: np.ndarray, flow_embedding: np.ndarray, prompt_feat: np.ndarray):
        prompt_feat_np = prompt_feat.astype(np.float32, copy=False)
        if prompt_feat_np.ndim == 2:
            prompt_feat_np = prompt_feat_np[np.newaxis, :]
        if prompt_feat_np.shape[1] == 0:
            prompt_feat_np = np.zeros((1, 1, 80), dtype=np.float32)
        return self.flow_enc_sess.run(None, {
            "token": all_tokens_np.astype(np.int64, copy=False),
            "token_len": np.array([all_tokens_np.shape[1]], dtype=np.int64),
            "embedding": flow_embedding.astype(np.float32, copy=False),
            "prompt_feat": prompt_feat_np,
        })

    # ------------------------------------------------------------------
    # Flow inference (encoder + estimator)
    # ------------------------------------------------------------------
    def _run_flow_estimator(self, mu, spks, cond, mask, n_timesteps: int):
        """Euler solver + classifier-free guidance (CFG) as in CosyVoice."""
        mu = mu.astype(np.float32, copy=False)
        spks = spks.astype(np.float32, copy=False)
        cond = cond.astype(np.float32, copy=False)
        mask = mask.astype(np.float32, copy=False)

        mel_len = int(mu.shape[2])
        z = self.rng.standard_normal((1, 80, mel_len), dtype=np.float32)

        t_span = np.linspace(0, 1, n_timesteps + 1, dtype=np.float32)
        t_span = 1.0 - np.cos(t_span * 0.5 * np.pi)

        cfg = float(self.flow_cfg_rate)
        for i in range(n_timesteps):
            t_cur = float(t_span[i])
            dt = float(t_span[i + 1] - t_span[i])

            if cfg == 0.0:
                out = self.flow_est_sess.run(None, {
                    "x": z,
                    "mask": mask,
                    "mu": mu,
                    "t": np.array([t_cur], dtype=np.float32),
                    "spks": spks,
                    "cond": cond,
                })[0]
                dphi_dt = out
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

                out = self.flow_est_sess.run(None, {
                    "x": x_in,
                    "mask": mask_in,
                    "mu": mu_in,
                    "t": t_in,
                    "spks": spks_in,
                    "cond": cond_in,
                })[0]
                dphi_dt = (1.0 + cfg) * out[0:1] - cfg * out[1:2]

            z = z + dphi_dt * dt

        return z.astype(np.float32)

    # ------------------------------------------------------------------
    # HiFT inference (PyTorch)
    # ------------------------------------------------------------------
    def _run_hift(self, mel_np):
        mel_tensor = torch.from_numpy(mel_np).float()
        with torch.no_grad():
            speech, _ = self.hift.inference(mel_tensor)
        return speech.cpu().numpy()

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def synthesize(self, text, prompt_wav_path=None, output_path="output.wav"):
        """Run full TTS pipeline and save waveform to output_path."""
        print(f"Input text: {text}")

        text_ids = self._tokenize_text(text)
        llm_speech_token = np.zeros((1, 0), dtype=np.int64)
        flow_speech_token = np.zeros((1, 0), dtype=np.int64)
        prompt_feat = np.zeros((1, 0, 80), dtype=np.float32)
        flow_embedding = np.zeros((1, 192), dtype=np.float32)

        if prompt_wav_path is not None:
            wav, sr = torchaudio.load(prompt_wav_path)
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            wav_np = wav.mean(dim=0).numpy()
            prompt_speech_token = self._extract_speech_tokens(wav_np)
            prompt_feat = self._compute_mel(wav.float()).astype(np.float32)
            if prompt_feat.ndim == 2:
                prompt_feat = prompt_feat[np.newaxis, :]
            llm_speech_token = prompt_speech_token
            flow_speech_token = prompt_speech_token
            embedding = self._extract_embedding(wav_np)
            flow_embedding = embedding

        # Step 1: LLM → speech tokens
        print("\n[1/4] Running LLM (Prefill + Decode)...")
        if text_ids.shape[1] == 0:
            text_ids = np.array([[1]], dtype=np.int64)
        if llm_speech_token.shape[1] == 0:
            llm_speech_token = np.zeros((1, 0), dtype=np.int64)

        generated_tokens = self._run_llm(text_ids, llm_speech_token)

        if len(generated_tokens) == 0:
            print("No speech tokens generated!")
            return

        # Step 2: Flow Encoder → mu/spks/cond/mask
        print("\n[2/4] Running Flow Encoder...")
        all_tokens = np.concatenate([flow_speech_token.flatten(), generated_tokens], axis=0)
        all_tokens_np = all_tokens.reshape(1, -1).astype(np.int64)
        if prompt_feat.ndim == 2:
            prompt_feat = prompt_feat[np.newaxis, :]

        mu, spks, cond, mask = self._run_flow_encoder(all_tokens_np, flow_embedding, prompt_feat)
        if float(mask.mean()) < 0.999:
            raise RuntimeError(
                "Flow encoder output mask is not all-ones in batch=1 inference. "
                "This will cause tail frames to remain noisy. "
                "Please re-export Flow Encoder ONNX using export_cosyvoice2_onnx.py."
            )

        # Step 3: Flow Estimator (CFM) → mel
        print("\n[3/4] Running Flow Estimator (CFM)...")
        mel_np = self._run_flow_estimator(mu, spks, cond, mask, n_timesteps=self.flow_steps)
        mel_len1 = prompt_feat.shape[1]
        mel_np = mel_np[:, :, mel_len1:]

        # Step 4: HiFT → waveform
        print("\n[4/4] Running HiFT vocoder...")
        speech_np = self._run_hift(mel_np)

        # Save
        speech_tensor = torch.from_numpy(speech_np).float()
        if speech_tensor.dim() == 1:
            speech_tensor = speech_tensor.unsqueeze(0)
        import soundfile as sf
        sf.write(output_path, speech_tensor.squeeze(0).numpy(), self.sample_rate)
        duration = speech_tensor.shape[-1] / self.sample_rate
        print(f"\nSaved to {output_path} ({duration:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description="CosyVoice2-0.5B ONNX inference")
    parser.add_argument("--onnx-dir", type=str, default="./cosyvoice2_onnx",
                        help="Directory with ONNX model files")
    parser.add_argument("--model-dir", type=str,
                        default="/Users/apple/git/models/models_weights/CosyVoice2-0.5B",
                        help="Path to CosyVoice2-0.5B weights")
    parser.add_argument("--model-code-dir", type=str,
                        default="/Users/apple/git/models/models_code/CosyVoice",
                        help="Path to CosyVoice source code")
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

    inferencer = CosyVoice2OnnxInferencer(
        onnx_dir=args.onnx_dir,
        model_dir=args.model_dir,
        model_code_dir=args.model_code_dir,
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
