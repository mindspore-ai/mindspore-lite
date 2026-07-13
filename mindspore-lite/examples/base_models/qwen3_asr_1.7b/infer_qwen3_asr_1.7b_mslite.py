"""End-to-end Qwen3-ASR 1.7B transcription with MindSpore Lite (separate prefill/decode + zero-copy)."""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Compatibility shim: qwen_asr 0.0.6 uses `@check_model_inputs()` (factory form),
# but transformers>=4.56 made it `check_model_inputs(func)` (direct decorator).
try:
    import transformers.utils.generic as _tug
    if not getattr(_tug.check_model_inputs, "_compat_patched", False):
        _orig_check_model_inputs = _tug.check_model_inputs

        def _compat_check_model_inputs(func=None):
            if func is not None:
                return _orig_check_model_inputs(func)

            def _deco(f):
                return _orig_check_model_inputs(f)

            return _deco

        _compat_check_model_inputs._compat_patched = True
        _tug.check_model_inputs = _compat_check_model_inputs
except Exception:
    pass

# qwen_asr imports `nagisa` (Japanese tokenizer) only for the forced-aligner path,
# which is unused here. Stub it so the package imports without the extra dep.
try:
    import nagisa  # noqa: F401  # pylint: disable=unused-import
except Exception:
    import sys
    import types
    if "nagisa" not in sys.modules:
        sys.modules["nagisa"] = types.ModuleType("nagisa")

from qwen_asr.inference.utils import (
    SAMPLE_RATE,
    merge_languages,
    normalize_audio_input,
    normalize_language_name,
    parse_asr_output,
    split_audio_into_chunks,
    validate_language,
)
from transformers import AutoTokenizer, WhisperFeatureExtractor


KV_CACHE_LEN = 1024
PREFILL_GEARS = [512, 640, 768]


def _require_mslite():
    try:
        import mindspore_lite as mslite
    except Exception as e:
        raise RuntimeError("mindspore_lite is required for MindIR inference") from e
    return mslite


_NP_TO_MSLITE_DTYPE = None


def _np_dtype_to_mslite(mslite, dt):
    global _NP_TO_MSLITE_DTYPE
    if _NP_TO_MSLITE_DTYPE is None:
        _NP_TO_MSLITE_DTYPE = {
            np.dtype(np.float32): mslite.DataType.FLOAT32,
            np.dtype(np.float16): mslite.DataType.FLOAT16,
            np.dtype(np.int32): mslite.DataType.INT32,
            np.dtype(np.int64): mslite.DataType.INT64,
        }
    return _NP_TO_MSLITE_DTYPE.get(np.dtype(dt), mslite.DataType.FLOAT32)


def _ensure_chat_template(tokenizer, model_path: str) -> None:
    if getattr(tokenizer, "chat_template", None):
        return
    p = os.path.join(model_path, "chat_template.json")
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        tpl = data.get("chat_template")
        if isinstance(tpl, str) and tpl.strip() != "":
            tokenizer.chat_template = tpl


def _load_audio_token_id(model_path: str) -> int:
    """Read audio_token_id from thinker_config.text_config in config.json."""
    p = os.path.join(model_path, "config.json")
    with open(p, "r", encoding="utf-8") as f:
        c = json.load(f)
    tc = c.get("thinker_config", {}).get("text_config", {})
    return int(tc.get("audio_token_id", 151676))


def _build_prompt(tokenizer, context: str, audio_token_len: int,
                  force_language: Optional[str]) -> str:
    """Render the ASR chat template prompt with the audio pad token expanded."""
    msgs = [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    base = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    audio_token = getattr(tokenizer, "audio_token", "<|audio_pad|>")
    prompt = base.replace(audio_token, audio_token * int(audio_token_len), 1)
    if force_language:
        prompt = prompt + f"language {force_language}{'<asr_text>'}"
    return prompt


def _get_eos_token_ids(tokenizer) -> List[int]:
    """Collect EOS token ids (im_end, endoftext, eot_id, and tokenizer.eos_token_id)."""
    eos_ids = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    for tok in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            continue
        if tid is None:
            continue
        tid = int(tid)
        if tid >= 0:
            eos_ids.add(tid)
    return sorted(eos_ids)


def _compute_position_ids(attention_mask: np.ndarray) -> np.ndarray:
    """mrope position_ids (3, b, s): cumsum-1, masked to 0 where mask==0, expand to 3."""
    pos = np.cumsum(attention_mask.astype(np.int32), axis=-1) - 1
    pos = np.where(attention_mask > 0, pos, 0).astype(np.int32)
    return np.repeat(pos[None, :, :], 3, axis=0)


def _select_prefill_gear(seq_len: int) -> int:
    for g in PREFILL_GEARS:
        if int(seq_len) <= g:
            return g
    return PREFILL_GEARS[-1]


class _Perf:
    """Per-module latency for the ASR pipeline.

    Modules:
      FeatureExt  (CPU)  — WhisperFeatureExtractor (FFT + mel + log-norm)
      AudioEncoder(Asc)  — audio encoder MindIR predict
      Prefill     (Asc)  — prompt assembly (numpy concat of pre-tokenized pieces,
                           <1ms) + prefill MindIR predict (embed + scatter +
                           transformer + last-pos lm_head)
      DecodeStep  (Asc)  — per-step decode MindIR predict (embed + transformer + lm_head)
      E2E         (wall) — whole infer_chunk wall time (FE → final decode)
    """

    def __init__(self) -> None:
        self.feature_ext_ms: List[float] = []
        self.audio_encoder_ms: List[float] = []
        self.prefill_ms: List[float] = []
        self.decode_step_ms: List[float] = []
        self.chunk_e2e_ms: List[float] = []
        self.decode_tokens: int = 0
        self.audio_token_lens: List[int] = []

    def add_feature_ext_ms(self, ms: float) -> None:
        self.feature_ext_ms.append(float(ms))

    def add_audio_encoder_ms(self, ms: float) -> None:
        self.audio_encoder_ms.append(float(ms))

    def add_prefill_ms(self, ms: float) -> None:
        self.prefill_ms.append(float(ms))

    def add_decode_step_ms(self, ms: float) -> None:
        self.decode_step_ms.append(float(ms))

    def add_chunk_e2e_ms(self, ms: float) -> None:
        self.chunk_e2e_ms.append(float(ms))

    def add_decode_tokens(self, n: int) -> None:
        self.decode_tokens += int(n)

    def add_audio_token_len(self, audio_token_len: int) -> None:
        self.audio_token_lens.append(int(audio_token_len))

    def summary(self) -> Dict[str, float]:
        """Aggregate per-stage latencies and throughput into a flat metrics dict."""
        def _agg(xs):
            if not xs:
                return 0.0, 0.0, 0.0
            return float(sum(xs) / len(xs)), float(min(xs)), float(max(xs))

        def _mean_int(xs):
            return int(round(float(sum(xs)) / len(xs))) if xs else 0

        fe_m, fe_i, fe_a = _agg(self.feature_ext_ms)
        ae_m, ae_i, ae_a = _agg(self.audio_encoder_ms)
        pf_m, pf_i, pf_a = _agg(self.prefill_ms)
        ds_m, ds_i, ds_a = _agg(self.decode_step_ms)
        e2e_m, e2e_i, e2e_a = _agg(self.chunk_e2e_ms)
        # Throughput = decode_tokens / E2E wall time (actual user-perceived).
        e2e_total_ms = float(sum(self.chunk_e2e_ms))
        e2e_total_s = e2e_total_ms / 1000.0 if e2e_total_ms > 0 else 0.0
        tok_s = float(self.decode_tokens / e2e_total_s) if e2e_total_s > 0 else 0.0
        return {
            "feature_ext_ms_mean": fe_m, "feature_ext_ms_min": fe_i, "feature_ext_ms_max": fe_a,
            "audio_encoder_ms_mean": ae_m, "audio_encoder_ms_min": ae_i, "audio_encoder_ms_max": ae_a,
            "prefill_ms_mean": pf_m, "prefill_ms_min": pf_i, "prefill_ms_max": pf_a,
            "decode_step_ms_mean": ds_m, "decode_step_ms_min": ds_i, "decode_step_ms_max": ds_a,
            "chunk_e2e_ms_mean": e2e_m, "chunk_e2e_ms_min": e2e_i, "chunk_e2e_ms_max": e2e_a,
            "decode_tokens": float(self.decode_tokens),
            "throughput_tok_s": tok_s,
            "audio_token_len": float(_mean_int(self.audio_token_lens)),
        }


def _find_mindir(base_dir: str, stem: str) -> str:
    """Find a single mindir under base_dir matching stem. Tries common suffixes."""
    suffixes = (".onnx.mindir", ".onnx_graph.mindir", "_graph.mindir", ".mindir")
    for suf in suffixes:
        for fname in sorted(os.listdir(base_dir)):
            if fname.startswith(stem) and fname.endswith(suf):
                return os.path.join(base_dir, fname)
    raise FileNotFoundError(f"No mindir found under {base_dir} for stem '{stem}'")


class _Qwen3AsrMslite:
    """Loads audio/prefill/decode MindIR and runs zero-copy autoregressive transcription."""

    def __init__(
        self,
        mindir_dir: str,
        tokenizer,
        feature_extractor,
        audio_token_id: int,
        device_id: int = 0,
        config_path: str = "",
        precision_mode: Optional[str] = None,
        kv_cache_len: int = KV_CACHE_LEN,
    ):
        mslite = _require_mslite()
        self.mslite = mslite
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.device_id = int(device_id)
        self.kv_cache_len = int(kv_cache_len)
        self._dev = f"ascend:{int(device_id)}"

        try:
            audio_path = _find_mindir(
                os.path.join(mindir_dir, "audio_encoder"), "qwen3_asr_audio_encoder",
            )
        except FileNotFoundError:
            audio_path = _find_mindir(mindir_dir, "qwen3_asr_audio_encoder")

        prefill_path = _find_mindir(
            os.path.join(mindir_dir, "prefill"), "qwen3_asr_text_prefill_fp32",
        )
        decode_path = _find_mindir(
            os.path.join(mindir_dir, "decode"), "qwen3_asr_text_decode_fp32",
        )

        self.audio_token_id = audio_token_id

        ctx = mslite.Context()
        if hasattr(ctx, "target"):
            ctx.target = ["ascend"]
        if hasattr(ctx, "ascend") and hasattr(ctx.ascend, "device_id"):
            ctx.ascend.device_id = int(device_id)

        cfg = self._build_config_dict(precision_mode)

        print(f"Loading audio encoder: {audio_path}")
        self.audio_model = self._build_model(audio_path, ctx, config_path, cfg)
        print(f"Loading prefill:       {prefill_path}")
        self.prefill_model = self._build_model(prefill_path, ctx, config_path, cfg)
        print(f"Loading decode:        {decode_path}")
        self.decode_model = self._build_model(decode_path, ctx, config_path, cfg)

        self._zc_inputs = None
        self._zc_outputs = None
        self._kv_np_dtype = np.float32
        self._mask_np_dtype = np.int32

        self._precompute_prompt_parts()

    def _precompute_prompt_parts(self):
        """Tokenize prompt template pieces once at init.

        Pays the Jinja2 template-compile + BPE regex-compile cost upfront
        (otherwise they show up as ~20ms on the first chunk) and lets
        runtime prompt assembly be a numpy concat instead of a full BPE pass.
        Handles context='' (the common case); non-empty context falls back
        to the slow template-render path.
        """
        tok = self.tokenizer
        audio_token = getattr(tok, "audio_token", "<|audio_pad|>")
        msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        base = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if audio_token not in base:
            raise RuntimeError(
                f"chat template does not contain {audio_token!r}; cannot pre-tokenize prompt"
            )
        pre_str, post_str = base.split(audio_token, 1)
        self._prompt_pre_audio = tok(
            pre_str, return_tensors="np", padding=False
        )["input_ids"].astype(np.int64)
        self._prompt_post_audio = tok(
            post_str, return_tensors="np", padding=False
        )["input_ids"].astype(np.int64)
        pad_ids = tok(audio_token, return_tensors="np", padding=False)["input_ids"].astype(np.int64)
        if pad_ids.shape[1] != 1:
            raise RuntimeError(
                f"{audio_token!r} tokenized to shape {pad_ids.shape}, expected (1, 1)"
            )
        self._audio_pad_id_arr = pad_ids.reshape(1, 1)
        self._lang_hint_ids = {}
        for lang in ("Chinese", "English"):
            hint = f"language {lang}<asr_text>"
            self._lang_hint_ids[lang] = tok(
                hint, return_tensors="np", padding=False
            )["input_ids"].astype(np.int64)

    def _assemble_prompt_ids(self, audio_token_len, force_language, context):
        """Build prompt input_ids via numpy concat of pre-tokenized pieces.

        Falls back to apply_chat_template + BPE when context is non-empty
        (rare; preserves correctness for arbitrary system prompts).
        """
        if context:
            prompt = _build_prompt(
                self.tokenizer, context, audio_token_len, force_language
            )
            return self.tokenizer(
                prompt, return_tensors="np", padding=False
            )["input_ids"].astype(np.int64)
        pieces = [self._prompt_pre_audio]
        if audio_token_len > 0:
            pieces.append(np.broadcast_to(self._audio_pad_id_arr, (1, audio_token_len)))
        pieces.append(self._prompt_post_audio)
        if force_language:
            cached = self._lang_hint_ids.get(force_language)
            if cached is not None:
                pieces.append(cached)
            else:
                hint = f"language {force_language}<asr_text>"
                pieces.append(
                    self.tokenizer(
                        hint, return_tensors="np", padding=False
                    )["input_ids"].astype(np.int64)
                )
        return np.concatenate(pieces, axis=1)

    @staticmethod
    def _build_config_dict(precision_mode):
        if precision_mode is None or str(precision_mode).strip() == "":
            return None
        return {"acl_init_options": {"ge.exec.precision_mode": str(precision_mode).strip()}}

    def _build_model(self, path, ctx, config_path, cfg_dict):
        """Build a mindspore_lite Model, preferring config_dict when supported."""
        model = self.mslite.Model()
        try:
            if cfg_dict is not None:
                try:
                    model.build_from_file(path, self.mslite.ModelType.MINDIR, ctx,
                                          config_path=str(config_path or ""),
                                          config_dict=cfg_dict)
                except TypeError:
                    model.build_from_file(path, self.mslite.ModelType.MINDIR, ctx,
                                          config_path=str(config_path or ""))
            else:
                model.build_from_file(path, self.mslite.ModelType.MINDIR, ctx,
                                      config_path=str(config_path or ""))
        except Exception as e:
            raise RuntimeError(f"build_from_file failed for {path}:\n{e}") from e
        return model

    # ------------------------------------------------------------------
    # Audio encoder (simple numpy in/out)
    # ------------------------------------------------------------------
    def run_audio_encoder(self, input_features: np.ndarray) -> np.ndarray:
        t_in = self.audio_model.get_inputs()[0]
        need_resize = any(int(d) == -1 for d in list(t_in.shape))
        if need_resize:
            self.audio_model.resize([t_in], [list(input_features.shape)])
            t_in = self.audio_model.get_inputs()[0]
        t_in.set_data_from_numpy(input_features.astype(np.float32, copy=False))
        out = self.audio_model.predict([t_in])
        return out[0].get_data_to_numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Prefill (zero-copy: KV cache stays on device)
    # ------------------------------------------------------------------
    def _make_prefill_inputs(self, input_ids, audio_features, attention_mask, position_ids):
        # Converter coerces ONNX int64 input_ids → int32 mindir tensor.
        t_ids = self.mslite.Tensor(input_ids.astype(np.int32, copy=False))
        t_aud = self.mslite.Tensor(audio_features.astype(np.float32, copy=False))
        t_am = self.mslite.Tensor(attention_mask.astype(self._mask_np_dtype, copy=False))
        t_pos = self.mslite.Tensor(position_ids.astype(self._mask_np_dtype, copy=False))
        return [t_ids, t_aud, t_am, t_pos]

    def run_prefill(self, input_ids, audio_features, attention_mask, position_ids):
        """Run prefill with device-resident output tensors (zero-copy).

        Returns (logits_np [1,1,vocab], (kv_k_dev, kv_v_dev), elapsed_ms).
        """
        dev = self._dev
        mslite = self.mslite
        t_in = self._make_prefill_inputs(input_ids, audio_features, attention_mask, position_ids)

        warmup_out = self.prefill_model.predict(t_in)
        out0_dev = mslite.Tensor(shape=list(warmup_out[0].shape),
                                 dtype=warmup_out[0].dtype, device=dev)
        kv_k_dev = mslite.Tensor(shape=list(warmup_out[1].shape),
                                 dtype=warmup_out[1].dtype, device=dev)
        kv_v_dev = mslite.Tensor(shape=list(warmup_out[2].shape),
                                 dtype=warmup_out[2].dtype, device=dev)
        out_devs = [out0_dev, kv_k_dev, kv_v_dev]

        t0 = time.perf_counter()
        self.prefill_model.predict(t_in, outputs=out_devs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        out_np = out0_dev.get_data_to_numpy()
        return out_np, (kv_k_dev, kv_v_dev), elapsed_ms

    # ------------------------------------------------------------------
    # Decode (zero-copy + ping-pong KV)
    # ------------------------------------------------------------------
    def _prime_decode(self, token_id, cur_attention_mask, valid_len, kv_dev):
        """One decode step to learn output shapes/dtype."""
        mslite = self.mslite
        decode_inputs_spec = self.decode_model.get_inputs()
        kv_np_dtype = np.float32
        for t in decode_inputs_spec:
            if getattr(t, "name", "") == "past_key_cache":
                kv_np_dtype = np.float16 if t.dtype == mslite.DataType.FLOAT16 else np.float32
                break

        position_ids_np = np.array([[[int(valid_len)]]], dtype=self._mask_np_dtype).repeat(3, axis=0)
        input_ids_np = np.array([[int(token_id)]], dtype=np.int32)

        prime_in = []
        for t in decode_inputs_spec:
            name = getattr(t, "name", "")
            if name == "input_ids":
                prime_in.append(mslite.Tensor(input_ids_np))
            elif name == "attention_mask":
                prime_in.append(mslite.Tensor(cur_attention_mask.astype(self._mask_np_dtype, copy=False)))
            elif name == "position_ids":
                prime_in.append(mslite.Tensor(position_ids_np))
            elif name == "past_key_cache":
                prime_in.append(kv_dev[0])
            elif name == "past_value_cache":
                prime_in.append(kv_dev[1])
        prime_out = self.decode_model.predict(prime_in)

        kv_shape = list(prime_out[1].shape)
        out0_shape = prime_out[0].shape
        out0_dtype = prime_out[0].dtype
        return kv_shape, out0_shape, out0_dtype, kv_np_dtype

    def _zc_setup(self, kv_shape, out0_shape, out0_dtype, kv_np_dtype, kv_dev):
        """Allocate device-resident input/output tensors for the zero-copy decode loop."""
        dev = self._dev
        mslite = self.mslite
        self._kv_np_dtype = np.dtype(kv_np_dtype)
        kv_mslite_dtype = _np_dtype_to_mslite(mslite, self._kv_np_dtype)
        out0_mslite_dtype = out0_dtype

        t_in0 = mslite.Tensor(shape=[1, 1], dtype=mslite.DataType.INT32, device=dev)
        t_attention_mask = mslite.Tensor(
            shape=[1, self.kv_cache_len], dtype=mslite.DataType.INT32, device=dev)
        t_position_ids = mslite.Tensor(
            shape=[3, 1, 1], dtype=mslite.DataType.INT32, device=dev)
        # Decode output KV buffers. After step 1, the prefill KV tensors
        # (kv_dev[0]/kv_dev[1]) are recycled as the second ping-pong buffer.
        t_out_k = mslite.Tensor(shape=kv_shape, dtype=kv_mslite_dtype, device=dev)
        t_out_v = mslite.Tensor(shape=kv_shape, dtype=kv_mslite_dtype, device=dev)
        out0_out = mslite.Tensor(shape=list(out0_shape), dtype=out0_mslite_dtype, device=dev)

        self._zc_inputs = [t_in0, t_attention_mask, t_position_ids, kv_dev[0], kv_dev[1]]
        self._zc_outputs = [out0_out, t_out_k, t_out_v]

    def decode_loop(self, first_token, kv_dev, valid_len,
                    eos_token_ids, max_new_tokens, perf: Optional[_Perf] = None):
        """Ping-pong decode with device-resident KV cache. Graph outputs logits directly."""
        cur_attention_mask = np.zeros((1, self.kv_cache_len), dtype=self._mask_np_dtype)
        if valid_len > 0:
            cur_attention_mask[0, :valid_len] = 1

        kv_shape, out0_shape, out0_dtype, kv_np_dtype = self._prime_decode(
            first_token, cur_attention_mask, valid_len, kv_dev
        )
        self._zc_setup(kv_shape, out0_shape, out0_dtype, kv_np_dtype, kv_dev)

        generated = [int(first_token)]

        for _ in range(int(max_new_tokens) - 1):
            if eos_token_ids and generated[-1] in eos_token_ids:
                break
            if valid_len >= self.kv_cache_len:
                break

            cur_attention_mask[0, valid_len] = 1
            next_in = np.array([[generated[-1]]], dtype=np.int32)
            self._zc_inputs[0].set_data_from_numpy(next_in)
            pos_np = np.array([[[int(valid_len)]]], dtype=self._mask_np_dtype).repeat(3, axis=0)
            self._zc_inputs[1].set_data_from_numpy(cur_attention_mask)
            self._zc_inputs[2].set_data_from_numpy(pos_np)

            t0 = time.perf_counter()
            self.decode_model.predict(self._zc_inputs, outputs=self._zc_outputs)
            if perf is not None:
                perf.add_decode_step_ms((time.perf_counter() - t0) * 1000.0)

            self._zc_inputs[3], self._zc_outputs[1] = self._zc_outputs[1], self._zc_inputs[3]
            self._zc_inputs[4], self._zc_outputs[2] = self._zc_outputs[2], self._zc_inputs[4]

            out0_np = self._zc_outputs[0].get_data_to_numpy()
            valid_len += 1
            next_id = int(np.argmax(out0_np[0, 0, :].astype(np.float32)))

            generated.append(next_id)
            if perf is not None:
                perf.add_decode_tokens(1)

        return generated

    # ------------------------------------------------------------------
    # Single chunk inference
    # ------------------------------------------------------------------
    def infer_chunk(self, wav: np.ndarray, context: str, force_language: Optional[str],
                    max_new_tokens: int, perf: Optional[_Perf] = None) -> Tuple[str, str]:
        """Transcribe one audio chunk end-to-end; returns (language, text)."""
        chunk_t0 = time.perf_counter()

        t0 = time.perf_counter()
        fe = self.feature_extractor(wav, sampling_rate=SAMPLE_RATE, return_attention_mask=False)
        input_features = np.asarray(fe["input_features"], dtype=np.float32)
        if perf is not None:
            perf.add_feature_ext_ms((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        audio_features = self.run_audio_encoder(input_features)
        if perf is not None:
            perf.add_audio_encoder_ms((time.perf_counter() - t0) * 1000.0)

        audio_token_len = int(audio_features.shape[1])
        if perf is not None:
            perf.add_audio_token_len(audio_token_len)

        t_prep = time.perf_counter()
        prompt_ids = self._assemble_prompt_ids(audio_token_len, force_language, context)
        prompt_len = int(prompt_ids.shape[1])

        # Pad prompt to nearest prefill gear
        gear = _select_prefill_gear(prompt_len)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        if gear > prompt_len:
            pad_n = gear - prompt_len
            pad_ids = np.full((1, pad_n), int(pad_token_id), dtype=prompt_ids.dtype)
            prompt_ids = np.concatenate([prompt_ids, pad_ids], axis=1)

        attention_mask = np.zeros((1, gear), dtype=self._mask_np_dtype)
        attention_mask[0, :prompt_len] = 1
        position_ids = _compute_position_ids(attention_mask)
        prep_ms = (time.perf_counter() - t_prep) * 1000.0

        # embed_tokens gather + audio scatter happen inside the prefill graph.
        # run_prefill returns Ascend-only predict ms (steady-state; per-gear
        # warmup predict stays hidden, matching the convention already used
        # by decode's _prime_decode). We fold prompt-assembly time into Prefill.
        logits_np, kv_dev, ascend_ms = self.run_prefill(
            prompt_ids, audio_features, attention_mask, position_ids
        )
        if perf is not None:
            perf.add_prefill_ms(prep_ms + ascend_ms)

        # logits_np shape: [1, 1, vocab]; graph already gathered at last valid pos.
        first_token = int(np.argmax(logits_np[0, 0, :].astype(np.float32)))

        eos_ids = _get_eos_token_ids(self.tokenizer)
        generated = self.decode_loop(
            first_token=first_token, kv_dev=kv_dev,
            valid_len=prompt_len, eos_token_ids=eos_ids, max_new_tokens=max_new_tokens, perf=perf,
        )

        decoded = self.tokenizer.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        lang, txt = parse_asr_output(decoded, user_language=force_language)

        if perf is not None:
            perf.add_chunk_e2e_ms((time.perf_counter() - chunk_t0) * 1000.0)
        return lang, txt


def transcribe_mindir(
    model_path: str,
    mindir_dir: str,
    audio: str,
    context: str = "",
    language: Optional[str] = None,
    max_chunk_sec: float = 30.0,
    max_new_tokens: int = 256,
    device_id: int = 0,
    config_path: str = "",
    precision_mode: Optional[str] = None,
    kv_cache_len: int = KV_CACHE_LEN,
    perf: Optional[_Perf] = None,
) -> Tuple[str, str]:
    """Top-level entry: load MindIR + tokenizer, split audio, transcribe each chunk."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    _ensure_chat_template(tokenizer, model_path)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    audio_token_id = _load_audio_token_id(model_path)

    inferencer = _Qwen3AsrMslite(
        mindir_dir=mindir_dir, tokenizer=tokenizer, feature_extractor=feature_extractor,
        audio_token_id=audio_token_id,
        device_id=device_id, config_path=config_path, precision_mode=precision_mode,
        kv_cache_len=kv_cache_len,
    )

    force_language = None
    if language is not None and str(language).strip() != "":
        ln = normalize_language_name(str(language))
        validate_language(ln)
        force_language = ln

    wav = normalize_audio_input(audio)
    chunks = split_audio_into_chunks(wav=wav, sr=SAMPLE_RATE, max_chunk_sec=float(max_chunk_sec))
    out_langs: List[str] = []
    out_texts: List[str] = []
    for cwav, _ in chunks:
        lang, txt = inferencer.infer_chunk(
            cwav, context=context, force_language=force_language,
            max_new_tokens=max_new_tokens, perf=perf,
        )
        out_langs.append(lang)
        out_texts.append(txt)
    return merge_languages(out_langs), "".join(out_texts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="./Qwen3-ASR-1.7B")
    ap.add_argument("--mindir-dir", type=str, default="./onnx")
    ap.add_argument("--audio", type=str, required=True)
    ap.add_argument("--context", type=str, default="")
    ap.add_argument("--language", type=str, default=None)
    ap.add_argument("--max-chunk-sec", type=float, default=30.0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--kv-cache-len", type=int, default=KV_CACHE_LEN)
    ap.add_argument("--config-path", type=str, default="")
    ap.add_argument("--precision-mode", type=str, default=None)
    args = ap.parse_args()

    perf = _Perf()
    lang, txt = transcribe_mindir(
        model_path=args.model_path, mindir_dir=args.mindir_dir, audio=args.audio,
        context=args.context, language=args.language, max_chunk_sec=args.max_chunk_sec,
        max_new_tokens=args.max_new_tokens, device_id=args.device_id,
        config_path=args.config_path, precision_mode=args.precision_mode,
        kv_cache_len=args.kv_cache_len, perf=perf,
    )
    print(lang)
    print(txt)
    s = perf.summary()
    dn = int(s["decode_tokens"])
    parts = [
        f"FeatureExt(ms) mean={s['feature_ext_ms_mean']:.2f}",
        f"AudioEncoder(ms) mean={s['audio_encoder_ms_mean']:.2f}",
        f"Prefill(ms) mean={s['prefill_ms_mean']:.2f}",
        f"DecodeStep(ms) mean={s['decode_step_ms_mean']:.2f} x{dn}",
        f"E2E(ms) mean={s['chunk_e2e_ms_mean']:.2f}",
        f"Throughput(tok/s)={s['throughput_tok_s']:.2f} [{dn}tok / E2E {s['chunk_e2e_ms_mean']:.2f}ms]",
        f"Tokens={dn}",
        f"AudioTokens={s['audio_token_len']:.0f}",
    ]
    print("Perf: " + "; ".join(parts))


if __name__ == "__main__":
    main()
