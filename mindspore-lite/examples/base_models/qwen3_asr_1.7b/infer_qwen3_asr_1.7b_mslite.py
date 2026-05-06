"""End-to-end Qwen3-ASR 1.7B transcription using MindSpore Lite MindIR models on Ascend."""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
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


def _require_mslite():
    try:
        import mindspore_lite as mslite
    except Exception as e:
        raise RuntimeError("mindspore_lite is required for MindIR inference") from e
    return mslite


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


def _build_prompt(
    tokenizer,
    context: str,
    audio_token_len: int,
    force_language: Optional[str],
) -> str:
    """Build chat-template prompt with repeated audio placeholder tokens."""
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
    """Return tokenizer EOS and common stop token ids for greedy decoding."""
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


class _Perf:
    """Collects simple latency and token counts for encoder and decode steps."""

    def __init__(self) -> None:
        self.audio_encoder_ms: List[float] = []
        self.decode_step_ms: List[float] = []
        self.decode_tokens: int = 0
        self.audio_token_lens: List[int] = []

    def add_audio_encoder_ms(self, ms: float) -> None:
        """Record one audio encoder forward latency in milliseconds."""
        self.audio_encoder_ms.append(float(ms))

    def add_decode_step_ms(self, ms: float) -> None:
        """Record one text decoder forward latency in milliseconds."""
        self.decode_step_ms.append(float(ms))

    def add_decode_tokens(self, n: int) -> None:
        """Add generated token count (typically 1 per decode step)."""
        self.decode_tokens += int(n)

    def add_audio_token_len(self, audio_token_len: int) -> None:
        """Record token lengths used by one audio chunk."""
        self.audio_token_lens.append(int(audio_token_len))

    def summary(self) -> Dict[str, float]:
        """Aggregate encoder/decode timings and approximate decode throughput."""
        def _agg(xs: List[float]) -> Tuple[float, float, float]:
            if not xs:
                return 0.0, 0.0, 0.0
            return float(sum(xs) / len(xs)), float(min(xs)), float(max(xs))

        def _mean_int(xs: List[int]) -> int:
            if not xs:
                return 0
            return int(round(float(sum(xs)) / len(xs)))

        a_mean, a_min, a_max = _agg(self.audio_encoder_ms)
        d_mean, d_min, d_max = _agg(self.decode_step_ms)
        decode_sec = float(sum(self.decode_step_ms) / 1000.0) if self.decode_step_ms else 0.0
        tok_s = float(self.decode_tokens / decode_sec) if decode_sec > 0 else 0.0
        return {
            "audio_encoder_ms_mean": a_mean,
            "audio_encoder_ms_min": a_min,
            "audio_encoder_ms_max": a_max,
            "decode_step_ms_mean": d_mean,
            "decode_step_ms_min": d_min,
            "decode_step_ms_max": d_max,
            "decode_tokens": float(self.decode_tokens),
            "throughput_tok_s": tok_s,
            "audio_token_len": float(_mean_int(self.audio_token_lens)),
        }


class _MsLiteRunner:
    """Thin wrapper around mindspore_lite.Model for a single MindIR graph."""

    def __init__(
        self,
        model_path: str,
        device_id: int = 1,
        config_path: str = "",
        precision_mode: Optional[str] = None,
    ):
        mslite = _require_mslite()
        self.mslite = mslite
        self.model = mslite.Model()

        ctx = mslite.Context()
        if hasattr(ctx, "target"):
            ctx.target = ["ascend"]
        if hasattr(ctx, "ascend") and hasattr(ctx.ascend, "device_id"):
            ctx.ascend.device_id = int(device_id)
        if not hasattr(self.model, "build_from_file"):
            raise RuntimeError("Unsupported mindspore_lite Model API")

        config_dict = None
        if precision_mode is not None and str(precision_mode).strip() != "":
            config_dict = {
                "acl_init_options": {"ge.exec.precision_mode": str(precision_mode).strip()}
            }

        try:
            if config_dict is not None:
                try:
                    self.model.build_from_file(
                        model_path,
                        mslite.ModelType.MINDIR,
                        ctx,
                        config_path=str(config_path or ""),
                        config_dict=config_dict,
                    )
                except TypeError:
                    self.model.build_from_file(
                        model_path,
                        mslite.ModelType.MINDIR,
                        ctx,
                        config_path=str(config_path or ""),
                    )
            else:
                self.model.build_from_file(
                    model_path,
                    mslite.ModelType.MINDIR,
                    ctx,
                    config_path=str(config_path or ""),
                )
        except Exception as e:
            raise RuntimeError(f"build_from_file failed:\n{e}") from e
        self.inputs = self.model.get_inputs()


    def run(self, feeds: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Resize inputs if needed, copy numpy feeds, run predict, return output arrays."""
        dims = []
        need_resize = False
        for t in self.inputs:
            if t.name not in feeds:
                dims.append(list(t.shape))
                continue
            new_shape = list(feeds[t.name].shape)
            dims.append(new_shape)
            if list(t.shape) != new_shape:
                need_resize = True
            if any(int(d) == -1 for d in list(t.shape)):
                need_resize = True

        if need_resize:
            self.model.resize(self.inputs, dims)
            self.inputs = self.model.get_inputs()

        for t in self.inputs:
            if t.name in feeds:
                arr = feeds[t.name]
                t_shape = list(t.shape)
                a_shape = list(arr.shape)
                if all(int(d) != -1 for d in t_shape) and t_shape != a_shape:
                    raise RuntimeError(
                        f"Input shape mismatch for {t.name}: numpy={a_shape}, tensor={t_shape}"
                    )
                try:
                    t.set_data_from_numpy(arr)
                except Exception as e:
                    t_dtype = getattr(t, "dtype", None)
                    raise RuntimeError(
                        "set_data_from_numpy failed: "
                        f"name={t.name}, "
                        f"numpy_shape={a_shape}, "
                        f"numpy_dtype={arr.dtype}, "
                        f"numpy_nbytes={arr.nbytes}, "
                        f"tensor_shape={t_shape}, tensor_dtype={t_dtype}"
                    ) from e
        outputs = self.model.predict(self.inputs)
        return [o.get_data_to_numpy() for o in outputs]


def _first_existing_path(base_dir: str, names: Sequence[str]) -> str:
    for name in names:
        p = os.path.join(base_dir, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(os.path.join(base_dir, " | ".join(names)))


def _load_mindir_runners(
    mindir_dir: str,
    device_id: int,
    precision_mode: Optional[str],
    config_path: str = "",
):
    """Load audio and text MindIR runners from known filenames under mindir_dir."""
    audio_path = _first_existing_path(
        mindir_dir,
        (
            "qwen3_asr_audio_encoder_fp32.onnx.mindir",
            "qwen3_asr_audio_encoder_fp32.onnx_graph.mindir",
            "qwen3_asr_audio_encoder_fp32.mindir",
        ),
    )
    text_path = _first_existing_path(
        mindir_dir,
        (
            "qwen3_asr_text_decoder_fp32.onnx_graph.mindir",
            "qwen3_asr_text_decoder_fp32.onnx_graph_graph.mindir",
            "qwen3_asr_text_decoder_fp32.onnx_graph_fp32_graph.mindir",
            "qwen3_asr_text_decoder_fp32.onnx.mindir",
            "qwen3_asr_text_decoder_fp32_graph.mindir",
        ),
    )
    if not config_path:
        default_cfg = "./config.ini"
        config_path = default_cfg if os.path.isfile(default_cfg) else ""
    runner_audio = _MsLiteRunner(
        audio_path,
        device_id=device_id,
        config_path=config_path,
        precision_mode=precision_mode,
    )
    runner_text = _MsLiteRunner(
        text_path,
        device_id=device_id,
        config_path=config_path,
        precision_mode=precision_mode,
    )
    return runner_audio, runner_text


def _greedy_decode(
    runner_text: _MsLiteRunner,
    input_ids: np.ndarray,
    audio_features: np.ndarray,
    eos_token_ids: Sequence[int],
    max_new_tokens: int,
    pad_token_id: int,
    perf: Optional[_Perf] = None,
) -> np.ndarray:
    """Greedy autoregressive decode with a fixed-length workspace and causal mask."""
    prompt_len = int(input_ids.shape[1])
    total_len = int(prompt_len + max_new_tokens)
    ids = np.full((1, total_len), int(pad_token_id), dtype=np.int32)
    ids[:, :prompt_len] = input_ids.astype(np.int32, copy=False)

    position_ids = np.arange(total_len, dtype=np.int32)[None, None, :]
    position_ids = np.repeat(position_ids, 3, axis=0)
    causal = np.triu(np.ones((total_len, total_len), dtype=np.float32), k=1)
    attention_mask = (causal * (-1e4)).reshape((1, 1, total_len, total_len))

    cur_len = prompt_len
    for _ in range(int(max_new_tokens)):
        t0 = time.perf_counter()
        logits = runner_text.run(
            {
                "input_ids": ids,
                "audio_features": audio_features.astype(np.float32, copy=False),
                "attention_mask": attention_mask.astype(np.float32, copy=False),
                "position_ids": position_ids,
            }
        )[0]
        if perf is not None:
            perf.add_decode_step_ms((time.perf_counter() - t0) * 1000.0)
        next_id = int(np.argmax(logits[0, cur_len - 1], axis=-1))
        if cur_len >= total_len:
            break
        ids[0, cur_len] = next_id
        cur_len += 1
        if perf is not None:
            perf.add_decode_tokens(1)
        if next_id in eos_token_ids:
            break
    return ids[:, :cur_len]


def _infer_one_chunk(
    feature_extractor,
    tokenizer,
    runner_audio: _MsLiteRunner,
    runner_text: _MsLiteRunner,
    wav: np.ndarray,
    context: str,
    force_language: Optional[str],
    max_new_tokens: int,
    perf: Optional[_Perf] = None,
) -> Tuple[str, str]:
    """Encode one waveform chunk and return (detected_language, transcript) strings."""
    fe = feature_extractor(wav, sampling_rate=SAMPLE_RATE, return_attention_mask=False)
    input_features = np.asarray(fe["input_features"], dtype=np.float32)

    t0 = time.perf_counter()
    audio_features = runner_audio.run({"input_features": input_features})[0].astype(np.float32)
    if perf is not None:
        perf.add_audio_encoder_ms((time.perf_counter() - t0) * 1000.0)
    audio_token_len = int(audio_features.shape[1])
    prompt = _build_prompt(
        tokenizer,
        context=context,
        audio_token_len=audio_token_len,
        force_language=force_language,
    )
    tok = tokenizer(prompt, return_tensors="np", padding=False)
    prompt_ids = tok["input_ids"].astype(np.int32)
    if perf is not None:
        perf.add_audio_token_len(audio_token_len)

    eos_ids = _get_eos_token_ids(tokenizer)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    full_ids = _greedy_decode(
        runner_text=runner_text,
        input_ids=prompt_ids,
        audio_features=audio_features,
        eos_token_ids=eos_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=int(pad_token_id),
        perf=perf,
    )

    gen_ids = full_ids[0, prompt_ids.shape[1] :]
    decoded = tokenizer.decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    lang, txt = parse_asr_output(decoded, user_language=force_language)
    return lang, txt


def transcribe_mindir(
    model_path: str,
    mindir_dir: str,
    audio: str,
    context: str = "",
    language: Optional[str] = None,
    max_chunk_sec: float = 30.0,
    max_new_tokens: int = 64,
    device_id: int = 1,
    config_path: str = "",
    precision_mode: Optional[str] = None,
    perf: Optional[_Perf] = None,
) -> Tuple[str, str]:
    """Transcribe audio from file or path; merge chunk languages and concatenated text."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    _ensure_chat_template(tokenizer, model_path)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    runner_audio, runner_text = _load_mindir_runners(
        mindir_dir,
        device_id=device_id,
        precision_mode=precision_mode,
        config_path=config_path,
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
        lang, txt = _infer_one_chunk(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            runner_audio=runner_audio,
            runner_text=runner_text,
            wav=cwav,
            context=context,
            force_language=force_language,
            max_new_tokens=max_new_tokens,
            perf=perf,
        )
        out_langs.append(lang)
        out_texts.append(txt)

    return merge_languages(out_langs), "".join(out_texts)


def main() -> None:
    """CLI entry: transcribe audio with MindIR models and print text plus perf stats."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="./Qwen3-ASR-1.7B")
    ap.add_argument("--mindir-dir", type=str, default="./mindir")
    ap.add_argument("--audio", type=str, required=True)
    ap.add_argument("--context", type=str, default="")
    ap.add_argument("--language", type=str, default=None)
    ap.add_argument("--max-chunk-sec", type=float, default=30.0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device-id", type=int, default=1)
    ap.add_argument("--config-path", type=str, default="")
    ap.add_argument("--precision-mode", type=str, default=None)
    args = ap.parse_args()

    perf = _Perf()
    lang, txt = transcribe_mindir(
        model_path=args.model_path,
        mindir_dir=args.mindir_dir,
        audio=args.audio,
        context=args.context,
        language=args.language,
        max_chunk_sec=args.max_chunk_sec,
        max_new_tokens=args.max_new_tokens,
        device_id=args.device_id,
        config_path=args.config_path,
        precision_mode=args.precision_mode,
        perf=perf,
    )
    print(lang)
    print(txt)
    s = perf.summary()
    audio_enc = (
        "AudioEncoder(ms) "
        f"mean={s['audio_encoder_ms_mean']:.2f}, "
        f"min={s['audio_encoder_ms_min']:.2f}, "
        f"max={s['audio_encoder_ms_max']:.2f}"
    )
    decode_step = (
        "DecodeStep(ms) "
        f"mean={s['decode_step_ms_mean']:.2f}, "
        f"min={s['decode_step_ms_min']:.2f}, "
        f"max={s['decode_step_ms_max']:.2f}"
    )
    throughput = f"Throughput(tok/s)={s['throughput_tok_s']:.2f}"
    token_len = f"TokenLength={s['audio_token_len']:.0f}"
    print(f"Perf: {audio_enc}; {decode_step}; {throughput}; {token_len}")


if __name__ == "__main__":
    main()
