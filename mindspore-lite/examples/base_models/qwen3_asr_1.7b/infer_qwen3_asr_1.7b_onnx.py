"""End-to-end Qwen3-ASR 1.7B transcription using exported ONNX models via ONNX Runtime."""

import argparse
import json
import os
from typing import List, Optional, Sequence, Tuple

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


def _require_onnxruntime():
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError("onnxruntime is required: pip install onnxruntime") from e
    return ort


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

def _load_ort_sessions(onnx_dir: str):
    """Create CPU InferenceSession objects for audio encoder and text decoder ONNX."""
    ort = _require_onnxruntime()
    audio_path = os.path.join(onnx_dir, "qwen3_asr_audio_encoder_fp32.onnx")
    text_path = os.path.join(onnx_dir, "qwen3_asr_text_decoder_fp32.onnx")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(audio_path)
    if not os.path.isfile(text_path):
        raise FileNotFoundError(text_path)
    providers = ["CPUExecutionProvider"]
    sess_audio = ort.InferenceSession(audio_path, providers=providers)
    sess_text = ort.InferenceSession(text_path, providers=providers)
    return sess_audio, sess_text


def _greedy_decode(
    sess_text,
    input_ids: np.ndarray,
    audio_features: np.ndarray,
    eos_token_ids: Sequence[int],
    max_new_tokens: int,
) -> np.ndarray:
    """Greedy autoregressive decode by extending input_ids each ONNX step."""
    ids = input_ids.astype(np.int64, copy=False)
    for _ in range(int(max_new_tokens)):
        seq = int(ids.shape[1])
        position_ids = np.arange(seq, dtype=np.int64)[None, None, :]
        position_ids = np.repeat(position_ids, 3, axis=0)
        causal = np.triu(np.ones((seq, seq), dtype=np.float32), k=1)
        attention_mask = (causal * (-1e4)).reshape((1, 1, seq, seq))

        logits = sess_text.run(
            None,
            {
                "input_ids": ids,
                "audio_features": audio_features,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
        )[0]
        next_id = int(np.argmax(logits[0, -1], axis=-1))
        ids = np.concatenate([ids, np.array([[next_id]], dtype=np.int64)], axis=1)
        if next_id in eos_token_ids:
            break
    return ids


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


def _infer_one_chunk(
    feature_extractor,
    tokenizer,
    sess_audio,
    sess_text,
    wav: np.ndarray,
    context: str,
    force_language: Optional[str],
    max_new_tokens: int,
) -> Tuple[str, str]:
    """Encode one waveform chunk and return (detected_language, transcript) strings."""
    fe = feature_extractor(wav, sampling_rate=SAMPLE_RATE, return_attention_mask=False)
    input_features = np.asarray(fe["input_features"], dtype=np.float32)
    audio_features = sess_audio.run(None, {"input_features": input_features})[0].astype(np.float32)
    audio_token_len = int(audio_features.shape[1])

    prompt = _build_prompt(
        tokenizer,
        context=context,
        audio_token_len=audio_token_len,
        force_language=force_language,
    )
    tok = tokenizer(prompt, return_tensors="np", padding=False)
    prompt_ids = tok["input_ids"].astype(np.int64)

    eos_ids = _get_eos_token_ids(tokenizer)
    full_ids = _greedy_decode(
        sess_text=sess_text,
        input_ids=prompt_ids,
        audio_features=audio_features,
        eos_token_ids=eos_ids,
        max_new_tokens=max_new_tokens,
    )

    gen_ids = full_ids[0, prompt_ids.shape[1] :]
    decoded = tokenizer.decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    lang, txt = parse_asr_output(decoded, user_language=force_language)
    return lang, txt


def transcribe_onnx(
    model_path: str,
    onnx_dir: str,
    audio: str,
    context: str = "",
    language: Optional[str] = None,
    max_chunk_sec: float = 30.0,
    max_new_tokens: int = 256,
) -> Tuple[str, str]:
    """Transcribe audio from file or path; merge chunk languages and concatenated text."""
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    _ensure_chat_template(tokenizer, model_path=model_path)
    sess_audio, sess_text = _load_ort_sessions(onnx_dir)

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
            sess_audio=sess_audio,
            sess_text=sess_text,
            wav=cwav,
            context=context,
            force_language=force_language,
            max_new_tokens=max_new_tokens,
        )
        out_langs.append(lang)
        out_texts.append(txt)

    return merge_languages(out_langs), "".join(out_texts)


def main():
    """CLI entry: transcribe audio with ONNX Runtime and print merged language and text."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="./Qwen3-ASR-1.7B")
    ap.add_argument("--onnx-dir", type=str, default="./onnx")
    ap.add_argument("--audio", type=str, required=True)
    ap.add_argument("--context", type=str, default="")
    ap.add_argument("--language", type=str, default=None)
    ap.add_argument("--max-chunk-sec", type=float, default=30.0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    lang, txt = transcribe_onnx(
        model_path=args.model_path,
        onnx_dir=args.onnx_dir,
        audio=args.audio,
        context=args.context,
        language=args.language,
        max_chunk_sec=args.max_chunk_sec,
        max_new_tokens=args.max_new_tokens,
    )
    print(lang)
    print(txt)


if __name__ == "__main__":
    main()
