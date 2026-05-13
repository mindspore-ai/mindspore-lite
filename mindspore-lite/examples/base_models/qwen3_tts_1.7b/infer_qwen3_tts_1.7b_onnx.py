"""ONNX Runtime demo for Qwen3-TTS (talker KV + code predictor + speech decoder)."""

from __future__ import annotations

import argparse
import importlib
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=None)
def _import_module(name: str) -> Any:
    """Import a module by name with caching."""
    return importlib.import_module(name)


def _torch_dtype(name: str):
    """Map a string dtype to torch dtype."""
    torch = _import_module("torch")
    val = (name or "float32").strip().lower()
    if val in ("float16", "fp16"):
        return torch.float16
    return torch.float32


def _apply_repetition_penalty(logits: Any, prev_tokens: Any, penalty: float):
    """Apply repetition penalty to logits for previously generated tokens."""
    torch = _import_module("torch")
    if penalty is None or float(penalty) == 1.0 or prev_tokens.numel() == 0:
        return logits
    penalty = float(penalty)
    unique = torch.unique(prev_tokens)
    selected = logits.index_select(dim=-1, index=unique)
    updated = torch.where(selected < 0, selected * penalty, selected / penalty)
    out = logits.clone()
    out.index_copy_(dim=-1, index=unique, source=updated)
    return out


def _sample_next_id(
    logits: Any,
    suppress_mask: Any,
    prev_tokens: Any,
    repetition_penalty: float = 1.05,
) -> int:
    """Select the next token id with suppression and repetition penalty."""
    torch = _import_module("torch")
    logits = logits.to(torch.float32)
    logits = logits.masked_fill(suppress_mask, float("-inf"))
    logits = _apply_repetition_penalty(logits, prev_tokens=prev_tokens, penalty=repetition_penalty)
    return int(torch.argmax(logits, dim=-1).item())


def _get_speaker_embed(model: Any, input_id: Any, speaker: str | None):
    """Get speaker embedding for the given speaker name."""
    torch = _import_module("torch")
    qwen = model.model
    talker = qwen.talker
    cfg = qwen.config

    if not speaker:
        return None
    spk_id = cfg.talker_config.spk_id[speaker.lower()]
    return talker.get_input_embeddings()(
        torch.tensor(spk_id, device=talker.device, dtype=input_id.dtype)
    )


def _resolve_language_id(model: Any, language: str | None, speaker: str | None) -> int | None:
    """Resolve language id from config with optional dialect override."""
    cfg = model.model.config
    if language is None or language.lower() == "auto":
        language_id = None
    else:
        language_id = cfg.talker_config.codec_language_id[language.lower()]

    if not speaker:
        return language_id
    if language is None or language.lower() not in ("chinese", "auto"):
        return language_id
    if cfg.talker_config.spk_is_dialect[speaker.lower()] is False:
        return language_id
    dialect = cfg.talker_config.spk_is_dialect[speaker.lower()]
    return cfg.talker_config.codec_language_id[dialect]


def _get_tts_special_embeds(model: Any, dtype: Any):
    """Get BOS/EOS/PAD embeddings used by talker prompt construction."""
    torch = _import_module("torch")
    qwen = model.model
    talker = qwen.talker
    cfg = qwen.config
    token_ids = torch.tensor(
        [[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]],
        device=talker.device,
        dtype=dtype,
    )
    return talker.text_projection(talker.get_text_embeddings()(token_ids)).chunk(3, dim=1)


def _get_codec_prefill_list(cfg, language_id: int | None) -> list[list[int]]:
    """Build codec prefill token sequences for the selected language."""
    if language_id is None:
        return [
            [
                cfg.talker_config.codec_nothink_id,
                cfg.talker_config.codec_think_bos_id,
                cfg.talker_config.codec_think_eos_id,
            ]
        ]
    return [
        [
            cfg.talker_config.codec_think_id,
            cfg.talker_config.codec_think_bos_id,
            language_id,
            cfg.talker_config.codec_think_eos_id,
        ]
    ]


def _get_codec_input_embedding(
    model: Any,
    input_dtype: Any,
    codec_prefill_list: list[list[int]],
    speaker_embed: Any,
):
    """Build codec input embedding sequence for the talker prompt."""
    torch = _import_module("torch")
    qwen = model.model
    talker = qwen.talker
    cfg = qwen.config

    embed_0 = talker.get_input_embeddings()(
        torch.tensor(codec_prefill_list, device=talker.device, dtype=input_dtype)
    )
    embed_1 = talker.get_input_embeddings()(
        torch.tensor(
            [[cfg.talker_config.codec_pad_id, cfg.talker_config.codec_bos_id]],
            device=talker.device,
            dtype=input_dtype,
        )
    )
    if speaker_embed is None:
        return torch.cat([embed_0, embed_1], dim=1)
    return torch.cat([embed_0, speaker_embed.view(1, 1, -1), embed_1], dim=1)


def _build_talker_input_embed(
    model: Any,
    input_id: Any,
    codec_input_embedding: Any,
    tts_bos_embed: Any,
    tts_pad_embed: Any,
):
    """Build talker prompt embeddings including role and codec embeddings."""
    torch = _import_module("torch")
    qwen = model.model
    talker = qwen.talker

    role_embed = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
    prompt_embed = torch.cat(
        (
            tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),
            tts_bos_embed,
        ),
        dim=1,
    ) + codec_input_embedding[:, :-1]
    prompt_embed = torch.cat((role_embed, prompt_embed), dim=1)
    last_embed = talker.text_projection(
        talker.get_text_embeddings()(input_id[:, 3:4])
    ) + codec_input_embedding[:, -1:]
    return torch.cat([prompt_embed, last_embed], dim=1)


def _build_trailing_text_hidden(model: Any, input_id: Any, tts_eos_embed: Any):
    """Build trailing text hidden states used to generate trailing steps."""
    torch = _import_module("torch")
    qwen = model.model
    talker = qwen.talker
    trailing = talker.text_projection(talker.get_text_embeddings()(input_id[:, 4:-5]))
    return torch.cat((trailing, tts_eos_embed), dim=1)


def _tokenize_assistant_input_id(model: Any, text: str, device: Any):
    """Tokenize assistant-formatted text and move the ids to the target device."""
    tokenize_texts = getattr(model, "_tokenize_texts")
    build_assistant_text = getattr(model, "_build_assistant_text")
    return tokenize_texts([build_assistant_text(text)])[0].to(device)


def _build_talker_prompt_tensors(model: Any, text: str, language: str, speaker: str):
    """Build prompt embeds, attention mask, and trailing hidden for prefill."""
    torch = _import_module("torch")
    input_id = _tokenize_assistant_input_id(model, text=text, device=model.model.talker.device)

    speaker_embed = _get_speaker_embed(model, input_id=input_id, speaker=speaker)
    language_id = _resolve_language_id(model, language=language, speaker=speaker)
    tts_bos_embed, tts_eos_embed, tts_pad_embed = _get_tts_special_embeds(
        model, dtype=input_id.dtype
    )
    codec_input_embedding = _get_codec_input_embedding(
        model,
        input_dtype=input_id.dtype,
        codec_prefill_list=_get_codec_prefill_list(model.model.config, language_id=language_id),
        speaker_embed=speaker_embed,
    )
    talker_input_embed = _build_talker_input_embed(
        model,
        input_id=input_id,
        codec_input_embedding=codec_input_embedding,
        tts_bos_embed=tts_bos_embed,
        tts_pad_embed=tts_pad_embed,
    )
    trailing_text_hidden = _build_trailing_text_hidden(
        model, input_id=input_id, tts_eos_embed=tts_eos_embed
    )
    attn_mask = torch.ones(
        (1, talker_input_embed.shape[1]),
        device=input_id.device,
        dtype=torch.int64,
    )
    return talker_input_embed, attn_mask, trailing_text_hidden, tts_pad_embed


@dataclass(frozen=True)
class _OnnxPaths:
    talker_prefill: str
    talker_step: str
    code_predictor: str
    speech_decoder: str


@dataclass(frozen=True)
class _GenArgs:
    text: str
    language: str
    speaker: str
    max_new_tokens: int | None
    repetition_penalty: float


@dataclass(frozen=True)
class _OrtSessions:
    prefill: Any
    step: Any
    gen: Any
    speech: Any
    gen_input_names: set[str]


@dataclass(frozen=True)
class _CodePredictorStep:
    hidden_last_t: Any
    last_id_hidden: Any
    trailing_step: Any
    next_id: int


@dataclass(frozen=True)
class _GenerateContext:
    talker: Any
    cfg: Any
    sessions: _OrtSessions
    trailing_text_hidden: Any
    tts_pad_embed: Any
    repetition_penalty: float


@dataclass(frozen=True)
class _GenerateState:
    hidden_last: Any
    logits_last: Any
    past_k: Any
    past_v: Any
    cache_base_pos: int
    max_new_tokens: int


@dataclass(frozen=True)
class _PromptPack:
    prompt_embeds: Any
    attn_mask: Any
    trailing_text_hidden: Any
    tts_pad_embed: Any


@dataclass(frozen=True)
class _PrefillPack:
    logits_last: Any
    hidden_last: Any
    past_k: Any
    past_v: Any
    cache_base_pos: int


@dataclass
class _GenLoopState:
    prev_tokens: Any
    next_id: int
    hidden_last_t: Any
    past_k: Any
    past_v: Any


@dataclass(frozen=True)
class _AdvanceInputs:
    loop: _GenLoopState
    step_embed: Any
    step_idx: int


def _create_suppress_mask(cfg) -> tuple[Any, int]:
    """Create suppression mask for codec vocabulary and return eos_id."""
    torch = _import_module("torch")
    eos_id = int(cfg.talker_config.codec_eos_token_id)
    vocab_size = int(cfg.talker_config.vocab_size)
    suppress_mask = torch.zeros((vocab_size,), dtype=torch.bool)
    suppress_from = max(vocab_size - 1024, 0)
    suppress_mask[suppress_from:vocab_size] = True
    suppress_mask[eos_id] = False
    return suppress_mask, eos_id


def _resolve_max_new_tokens(trailing_text_hidden: Any, max_new_tokens: int | None) -> int:
    """Resolve max_new_tokens from args with a trailing-length heuristic."""
    if max_new_tokens is not None:
        return int(max_new_tokens)
    trailing_len = int(trailing_text_hidden.shape[1])
    return int(max(32, min(256, trailing_len + 32)))


def _run_prefill(
    sess_prefill: Any,
    prompt_embeds: Any,
    attn_mask: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Run talker prefill and return logits/hidden, KV cache, and prompt_len."""
    np = _import_module("numpy")
    torch = _import_module("torch")
    prompt_embeds_np = prompt_embeds.to(torch.float32).detach().cpu().numpy()
    attn_mask_np = attn_mask.cpu().numpy().astype(np.int64)
    logits_last, hidden_last, past_k, past_v, prompt_len = sess_prefill.run(
        None, {"inputs_embeds": prompt_embeds_np, "attention_mask": attn_mask_np}
    )
    return logits_last, hidden_last, past_k, past_v, prompt_len


def _make_trailing_step(
    trailing_text_hidden: Any,
    step_idx: int,
    tts_pad_embed: Any,
):
    """Create the trailing step embedding for the current generation step."""
    torch = _import_module("torch")
    if step_idx < int(trailing_text_hidden.shape[1]):
        return trailing_text_hidden[:, step_idx].unsqueeze(1).to(torch.float32)
    return tts_pad_embed.to(torch.float32)


def _run_code_predictor_onnx(
    sess_gen: Any,
    gen_input_names: set[str],
    step: _CodePredictorStep,
) -> tuple[Any, Any]:
    """Run code predictor ONNX and return (codec_ids, step_embed)."""
    np = _import_module("numpy")
    torch = _import_module("torch")
    gen_feed = {
        "inputs_embeds": torch.cat((step.hidden_last_t, step.last_id_hidden), dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "next_id": np.array([[step.next_id]], dtype=np.int64),
        "last_id_hidden": step.last_id_hidden.detach().cpu().numpy().astype(np.float32),
        "trailing_step": step.trailing_step.detach().cpu().numpy().astype(np.float32),
    }
    gen_feed = {k: v for k, v in gen_feed.items() if k in gen_input_names}
    gen_out = sess_gen.run(None, gen_feed)
    if len(gen_out) != 2:
        raise RuntimeError(
            f"Expected code predictor outputs: (codec_ids, step_embed), got {len(gen_out)}."
        )
    codec_ids, step_embed = gen_out
    codec_ids_t = torch.from_numpy(codec_ids).to(torch.long)
    step_embed_t = torch.from_numpy(step_embed).to(torch.float32)
    return codec_ids_t, step_embed_t


def _run_step_onnx(
    sess_step: Any,
    step_embed: Any,
    past_k: Any,
    past_v: Any,
    cache_pos: int,
) -> tuple[Any, Any, Any, Any]:
    """Run one talker_step inference and return updated KV cache."""
    feed = _build_step_feed(
        step_embed=step_embed,
        past_k=past_k,
        past_v=past_v,
        cache_pos=cache_pos,
    )
    logits_last, hidden_last, past_k, past_v = sess_step.run(
        None,
        feed,
    )
    return logits_last, hidden_last, past_k, past_v


def _build_step_feed(
    step_embed: Any,
    past_k: Any,
    past_v: Any,
    cache_pos: int,
) -> dict[str, Any]:
    """Build ONNX Runtime feed dict for one talker step."""
    np = _import_module("numpy")
    pos = np.array([[[cache_pos]], [[cache_pos]], [[cache_pos]]], dtype=np.int64)
    cache_len = np.array([cache_pos], dtype=np.int64)
    return {
        "step_embed": step_embed.detach().cpu().numpy().astype(np.float32),
        "past_k": past_k,
        "past_v": past_v,
        "position_ids_step": pos,
        "cache_len": cache_len,
    }


def _init_generation_runtime(
    ctx: _GenerateContext,
    state: _GenerateState,
) -> tuple[Any, int, _GenLoopState]:
    """Initialize generation loop state after prefill."""
    torch = _import_module("torch")
    suppress_mask, eos_id = _create_suppress_mask(ctx.cfg)
    past_k = state.past_k
    past_v = state.past_v
    prev_tokens = torch.empty((0,), dtype=torch.long)
    logits_t = torch.from_numpy(state.logits_last[0])
    next_id = _sample_next_id(
        logits=logits_t,
        suppress_mask=suppress_mask,
        prev_tokens=prev_tokens,
        repetition_penalty=ctx.repetition_penalty,
    )
    prev_tokens = torch.tensor([next_id], dtype=torch.long)
    hidden_last_t = torch.from_numpy(state.hidden_last).to(ctx.talker.device).to(torch.float32)
    return suppress_mask, eos_id, _GenLoopState(
        prev_tokens=prev_tokens,
        next_id=next_id,
        hidden_last_t=hidden_last_t,
        past_k=past_k,
        past_v=past_v,
    )


def _predict_codec_ids_and_step_embed(
    ctx: _GenerateContext,
    loop: _GenLoopState,
    step_idx: int,
) -> tuple[Any, Any]:
    """Predict codec ids and step embedding for the current loop iteration."""
    torch = _import_module("torch")
    input_ids = torch.tensor([[loop.next_id]], device=ctx.talker.device, dtype=torch.long)
    last_id_hidden = ctx.talker.get_input_embeddings()(input_ids).to(torch.float32)
    trailing_step = _make_trailing_step(
        ctx.trailing_text_hidden,
        step_idx=step_idx,
        tts_pad_embed=ctx.tts_pad_embed,
    )
    return _run_code_predictor_onnx(
        sess_gen=ctx.sessions.gen,
        gen_input_names=ctx.sessions.gen_input_names,
        step=_CodePredictorStep(
            hidden_last_t=loop.hidden_last_t,
            last_id_hidden=last_id_hidden,
            trailing_step=trailing_step,
            next_id=loop.next_id,
        ),
    )


def _advance_loop_state(
    ctx: _GenerateContext,
    state: _GenerateState,
    suppress_mask: Any,
    adv: _AdvanceInputs,
) -> _GenLoopState:
    """Advance loop state with one talker_step inference."""
    torch = _import_module("torch")
    cache_pos = int(state.cache_base_pos + adv.step_idx)
    logits_last, hidden_last, past_k, past_v = _run_step_onnx(
        sess_step=ctx.sessions.step,
        step_embed=adv.step_embed,
        past_k=adv.loop.past_k,
        past_v=adv.loop.past_v,
        cache_pos=cache_pos,
    )
    hidden_last_t = torch.from_numpy(hidden_last).to(ctx.talker.device).to(torch.float32)
    logits_t = torch.from_numpy(logits_last[0])
    next_id = _sample_next_id(
        logits=logits_t,
        suppress_mask=suppress_mask,
        prev_tokens=adv.loop.prev_tokens,
        repetition_penalty=ctx.repetition_penalty,
    )
    prev_tokens = torch.cat(
        [adv.loop.prev_tokens, torch.tensor([next_id], dtype=torch.long)],
        dim=0,
    )
    return _GenLoopState(
        prev_tokens=prev_tokens,
        next_id=next_id,
        hidden_last_t=hidden_last_t,
        past_k=past_k,
        past_v=past_v,
    )


def _generate_one_code_step(
    ctx: _GenerateContext,
    state: _GenerateState,
    suppress_mask: Any,
    loop: _GenLoopState,
    step_idx: int,
) -> tuple[Any, _GenLoopState]:
    """Generate one codec step and return (codec_ids, next_loop_state)."""
    codec_ids_t, step_embed = _predict_codec_ids_and_step_embed(ctx, loop=loop, step_idx=step_idx)
    next_loop = _advance_loop_state(
        ctx=ctx,
        state=state,
        suppress_mask=suppress_mask,
        adv=_AdvanceInputs(loop=loop, step_embed=step_embed, step_idx=step_idx),
    )
    return codec_ids_t, next_loop


def _generate_codes_onnx(
    ctx: _GenerateContext,
    state: _GenerateState,
) -> list[Any]:
    """Generate all codec steps until EOS or max_new_tokens is reached."""
    suppress_mask, eos_id, loop = _init_generation_runtime(ctx, state)
    all_codes: list[Any] = []
    torch = _import_module("torch")
    with torch.no_grad():
        for step_idx in range(int(state.max_new_tokens)):
            if loop.next_id == eos_id:
                break
            codec_ids_t, loop = _generate_one_code_step(
                ctx=ctx,
                state=state,
                suppress_mask=suppress_mask,
                loop=loop,
                step_idx=step_idx,
            )
            all_codes.append(codec_ids_t.squeeze(0).cpu())
    return all_codes


def _create_ort_sessions(onnx_paths: _OnnxPaths, providers: list[str]) -> _OrtSessions:
    """Create ONNX Runtime sessions for the four sub-models."""
    ort = _import_module("onnxruntime")
    sess_prefill = ort.InferenceSession(onnx_paths.talker_prefill, providers=providers)
    sess_step = ort.InferenceSession(onnx_paths.talker_step, providers=providers)
    sess_gen = ort.InferenceSession(onnx_paths.code_predictor, providers=providers)
    sess_speech = ort.InferenceSession(onnx_paths.speech_decoder, providers=providers)
    gen_input_names = {i.name for i in sess_gen.get_inputs()}
    return _OrtSessions(
        prefill=sess_prefill,
        step=sess_step,
        gen=sess_gen,
        speech=sess_speech,
        gen_input_names=gen_input_names,
    )


def _decode_speech(
    sess_speech: Any,
    codes_bkt: Any,
    upsample_rate: int,
    sample_rate: int,
) -> tuple[Any, int]:
    """Decode bucketed codec ids into waveform."""
    np = _import_module("numpy")
    wav = sess_speech.run(None, {"codes": codes_bkt.detach().cpu().numpy().astype(np.int64)})[0]
    wav = np.asarray(wav)
    if wav.ndim == 3:
        wav = wav[:, 0, :]
    wav = wav[0].astype(np.float32, copy=False)
    t = int(codes_bkt.shape[-1])
    up = int(upsample_rate)
    if t > 0 and up > 0:
        wav = wav[: int(t * up)]
    return wav, int(sample_rate)


def _build_prompt_pack(model: Any, gen_args: _GenArgs) -> _PromptPack:
    prompt_embeds, attn_mask, trailing_text_hidden, tts_pad_embed = _build_talker_prompt_tensors(
        model,
        text=gen_args.text,
        language=gen_args.language,
        speaker=gen_args.speaker,
    )
    return _PromptPack(
        prompt_embeds=prompt_embeds,
        attn_mask=attn_mask,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
    )


def _run_prefill_pack(sessions: _OrtSessions, prompt: _PromptPack) -> _PrefillPack:
    np = _import_module("numpy")
    logits_last, hidden_last, past_k, past_v, prompt_len = _run_prefill(
        sessions.prefill,
        prompt.prompt_embeds,
        prompt.attn_mask,
    )
    prompt_len_i = int(np.maximum(prompt_len.astype(np.int64)[0], 1))
    return _PrefillPack(
        logits_last=logits_last,
        hidden_last=hidden_last,
        past_k=past_k,
        past_v=past_v,
        cache_base_pos=prompt_len_i,
    )


def _infer_codes_bkt_onnx(
    model: Any,
    sessions: _OrtSessions,
    gen_args: _GenArgs,
) -> Any:
    """Infer bucketed codec ids using talker KV and code predictor sessions."""
    torch = _import_module("torch")
    talker = model.model.talker
    cfg = model.model.config
    prompt = _build_prompt_pack(model, gen_args=gen_args)
    max_new_tokens_i = _resolve_max_new_tokens(prompt.trailing_text_hidden, gen_args.max_new_tokens)
    prefill = _run_prefill_pack(sessions, prompt=prompt)
    all_codes = _generate_codes_onnx(
        ctx=_GenerateContext(
            talker=talker,
            cfg=cfg,
            sessions=sessions,
            trailing_text_hidden=prompt.trailing_text_hidden,
            tts_pad_embed=prompt.tts_pad_embed,
            repetition_penalty=gen_args.repetition_penalty,
        ),
        state=_GenerateState(
            hidden_last=prefill.hidden_last,
            logits_last=prefill.logits_last,
            past_k=prefill.past_k,
            past_v=prefill.past_v,
            cache_base_pos=prefill.cache_base_pos,
            max_new_tokens=max_new_tokens_i,
        ),
    )
    if all_codes:
        codes = torch.stack(all_codes, dim=0)
    else:
        codes = torch.empty((0, int(cfg.talker_config.num_code_groups)), dtype=torch.long)
    return codes.transpose(0, 1).unsqueeze(0).contiguous()


def _run_talker_onnx_kv(
    model: Any,
    providers: list[str] | None = None,
    onnx_paths: _OnnxPaths | None = None,
    gen_args: _GenArgs | None = None,
):
    """Run full ONNX pipeline (talker KV + decoder) and return waveform."""
    if onnx_paths is None:
        raise ValueError("onnx_paths is required")
    if gen_args is None:
        raise ValueError("gen_args is required")

    if providers is None:
        providers = ["CPUExecutionProvider"]
    sessions = _create_ort_sessions(onnx_paths=onnx_paths, providers=providers)
    codes_bkt = _infer_codes_bkt_onnx(model, sessions=sessions, gen_args=gen_args)
    upsample_rate = int(model.model.speech_tokenizer.get_decode_upsample_rate())
    sample_rate = int(model.model.speech_tokenizer.get_output_sample_rate())
    return _decode_speech(
        sessions.speech,
        codes_bkt,
        upsample_rate=upsample_rate,
        sample_rate=sample_rate,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Qwen3-TTS ONNX Runtime demo (KV cache).")
    parser.add_argument("--model_path", type=str, default="../Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--device_map", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])

    parser.add_argument("--onnx_dir", type=str, default="onnx_models_talker_core_fp32_no_custom")
    parser.add_argument("--talker_prefill_onnx", type=str, default="")
    parser.add_argument("--talker_step_onnx", type=str, default="")
    parser.add_argument("--code_predictor_onnx", type=str, default="")
    parser.add_argument("--speech_decoder_onnx", type=str, default="")

    parser.add_argument("--text", type=str, default="其实我真的有发现，我是一个特别善于观察别人情绪的人。")
    parser.add_argument("--language", type=str, default="Chinese")
    parser.add_argument("--speaker", type=str, default="Vivian")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--output_wav", type=str, default="output_custom_voice.onnx.wav")
    parser.add_argument(
        "--ort_providers",
        type=str,
        default="CPUExecutionProvider",
        help=(
            "Comma-separated ORT providers, e.g. "
            "CPUExecutionProvider or CUDAExecutionProvider,CPUExecutionProvider"
        ),
    )
    return parser.parse_args(argv)


def _resolve_path(value: str, env_key: str, default_value: str) -> str:
    val = (value or "").strip()
    if val:
        return val
    env_val = (os.getenv(env_key, "") or "").strip()
    if env_val:
        return env_val
    return default_value


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _parse_args(argv)

    qwen_tts = _import_module("qwen_tts")
    sf = _import_module("soundfile")
    model = qwen_tts.Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=args.device_map,
        dtype=_torch_dtype(args.dtype),
    )

    onnx_dir = _resolve_path(
        args.onnx_dir,
        "QWEN3_TTS_ONNX_DIR",
        "onnx_models_talker_core_fp32_no_custom",
    )
    merged_code_predictor_default = os.path.join("onnx_models_talker_core", "generate_process.onnx")

    talker_prefill_onnx = _resolve_path(
        args.talker_prefill_onnx,
        "QWEN3_TTS_TALKER_PREFILL_ONNX",
        os.path.join(onnx_dir, "talker_prefill.onnx"),
    )
    talker_step_onnx = _resolve_path(
        args.talker_step_onnx,
        "QWEN3_TTS_TALKER_STEP_ONNX",
        os.path.join(onnx_dir, "talker_step.onnx"),
    )
    code_predictor_onnx = _resolve_path(
        args.code_predictor_onnx,
        "QWEN3_TTS_CODE_PREDICTOR_ONNX",
        merged_code_predictor_default
        if os.path.exists(merged_code_predictor_default)
        else os.path.join(onnx_dir, "generate_process.onnx"),
    )
    speech_decoder_onnx = _resolve_path(
        args.speech_decoder_onnx,
        "QWEN3_TTS_SPEECH_DECODER_ONNX",
        os.path.join("onnx_models_speech_tokenizer", "speech_decoder.onnx"),
    )

    providers = [p.strip() for p in (args.ort_providers or "").split(",") if p.strip()]
    if not providers:
        providers = ["CPUExecutionProvider"]

    wav, sr = _run_talker_onnx_kv(
        model,
        providers=providers,
        onnx_paths=_OnnxPaths(
            talker_prefill=talker_prefill_onnx,
            talker_step=talker_step_onnx,
            code_predictor=code_predictor_onnx,
            speech_decoder=speech_decoder_onnx,
        ),
        gen_args=_GenArgs(
            text=args.text,
            language=args.language,
            speaker=args.speaker,
            max_new_tokens=int(args.max_new_tokens),
            repetition_penalty=float(args.repetition_penalty),
        ),
    )
    sf.write(args.output_wav, wav, sr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
