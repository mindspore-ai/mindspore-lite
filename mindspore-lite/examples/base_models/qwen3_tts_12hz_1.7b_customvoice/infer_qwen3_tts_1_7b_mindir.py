"""MindSpore Lite (MindIR) demo for Qwen3-TTS (talker KV + generate_process + speech decoder)."""

from __future__ import annotations

import argparse
import importlib
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=None)
def _import_module(name: str) -> Any:
    return importlib.import_module(name)

np = _import_module("numpy")
ort = _import_module("onnxruntime")


def _torch_dtype(name: str):
    torch = _import_module("torch")
    val = (name or "float32").strip().lower()
    if val in ("float16", "fp16"):
        return torch.float16
    return torch.float32


def _apply_repetition_penalty(logits: Any, prev_tokens: Any, penalty: float):
    """Apply repetition penalty in-place style to logits for tokens seen in prev_tokens."""
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
    """Sample the next token id from logits with a suppress mask and repetition penalty."""
    torch = _import_module("torch")
    logits = logits.to(torch.float32)
    logits = logits.masked_fill(suppress_mask, float("-inf"))
    logits = _apply_repetition_penalty(logits, prev_tokens=prev_tokens, penalty=repetition_penalty)
    return int(torch.argmax(logits, dim=-1).item())


def _get_speaker_embed(model: Any, input_id: Any, speaker: str | None):
    torch = _import_module("torch")
    qwen = model.model
    talker = qwen.talker
    cfg = qwen.config
    if not speaker:
        return None
    spk_id = cfg.talker_config.spk_id[speaker.lower()]
    return talker.get_input_embeddings()(torch.tensor(spk_id, device=talker.device, dtype=input_id.dtype))


def _resolve_language_id(model: Any, language: str | None, speaker: str | None) -> int | None:
    """Resolve codec language id with dialect override when speaker is dialect."""
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
    """Return (tts_bos_embed, tts_eos_embed, tts_pad_embed) projected to talker hidden size."""
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
    """Build initial codec input embedding with optional speaker embedding inserted."""
    torch = _import_module("torch")
    qwen = model.model
    talker = qwen.talker
    cfg = qwen.config
    embed_0 = talker.get_input_embeddings()(torch.tensor(codec_prefill_list, device=talker.device, dtype=input_dtype))
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
    """Build talker prompt input embeddings by combining role/text and codec embeddings."""
    torch = _import_module("torch")
    qwen = model.model
    talker = qwen.talker
    role_embed = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
    prompt_embed = torch.cat(
        (tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1), tts_bos_embed),
        dim=1,
    ) + codec_input_embedding[:, :-1]
    prompt_embed = torch.cat((role_embed, prompt_embed), dim=1)
    last_embed = talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:4])) + codec_input_embedding[:, -1:]
    return torch.cat([prompt_embed, last_embed], dim=1)


def _build_trailing_text_hidden(model: Any, input_id: Any, tts_eos_embed: Any):
    torch = _import_module("torch")
    qwen = model.model
    talker = qwen.talker
    trailing = talker.text_projection(talker.get_text_embeddings()(input_id[:, 4:-5]))
    return torch.cat((trailing, tts_eos_embed), dim=1)


def _tokenize_assistant_input_id(model: Any, text: str, device: Any):
    tokenize_texts = getattr(model, "_tokenize_texts")
    build_assistant_text = getattr(model, "_build_assistant_text")
    return tokenize_texts([build_assistant_text(text)])[0].to(device)


def _build_talker_prompt_tensors(model: Any, text: str, language: str, speaker: str):
    """Build (prompt_embeds, attention_mask, trailing_text_hidden, tts_pad_embed) for MindIR prefill."""
    torch = _import_module("torch")
    input_id = _tokenize_assistant_input_id(model, text=text, device=model.model.talker.device)
    speaker_embed = _get_speaker_embed(model, input_id=input_id, speaker=speaker)
    language_id = _resolve_language_id(model, language=language, speaker=speaker)
    tts_bos_embed, tts_eos_embed, tts_pad_embed = _get_tts_special_embeds(model, dtype=input_id.dtype)
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
    trailing_text_hidden = _build_trailing_text_hidden(model, input_id=input_id, tts_eos_embed=tts_eos_embed)
    attn_mask = torch.ones((1, talker_input_embed.shape[1]), device=input_id.device, dtype=torch.int64)
    return talker_input_embed, attn_mask, trailing_text_hidden, tts_pad_embed


def _create_suppress_mask(cfg) -> tuple[Any, int]:
    torch = _import_module("torch")
    eos_id = int(cfg.talker_config.codec_eos_token_id)
    vocab_size = int(cfg.talker_config.vocab_size)
    suppress_mask = torch.zeros((vocab_size,), dtype=torch.bool)
    suppress_from = max(vocab_size - 1024, 0)
    suppress_mask[suppress_from:vocab_size] = True
    suppress_mask[eos_id] = False
    return suppress_mask, eos_id


def _resolve_max_new_tokens(trailing_text_hidden: Any, max_new_tokens: int | None) -> int:
    if max_new_tokens is not None:
        return int(max_new_tokens)
    trailing_len = int(trailing_text_hidden.shape[1])
    return int(max(32, min(256, trailing_len + 32)))


def _make_trailing_step(trailing_text_hidden: Any, step_idx: int, tts_pad_embed: Any):
    torch = _import_module("torch")
    if step_idx < int(trailing_text_hidden.shape[1]):
        return trailing_text_hidden[:, step_idx].unsqueeze(1).to(torch.float32)
    return tts_pad_embed.to(torch.float32)


def _build_step_feed(step_embed: Any, past_k: Any, past_v: Any, cache_pos: int) -> dict[str, Any]:
    pos = np.array([[[int(cache_pos)]], [[int(cache_pos)]], [[int(cache_pos)]]], dtype=np.int64)
    cache_len = np.array([int(cache_pos)], dtype=np.int64)
    step_embed_np = (
        step_embed.detach().cpu().numpy().astype(np.float32)
        if hasattr(step_embed, "detach")
        else np.asarray(step_embed, dtype=np.float32)
    )
    return {
        "step_embed": step_embed_np,
        "past_k": past_k,
        "past_v": past_v,
        "position_ids_step": pos,
        "cache_len": cache_len,
    }


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


def _cosine_similarity(a: Any, b: Any) -> float:
    a = np.asarray(a).reshape(-1).astype(np.float64, copy=False)
    b = np.asarray(b).reshape(-1).astype(np.float64, copy=False)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    if denom == 0.0:
        return 1.0 if float(np.linalg.norm(a - b)) == 0.0 else 0.0
    return float(np.dot(a, b) / denom)


def _diff_stats(a: Any, b: Any) -> dict:
    """Compute simple difference statistics between two numpy-like arrays."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {"shape_a": tuple(a.shape), "shape_b": tuple(b.shape), "same_shape": False}
    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        neq = int(np.count_nonzero(a != b))
        total = int(a.size)
        return {
            "same_shape": True,
            "dtype_a": str(a.dtype),
            "dtype_b": str(b.dtype),
            "neq": neq,
            "total": total,
        }
    af = a.astype(np.float32, copy=False)
    bf = b.astype(np.float32, copy=False)
    diff = np.abs(af - bf)
    return {
        "same_shape": True,
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "cos": _cosine_similarity(af, bf),
    }


def _get_out_by_name(
    out_map: dict[str, Any],
    *,
    names: tuple[str, ...],
    contains: tuple[str, ...] = (),
    index_fallback: int | None = None,
) -> Any:
    """Pick a tensor/ndarray from an output map using name/substring heuristics."""
    for n in names:
        if n in out_map:
            return out_map[n]
    if contains:
        for k, v in out_map.items():
            kl = str(k).lower()
            if any(s in kl for s in contains):
                return v
    if index_fallback is not None:
        vals = list(out_map.values())
        if 0 <= int(index_fallback) < len(vals):
            return vals[int(index_fallback)]
    raise KeyError(f"Output not found. names={names}, contains={contains}, keys={list(out_map.keys())}")


def _to_numpy_any(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "get_data_to_numpy"):
        return np.asarray(x.get_data_to_numpy())
    return np.asarray(x)


class _LiteSession:
    """A minimal MindSpore Lite session wrapper for MindIR inference."""

    def __init__(
        self,
        mindir_path: str,
        device_id: int = 0,
        target: str = "ascend",
        *,
        alloc_output_seq_len: int | None = None,
        alloc_kv_cache_seq_len: int | None = None,
        graph_kind: str | None = None,
    ):
        mslite = _import_module("mindspore_lite")
        os.environ.setdefault("DEVICE_ID", str(int(device_id)))
        os.environ.setdefault("ASCEND_DEVICE_ID", str(int(device_id)))
        ctx = mslite.Context()
        ctx.target = [str(target)]
        self._mslite = mslite
        self._tensor_type = getattr(mslite, "Tensor")
        self._device_id = int(device_id)
        self._target = str(target)
        self._alloc_output_seq_len = None if alloc_output_seq_len is None else int(alloc_output_seq_len)
        self._alloc_kv_cache_seq_len = None if alloc_kv_cache_seq_len is None else int(alloc_kv_cache_seq_len)
        self._graph_kind = "" if graph_kind is None else str(graph_kind).strip().lower()
        self.model = mslite.Model()
        self.model.build_from_file(mindir_path, mslite.ModelType.MINDIR, context=ctx)
        self._refresh_io()
        self._output_buffers = self._alloc_outputs_for_predict()

    def _refresh_io(self):
        self.inputs = list(self.model.get_inputs())
        self.outputs = list(self.model.get_outputs())
        self.input_by_name = {t.name: t for t in self.inputs}
        self.output_names = [t.name for t in self.outputs]

    @staticmethod
    def _cast_input_array(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.dtype == np.int64:
            return arr.astype(np.int32, copy=False)
        return arr

    def _prepare_active_inputs(self, feed: dict[str, Any]) -> dict[str, Any]:
        """Filter feed to only model inputs, and cast numpy int64 to int32 when needed."""
        active: dict[str, Any] = {}
        for tensor in self.inputs:
            if tensor.name not in feed:
                continue
            val = feed[tensor.name]
            if isinstance(val, np.ndarray):
                active[tensor.name] = self._cast_input_array(val)
            else:
                active[tensor.name] = val
        return active

    @staticmethod
    def _tensor_shape(tensor) -> tuple[int, ...] | None:
        try:
            shape = getattr(tensor, "shape", None) or []
            return tuple(int(x) for x in shape)
        except (TypeError, ValueError, AttributeError):
            return None

    @staticmethod
    def _tensor_data_size(tensor) -> int | None:
        try:
            val = int(getattr(tensor, "data_size", 0) or 0)
            return val if val > 0 else None
        except (TypeError, ValueError, AttributeError):
            return None

    def _needs_resize(self, active: dict[str, Any]) -> tuple[bool, list[list[int]]]:
        """Check whether MindSpore Lite model inputs need resizing."""
        shapes: list[list[int]] = []
        need_resize = False
        for tensor in self.inputs:
            if tensor.name not in active:
                continue
            val = active[tensor.name]
            if isinstance(val, self._tensor_type):
                exp_shape = self._tensor_shape(val)
                if not exp_shape:
                    need_resize = True
                    exp_shape = self._tensor_shape(tensor) or ()
            else:
                exp_shape = tuple(int(x) for x in val.shape)
            shapes.append(list(exp_shape))
            cur_shape = self._tensor_shape(tensor)
            if not cur_shape or cur_shape != exp_shape:
                need_resize = True
            cur_size = self._tensor_data_size(tensor)
            exp_size = self._tensor_data_size(val) if isinstance(val, self._tensor_type) else int(val.nbytes)
            if cur_size is None or cur_size != int(exp_size):
                need_resize = True
        return need_resize, shapes

    def _resize_if_needed(self, active: dict[str, Any]) -> None:
        need_resize, shapes = self._needs_resize(active)
        if need_resize and shapes:
            self.model.resize(self.inputs, shapes)
            self._refresh_io()

    def _set_inputs(self, active: dict[str, Any]) -> list[Any]:
        """Set input tensor data and return the inputs list for model.predict()."""
        inputs_for_predict: list[Any] = []
        for tensor in self.inputs:
            if tensor.name in active:
                val = active[tensor.name]
                if isinstance(val, self._tensor_type):
                    inputs_for_predict.append(val)
                    continue
                tensor.set_data_from_numpy(val)
                inputs_for_predict.append(tensor)
            else:
                inputs_for_predict.append(tensor)
        return inputs_for_predict

    def _collect_outputs(self, outs, *, return_tensors: bool) -> dict[str, Any]:
        out_map: dict[str, Any] = {}
        for idx, tensor in enumerate(outs):
            name = self.output_names[idx] if idx < len(self.output_names) else f"output_{idx}"
            out_map[name] = tensor if return_tensors else tensor.get_data_to_numpy()
        return out_map

    def _prealloc_enabled(self) -> bool:
        return (
            str(self._target).lower() == "ascend"
            and self._alloc_output_seq_len is not None
            and self._mslite is not None
        )

    def _prealloc_dtype(self) -> tuple[Any, Any]:
        dt = getattr(self._mslite, "DataType", None)
        float32 = getattr(dt, "FLOAT32", None) if dt is not None else None
        int32 = getattr(dt, "INT32", None) if dt is not None else None
        return float32, int32

    @staticmethod
    def _make_shape_positive(shape: list[int]) -> list[int]:
        out = [int(x) for x in shape]
        for i, d in enumerate(out):
            if int(d) <= 0:
                out[i] = 1
        return out

    def _expected_output_specs(
        self,
        *,
        kv_len: int,
        float32: Any,
        int32: Any,
    ) -> dict[str, list[tuple[list[int], Any]]]:
        """Return expected output shapes/dtypes for known prefill/step graphs."""
        layers = 28
        num_kv_heads = 8
        head_dim = 128
        hidden_size = 2048
        vocab_size = 3072
        prefill_expected: list[tuple[list[int], Any]] = [
            ([1, vocab_size], float32),
            ([1, 1, hidden_size], float32),
            ([layers, 1, num_kv_heads, kv_len, head_dim], float32),
            ([layers, 1, num_kv_heads, kv_len, head_dim], float32),
            ([1], int32),
        ]
        step_expected: list[tuple[list[int], Any]] = [
            ([1, vocab_size], float32),
            ([1, 1, hidden_size], float32),
            ([layers, 1, num_kv_heads, kv_len, head_dim], float32),
            ([layers, 1, num_kv_heads, kv_len, head_dim], float32),
        ]
        return {"prefill": prefill_expected, "step": step_expected}

    def _fallback_output_spec(self, *, idx: int, ref: Any, float32: Any, int32: Any) -> tuple[list[int], Any]:
        name = self.output_names[idx] if idx < len(self.output_names) else f"output_{idx}"
        name_l = str(name).lower()
        ref_shape = list(getattr(ref, "shape", None) or [])
        shape = self._make_shape_positive(ref_shape if ref_shape else [1, 1, 1])
        out_dtype = getattr(ref, "dtype", None) or float32
        if out_dtype is None and int32 is not None and any(x in name_l for x in ("id", "mask", "pos", "len")):
            out_dtype = int32
        return shape, out_dtype

    def _alloc_outputs_for_predict(self) -> list[Any] | None:
        """Pre-allocate output tensors on Ascend for model.predict(out_tensors=...)."""
        if not self._prealloc_enabled():
            return None
        dev = f"ascend:{int(self._device_id)}"
        seq_len = int(self._alloc_output_seq_len)
        kv_len = int(self._alloc_kv_cache_seq_len) if self._alloc_kv_cache_seq_len is not None else int(seq_len)
        float32, int32 = self._prealloc_dtype()
        expected = self._expected_output_specs(kv_len=kv_len, float32=float32, int32=int32)
        kind = str(self._graph_kind).strip().lower()

        outs: list[Any] = []
        for idx, ref in enumerate(self.outputs):
            if kind in expected and idx < len(expected[kind]):
                shape, out_dtype = expected[kind][idx]
            else:
                shape, out_dtype = self._fallback_output_spec(idx=idx, ref=ref, float32=float32, int32=int32)
            outs.append(self._mslite.Tensor(shape=list(shape), dtype=out_dtype, device=dev))
        return outs

    def run(self, feed: dict[str, Any], *, return_tensors: bool = False) -> dict[str, Any]:
        """Run a single inference and return a name->output mapping."""
        active = self._prepare_active_inputs(feed)
        self._resize_if_needed(active)
        inputs_for_predict = self._set_inputs(active)
        out_tensors = self._output_buffers
        outs = self.model.predict(inputs_for_predict, out_tensors)
        if out_tensors is not None:
            outs = out_tensors if outs is None else outs
        return self._collect_outputs(outs, return_tensors=bool(return_tensors))

    def get_io_names(self) -> tuple[list[str], list[str]]:
        """Return (input_names, output_names) for the underlying model."""
        return [t.name for t in self.inputs], list(self.output_names)


@dataclass(frozen=True)
class _InferArgs:
    mindir_dir: str
    gen_args: Any
    device_id: int


def _bucket_seq_len(seq_len: int, *, bucket: int = 10, min_len: int = 10, max_len: int = 260) -> int:
    """Round seq_len up to a fixed bucket for MindIR dynamic shape buckets."""
    seq_len_i = int(seq_len)
    if seq_len_i <= int(min_len):
        return int(min_len)
    padded = ((seq_len_i + int(bucket) - 1) // int(bucket)) * int(bucket)
    if padded > int(max_len):
        raise ValueError(f"seq_len={seq_len_i} exceeds max_len={int(max_len)} for bucketed MindIR inputs.")
    return int(padded)


_PREFILL_MAX_GEAR_SEQ_LEN = 260
_KV_CACHE_TOTAL_SEQ_LEN = 512


def _pad_seq_len(arr: np.ndarray, padded_len: int) -> np.ndarray:
    """Pad (or slice) an array along axis=1 to match padded_len."""
    if arr.ndim < 2:
        raise ValueError(f"Expected an array with a seq_len dimension, got shape={arr.shape}.")
    seq_len = int(arr.shape[1])
    if seq_len == int(padded_len):
        return arr
    if seq_len > int(padded_len):
        return arr[:, : int(padded_len), ...]
    pad_width = [(0, 0)] * int(arr.ndim)
    pad_width[1] = (0, int(padded_len) - seq_len)
    return np.pad(arr, pad_width=pad_width, mode="constant", constant_values=0)


def _slice_seq_len(arr: Any, seq_len: int, padded_len: int) -> Any:
    """Slice numpy outputs back to seq_len for bucketed MindIR runs."""
    del padded_len
    if not isinstance(arr, np.ndarray):
        return arr
    if arr.ndim >= 3 and int(arr.shape[1]) >= int(seq_len):
        return arr[:, : int(seq_len), ...]
    return arr


def _prefill_mindir(
    sess_prefill: _LiteSession,
    prompt_embeds: Any,
    attn_mask: Any,
) -> tuple[np.ndarray, np.ndarray, Any, Any, int]:
    """Run talker prefill MindIR with bucket padding and slice outputs back."""
    torch = _import_module("torch")
    prompt_embeds_np = prompt_embeds.to(torch.float32).detach().cpu().numpy()
    attn_mask_np = attn_mask.cpu().numpy().astype(np.int64)
    seq_len = int(prompt_embeds_np.shape[1])
    padded_len = _bucket_seq_len(seq_len)
    prompt_embeds_np = _pad_seq_len(prompt_embeds_np, padded_len)
    attn_mask_np = _pad_seq_len(attn_mask_np, padded_len).astype(np.int64, copy=False)
    out_prefill = sess_prefill.run(
        {
            "inputs_embeds": prompt_embeds_np,
            "attention_mask": attn_mask_np,
        },
        return_tensors=True,
    )
    logits_last_t = _get_out_by_name(
        out_prefill,
        names=("logits_last", "logits"),
        contains=("logits",),
        index_fallback=0,
    )
    hidden_last_t = _get_out_by_name(
        out_prefill,
        names=("hidden_last", "hidden"),
        contains=("hidden",),
        index_fallback=1,
    )
    past_k = _get_out_by_name(
        out_prefill,
        names=("past_k", "present_k", "k_cache"),
        contains=("past_k", "present_k", "k_cache"),
        index_fallback=2,
    )
    past_v = _get_out_by_name(
        out_prefill,
        names=("past_v", "present_v", "v_cache"),
        contains=("past_v", "present_v", "v_cache"),
        index_fallback=3,
    )
    logits_last = _slice_seq_len(_to_numpy_any(logits_last_t), seq_len, padded_len)
    hidden_last = _slice_seq_len(_to_numpy_any(hidden_last_t), seq_len, padded_len)
    cache_base_pos = int(np.maximum(np.int64(seq_len), 1))
    return logits_last, hidden_last, past_k, past_v, cache_base_pos


@dataclass(frozen=True)
class _CodePredictorInputs:
    """Inputs for the generate_process MindIR."""

    hidden_last_t: Any
    last_id_hidden: Any
    trailing_step: Any
    next_id: int


def _to_numpy_f32(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32)


def _run_code_predictor_mindir(
    sess_gen: _LiteSession,
    gen_input_names: set[str],
    inputs: _CodePredictorInputs,
) -> tuple[Any, Any]:
    """Run generate_process MindIR and return (codec_ids, step_embed)."""
    torch = _import_module("torch")
    active_items: list[tuple[str, np.ndarray]] = []
    if "inputs_embeds" in gen_input_names:
        active_items.append(
            (
                "inputs_embeds",
                _to_numpy_f32(torch.cat((inputs.hidden_last_t, inputs.last_id_hidden), dim=1)),
            )
        )
    if "next_id" in gen_input_names:
        active_items.append(("next_id", np.array([[int(inputs.next_id)]], dtype=np.int64)))
    if "last_id_hidden" in gen_input_names:
        active_items.append(("last_id_hidden", _to_numpy_f32(inputs.last_id_hidden)))
    if "trailing_step" in gen_input_names:
        active_items.append(("trailing_step", _to_numpy_f32(inputs.trailing_step)))
    gen_feed = dict(active_items)
    out_gen = sess_gen.run(gen_feed)
    codec_ids = out_gen.get("codec_ids")
    step_embed = out_gen.get("step_embed")
    if codec_ids is None or step_embed is None:
        values = list(out_gen.values())
        if len(values) != 2:
            raise RuntimeError(
                f"Expected generate_process outputs (codec_ids, step_embed), got {len(values)}."
            )
        codec_ids, step_embed = values
    codec_ids_t = torch.from_numpy(np.asarray(codec_ids)).to(torch.long)
    step_embed_t = torch.from_numpy(np.asarray(step_embed)).to(torch.float32)
    return codec_ids_t, step_embed_t


def _run_step_mindir(
    sess_step: _LiteSession,
    step_embed: Any,
    past_k: Any,
    past_v: Any,
    cache_pos: int,
) -> tuple[np.ndarray, np.ndarray, Any, Any]:
    """Run talker_step MindIR once and return updated KV cache."""
    feed = _build_step_feed(step_embed=step_embed, past_k=past_k, past_v=past_v, cache_pos=cache_pos)
    out_step = sess_step.run(feed, return_tensors=True)
    logits_last_t = _get_out_by_name(
        out_step,
        names=("logits_last", "logits"),
        contains=("logits",),
        index_fallback=0,
    )
    hidden_last_t = _get_out_by_name(
        out_step,
        names=("hidden_last", "hidden"),
        contains=("hidden",),
        index_fallback=1,
    )
    past_k_out = _get_out_by_name(
        out_step,
        names=("past_k", "present_k", "k_cache"),
        contains=("past_k", "present_k", "k_cache"),
        index_fallback=2,
    )
    past_v_out = _get_out_by_name(
        out_step,
        names=("past_v", "present_v", "v_cache"),
        contains=("past_v", "present_v", "v_cache"),
        index_fallback=3,
    )
    logits_last = _to_numpy_any(logits_last_t)
    hidden_last = _to_numpy_any(hidden_last_t)
    return logits_last, hidden_last, past_k_out, past_v_out


def _decode_speech_chunked(
    sess_speech: _LiteSession,
    codes_bkt: Any,
    upsample_rate: int,
    chunk: int = 60,
) -> np.ndarray:
    """Decode speech in chunks to avoid large dynamic shapes."""
    t_total = int(codes_bkt.shape[-1])
    if t_total <= 0:
        return np.zeros((0,), dtype=np.float32)
    up = int(upsample_rate)
    wav_chunks: list[np.ndarray] = []
    for start in range(0, t_total, int(chunk)):
        end = min(start + int(chunk), t_total)
        codes_chunk = codes_bkt[..., start:end].contiguous()
        out_speech = sess_speech.run({"codes": codes_chunk.detach().cpu().numpy().astype(np.int32)})
        wav = np.asarray(list(out_speech.values())[0])
        if wav.ndim == 3:
            wav = wav[:, 0, :]
        wav = wav[0].astype(np.float32, copy=False)
        t = int(end - start)
        if t > 0 and up > 0:
            wav = wav[: int(t * up)]
        wav_chunks.append(wav)
    return np.concatenate(wav_chunks, axis=0) if wav_chunks else np.zeros((0,), dtype=np.float32)


def _create_mindir_sessions(mindir_dir: str, device_id: int) -> Any:
    """Create MindSpore Lite sessions from MindIR files under mindir_dir."""
    mindir_files = {
        "prefill": "talker_prefill_graph.mindir",
        "step": "talker_step_graph.mindir",
        "gen": "generate_process.mindir",
        "speech": "speech_decoder.mindir",
    }
    prefill_path = os.path.join(mindir_dir, mindir_files["prefill"])
    step_path = os.path.join(mindir_dir, mindir_files["step"])
    gen_path = os.path.join(mindir_dir, mindir_files["gen"])
    speech_path = os.path.join(mindir_dir, mindir_files["speech"])
    sess_prefill = _LiteSession(
        prefill_path,
        device_id=device_id,
        alloc_output_seq_len=int(_PREFILL_MAX_GEAR_SEQ_LEN),
        alloc_kv_cache_seq_len=int(_KV_CACHE_TOTAL_SEQ_LEN),
        graph_kind="prefill",
    )
    sess_step = _LiteSession(
        step_path,
        device_id=device_id,
        alloc_output_seq_len=int(_KV_CACHE_TOTAL_SEQ_LEN),
        alloc_kv_cache_seq_len=int(_KV_CACHE_TOTAL_SEQ_LEN),
        graph_kind="step",
    )
    sess_gen = _LiteSession(gen_path, device_id=device_id)
    sess_speech = _LiteSession(speech_path, device_id=device_id)
    gen_input_names = set(sess_gen.input_by_name.keys())
    payload = {
        "prefill": sess_prefill,
        "step": sess_step,
        "gen": sess_gen,
        "speech": sess_speech,
        "gen_input_names": gen_input_names,
    }
    return _OrtSessions(**payload)


def _load_talker_objects(model: Any) -> tuple[Any, Any]:
    cfg = model.model.config
    talker = model.model.talker
    return cfg, talker


def _sample_first_token_id(
    cfg: Any,
    logits_last: np.ndarray,
    repetition_penalty: float,
) -> tuple[Any, int, int]:
    """Sample the first token id from prefill logits and return (suppress_mask, eos_id, next_id)."""
    torch = _import_module("torch")
    suppress_mask, eos_id = _create_suppress_mask(cfg)
    prev_tokens = torch.empty((0,), dtype=torch.long)
    logits_np = np.asarray(logits_last)
    vocab_size = int(cfg.talker_config.vocab_size)
    if logits_np.ndim == 3 and logits_np.shape[-1] == vocab_size:
        logits_vec = logits_np[0, -1, :]
    elif logits_np.ndim == 2 and logits_np.shape[-1] == vocab_size:
        logits_vec = logits_np[-1, :]
    elif logits_np.ndim == 2 and logits_np.shape[0] == vocab_size:
        logits_vec = logits_np[:, -1]
    elif logits_np.ndim == 1 and logits_np.shape[0] == vocab_size:
        logits_vec = logits_np
    else:
        logits_vec = logits_np.reshape(-1)
    logits_t = torch.from_numpy(np.asarray(logits_vec)).to(torch.float32)
    next_id = _sample_next_id(
        logits=logits_t,
        suppress_mask=suppress_mask,
        prev_tokens=prev_tokens,
        repetition_penalty=float(repetition_penalty),
    )
    return suppress_mask, int(eos_id), int(next_id)


def _prefill_and_init_loop(
    model: Any,
    sessions: Any,
    args: _InferArgs,
) -> dict[str, Any]:
    """Run prefill and construct the initial loop state for generation."""
    cfg, talker = _load_talker_objects(model)
    gen_args = args.gen_args
    prompt = _build_talker_prompt_tensors(
        model,
        text=gen_args.text,
        language=gen_args.language,
        speaker=gen_args.speaker,
    )
    max_new_tokens_i = int(_resolve_max_new_tokens(prompt[2], gen_args.max_new_tokens))
    prefill = _prefill_mindir(sessions.prefill, prompt[0], prompt[1])
    suppress_mask, eos_id, next_id = _sample_first_token_id(
        cfg, logits_last=prefill[0], repetition_penalty=gen_args.repetition_penalty
    )
    torch = _import_module("torch")
    hidden_last_t = torch.from_numpy(np.asarray(prefill[1])).to(talker.device).to(torch.float32)
    return {
        "cfg": cfg,
        "talker": talker,
        "trailing_text_hidden": prompt[2],
        "tts_pad_embed": prompt[3],
        "max_new_tokens": max_new_tokens_i,
        "suppress_mask": suppress_mask,
        "eos_id": eos_id,
        "next_id": next_id,
        "prev_tokens": torch.tensor([int(next_id)], dtype=torch.long),
        "hidden_last_t": hidden_last_t,
        "past_k": prefill[2],
        "past_v": prefill[3],
        "cache_base_pos": prefill[4],
    }


def _run_mindir_loop(
    sessions: Any,
    state: dict[str, Any],
    repetition_penalty: float,
) -> Any:
    """Run the MindIR generation loop and return bucketed codec ids."""
    torch = _import_module("torch")
    talker = state["talker"]

    all_codes: list[Any] = []
    with torch.no_grad():
        for step_idx in range(int(state["max_new_tokens"])):
            next_id = int(state["next_id"])
            if next_id == int(state["eos_id"]):
                break
            last_id_hidden = talker.get_input_embeddings()(
                torch.tensor([[next_id]], device=talker.device, dtype=torch.long)
            ).to(torch.float32)
            trailing_step = _make_trailing_step(
                state["trailing_text_hidden"],
                step_idx=step_idx,
                tts_pad_embed=state["tts_pad_embed"],
            )
            gen_out = _run_code_predictor_mindir(
                sess_gen=sessions.gen,
                gen_input_names=sessions.gen_input_names,
                inputs=_CodePredictorInputs(
                    hidden_last_t=state["hidden_last_t"],
                    last_id_hidden=last_id_hidden,
                    trailing_step=trailing_step,
                    next_id=next_id,
                ),
            )
            all_codes.append(gen_out[0].squeeze(0).cpu())
            step_out = _run_step_mindir(
                sess_step=sessions.step,
                step_embed=gen_out[1],
                past_k=state["past_k"],
                past_v=state["past_v"],
                cache_pos=int(int(state["cache_base_pos"]) + step_idx),
            )
            state["past_k"] = step_out[2]
            state["past_v"] = step_out[3]
            state["hidden_last_t"] = (
                torch.from_numpy(np.asarray(step_out[1])).to(talker.device).to(torch.float32)
            )
            state["next_id"] = _sample_next_id(
                logits=torch.from_numpy(np.asarray(step_out[0])[0]),
                suppress_mask=state["suppress_mask"],
                prev_tokens=state["prev_tokens"],
                repetition_penalty=float(repetition_penalty),
            )
            state["prev_tokens"] = torch.cat(
                [
                    state["prev_tokens"],
                    torch.tensor([int(state["next_id"])], dtype=torch.long),
                ],
                dim=0,
            )

    if all_codes:
        codes = torch.stack(all_codes, dim=0)
    else:
        num_groups = int(state["cfg"].talker_config.num_code_groups)
        codes = torch.empty((0, num_groups), dtype=torch.long)
    return codes.transpose(0, 1).unsqueeze(0).contiguous()


def _infer_codes_bkt_mindir(model: Any, sessions: Any, args: _InferArgs) -> Any:
    state = _prefill_and_init_loop(model, sessions=sessions, args=args)
    penalty = float(args.gen_args.repetition_penalty)
    return _run_mindir_loop(sessions, state=state, repetition_penalty=penalty)


def _run_talker_mindir_kv(model: Any, args: _InferArgs) -> tuple[np.ndarray, int]:
    """Run MindIR end-to-end (talker KV + generate_process + speech) and return (wav, sr)."""
    sessions = _create_mindir_sessions(args.mindir_dir, device_id=args.device_id)
    import time
    tic = time.perf_counter()
    codes_bkt = _infer_codes_bkt_mindir(model, sessions=sessions, args=args)
    up = int(model.model.speech_tokenizer.get_decode_upsample_rate())
    wav = _decode_speech_chunked(sessions.speech, codes_bkt, upsample_rate=up, chunk=60)
    sr = int(model.model.speech_tokenizer.get_output_sample_rate())
    toc = time.perf_counter()
    print(f"MindIR inference time: {toc - tic:.4f} s")
    return wav, sr


def _compare_one(
    name: str,
    sess_onnx: ort.InferenceSession,
    sess_mindir: _LiteSession,
    feed: dict[str, np.ndarray],
) -> list[dict]:
    """Compare one ONNX session and one MindIR session on the same feed."""
    onnx_input_names = {i.name for i in sess_onnx.get_inputs()}
    onnx_feed = {k: v for k, v in feed.items() if k in onnx_input_names}
    out_onnx = sess_onnx.run(None, onnx_feed)
    out_mindir_map = sess_mindir.run(feed)
    out_mindir = list(out_mindir_map.values())
    n = min(len(out_onnx), len(out_mindir))
    rows = []
    for i in range(n):
        a = np.asarray(out_onnx[i])
        b = np.asarray(out_mindir[i])
        stats = _diff_stats(a, b)
        stats["model"] = name
        stats["output_index"] = i
        rows.append(stats)
    return rows


def compare_accuracy(
    onnx_dir: str,
    mindir_dir: str,
    device_id: int = 0,
    seed: int = 0,
):
    """Compare ONNXRuntime and MindSpore Lite outputs on random inputs."""
    ctx = _prepare_compare_context(
        onnx_dir=onnx_dir,
        mindir_dir=mindir_dir,
        device_id=device_id,
        seed=seed,
    )

    rows: list[dict] = []
    rows.extend(
        _compare_one(
            "talker_prefill",
            ctx["sess_prefill_onnx"],
            ctx["sess_prefill_ms"],
            {
                "inputs_embeds": ctx["prompt_embeds"],
                "attention_mask": ctx["attention_mask"],
            },
        )
    )
    rows.extend(
        _compare_one(
            "talker_step",
            ctx["sess_step_onnx"],
            ctx["sess_step_ms"],
            {
                "step_embed": ctx["step_embed"],
                "past_k": ctx["past_k"],
                "past_v": ctx["past_v"],
                "position_ids_step": ctx["position_ids_step"],
                "cache_len": ctx["cache_len"],
            },
        )
    )
    rows.extend(
        _compare_one(
            "generate_process",
            ctx["sess_gen_onnx"],
            ctx["sess_gen_ms"],
            {"inputs_embeds": ctx["inputs_embeds"]},
        )
    )
    rows.extend(
        _compare_one(
            "speech_decoder",
            ctx["sess_speech_onnx"],
            ctx["sess_speech_ms"],
            {"codes": ctx["codes"]},
        )
    )
    _print_compare_rows(rows)


def _prepare_compare_context(
    onnx_dir: str,
    mindir_dir: str,
    device_id: int,
    seed: int,
) -> dict[str, Any]:
    """Prepare sessions and random inputs for ONNX vs MindIR comparison."""
    rng = np.random.default_rng(int(seed))
    sess_onnx = _build_onnx_sessions(onnx_dir)
    sess_ms = _build_mindir_sessions(mindir_dir, device_id=device_id)
    prefill_inputs = _make_prefill_inputs(rng, seq=20, hidden=2048)
    step_inputs = _make_step_inputs(rng, hidden=2048, cache_pos=10)
    inputs_embeds = _make_gen_inputs(rng, seq=2, hidden=2048)
    codes = _make_speech_inputs(rng, t_codes=20, num_groups=16, vocab=1024)
    past_k, past_v = _make_past_kv(rng, sess_onnx[1])
    return {
        "sess_prefill_onnx": sess_onnx[0],
        "sess_step_onnx": sess_onnx[1],
        "sess_gen_onnx": sess_onnx[2],
        "sess_speech_onnx": sess_onnx[3],
        "sess_prefill_ms": sess_ms[0],
        "sess_step_ms": sess_ms[1],
        "sess_gen_ms": sess_ms[2],
        "sess_speech_ms": sess_ms[3],
        "prompt_embeds": prefill_inputs[0],
        "attention_mask": prefill_inputs[1],
        "step_embed": step_inputs[0],
        "position_ids_step": step_inputs[1],
        "cache_len": step_inputs[2],
        "inputs_embeds": inputs_embeds,
        "codes": codes,
        "past_k": past_k,
        "past_v": past_v,
    }


def _build_onnx_sessions(onnx_dir: str):
    """Create ONNX Runtime sessions for comparison."""
    providers = ["CPUExecutionProvider"]
    sess_prefill = ort.InferenceSession(
        os.path.join(onnx_dir, "talker_prefill.onnx"),
        providers=providers,
    )
    sess_step = ort.InferenceSession(
        os.path.join(onnx_dir, "talker_step.onnx"),
        providers=providers,
    )
    sess_gen = ort.InferenceSession(
        os.path.join(onnx_dir, "generate_process.onnx"),
        providers=providers,
    )
    speech_path = "/data/xp/qwen3-tts/Qwen3-TTS/onnx_models_speech_tokenizer/speech_decoder.onnx"
    sess_speech = ort.InferenceSession(speech_path, providers=providers)
    return sess_prefill, sess_step, sess_gen, sess_speech


def _build_mindir_sessions(mindir_dir: str, device_id: int):
    """Create MindSpore Lite sessions for comparison."""
    sess_prefill = _LiteSession(
        os.path.join(mindir_dir, "talker_prefill_graph.mindir"),
        device_id=device_id,
        alloc_output_seq_len=int(_PREFILL_MAX_GEAR_SEQ_LEN),
        alloc_kv_cache_seq_len=int(_KV_CACHE_TOTAL_SEQ_LEN),
        graph_kind="prefill",
    )
    sess_step = _LiteSession(
        os.path.join(mindir_dir, "talker_step_graph.mindir"),
        device_id=device_id,
        alloc_output_seq_len=int(_KV_CACHE_TOTAL_SEQ_LEN),
        alloc_kv_cache_seq_len=int(_KV_CACHE_TOTAL_SEQ_LEN),
        graph_kind="step",
    )
    sess_gen = _LiteSession(
        os.path.join(mindir_dir, "generate_process.mindir"),
        device_id=device_id,
    )
    sess_speech = _LiteSession(
        os.path.join(mindir_dir, "speech_decoder.mindir"),
        device_id=device_id,
    )
    return sess_prefill, sess_step, sess_gen, sess_speech


def _make_prefill_inputs(rng: np.random.Generator, seq: int, hidden: int):
    prompt_embeds = rng.standard_normal((1, int(seq), int(hidden)), dtype=np.float32)
    attention_mask = np.ones((1, int(seq)), dtype=np.int64)
    return prompt_embeds, attention_mask


def _make_step_inputs(rng: np.random.Generator, hidden: int, cache_pos: int):
    step_embed = rng.standard_normal((1, 1, int(hidden)), dtype=np.float32)
    pos = int(cache_pos)
    position_ids_step = np.array([[[pos]], [[pos]], [[pos]]], dtype=np.int64)
    cache_len = np.array([pos], dtype=np.int64)
    return step_embed, position_ids_step, cache_len


def _make_gen_inputs(rng: np.random.Generator, seq: int, hidden: int):
    return rng.standard_normal((1, int(seq), int(hidden)), dtype=np.float32)


def _make_speech_inputs(rng: np.random.Generator, t_codes: int, num_groups: int, vocab: int):
    return rng.integers(
        low=0,
        high=int(vocab),
        size=(1, int(num_groups), int(t_codes)),
        dtype=np.int64,
    )


def _make_past_kv(rng: np.random.Generator, sess_step_onnx: ort.InferenceSession):
    """Create past_k/past_v inputs for step graph comparison."""
    pk_shape = None
    pv_shape = None
    try:
        pk_shape = tuple(int(x) for x in sess_step_onnx.get_inputs()[1].shape)
        pv_shape = tuple(int(x) for x in sess_step_onnx.get_inputs()[2].shape)
    except (TypeError, ValueError, AttributeError):
        pk_shape = None
        pv_shape = None

    if pk_shape and -1 not in pk_shape:
        past_k = rng.standard_normal(pk_shape, dtype=np.float32)
    else:
        past_k = rng.standard_normal((28, 1, 8, 512, 128), dtype=np.float32)
    if pv_shape and -1 not in pv_shape:
        past_v = rng.standard_normal(pv_shape, dtype=np.float32)
    else:
        past_v = rng.standard_normal((28, 1, 8, 512, 128), dtype=np.float32)
    return past_k, past_v


def _print_compare_rows(rows: list[dict]) -> None:
    for row in rows:
        items = [f"{k}={row[k]}" for k in row.keys() if k not in ("model", "output_index")]
        print(f"{row['model']}[{row['output_index']}]: " + ", ".join(items))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for MindIR inference and compare mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=os.getenv("QWEN3_TTS_MODE", "infer"),
        choices=["infer", "compare"],
    )
    parser.add_argument("--model_path", type=str, default="../Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--device_map", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument(
        "--mindir_dir",
        type=str,
        default=os.getenv("QWEN3_TTS_MINDIR_DIR", "./mindir"),
    )
    parser.add_argument(
        "--onnx_dir",
        type=str,
        default=os.getenv("QWEN3_TTS_ONNX_DIR", "./onnx_models_talker_core_fp32_no_custom"),
    )
    parser.add_argument("--device_id", type=int, default=int(os.getenv("DEVICE_ID", "0") or "0"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="output_custom_voice.mindir.wav")
    parser.add_argument("--text", type=str, default="其实我真的有发现，我是一个特别善于观察别人情绪的人。")
    parser.add_argument("--language", type=str, default="Chinese")
    parser.add_argument("--speaker", type=str, default="Vivian")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run MindIR inference or compare outputs against ONNXRuntime."""
    args = _parse_args(argv)
    qwen_tts = _import_module("qwen_tts")
    qwen3_tts_model_cls = getattr(qwen_tts, "Qwen3TTSModel")
    model = qwen3_tts_model_cls.from_pretrained(
        args.model_path,
        device_map=args.device_map,
        dtype=_torch_dtype(args.dtype),
    )

    if args.mode == "compare":
        compare_accuracy(args.onnx_dir, args.mindir_dir, device_id=args.device_id, seed=args.seed)
        return 0

    gen_args = _GenArgs(
        text=str(args.text),
        language=str(args.language),
        speaker=str(args.speaker),
        max_new_tokens=int(args.max_new_tokens) if args.max_new_tokens is not None else None,
        repetition_penalty=1.05,
    )
    infer_args = _InferArgs(
        mindir_dir=str(args.mindir_dir),
        gen_args=gen_args,
        device_id=int(args.device_id),
    )
    wav, sr = _run_talker_mindir_kv(model, infer_args)
    sf = _import_module("soundfile")
    sf.write(args.output, wav, sr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
