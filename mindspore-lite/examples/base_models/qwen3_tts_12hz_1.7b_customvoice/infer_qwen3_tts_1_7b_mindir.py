"""MindSpore Lite (MindIR) demo for Qwen3-TTS (talker KV + generate_process + speech decoder)."""

from __future__ import annotations

import argparse
import importlib
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=None)
def _import_module(name: str) -> Any:
    return importlib.import_module(name)

np = _import_module("numpy")
ml_dtypes = _import_module("ml_dtypes")
bf16 = getattr(ml_dtypes, "bfloat16", None)

# Module-level input dtype, overridden by --input_dtype in main().
# Default matches the argparse default ("float32") for the fp32 MindIR model sets.
_INPUT_DTYPE = "float32"


def _input_np_dtype():
    """Return numpy dtype for MindIR inputs, controlled by --input_dtype."""
    if _INPUT_DTYPE == "float32":
        return np.float32
    return bf16


def _input_torch_dtype():
    """Return torch dtype for MindIR inputs, controlled by --input_dtype."""
    torch = _import_module("torch")
    if _INPUT_DTYPE == "float32":
        return torch.float32
    return torch.bfloat16


def _torch_from_numpy(arr: Any) -> Any:
    """Convert a numpy array to a torch tensor, handling ml_dtypes.bfloat16 arrays."""
    torch = _import_module("torch")
    arr = np.asarray(arr)
    if bf16 is not None and arr.dtype == bf16:
        # torch.from_numpy cannot consume ml_dtypes.bfloat16; float32 is lossless for bf16.
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr)


def _torch_to_numpy(tensor: Any) -> np.ndarray:
    """Convert a torch tensor to numpy, keeping bfloat16 as ml_dtypes.bfloat16."""
    torch = _import_module("torch")
    arr = tensor.detach().cpu()
    if arr.dtype == torch.bfloat16:
        arr32 = arr.float().numpy()
        return arr32.astype(bf16) if bf16 is not None else arr32
    return arr.numpy()


def _torch_dtype(name: str):
    torch = _import_module("torch")
    val = (name or "bfloat16").strip().lower()
    if val in ("float16", "fp16"):
        return torch.float16
    if val in ("bfloat16", "bf16"):
        return torch.bfloat16
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
    logits = logits.to(_input_torch_dtype())
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
    if step_idx < int(trailing_text_hidden.shape[1]):
        return trailing_text_hidden[:, step_idx].unsqueeze(1).to(_input_torch_dtype())
    return tts_pad_embed.to(_input_torch_dtype())


def _build_step_feed(step_embed: Any, past_k: Any, past_v: Any, cache_pos: int) -> dict[str, Any]:
    """Build the step feed dict for the talker_step MindIR graph."""
    # MindSpore Lite 不支持 int64 输入，必须用 int32
    pos = np.array([[[int(cache_pos)]], [[int(cache_pos)]], [[int(cache_pos)]]], dtype=np.int32)
    # cache_len 期望 shape [1, 1] (与 converter_lite config 一致)
    cache_len = np.array([[int(cache_pos)]], dtype=np.int32)
    step_embed_np = (
        _torch_to_numpy(step_embed).astype(_input_np_dtype())
        if hasattr(step_embed, "detach")
        else np.asarray(step_embed, dtype=_input_np_dtype())
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
        # 累计模型推理时延（ms）与调用次数，用于端到端性能统计
        self._infer_ms = 0.0
        self._run_count = 0

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
        if _INPUT_DTYPE == "float32":
            ftype = getattr(dt, "FLOAT32", None) if dt is not None else None
        else:
            ftype = getattr(dt, "BFLOAT16", None) if dt is not None else None
        int32 = getattr(dt, "INT32", None) if dt is not None else None
        return ftype, int32

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
        bf16_dtype: Any,
        int32: Any,
    ) -> dict[str, list[tuple[list[int], Any]]]:
        """Return expected output shapes/dtypes for known prefill/step graphs.

        预分配输出缓冲的 batch 取 5，与 MindIR 动态分档的最大档位（ge.dynamicDims 的 5）
        一致：模型实际执行按档位 5 产出输出，若缓冲按 batch=1 预分配会小于实际输出，
        设备侧拷贝越界（SMMU page table error）。缓冲按最大档位分配后，实际输出写入
        只会占用前部空间，安全且免每次推理重新分配。
        """
        layers = 28
        num_kv_heads = 8
        head_dim = 128
        hidden_size = 2048
        vocab_size = 3072
        batch = 5
        prefill_expected: list[tuple[list[int], Any]] = [
            ([batch, vocab_size], bf16_dtype),
            ([batch, 1, hidden_size], bf16_dtype),
            ([layers, batch, num_kv_heads, kv_len, head_dim], bf16_dtype),
            ([layers, batch, num_kv_heads, kv_len, head_dim], bf16_dtype),
            ([batch], int32),
        ]
        step_expected: list[tuple[list[int], Any]] = [
            ([batch, vocab_size], bf16_dtype),
            ([batch, 1, hidden_size], bf16_dtype),
            ([layers, batch, num_kv_heads, kv_len, head_dim], bf16_dtype),
            ([layers, batch, num_kv_heads, kv_len, head_dim], bf16_dtype),
        ]
        return {"prefill": prefill_expected, "step": step_expected}

    def _fallback_output_spec(self, *, idx: int, ref: Any, bf16_dtype: Any, int32: Any) -> tuple[list[int], Any]:
        name = self.output_names[idx] if idx < len(self.output_names) else f"output_{idx}"
        name_l = str(name).lower()
        ref_shape = list(getattr(ref, "shape", None) or [])
        shape = self._make_shape_positive(ref_shape if ref_shape else [1, 1, 1])
        out_dtype = getattr(ref, "dtype", None) or bf16_dtype
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
        bf16_dtype, int32 = self._prealloc_dtype()
        expected = self._expected_output_specs(kv_len=kv_len, bf16_dtype=bf16_dtype, int32=int32)
        kind = str(self._graph_kind).strip().lower()

        outs: list[Any] = []
        for idx, ref in enumerate(self.outputs):
            if kind in expected and idx < len(expected[kind]):
                shape, out_dtype = expected[kind][idx]
            else:
                shape, out_dtype = self._fallback_output_spec(idx=idx, ref=ref, bf16_dtype=bf16_dtype, int32=int32)
            outs.append(self._mslite.Tensor(shape=list(shape), dtype=out_dtype, device=dev))
        return outs

    def run(self, feed: dict[str, Any], *, return_tensors: bool = False) -> dict[str, Any]:
        """Run a single inference and return a name->output mapping."""
        active = self._prepare_active_inputs(feed)
        self._resize_if_needed(active)
        inputs_for_predict = self._set_inputs(active)
        out_tensors = self._output_buffers
        t0 = time.perf_counter()
        outs = self.model.predict(inputs_for_predict, out_tensors)
        self._infer_ms += (time.perf_counter() - t0) * 1000.0
        self._run_count += 1
        if out_tensors is not None:
            outs = out_tensors if outs is None else outs
        return self._collect_outputs(outs, return_tensors=bool(return_tensors))


@dataclass(frozen=True)
class _InferArgs:
    mindir_dir: str
    gen_args: Any
    device_id: int
    dump_calib: str | None = None


def _bucket_seq_len(seq_len: int, *, bucket: int = 32, min_len: int = 32, max_len: int = 256) -> int:
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
    prompt_embeds_np = _torch_to_numpy(prompt_embeds.to(_input_torch_dtype()))
    attn_mask_np = attn_mask.cpu().numpy().astype(np.int64)
    seq_len = int(prompt_embeds_np.shape[1])
    padded_len = _bucket_seq_len(seq_len)
    prompt_embeds_np = _pad_seq_len(prompt_embeds_np, padded_len)
    attn_mask_np = _pad_seq_len(attn_mask_np, padded_len).astype(np.int64, copy=False)
    feed = {
        "inputs_embeds": prompt_embeds_np,
        "attention_mask": attn_mask_np,
    }
    out_prefill = sess_prefill.run(feed, return_tensors=True)
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


def _to_numpy_bf16(tensor: Any) -> np.ndarray:
    return _torch_to_numpy(tensor).astype(_input_np_dtype())


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
                _to_numpy_bf16(torch.cat((inputs.hidden_last_t, inputs.last_id_hidden), dim=1)),
            )
        )
    if "next_id" in gen_input_names:
        active_items.append(("next_id", np.array([[int(inputs.next_id)]], dtype=np.int64)))
    if "last_id_hidden" in gen_input_names:
        active_items.append(("last_id_hidden", _to_numpy_bf16(inputs.last_id_hidden)))
    if "trailing_step" in gen_input_names:
        active_items.append(("trailing_step", _to_numpy_bf16(inputs.trailing_step)))
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
    codec_ids_t = _torch_from_numpy(codec_ids).to(torch.long)
    step_embed_t = _torch_from_numpy(step_embed).to(_input_torch_dtype())
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
        return np.zeros((0,), dtype=_input_np_dtype())
    up = int(upsample_rate)
    wav_chunks: list[np.ndarray] = []
    for start in range(0, t_total, int(chunk)):
        end = min(start + int(chunk), t_total)
        codes_chunk = codes_bkt[..., start:end].contiguous()
        feed = {"codes": codes_chunk.detach().cpu().numpy().astype(np.int32)}
        out_speech = sess_speech.run(feed)
        wav = np.asarray(list(out_speech.values())[0])
        if wav.ndim == 3:
            wav = wav[:, 0, :]
        wav = wav[0].astype(_input_np_dtype(), copy=False)
        t = int(end - start)
        if t > 0 and up > 0:
            wav = wav[: int(t * up)]
        wav_chunks.append(wav)
    return np.concatenate(wav_chunks, axis=0) if wav_chunks else np.zeros((0,), dtype=_input_np_dtype())


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
    logits_t = _torch_from_numpy(logits_vec).to(_input_torch_dtype())
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
    hidden_last_t = _torch_from_numpy(prefill[1]).to(talker.device).to(_input_torch_dtype())
    return {
        # 保存 prefill 输入用于校准数据收集
        "_calib_prompt_embeds": prompt[0],
        "_calib_attn_mask": prompt[1],
        "_calib_text": gen_args.text,
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
            ).to(_input_torch_dtype())
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
            # import pdb;pdb.set_trace()
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
                _torch_from_numpy(step_out[1]).to(talker.device).to(_input_torch_dtype())
            )
            state["next_id"] = _sample_next_id(
                logits=_torch_from_numpy(step_out[0])[0],
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
    codes = _run_mindir_loop(sessions, state=state, repetition_penalty=penalty)

    # ── 收集量化矫正数据 (JSONL) ──
    if args.dump_calib:
        _save_calib_record(state, codes, args)

    return codes


def _save_calib_record(state: dict[str, Any], codes: Any, args: _InferArgs) -> None:
    """Append one calibration record to JSONL file for PTQ calibration."""
    import json as _json
    prompt_embeds = state.get("_calib_prompt_embeds")
    attn_mask = state.get("_calib_attn_mask")
    text = state.get("_calib_text", "")
    if prompt_embeds is None or attn_mask is None:
        return
    embeds_np = _torch_to_numpy(prompt_embeds) if hasattr(prompt_embeds, "detach") else np.asarray(prompt_embeds)
    mask_np = attn_mask.cpu().numpy() if hasattr(attn_mask, "cpu") else np.asarray(attn_mask)
    codes_np = codes.detach().cpu().numpy() if hasattr(codes, "detach") else np.asarray(codes)
    if codes_np.ndim == 3:
        sequence = codes_np[0, 0, :].tolist()
    elif codes_np.ndim == 2:
        sequence = codes_np[0, :].tolist()
    else:
        sequence = []
    record = {
        "meta": {"prompt": str(text), "sequence": [int(x) for x in sequence]},
        "input": {
            "inputs_embeds": embeds_np.astype(_input_np_dtype()).tolist(),
            "attention_mask": mask_np.astype(np.int64).tolist(),
        },
    }
    path = str(args.dump_calib)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(_json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  [calib] Appended record to {path} ({len(sequence)} codes)")


def _print_perf_table(sessions: Any, total_ms: float) -> None:
    """Print per-module latency (ms) of the zero-copy MindIR pipeline."""

    def _fmt_ms(v: float) -> str:
        return str(int(round(v)))

    def _fmt_avg(v: float) -> str:
        s = f"{v:.2f}"
        return s.rstrip("0").rstrip(".") if "." in s else s

    speech_ms = float(getattr(sessions.speech, "_infer_ms", 0.0))
    prefill_ms = float(getattr(sessions.prefill, "_infer_ms", 0.0))
    decode_ms = float(getattr(sessions.step, "_infer_ms", 0.0))
    gen_ms = float(getattr(sessions.gen, "_infer_ms", 0.0))
    decode_steps = int(getattr(sessions.step, "_run_count", 0))
    gen_steps = int(getattr(sessions.gen, "_run_count", 0))
    avg_decode = decode_ms / max(decode_steps, 1)
    avg_gen = gen_ms / max(gen_steps, 1)
    throughput = decode_steps / (max(total_ms, 1e-6) / 1000.0)
    rows = [
        f"| speech_decoder (ms)              | {_fmt_ms(speech_ms)} |",
        f"| Prefill (ms)             | {_fmt_ms(prefill_ms)} |",
        f"| Total Decode (ms)        | {_fmt_ms(decode_ms)} |",
        f"| **Avg decode step (ms)** | **{_fmt_avg(avg_decode)}** |",
        f"| Total generate_process (ms)        | {_fmt_ms(gen_ms)} |",
        f"| **Avg generate_process step (ms)** | **{_fmt_avg(avg_gen)}** |",
        f"| Total (ms)               | {_fmt_ms(total_ms)} |",
        f"| **Throughput (tok/s)**   | **{throughput:.1f}** |",
    ]
    print("\n".join(rows))


def _run_talker_mindir_kv(model: Any, args: _InferArgs) -> tuple[np.ndarray, int]:
    """Run MindIR end-to-end (talker KV + generate_process + speech) and return (wav, sr)."""
    sessions = _create_mindir_sessions(args.mindir_dir, device_id=args.device_id)
    tic = time.perf_counter()
    codes_bkt = _infer_codes_bkt_mindir(model, sessions=sessions, args=args)
    up = int(model.model.speech_tokenizer.get_decode_upsample_rate())
    wav = _decode_speech_chunked(sessions.speech, codes_bkt, upsample_rate=up, chunk=60)
    sr = int(model.model.speech_tokenizer.get_output_sample_rate())
    toc = time.perf_counter()
    _print_perf_table(sessions, total_ms=(toc - tic) * 1000.0)
    return wav, sr


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for MindIR end-to-end inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--device_map", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--input_dtype", type=str, default="float32", choices=["float32", "bfloat16"],
                        help="Dtype for MindIR model inputs (bf16 or fp32).")
    parser.add_argument(
        "--mindir_dir",
        type=str,
        default=os.getenv("QWEN3_TTS_MINDIR_DIR", "./mindir"),
    )
    parser.add_argument("--device_id", type=int, default=int(os.getenv("DEVICE_ID", "0") or "0"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="output_custom_voice.mindir.wav")
    parser.add_argument(
        "--dump-calib",
        type=str,
        default="",
        help="Append one PTQ calibration record to this JSONL file "
             "(inputs_embeds + generated sequence).",
    )
    parser.add_argument("--text", type=str, default="其实我真的有发现，我是一个特别善于观察别人情绪的人。")
    parser.add_argument("--language", type=str, default="Chinese")
    parser.add_argument("--speaker", type=str, default="Vivian")
    parser.add_argument("--max_new_tokens", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run MindIR end-to-end inference and write the output wav."""
    args = _parse_args(argv)
    # Set module-level input dtype
    global _INPUT_DTYPE
    _INPUT_DTYPE = str(args.input_dtype)
    qwen_tts = _import_module("qwen_tts")
    qwen3_tts_model_cls = getattr(qwen_tts, "Qwen3TTSModel")
    model = qwen3_tts_model_cls.from_pretrained(
        args.model_path,
        device_map=args.device_map,
        dtype=_torch_dtype(args.dtype),
    )

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
        dump_calib=(str(args.dump_calib).strip() or None),
    )
    wav, sr = _run_talker_mindir_kv(model, infer_args)
    sf = _import_module("soundfile")
    # soundfile 不支持 bfloat16（仅 float32/float64/int16/int32），bf16 模式先转 float32 再写盘
    sf.write(args.output, np.asarray(wav, dtype=np.float32), sr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
