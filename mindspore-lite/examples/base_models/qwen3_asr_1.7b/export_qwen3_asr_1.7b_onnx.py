"""Export Qwen3-ASR 1.7B audio encoder and text decoder to ONNX for MindSpore Lite."""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from qwen_asr.core.transformers_backend import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel, AutoTokenizer


AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)


def _audio_token_len_from_feat_frames(n_frames: int) -> int:
    input_lengths_leave = n_frames % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (n_frames // 100) * 13
    return int(output_lengths)


@dataclass(frozen=True)
class _AudioOeSpec:
    n_mels: int
    n_frames: int
    chunk_size: int
    chunks: int
    aftercnn_per_chunk: int
    aftercnn_total: int
    window_aftercnn: int
    cu_seqlens: Tuple[int, ...]


def _build_audio_spec(
    n_mels: int = 128,
    n_frames: int = 3000,
    n_window: int = 100,
    n_window_infer: int = 400,
):
    """Compute mel-frame layout, chunking, and cu_seqlens for ONNX audio export."""
    chunk_size = int(n_window * 2)
    if n_frames % chunk_size != 0:
        raise ValueError(
            "n_frames must be divisible by chunk_size, got "
            f"n_frames={n_frames}, chunk_size={chunk_size}"
        )
    chunks = int(n_frames // chunk_size)
    aftercnn_per_chunk = _audio_token_len_from_feat_frames(chunk_size)
    aftercnn_total = int(aftercnn_per_chunk * chunks)
    ratio = int(n_window_infer // (n_window * 2))
    window_aftercnn = int(aftercnn_per_chunk * ratio)
    if aftercnn_total % window_aftercnn == 0:
        parts = [window_aftercnn] * (aftercnn_total // window_aftercnn)
    else:
        parts = [window_aftercnn] * (aftercnn_total // window_aftercnn) + [
            aftercnn_total % window_aftercnn
        ]
    cu = [0]
    for p in parts:
        cu.append(cu[-1] + p)
    return _AudioOeSpec(
        n_mels=n_mels,
        n_frames=n_frames,
        chunk_size=chunk_size,
        chunks=chunks,
        aftercnn_per_chunk=aftercnn_per_chunk,
        aftercnn_total=aftercnn_total,
        window_aftercnn=window_aftercnn,
        cu_seqlens=tuple(int(x) for x in cu),
    )


class Qwen3AsrAudioEncoderOnnx(torch.nn.Module):
    """Wraps the Qwen3-ASR audio tower for ONNX export (mel features -> audio embeddings)."""

    def __init__(self, audio_tower: torch.nn.Module, spec: _AudioOeSpec):
        super().__init__()
        self.audio_tower = audio_tower
        self.spec = spec

        self.register_buffer(
            "_cu_seqlens",
            torch.tensor(spec.cu_seqlens, dtype=torch.int32),
            persistent=False,
        )
        attn = torch.full(
            (spec.aftercnn_total, spec.aftercnn_total),
            fill_value=torch.finfo(torch.float32).min,
            dtype=torch.float32,
        )
        for i in range(len(spec.cu_seqlens) - 1):
            s = spec.cu_seqlens[i]
            e = spec.cu_seqlens[i + 1]
            attn[s:e, s:e] = 0.0
        self.register_buffer(
            "_attn_mask_4d",
            attn[None, None, :, :],
            persistent=False,
        )

        if hasattr(self.audio_tower, "config") and hasattr(
            self.audio_tower.config,
            "_attn_implementation",
        ):
            self.audio_tower.config._attn_implementation = "eager"

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run convs, positional encoding, transformer layers, and projection."""
        x = input_features[:, :, : self.spec.n_frames]
        x = x.reshape(1, self.spec.n_mels, self.spec.chunks, self.spec.chunk_size)
        x = x.permute(0, 2, 1, 3).reshape(
            self.spec.chunks,
            self.spec.n_mels,
            self.spec.chunk_size,
        )

        x = x.unsqueeze(1)
        x = torch.nn.functional.gelu(self.audio_tower.conv2d1(x))
        x = torch.nn.functional.gelu(self.audio_tower.conv2d2(x))
        x = torch.nn.functional.gelu(self.audio_tower.conv2d3(x))
        b, c, f, t = x.size()
        x = (
            x.permute(0, 3, 1, 2)
            .contiguous()
            .view(b, t, c * f)
        )
        x = self.audio_tower.conv_out(x)

        pos = self.audio_tower.positional_embedding.positional_embedding[: x.shape[1], :]
        pos = pos.unsqueeze(0).to(x.dtype)
        x = x + pos
        x = x.reshape(-1, x.shape[-1]).contiguous()

        cu_seqlens = self._cu_seqlens.to(x.device)
        attn_mask_4d = self._attn_mask_4d.to(x.device, x.dtype)
        for layer in self.audio_tower.layers:
            x = layer(x, cu_seqlens=cu_seqlens, attention_mask=attn_mask_4d)[0]

        x = self.audio_tower.ln_post(x)
        x = self.audio_tower.proj1(x)
        x = self.audio_tower.act(x)
        x = self.audio_tower.proj2(x)
        return x.unsqueeze(0)


class Qwen3AsrTextDecoderOnnx(torch.nn.Module):
    """Wraps the thinker (decoder + lm_head) for ONNX export with fused audio embeddings."""

    def __init__(self, thinker: torch.nn.Module):
        super().__init__()
        self.thinker = thinker
        self.audio_token_id = int(thinker.config.audio_token_id)

        if hasattr(self.thinker.model, "config") and hasattr(
            self.thinker.model.config,
            "_attn_implementation",
        ):
            self.thinker.model.config._attn_implementation = "eager"

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_features: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter audio features into token embeddings and return logits."""
        inputs_embeds = self.thinker.get_input_embeddings()(input_ids)
        audio_mask = (input_ids == self.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        hidden_states = inputs_embeds
        position_embeddings = self.thinker.model.rotary_emb(hidden_states, position_ids)
        text_position_ids = position_ids[0]

        for layer in self.thinker.model.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
            )

        hidden_states = self.thinker.model.norm(hidden_states)
        return self.thinker.lm_head(hidden_states)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


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


def main():
    """CLI entry: load Qwen3-ASR, trace wrappers, and write encoder/decoder ONNX files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="./Qwen3-ASR-1.7B")
    ap.add_argument("--output-dir", type=str, default="./onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    _ensure_dir(args.output_dir)

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, fix_mistral_regex=True)
    _ensure_chat_template(tokenizer, model_path=args.model_path)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
    )
    model.eval()

    spec = _build_audio_spec(
        n_mels=int(getattr(feature_extractor, "feature_size", 128)),
        n_frames=int(getattr(feature_extractor, "nb_max_frames", 3000)),
        n_window=int(getattr(model.thinker.audio_tower, "n_window", 100)),
        n_window_infer=int(getattr(model.thinker.audio_tower, "n_window_infer", 400)),
    )
    audio_token_len = spec.aftercnn_total

    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    base = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    audio_token = getattr(tokenizer, "audio_token", "<|audio_pad|>")
    prompt = base.replace(audio_token, audio_token * audio_token_len, 1)

    tok = tokenizer(prompt, return_tensors="pt", padding=False)
    input_ids = tok["input_ids"].to(torch.long)

    wav = np.zeros((int(getattr(feature_extractor, "n_samples", 480000)),), dtype=np.float32)
    fe = feature_extractor(wav, sampling_rate=16000, return_attention_mask=True)
    input_features = torch.tensor(fe["input_features"], dtype=torch.float32)

    audio_encoder = Qwen3AsrAudioEncoderOnnx(model.thinker.audio_tower, spec)
    text_decoder = Qwen3AsrTextDecoderOnnx(model.thinker)

    with torch.no_grad():
        audio_features = audio_encoder(input_features)
        seq = input_ids.shape[1]
        causal = torch.triu(torch.ones((seq, seq), dtype=torch.bool), diagonal=1)
        attn = torch.zeros((seq, seq), dtype=torch.float32)
        attn = attn.masked_fill(causal, -1e4)
        attn = attn[None, None, :, :]
        pos = torch.arange(seq, dtype=torch.long).view(1, -1).expand(1, -1)
        pos = pos.unsqueeze(0).expand(3, -1, -1)
        _ = text_decoder(input_ids, audio_features, attn, pos)

    audio_onnx_path = os.path.join(args.output_dir, "qwen3_asr_audio_encoder_fp32.onnx")
    text_onnx_path = os.path.join(args.output_dir, "qwen3_asr_text_decoder_fp32.onnx")

    torch.onnx.export(
        audio_encoder,
        (input_features,),
        audio_onnx_path,
        input_names=["input_features"],
        output_names=["audio_features"],
        opset_version=int(args.opset),
        dynamo=False,
        do_constant_folding=True,
    )

    torch.onnx.export(
        text_decoder,
        (input_ids, audio_features, attn, pos),
        text_onnx_path,
        input_names=["input_ids", "audio_features", "attention_mask", "position_ids"],
        output_names=["logits"],
        opset_version=int(args.opset),
        dynamo=False,
        do_constant_folding=True,
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "attention_mask": {2: "seq_len", 3: "seq_len"},
            "position_ids": {2: "seq_len"},
            "logits": {1: "seq_len"},
        },
        external_data=True,
    )

    print(f"Exported:\n- {audio_onnx_path}\n- {text_onnx_path}")

if __name__ == "__main__":
    main()
