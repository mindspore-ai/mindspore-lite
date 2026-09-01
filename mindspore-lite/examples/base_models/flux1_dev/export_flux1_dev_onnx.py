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
"""Export FLUX.1-dev components to fixed-shape, standard-operator ONNX."""

import argparse
import gc
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKL, FluxTransformer2DModel
from diffusers.models.attention_processor import AttnProcessor
from diffusers.models.transformers import transformer_flux
from transformers import CLIPTextModel, T5EncoderModel

from flux1_utils import CLIP_SEQUENCE_LENGTH, FluxShape


_OPSET = 18


@dataclass(frozen=True)
class ExportConfig:
    """Locations and fixed shapes shared by component exporters."""

    model_dir: Path
    output_dir: Path
    shape: FluxShape
    device: torch.device


class _TransformerWrapper(torch.nn.Module):
    """Expose only the FLUX noise prediction tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, encoder_hidden_states, pooled_projections,
                timestep, img_ids, txt_ids, guidance):
        """Run the transformer with the positional input order used by ONNX."""
        return self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]


class _VaeDecoderWrapper(torch.nn.Module):
    """Expose only the decoded image while retaining the VAE as a submodule."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        """Decode a latent tensor."""
        return self.vae.decode(latents, return_dict=False)[0]


class _T5Wrapper(torch.nn.Module):
    """Expose only the T5 last hidden state."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        """Encode fixed-length T5 token IDs."""
        return self.model(input_ids=input_ids, return_dict=False)[0]


class _ClipWrapper(torch.nn.Module):
    """Expose only the CLIP pooled prompt embedding."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        """Encode CLIP token IDs and return pooler output."""
        return self.model(input_ids=input_ids, return_dict=False)[1]


def _patch_rmsnorm():
    """Decompose torch RMSNorm into converter-friendly standard operators."""

    def _forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + float(self.eps))
        if getattr(self, "weight", None) is not None:
            return hidden_states.to(self.weight.dtype) * self.weight
        return hidden_states.to(input_dtype)

    torch.nn.RMSNorm.forward = _forward


def _patch_layernorm():
    """Decompose LayerNorm because the Atlas 300I Duo GE builder cannot infer its ONNX op."""

    def _forward(self, inputs):
        input_dtype = inputs.dtype
        axes = tuple(range(-len(self.normalized_shape), 0))
        values = inputs.float()
        mean = values.mean(axes, keepdim=True)
        centered = values - mean
        variance = (centered * centered).mean(axes, keepdim=True)
        values = centered * torch.rsqrt(variance + float(self.eps))
        if getattr(self, "weight", None) is not None:
            values = values * self.weight.float()
        if getattr(self, "bias", None) is not None:
            values = values + self.bias.float()
        return values.to(input_dtype)

    torch.nn.LayerNorm.forward = _forward


def _patch_flux_attention():
    """Replace SDPA dispatch with full standard-op attention in BNSD layout."""

    def _dispatch(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None, enable_gqa=False,
                  attention_kwargs=None, *, backend=None, parallel_config=None):
        del attn_mask, dropout_p, is_causal, enable_gqa, attention_kwargs
        del backend, parallel_config
        head_dim = int(value.shape[-1])
        scale_value = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale_value
        probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        return torch.matmul(probs, value).transpose(1, 2)

    transformer_flux.dispatch_attention_fn = _dispatch


def _patch_flux_pos_embed():
    """Keep rotary-frequency math in float32 for ONNX and Ascend compatibility."""

    def _forward(self, ids):
        position = ids.float()
        cosines = []
        sines = []
        for axis in range(ids.shape[-1]):
            cosine, sine = transformer_flux.get_1d_rotary_pos_embed(
                self.axes_dim[axis],
                position[:, axis:axis + 1].reshape(-1),
                theta=self.theta,
                repeat_interleave_real=False,
                use_real=True,
                freqs_dtype=torch.float32,
            )
            token_count, width = cosine.shape
            cosine = cosine.reshape(token_count, 2, width // 2)
            cosine = cosine.transpose(1, 2).reshape(token_count, width)
            sine = sine.reshape(token_count, 2, width // 2)
            sine = sine.transpose(1, 2).reshape(token_count, width)
            cosines.append(cosine)
            sines.append(sine)
        return torch.cat(cosines, dim=-1), torch.cat(sines, dim=-1)

    transformer_flux.FluxPosEmbed.forward = _forward


def _freeze(model):
    """Freeze and place a component in inference mode."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _clear_cache():
    """Release component references between the four large exports."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _export(wrapper, inputs, names, output_name, path):
    """Export one fixed-shape graph with external tensor data enabled."""
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper.eval()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            tuple(inputs),
            str(path),
            input_names=names,
            output_names=[output_name],
            opset_version=_OPSET,
            dynamo=True,
            external_data=True,
        )
    print(f"[export] saved {path}")


def export_transformer(config):
    """Export the 12B FLUX denoiser at the configured fixed image shape."""
    _patch_rmsnorm()
    _patch_layernorm()
    _patch_flux_attention()
    _patch_flux_pos_embed()
    model = FluxTransformer2DModel.from_pretrained(
        config.model_dir, subfolder="transformer", torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(config.device)
    wrapper = _TransformerWrapper(_freeze(model))
    shape = config.shape
    dtype = torch.float32
    device = config.device
    inputs = (
        torch.randn(1, shape.image_sequence_length, 64, dtype=dtype, device=device),
        torch.randn(1, shape.t5_sequence_length, 4096, dtype=dtype, device=device),
        torch.randn(1, 768, dtype=dtype, device=device),
        torch.tensor([0.5], dtype=dtype, device=device),
        torch.zeros(shape.image_sequence_length, 3, dtype=dtype, device=device),
        torch.zeros(shape.t5_sequence_length, 3, dtype=dtype, device=device),
        torch.tensor([3.5], dtype=dtype, device=device),
    )
    names = ["hidden_states", "encoder_hidden_states", "pooled_projections",
             "timestep", "img_ids", "txt_ids", "guidance"]
    path = config.output_dir / "flux1_transformer.onnx"
    _export(wrapper, inputs, names, "noise_pred", path)
    del wrapper, model, inputs
    _clear_cache()


def export_vae(config):
    """Export the FLUX AutoencoderKL decoder."""
    vae = AutoencoderKL.from_pretrained(
        config.model_dir, subfolder="vae", torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(config.device)
    # PyTorch SDPA exports a four-dimensional Reshape whose int64 shape input is
    # misread by the Atlas 300I Duo GE builder. The legacy processor spells the same
    # single-head VAE attention as standard BMM + Softmax + BMM operations.
    vae.set_attn_processor(AttnProcessor())
    wrapper = _VaeDecoderWrapper(_freeze(vae))
    shape = config.shape
    latents = torch.randn(
        1, 16, shape.latent_height, shape.latent_width,
        dtype=torch.float32, device=config.device,
    )
    path = config.output_dir / "flux1_vae_decoder.onnx"
    _export(wrapper, (latents,), ["latents"], "image", path)
    del wrapper, vae, latents
    _clear_cache()


def export_t5(config):
    """Export the FLUX T5-XXL text encoder."""
    model = T5EncoderModel.from_pretrained(
        config.model_dir / "text_encoder_2", torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(config.device)
    wrapper = _T5Wrapper(_freeze(model))
    ids = torch.zeros(
        (1, config.shape.t5_sequence_length), dtype=torch.int64,
        device=config.device,
    )
    path = config.output_dir / "flux1_t5_encoder.onnx"
    _export(wrapper, (ids,), ["input_ids"], "last_hidden_state", path)
    del wrapper, model, ids
    _clear_cache()


def export_clip(config):
    """Export the FLUX CLIP-L pooled text encoder."""
    _patch_layernorm()
    model = CLIPTextModel.from_pretrained(
        config.model_dir / "text_encoder", torch_dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager",
    ).to(config.device)
    wrapper = _ClipWrapper(_freeze(model))
    ids = torch.zeros(
        (1, CLIP_SEQUENCE_LENGTH), dtype=torch.int64, device=config.device,
    )
    path = config.output_dir / "flux1_clip_encoder.onnx"
    _export(wrapper, (ids,), ["input_ids"], "pooled_projections", path)
    del wrapper, model, ids
    _clear_cache()


def _parse_args():
    """Parse export options."""
    parser = argparse.ArgumentParser(description="Export FLUX.1-dev components to ONNX")
    parser.add_argument("--model-dir", required=True, help="local FLUX.1-dev Diffusers directory")
    parser.add_argument("--output-dir", default="./flux1_dev_onnx", help="ONNX output directory")
    parser.add_argument("--parts", default="transformer,vae,t5,clip",
                        help="comma-separated subset: transformer,vae,t5,clip")
    parser.add_argument("--height", type=int, default=512, help="fixed output height")
    parser.add_argument("--width", type=int, default=512, help="fixed output width")
    parser.add_argument("--t5-seq-len", type=int, default=256, help="fixed T5 sequence length")
    parser.add_argument(
        "--device", default="cpu", help="PyTorch export device, for example cpu or cuda:0",
    )
    return parser.parse_args()


def main():
    """Export the requested FLUX.1-dev sub-models one at a time."""
    args = _parse_args()
    shape = FluxShape(args.height, args.width, args.t5_seq_len)
    config = ExportConfig(
        Path(args.model_dir), Path(args.output_dir), shape, torch.device(args.device),
    )
    exporters = {
        "transformer": export_transformer,
        "vae": export_vae,
        "t5": export_t5,
        "clip": export_clip,
    }
    parts = [part.strip() for part in args.parts.split(",") if part.strip()]
    unknown = sorted(set(parts) - set(exporters))
    if unknown:
        raise ValueError(f"unknown components: {', '.join(unknown)}")
    print(f"[export] fixed shape: {shape.height}x{shape.width}, "
          f"image tokens={shape.image_sequence_length}, T5 tokens={shape.t5_sequence_length}")
    for part in parts:
        print(f"[export] starting {part}")
        exporters[part](config)


if __name__ == "__main__":
    main()
