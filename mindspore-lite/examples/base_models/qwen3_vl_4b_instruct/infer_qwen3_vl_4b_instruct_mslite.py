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
Infer Qwen3-VL-4B-Instruct on Ascend with MindSpore Lite.
"""

import sys
import argparse
import urllib.request
from io import BytesIO
import numpy as np
from PIL import Image

try:
    import mindspore_lite as mslite
    from transformers import AutoConfig, AutoProcessor
except ImportError:
    print("Error: mindspore_lite or transformers package not found.")
    print("Please install them first.")
    sys.exit(1)


def _load_image(image_path_or_url: str) -> Image.Image:
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith(
        "https://"
    ):
        with urllib.request.urlopen(image_path_or_url) as resp:
            data = resp.read()
        return Image.open(BytesIO(data)).convert("RGB")
    return Image.open(image_path_or_url).convert("RGB")


def _pad_to_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    out = Image.new("RGB", (side, side), (0, 0, 0))
    out.paste(image, ((side - w) // 2, (side - h) // 2))
    return out


def _get_vision_position_ids(start_position, grid_thw, spatial_merge_size, device):
    """
    Get vision position ids for Qwen3-VL-4B-Instruct.

    Args:
        start_position (int): Start position for vision position ids.
        grid_thw (torch.Tensor): Grid size (t, h, w) for vision position ids.
        spatial_merge_size (int): Spatial merge size for vision position ids.
        device (torch.device): Device for vision position ids.

    Returns:
        torch.Tensor: Vision position ids.
    """
    import torch

    llm_grid_t = int(grid_thw[0].item())
    llm_grid_h = int(grid_thw[1].item()) // spatial_merge_size
    llm_grid_w = int(grid_thw[2].item()) // spatial_merge_size
    image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
    position_width = torch.arange(
        start_position, start_position + llm_grid_w, device=device
    ).repeat(llm_grid_h * llm_grid_t)
    position_height = torch.arange(
        start_position, start_position + llm_grid_h, device=device
    ).repeat_interleave(llm_grid_w * llm_grid_t)
    position_temporal = torch.full(
        (image_seq_length,), start_position, device=device, dtype=torch.long
    )
    return torch.stack([position_temporal, position_height, position_width], dim=0)


def _get_rope_index(
    input_ids, mm_token_type_ids, image_grid_thw, attention_mask, spatial_merge_size
):
    """
    Get rope index for Qwen3-VL-4B-Instruct.

    Args:
        input_ids (torch.Tensor): Input ids.
        mm_token_type_ids (torch.Tensor): MM token type ids.
        image_grid_thw (torch.Tensor): Image grid (t, h, w) for rope index.
        attention_mask (torch.Tensor): Attention mask for rope index.
        spatial_merge_size (int): Spatial merge size for rope index.

    Returns:
        torch.Tensor: Rope index.
    """
    import torch

    bsz, seq_len = input_ids.shape
    position_ids = torch.zeros(
        (3, bsz, seq_len), dtype=torch.long, device=input_ids.device
    )
    mrope_position_deltas = []
    image_iter = iter(image_grid_thw) if image_grid_thw is not None else iter([])

    for b in range(bsz):
        cur_types = mm_token_type_ids[b]
        cur_mask = attention_mask[b].bool() if attention_mask is not None else None
        if cur_mask is not None:
            cur_types = cur_types[cur_mask]
        cur_types_list = cur_types.tolist()
        groups = []
        start = 0
        for i in range(1, len(cur_types_list) + 1):
            if i == len(cur_types_list) or cur_types_list[i] != cur_types_list[start]:
                groups.append((cur_types_list[start], start, i))
                start = i

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in groups:
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device)
                    .view(1, -1)
                    .expand(3, -1)
                    + current_pos
                )
                current_pos += text_len
            elif modality_type == 1:
                grid = next(image_iter)
                vision_pos = _get_vision_position_ids(
                    current_pos, grid, spatial_merge_size, device=input_ids.device
                )
                llm_pos_ids_list.append(vision_pos)
                current_pos += (
                    max(int(grid[1].item()), int(grid[2].item())) // spatial_merge_size
                )
            else:
                raise ValueError(
                    f"Unsupported modality_type in this tutorial: {modality_type}"
                )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if attention_mask is not None:
            position_ids[:, b, attention_mask[b].bool()] = llm_positions.to(
                position_ids.device
            )
        else:
            position_ids[:, b] = llm_positions.to(position_ids.device)

        n_tokens = (
            int(attention_mask[b].sum().item())
            if attention_mask is not None
            else seq_len
        )
        mrope_position_deltas.append(llm_positions.max() + 1 - n_tokens)

    mrope_position_deltas = torch.tensor(
        mrope_position_deltas, device=input_ids.device, dtype=torch.long
    ).unsqueeze(1)
    return position_ids, mrope_position_deltas


def _build_position_ids(
    cfg, input_ids, attention_mask, mm_token_type_ids, image_grid_thw
):
    """
    Build position ids for Qwen3-VL-4B-Instruct.

    Args:
        cfg (Qwen3VLConfig): Qwen3-VL-4B-Instruct configuration.
        input_ids (torch.Tensor): Input ids.
        attention_mask (torch.Tensor): Attention mask.
        mm_token_type_ids (torch.Tensor): MM token type ids.
        image_grid_thw (torch.Tensor): Image grid (t, h, w) for position ids.

    Returns:
        torch.Tensor: Position ids.
    """
    import torch

    position_ids_3, rope_deltas = _get_rope_index(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
        spatial_merge_size=cfg.vision_config.spatial_merge_size,
    )
    text_pos = attention_mask.long().cumsum(-1) - 1
    text_pos = text_pos.masked_fill(attention_mask == 0, 0)
    position_ids_4 = torch.cat([text_pos.unsqueeze(0), position_ids_3], dim=0).to(
        torch.long
    )
    return position_ids_4, rope_deltas


def _force_processor_image_size(processor, image_size: int):
    if hasattr(processor, "image_processor") and hasattr(
        processor.image_processor, "size"
    ):
        size_pixels = int(image_size) * int(image_size)
        processor.image_processor.size = {
            "shortest_edge": size_pixels,
            "longest_edge": size_pixels,
        }


def _mslite_tensor(np_array: np.ndarray) -> mslite.Tensor:
    return mslite.Tensor(np_array)


def _build_mslite_inputs(model: mslite.Model, feed_dict, preferred_order=None):
    """
    Build MindSpore Lite inputs for Qwen3-VL-4B-Instruct.

    Args:
        model (mslite.Model): MindSpore Lite model.
        feed_dict (dict): Feed dictionary.
        preferred_order (list, optional): Preferred order of inputs. Defaults to None.

    Returns:
        list: MindSpore Lite tensors.
    """
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
        "input mismatch. model inputs={[getattr(x, 'name', '') for x in inputs]} feed keys={list(feed_dict.keys())}"
    )


class Qwen3VLInferencer:
    """
    Qwen3-VL-4B-Instruct inferencer.
    """

    def __init__(
        self,
        vision_model_path: str,
        prefill_model_path: str,
        decode_model_path: str,
        processor_id: str,
        device: str = "ascend",
        device_id: int = 0,
        image_size: int = 128,
    ):
        """
        Initialize Qwen3-VL-4B-Instruct inferencer.
        """
        if device not in ["cpu", "ascend"]:
            raise ValueError("device must be cpu or ascend")

        if device == "ascend":
            print("Initializing MindSpore Lite context for Ascend...")
        else:
            print("Initializing MindSpore Lite context for CPU...")

        self.context = mslite.Context()
        self.context.target = [device]
        if device == "ascend":
            self.context.ascend.device_id = device_id
            self.context.ascend.precision_mode = "enforce_fp16"

        # Load vision model
        print(f"Loading vision model from {vision_model_path}...")
        self.vision_model = mslite.Model()
        self.vision_model.build_from_file(
            vision_model_path, mslite.ModelType.MINDIR, self.context
        )

        # Load prefill model
        print(f"Loading prefill model from {prefill_model_path}...")
        self.prefill_model = mslite.Model()
        self.prefill_model.build_from_file(
            prefill_model_path, mslite.ModelType.MINDIR, self.context
        )

        # Load decode model
        print(f"Loading decode model from {decode_model_path}...")
        self.decode_model = mslite.Model()
        self.decode_model.build_from_file(
            decode_model_path, mslite.ModelType.MINDIR, self.context
        )

        # Load processor
        print(f"Loading processor from {processor_id}...")
        self.cfg = AutoConfig.from_pretrained(processor_id)
        self.processor = AutoProcessor.from_pretrained(processor_id)
        _force_processor_image_size(self.processor, image_size)

    def _prepare_inputs(self, image_path_or_url: str, prompt: str):
        """
        Prepare inputs for Qwen3-VL-4B-Instruct.

        Args:
            image_path_or_url (str): Path or URL to the image.
            prompt (str): Prompt for the image.

        Returns:
            tuple: Input ids, attention mask, mm token type ids, pixel values, image grid thw.
        """
        import torch

        image = _pad_to_square(_load_image(image_path_or_url))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(torch.long)
        attention_mask = inputs.attention_mask.to(torch.long)
        mm_token_type_ids = inputs.mm_token_type_ids.to(torch.int64)
        pixel_values = inputs.pixel_values.to(torch.float16)
        image_grid_thw = inputs.image_grid_thw.to(torch.long)
        return (
            input_ids,
            attention_mask,
            mm_token_type_ids,
            pixel_values,
            image_grid_thw,
        )

    def infer(
        self, image_path_or_url: str, text_prompt: str, max_new_tokens: int = 128
    ):
        """
        Infer Qwen3-VL-4B-Instruct on MindSpore Lite.
        """
        input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw = (
            self._prepare_inputs(image_path_or_url, text_prompt)
        )
        position_ids_4, rope_deltas = _build_position_ids(
            self.cfg, input_ids, attention_mask, mm_token_type_ids, image_grid_thw
        )

        pixel_values_np = pixel_values.cpu().numpy()
        image_grid_thw_np = image_grid_thw.cpu().numpy()

        vision_inputs = self.vision_model.get_inputs()
        if len(vision_inputs) == 1:
            feed = {"pixel_values": pixel_values_np}
            preferred = ["pixel_values"]
        else:
            feed = {"pixel_values": pixel_values_np, "grid_thw": image_grid_thw_np}
            preferred = ["pixel_values", "grid_thw"]
        vision_out = self.vision_model.predict(
            _build_mslite_inputs(self.vision_model, feed, preferred_order=preferred)
        )
        image_embeds = vision_out[0].get_data_to_numpy()
        deepstack_embeds = vision_out[1].get_data_to_numpy()

        image_token_cnt = int((input_ids == int(self.cfg.image_token_id)).sum().item())
        if int(image_embeds.shape[0]) != image_token_cnt:
            raise RuntimeError(
                f"image_embeds length mismatch: embeds={image_embeds.shape[0]} vs image_token_cnt={image_token_cnt}. "
                f"grid_thw={image_grid_thw_np.tolist()}"
            )

        prefill_feed = {
            "input_ids": input_ids.cpu().numpy().astype(np.int32),
            "attention_mask": attention_mask.cpu().numpy().astype(np.int32),
            "position_ids": position_ids_4.cpu().numpy().astype(np.int32),
            "image_embeds": image_embeds.astype(np.float16),
            "deepstack_embeds": deepstack_embeds.astype(np.float16),
        }
        prefill_out = self.prefill_model.predict(
            _build_mslite_inputs(
                self.prefill_model,
                prefill_feed,
                preferred_order=[
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "image_embeds",
                    "deepstack_embeds",
                ],
            )
        )
        logits = prefill_out[0].get_data_to_numpy()
        past_kv = prefill_out[1].get_data_to_numpy()

        eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        generated = []
        next_token = int(np.argmax(logits[0, -1]))
        generated.append(next_token)

        attn_mask_np = attention_mask.cpu().numpy().astype(np.int32)
        rope_deltas_np = rope_deltas.cpu().numpy().astype(np.int32)

        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and generated[-1] == int(eos_token_id):
                break

            step_id = np.array([[generated[-1]]], dtype=np.int32)
            attn_mask_np = np.concatenate(
                [attn_mask_np, np.ones((1, 1), dtype=np.int32)], axis=1
            )
            total_len = int(attn_mask_np.shape[1])

            text_pos_step = np.array([[[total_len - 1]]], dtype=np.int32)
            mm_pos_step = (text_pos_step + rope_deltas_np.reshape(1, 1, 1)).repeat(
                3, axis=0
            )
            position_ids_step = np.concatenate(
                [text_pos_step, mm_pos_step], axis=0
            ).astype(np.int32)

            decode_feed = {
                "input_ids": step_id,
                "attention_mask": attn_mask_np,
                "position_ids": position_ids_step,
                "past_key_values": past_kv.astype(np.float16),
            }
            decode_out = self.decode_model.predict(
                _build_mslite_inputs(
                    self.decode_model,
                    decode_feed,
                    preferred_order=[
                        "input_ids",
                        "attention_mask",
                        "position_ids",
                        "past_key_values",
                    ],
                )
            )
            logits = decode_out[0].get_data_to_numpy()
            past_kv = decode_out[1].get_data_to_numpy()
            generated.append(int(np.argmax(logits[0, -1])))

        return self.processor.tokenizer.decode(generated, skip_special_tokens=True)


def main():
    """
    Main function for Qwen3-VL-4B-Instruct inference on Ascend with MindSpore Lite.
    """
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-4B-Instruct MindSpore Lite inference (vision + prefill + decode)"
    )
    parser.add_argument(
        "--vision-model", type=str, required=True, help="Path to qwen3_vl_vision.mindir"
    )
    parser.add_argument(
        "--prefill-model",
        type=str,
        required=True,
        help="Path to qwen3_vl_llm_prefill.mindir",
    )
    parser.add_argument(
        "--decode-model",
        type=str,
        required=True,
        help="Path to qwen3_vl_llm_decode.mindir",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default="../Qwen/Qwen3-VL-4B-Instruct",
        help="Processor ID or local path",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        help="Image URL or path (http/https or local path)",
    )
    parser.add_argument(
        "--prompt", type=str, default="Describe this image.", help="Text prompt"
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Force processor image size (must match exported vision model)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="ascend",
        choices=["ascend", "cpu"],
        help="MindSpore Lite target device",
    )
    parser.add_argument("--device-id", type=int, default=0, help="Ascend device ID")

    args = parser.parse_args()

    inferencer = Qwen3VLInferencer(
        vision_model_path=args.vision_model,
        prefill_model_path=args.prefill_model,
        decode_model_path=args.decode_model,
        processor_id=args.processor,
        device=args.device,
        device_id=args.device_id,
        image_size=args.image_size,
    )
    result = inferencer.infer(
        args.image, args.prompt, max_new_tokens=args.max_new_tokens
    )

    print("\n" + "=" * 50)
    print(f"Input Prompt: {args.prompt}")
    print(f"Generated Response: {result}")
    print("=" * 50)


if __name__ == "__main__":
    main()
