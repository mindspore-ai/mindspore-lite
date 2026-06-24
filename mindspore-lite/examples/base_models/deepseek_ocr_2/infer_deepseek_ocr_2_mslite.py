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
Infer DeepSeek-OCR-2 on Ascend with MindSpore Lite (vision + prefill + decode).

Pure numpy / PIL pipeline (no torch). Mirrors the original model.infer() image
preprocessing (global 1024x1024 view + N local 768x768 crops, normalize 0.5/0.5)
and prompt layout (image tokens 128815, plain chat template).

NOTE: the export used here flattens the MoE and fixes the crop count; this script
must pass the same --n-crops / --image-size used at export time.
"""

import argparse
import sys
import time
import urllib.request
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

try:
    import mindspore_lite as mslite
    from transformers import AutoTokenizer
except ImportError:
    print("Error: mindspore_lite or transformers package not found.")
    sys.exit(1)


IMAGE_TOKEN_ID = 128815
BOS_ID = 0
EOS_ID = 1
BASE_SIZE = 1024          # global view size
CROP_SIZE = 768           # local crop size
PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 4
NUM_QUERIES = 12          # (CROP_SIZE // PATCH_SIZE) // DOWNSAMPLE_RATIO  -> 12 (=> 144 tokens/crop)
NUM_QUERIES_BASE = 16     # (BASE_SIZE // PATCH_SIZE) // DOWNSAMPLE_RATIO  -> 16 (=> 256 tokens/global)


# ---------------------------------------------------------------------------
# Image loading + preprocessing (matches DeepseekOCR2ForCausalLM.infer).
# ---------------------------------------------------------------------------


def _load_image(path):
    """Load an image from path/URL and apply EXIF orientation."""
    if path.startswith("http://") or path.startswith("https://"):
        with urllib.request.urlopen(path) as resp:
            data = resp.read()
        return Image.open(BytesIO(data)).convert("RGB")
    return Image.open(path).convert("RGB")


def _to_norm_tensor(img):
    """PIL -> numpy (1,3,H,W) normalized by (x/255 - 0.5)/0.5, matching BasicImageTransform."""
    arr = np.asarray(img, dtype=np.float32) / 255.0      # (H,W,3)
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))[None, ...]         # (1,3,H,W)
    return arr


def _find_closest_aspect_ratio(aspect, min_num, max_num, h, w, image_size):
    """Replicate dynamic_preprocess aspect-ratio selection."""
    best, best_diff = (1, 1), float("inf")
    area = h * w
    for i in range(min_num, max_num + 1):
        for j in range(1, i + 1):
            if i * j > max_num or i * j < min_num:
                continue
            r = j / i
            d = abs(aspect - r)
            if d < best_diff or (d == best_diff and area > 0.5 * image_size * image_size * j * i):
                best_diff, best = d, (j, i)
    return best


def _dynamic_preprocess(image, min_num=2, max_num=6, image_size=CROP_SIZE):
    """Split image into min_num..max_num crops of image_size x image_size (matches the model)."""
    w, h = image.size
    aspect = w / h
    cw, ch = _find_closest_aspect_ratio(aspect, min_num, max_num, h, w, image_size)
    target_w, target_h = image_size * cw, image_size * ch
    resized = image.resize((target_w, target_h))
    crops = []
    for i in range(cw * ch):
        bx = (i % cw) * image_size
        by = (i // cw) * image_size
        crops.append(resized.crop((bx, by, bx + image_size, by + image_size)))
    return crops, (cw, ch)


def _prepare_image(image, n_crops):
    """Return (global_tensor (1,3,1024,1024), crops_tensor (N,3,768,768), crop_count)."""
    global_view = ImageOps.pad(image, (BASE_SIZE, BASE_SIZE), color=(128, 128, 128))
    global_t = _to_norm_tensor(global_view)              # (1,3,1024,1024)
    if min(image.size) <= CROP_SIZE:
        crops, _ = [], (1, 1)
    else:
        crops, (cw, ch) = _dynamic_preprocess(image)
    # Pad / truncate crops to exactly n_crops to match the exported vision model.
    crops = crops[:n_crops] + [ImageOps.pad(image, (CROP_SIZE, CROP_SIZE), color=(128, 128, 128))] * max(0, n_crops - len(crops))
    crops_t = np.concatenate([_to_norm_tensor(c) for c in crops[:n_crops]], axis=0) if n_crops > 0 else \
        np.zeros((0, 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    return global_t.astype(np.float32), crops_t.astype(np.float32), len(crops[:n_crops])


# ---------------------------------------------------------------------------
# Prompt / token layout (matches model.infer plain template).
# ---------------------------------------------------------------------------


def _build_inputs(tokenizer, prompt_text, n_crops):
    """Build (input_ids, attention_mask) with image-token placeholder layout."""
    # image placeholder split: "<image>\n" is the marker; here we use the prompt with <image>.
    marker = "<image>"
    parts = prompt_text.split(marker)
    head_ids = tokenizer.encode(parts[0], add_special_tokens=False) if parts[0] else []
    tail_ids = tokenizer.encode(parts[1], add_special_tokens=False) if len(parts) > 1 and parts[1] else []
    # global(256) + sep(1) + local(144*n_crops)
    img_tokens = ([IMAGE_TOKEN_ID] * NUM_QUERIES_BASE) * NUM_QUERIES_BASE
    img_tokens += [IMAGE_TOKEN_ID]
    img_tokens += ([IMAGE_TOKEN_ID] * (NUM_QUERIES * NUM_QUERIES)) * n_crops
    ids = [BOS_ID] + head_ids + img_tokens + tail_ids
    input_ids = np.array([ids], dtype=np.int32)
    attention_mask = np.ones_like(input_ids)
    return input_ids, attention_mask


# ---------------------------------------------------------------------------
# MindSpore Lite helpers.
# ---------------------------------------------------------------------------


def _build_inputs_tensors(model, feed, order):
    """Build mslite input tensors by name (fallback to order)."""
    inputs = model.get_inputs()
    if inputs and all(getattr(t, "name", None) in feed for t in inputs):
        return [mslite.Tensor(feed[t.name]) for t in inputs]
    return [mslite.Tensor(feed[k]) for k in order]


class DeepseekOcrInferencer:
    """Run DeepSeek-OCR-2 end-to-end on MindSpore Lite (vision + prefill + decode)."""

    def __init__(self, vision_model, prefill_model, decode_model, tokenizer_dir,
                 device="ascend", device_id=0, n_crops=2, kv_cache_len=2048,
                 prefill_seq=1152):
        """Build the three MindIR models and load the tokenizer."""
        self.device = str(device)
        self.device_id = int(device_id)
        self.n_crops = int(n_crops)
        self.kv_cache_len = int(kv_cache_len)
        self.prefill_seq = int(prefill_seq)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

        self.context = mslite.Context()
        self.context.target = [self.device]
        if self.device == "ascend":
            self.context.ascend.device_id = self.device_id
        self.vision_model = mslite.Model()
        self.vision_model.build_from_file(vision_model, mslite.ModelType.MINDIR, self.context)
        self.prefill_model = mslite.Model()
        self.prefill_model.build_from_file(prefill_model, mslite.ModelType.MINDIR, self.context)
        self.decode_model = mslite.Model()
        self.decode_model.build_from_file(decode_model, mslite.ModelType.MINDIR, self.context)
        # Match decode past_key_values input dtype.
        self.kv_dtype = np.float16
        for _inp in self.decode_model.get_inputs():
            if getattr(_inp, "name", "") == "past_key_values":
                self.kv_dtype = np.float16 if str(_inp.dtype).endswith("FLOAT16") else np.float32
                break

    def _run_vision(self, global_t, crops_t):
        """Run vision model -> image_embeds (n_crops*144 + 257, 1280)."""
        feed = {"global_image": global_t, "crops": crops_t}
        out = self.vision_model.predict(
            _build_inputs_tensors(self.vision_model, feed, ["global_image", "crops"]))
        emb = out[0].get_data_to_numpy()
        return emb.astype(self.kv_dtype)

    def infer(self, image_path, prompt, max_new_tokens=256):
        """Full pipeline: preprocess -> vision -> prefill -> decode loop."""
        t0 = time.perf_counter()
        image = _load_image(image_path)
        global_t, crops_t, nc = _prepare_image(image, self.n_crops)
        input_ids, attention_mask = _build_inputs(self.tokenizer, prompt, nc)
        prompt_len = int(input_ids.shape[1])
        if prompt_len > self.prefill_seq:
            raise RuntimeError(f"prompt_len={prompt_len} > prefill_seq={self.prefill_seq}")

        # Pad to fixed prefill seq.
        pad = self.prefill_seq - prompt_len
        pad_id = int(self.tokenizer.pad_token_id or 0)
        input_ids = np.concatenate([input_ids, np.full((1, pad), pad_id, np.int32)], axis=1)
        attention_mask = np.concatenate([attention_mask, np.zeros((1, pad), np.int32)], axis=1)
        position_ids = np.arange(self.prefill_seq, dtype=np.int32)[None, :]

        image_embeds, = (self._run_vision(global_t, crops_t),)
        t_vis = time.perf_counter() - t0

        prefill_feed = {"input_ids": input_ids, "attention_mask": attention_mask,
                        "position_ids": position_ids, "image_embeds": image_embeds}
        tp0 = time.perf_counter()
        prefill_out = self.prefill_model.predict(
            _build_inputs_tensors(self.prefill_model, prefill_feed,
                                  ["input_ids", "attention_mask", "position_ids", "image_embeds"]))
        t_prefill = time.perf_counter() - tp0
        logits = prefill_out[0].get_data_to_numpy()
        past_kv = prefill_out[1].get_data_to_numpy()

        generated = [int(np.argmax(logits[0, prompt_len - 1]))]
        past_kv_fixed = np.zeros(
            (past_kv.shape[0], past_kv.shape[1], past_kv.shape[2], self.kv_cache_len, past_kv.shape[4]),
            dtype=self.kv_dtype)
        past_kv_fixed[:, :, :, : self.prefill_seq, :] = past_kv.astype(self.kv_dtype)
        attn = np.zeros((1, self.kv_cache_len), dtype=np.int32)
        cache_pos = prompt_len

        t_dec = 0.0
        steps = 0
        eos = int(self.tokenizer.eos_token_id or EOS_ID)
        for _ in range(max_new_tokens - 1):
            if generated[-1] == eos or cache_pos >= self.kv_cache_len:
                break
            attn[0, : cache_pos + 1] = 1
            step_id = np.array([[generated[-1]]], dtype=np.int32)
            step_pos = np.array([[cache_pos]], dtype=np.int32)
            cache_pos_arr = np.array([cache_pos], dtype=np.int32)
            feed = {"input_ids": step_id, "attention_mask": attn, "position_ids": step_pos,
                    "past_key_values": past_kv_fixed, "cache_pos": cache_pos_arr}
            td0 = time.perf_counter()
            dout = self.decode_model.predict(
                _build_inputs_tensors(self.decode_model, feed,
                                      ["input_ids", "attention_mask", "position_ids", "past_key_values", "cache_pos"]))
            t_dec += time.perf_counter() - td0
            past_kv_fixed = dout[1].get_data_to_numpy()
            generated.append(int(np.argmax(dout[0].get_data_to_numpy()[0, -1])))
            cache_pos += 1
            steps += 1

        t_e2e = time.perf_counter() - t0
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        timing = {"vision": t_vis * 1000, "prefill": t_prefill * 1000, "decode": t_dec * 1000,
                  "steps": steps, "e2e": t_e2e * 1000, "prompt_len": prompt_len}
        return text, timing


def main():
    """Parse args and run DeepSeek-OCR-2 inference on MindSpore Lite."""
    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 MindSpore Lite inference")
    parser.add_argument("--vision-model", required=True)
    parser.add_argument("--prefill-model", required=True)
    parser.add_argument("--decode-model", required=True)
    parser.add_argument("--tokenizer", required=True, help="Tokenizer / model dir")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="<image>\nFree OCR. ")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--n-crops", type=int, default=2, help="Must match export --n-crops")
    parser.add_argument("--kv-cache-len", type=int, default=2048)
    parser.add_argument("--prefill-seq", type=int, default=1152)
    parser.add_argument("--device", default="ascend", choices=["ascend", "cpu"])
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    inf = DeepseekOcrInferencer(
        args.vision_model, args.prefill_model, args.decode_model, args.tokenizer,
        device=args.device, device_id=args.device_id, n_crops=args.n_crops,
        kv_cache_len=args.kv_cache_len, prefill_seq=args.prefill_seq)
    text, tm = inf.infer(args.image, args.prompt, max_new_tokens=args.max_new_tokens)
    avg = tm["decode"] / tm["steps"] if tm["steps"] else 0.0
    print(f"\n--- Performance ---\n"
          f"  Vision:       {tm['vision']:.1f} ms\n"
          f"  Prefill:      {tm['prefill']:.1f} ms (prompt_len={tm['prompt_len']})\n"
          f"  Decode:       {tm['decode']:.1f} ms / {tm['steps']} steps (avg {avg:.1f} ms)\n"
          f"  Total:        {tm['e2e']:.1f} ms")
    print("\n" + "=" * 60 + f"\nResponse:\n{text}\n" + "=" * 60)


if __name__ == "__main__":
    main()
