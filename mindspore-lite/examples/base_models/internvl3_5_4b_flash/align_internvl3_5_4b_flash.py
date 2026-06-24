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
"""Accuracy alignment for InternVL3.5-4B-Flash: HF reference vs MindSpore Lite.

The HF reference (model ``chat`` answer + prefill next-token logits over the fused
multimodal embeds) is computed with the transformers-compatible export env. The
MindSpore Lite cosine check runs only if ``mindspore_lite`` is importable in the
same process; otherwise the dumps are written and the check can be run separately
in the runtime env. Run this in the export conda env (transformers 4.51.x) for the
HF side.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PREFILL_SEQ = 320


def _preprocess(image_path, size=448):
    """Resize to a size square and normalize with CLIP mean/std -> [1,3,size,size]."""
    image = Image.open(image_path).convert("RGB").resize((size, size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[None, :].astype(np.float32)


def _hf_reference(model_dir, image_path, question, max_new_tokens, dump_dir):
    """Run HF InternVL ``chat`` + dump prefill next-token logits for cosine check."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float32,
                                      trust_remote_code=True, low_cpu_mem_usage=True).eval()
    pixel_values = torch.from_numpy(_preprocess(image_path))
    answer = model.chat(tok, pixel_values, question,
                        {"do_sample": False, "max_new_tokens": max_new_tokens})
    print(f"[align] HF answer: {answer!r}")

    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    query = f"<image>\n{question}"
    text = tok.apply_chat_template([{"role": "user", "content": query}],
                                   tokenize=False, add_generation_prompt=True)
    text = text.replace("<image>", "<IMG_CONTEXT>" * 256)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        img_emb = model.extract_feature(pixel_values)[0]
    fused = model.language_model.get_input_embeddings().weight[ids[0]].clone()
    sel = (ids[0] == model.img_context_token_id).numpy()
    fused[sel] = img_emb
    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(dump_dir) / "hf_fused_embeds.npy", fused.float().detach().numpy())
    with torch.no_grad():
        logits = model.language_model(
            inputs_embeds=fused[None],
            attention_mask=torch.ones(1, ids.shape[1], dtype=torch.int64),
            position_ids=torch.arange(ids.shape[1])[None]).logits[0, -1]
    np.save(Path(dump_dir) / "hf_first_logits.npy", logits.float().detach().numpy())
    top = int(torch.argmax(logits))
    print(f"[align] HF first token: {top} {tok.decode([top])!r}")
    return answer


def _mslite_cosine(mindir_dir, dump_dir, seq_len=PREFILL_SEQ):
    """Load HF dumped logits/embeds and the MSLite prefill, report next-token cosine."""
    import mindspore_lite as mslite
    hf_logits = np.load(Path(dump_dir) / "hf_first_logits.npy")
    hf_fused = np.load(Path(dump_dir) / "hf_fused_embeds.npy")
    ctx = mslite.Context()
    ctx.target = ["ascend"]
    ctx.ascend.device_id = 0
    model = mslite.Model()
    model.build_from_file(str(Path(mindir_dir) / "internvl_llm_prefill_graph.mindir"),
                          mslite.ModelType.MINDIR, ctx)
    n = hf_fused.shape[1]
    fused = np.concatenate([hf_fused, np.zeros((1, seq_len - n, hf_fused.shape[2]), np.float32)], axis=1)
    am = np.concatenate([np.ones((1, n), np.int32), np.zeros((1, seq_len - n), np.int32)], axis=1)
    pos = np.arange(seq_len, dtype=np.int32)[None, :]
    out = model.predict([mslite.Tensor(fused.astype(np.float32)), mslite.Tensor(am), mslite.Tensor(pos)])
    ms_logits = out[0].get_data_to_numpy()[0, n - 1]
    cos = float((ms_logits * hf_logits).sum() / (np.linalg.norm(ms_logits) * np.linalg.norm(hf_logits) + 1e-9))
    print(f"[align] MSLite prefill next-token cosine vs HF = {cos:.5f} "
          f"(MS argmax={int(ms_logits.argmax())}, HF argmax={int(hf_logits.argmax())})")


def main():
    """Parse arguments and run HF alignment (+ optional MSLite cosine check)."""
    parser = argparse.ArgumentParser(description="InternVL3.5-4B-Flash HF vs MSLite alignment")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Describe this image in detail.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--mindir-dir", default="./internvl3_5_4b_flash_onnx")
    parser.add_argument("--dump-dir", default="./align_dump")
    parser.add_argument("--check-mslite", action="store_true",
                        help="also run the MSLite prefill cosine check (needs mindspore_lite)")
    args = parser.parse_args()

    _hf_reference(args.model_dir, args.image, args.prompt, args.max_new_tokens, args.dump_dir)
    if args.check_mslite:
        try:
            _mslite_cosine(args.mindir_dir, args.dump_dir)
        except ImportError:
            print("[align] mindspore_lite not importable in this env; "
                  "re-run with --check-mslite in the runtime env.")


if __name__ == "__main__":
    main()
