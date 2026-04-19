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
Infer Qwen3 VL Embedding 2B model with MindSpore Lite.
"""

import argparse
import os
from typing import Dict, List, Optional

import numpy as np

try:
    import mindspore_lite as mslite
except ImportError:
    print("Error: mindspore_lite package not found.")
    print(
        "Please install: pip install mindspore-lite (or install the wheel built from MindSpore Lite)"
    )
    raise

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from transformers import AutoProcessor
except ImportError:
    print("Error: transformers package not found.")
    print("Please install: pip install transformers")
    raise


def _load_image(path: str):
    """
    Load image from path.
    """
    if Image is None:
        raise RuntimeError("Pillow not installed. Please install: pip install pillow")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _create_context(device: str, device_id: int) -> mslite.Context:
    """
    Create MindSpore Lite context.
    """
    context = mslite.Context()
    context.target = [device]
    if device == "ascend":
        context.ascend.device_id = int(device_id)
        context.ascend.precision_mode = "preferred_fp16"
    return context


def _describe_model_io(model: mslite.Model):
    """
    Describe model inputs and outputs.
    """
    inputs = model.get_inputs()
    outputs = model.get_outputs()
    print("Model Inputs:")
    for t in inputs:
        name = getattr(t, "name", "")
        shape = getattr(t, "shape", None)
        dtype = getattr(t, "data_type", None)
        print(f"  - {name}\tshape={shape}\tdtype={dtype}")
    print("Model Outputs:")
    for t in outputs:
        name = getattr(t, "name", "")
        shape = getattr(t, "shape", None)
        dtype = getattr(t, "data_type", None)
        print(f"  - {name}\tshape={shape}\tdtype={dtype}")


def _normalize_feed_keys(feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Normalize feed keys to match model input names.
    """
    aliases = {
        "grid_thw": "image_grid_thw",
        "image_grid_thw": "image_grid_thw",
    }
    out = dict(feed)
    if "image_grid_thw" in out and "grid_thw" not in out:
        out["grid_thw"] = out["image_grid_thw"]
    if "grid_thw" in out and "image_grid_thw" not in out:
        out["image_grid_thw"] = out["grid_thw"]
    for k, v in list(out.items()):
        if k in aliases:
            out[aliases[k]] = v
    return out


def _build_mslite_inputs(
    model: mslite.Model, feed: Dict[str, np.ndarray]
) -> List[mslite.Tensor]:
    """
    Build MindSpore Lite model inputs from feed.
    """
    feed = _normalize_feed_keys(feed)
    inputs = model.get_inputs()
    if not inputs:
        raise RuntimeError("Model has no inputs.")

    missing = []
    tensors = []
    for t in inputs:
        name = getattr(t, "name", None)
        if not name or name not in feed:
            missing.append(str(name))
            continue
        tensors.append(mslite.Tensor(np.ascontiguousarray(feed[name])))

    if missing:
        model_input_names = [getattr(t, "name", "") for t in inputs]
        raise ValueError(
            f"Missing model inputs: {missing}. "
            f"model_inputs={model_input_names} feed_keys={list(feed.keys())}. "
            f"Tip: if the model expects vision inputs, pass --image to enable pixel_values/image_grid_thw."
        )
    return tensors


def _masked_mean_pool(
    last_hidden_state: np.ndarray, attention_mask: np.ndarray
) -> np.ndarray:
    """
    Apply masked mean pooling to the last hidden state.
    """
    hs = last_hidden_state.astype(np.float32, copy=False)
    mask = attention_mask.astype(np.float32, copy=False)
    mask3 = mask[:, :, None]
    summed = (hs * mask3).sum(axis=1)
    denom = np.clip(mask3.sum(axis=1), 1e-6, None)
    return summed / denom


def run_inference(
    model_path: str,
    processor_id: str,
    texts: List[str],
    image_path: Optional[str],
    device: str,
    device_id: int,
) -> np.ndarray:
    """
    Run inference on the Qwen3 VL Embedding 2B model with MindSpore Lite.
    """
    print(f"Loading processor from {processor_id}...")
    processor = AutoProcessor.from_pretrained(processor_id, trust_remote_code=True)

    print(f"Loading MindIR model from {model_path}...")
    context = _create_context(device=device, device_id=device_id)
    model = mslite.Model()
    model.build_from_file(model_path, mslite.ModelType.MINDIR, context)
    _describe_model_io(model)

    image = None
    if image_path:
        print(f"Loading image from {image_path}...")
        image = _load_image(image_path)

    print(f"Processing {len(texts)} texts...")
    if image is None:
        inputs = processor(
            text=texts, return_tensors="np", padding=True, truncation=True
        )
    else:
        images = [image] * len(texts)
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="np",
            padding=True,
            truncation=True,
        )

    feed: Dict[str, np.ndarray] = {}
    if "input_ids" in inputs:
        feed["input_ids"] = inputs["input_ids"].astype(np.int64)
    if "attention_mask" in inputs:
        feed["attention_mask"] = inputs["attention_mask"].astype(np.int64)
    if "pixel_values" in inputs:
        feed["pixel_values"] = inputs["pixel_values"].astype(np.float16)
    if "image_grid_thw" in inputs:
        feed["image_grid_thw"] = inputs["image_grid_thw"].astype(np.int64)
    feed = _normalize_feed_keys(feed)

    outputs = model.predict(_build_mslite_inputs(model, feed))
    last_hidden_state = outputs[0].get_data_to_numpy()

    if "attention_mask" not in feed:
        embeddings = last_hidden_state.mean(axis=1).astype(np.float32)
    else:
        embeddings = _masked_mean_pool(
            last_hidden_state, feed["attention_mask"]
        ).astype(np.float32)

    print("=" * 50)
    print("Embeddings computed successfully!")
    print("=" * 50)
    print(f"last_hidden_state shape: {tuple(last_hidden_state.shape)}")
    print(f"embeddings shape:        {tuple(embeddings.shape)}")
    return embeddings


def compute_similarity(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute similarity matrix between embeddings.
    """
    x = embeddings.astype(np.float32, copy=False)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    x = x / denom
    sim = x @ x.T

    print("\nSimilarity Matrix:")
    print("=" * 50)
    n = int(sim.shape[0])
    for i in range(n):
        for j in range(n):
            if i < j:
                print(f"Text {i + 1} vs Text {j + 1}: {float(sim[i, j]):.4f}")
    return sim


def main():
    """
    Main function for running inference with the Qwen3 VL Embedding 2B model with MindSpore Lite.
    """
    parser = argparse.ArgumentParser(
        description="Inference with Qwen3-VL-Embedding-2B MindIR model (MindSpore Lite)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b.mindir",
        help="Path to MindIR model",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default="Qwen/Qwen3-VL-Embedding-2B",
        help="HuggingFace processor id or local path",
    )
    parser.add_argument(
        "--texts",
        type=str,
        nargs="+",
        default=["Hello world", "Hi there", "Good morning"],
        help="List of texts to embed",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional image path for multimodal embedding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "ascend"],
        help="Device for inference",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Ascend device id (only for --device ascend)",
    )
    parser.add_argument(
        "--compute-similarity", action="store_true", help="Compute similarity matrix"
    )
    args = parser.parse_args()

    embeddings = run_inference(
        model_path=args.model,
        processor_id=args.processor,
        texts=args.texts,
        image_path=args.image,
        device=args.device,
        device_id=args.device_id,
    )

    if args.compute_similarity:
        compute_similarity(embeddings)


if __name__ == "__main__":
    main()
