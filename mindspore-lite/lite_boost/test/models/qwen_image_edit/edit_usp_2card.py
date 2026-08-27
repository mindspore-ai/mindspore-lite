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
"""Two-card USP inference check for Qwen-Image-Edit on Ascend NPUs.

Initializes the HCCL distributed environment, boosts the Diffusers pipeline
in-place via BoostManager (with per-module parallel settings selected by
qwen_image_edit.yaml), then runs one image-edit request and saves the
result image on rank 0.
"""
import os
import torch
from PIL import Image
import torch_npu
# Deliberate side-effect import: redirects CUDA calls inside transformers and
# diffusers to their NPU equivalents; never referenced directly in this file.
from torch_npu.contrib import transfer_to_npu  # pylint: disable=unused-import
from diffusers import QwenImageEditPipeline
from lite_boost.parallel import initialize_usp
from lite_boost import BoostManager

DTYPE = torch.bfloat16

# Per-module optimization config: Parallel.dit (CP) + Parallel.vae (DP).
CONFIG_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "python", "model", "qwen_image_edit", "qwen_image_edit.yaml",
)

local_rank = int(os.getenv("LOCAL_RANK", os.getenv("RANK", "0")))
torch_npu.npu.set_device(local_rank)
initialize_usp()

pipe = QwenImageEditPipeline.from_pretrained("qwen-image-edit", torch_dtype=DTYPE)

boost_manager = BoostManager()
pipe = boost_manager(pipe, config=CONFIG_YAML)
print(f"rank {local_rank}: pipeline boosted")

pipe.to("npu")

for comp in pipe.components.values():
    if hasattr(comp, "parameters"):
        for p in comp.parameters():
            if p.dtype == torch.float32:
                p.data = p.data.to(dtype=DTYPE)

# Transformers offers no public setter for the attention implementation of a
# loaded model's sub-configs, so patch the private field directly.
# pylint: disable=protected-access
pipe.text_encoder.config._attn_implementation = "eager"
pipe.text_encoder.config.vision_config._attn_implementation = "eager"
# pylint: enable=protected-access

image1 = Image.open("image1.jpg")
prompt = "给图中的人头上戴一顶帽子"

inputs = {
    "image": [image1, ],
    "prompt": prompt,
    "generator": torch.manual_seed(42),
    "true_cfg_scale": 4.0,
    "negative_prompt": "模糊，低质量，失真，变形，水印，噪点，文字，过度曝光，细节缺失",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1
}
with torch.inference_mode():
    output = pipe(**inputs)
    output_images = output.images[0]
if int(local_rank) == 0:
    output_images.save("output.png")
