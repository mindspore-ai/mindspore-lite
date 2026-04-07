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
Infer Qwen3-VL-2B on Ascend with MindSpore Lite.
"""

import sys
import argparse
import numpy as np
from PIL import Image

try:
    import mindspore_lite as mslite
    from transformers import AutoProcessor
except ImportError:
    print("Error: mindspore_lite or transformers package not found.")
    print("Please install them first.")
    sys.exit(1)


class Qwen3VLInferencer:
    """
    Qwen3-VL-2B inferencer.
    """

    def __init__(self, vision_model_path, llm_model_path, processor_id, device_id=0):
        """
        Initialize Qwen3-VL-2B inferencer.
        """
        print("Initializing MindSpore Lite context for Ascend...")

        # Configure Ascend context
        self.context = mslite.Context()
        self.context.target = ["ascend"]
        self.context.ascend.device_id = device_id
        self.context.ascend.precision_mode = "preferred_fp16"

        # Load vision model
        print(f"Loading vision model from {vision_model_path}...")
        self.vision_model = mslite.Model()
        self.vision_model.build_from_file(
            vision_model_path, mslite.ModelType.MINDIR, self.context
        )

        # Load LLM model
        print(f"Loading LLM model from {llm_model_path}...")
        self.llm_model = mslite.Model()
        self.llm_model.build_from_file(
            llm_model_path, mslite.ModelType.MINDIR, self.context
        )

        # Load processor
        print(f"Loading processor from {processor_id}...")
        self.processor = AutoProcessor.from_pretrained(processor_id)

    def preprocess_image(self, image_url):
        """
        Preprocess image from URL.
        """
        image = Image.open(image_url).convert("RGB")
        return image

    def infer(self, image_url, text_prompt):
        """
        Infer Qwen3-VL-2B on AscSpore Lite.
        """
        image = self.preprocess_image(image_url)

        # Prepare multimodal inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
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

        # Step 1: Vision Inference (extract image embeddings)
        # Note: In a real split, the vision features would be mapped to LLM space
        # Here we assume the split logic in export script
        pixel_values = inputs.pixel_values.numpy()
        grid_thw = inputs.image_grid_thw.numpy()

        vision_inputs = [mslite.Tensor(pixel_values), mslite.Tensor(grid_thw)]
        vision_outputs = self.vision_model.predict(vision_inputs)
        image_embeds = vision_outputs[0].get_data_to_numpy()
        print("Image embeddings shape: ", image_embeds.shape)

        # Step 2: LLM Generation Loop
        input_ids = inputs.input_ids.numpy()
        attention_mask = inputs.attention_mask.numpy()
        position_ids = inputs.position_ids.numpy()

        print("Starting LLM generation...")
        generated_ids = []
        max_new_tokens = 128

        # Simplified loop (non-incremental for demo)
        # In production, we'd use KV cache and incremental decoding
        for _ in range(max_new_tokens):
            llm_inputs = [
                mslite.Tensor(input_ids),
                mslite.Tensor(attention_mask),
                mslite.Tensor(position_ids),
            ]

            llm_outputs = self.llm_model.predict(llm_inputs)
            logits = llm_outputs[0].get_data_to_numpy()

            # Greedy search
            next_token_id = np.argmax(logits[0, -1, :])
            generated_ids.append(next_token_id)

            if next_token_id == self.processor.tokenizer.eos_token_id:
                break

            # Update inputs for next step
            input_ids = np.concatenate([input_ids, [[next_token_id]]], axis=1)
            attention_mask = np.concatenate([attention_mask, [[1]]], axis=1)
            position_ids = np.concatenate(
                [position_ids, [[position_ids[0, -1] + 1]]], axis=1
            )

        output_text = self.processor.batch_decode(
            [generated_ids], skip_special_tokens=True
        )[0]
        return output_text


def main():
    """
    Main function for Qwen3-VL-2B inference on Ascend with MindSpore Lite.
    """
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-2B Inference on Ascend with MindSpore Lite"
    )
    parser.add_argument(
        "--vision-model", type=str, required=True, help="Path to qwen3_vl_vision.ms"
    )
    parser.add_argument(
        "--llm-model", type=str, required=True, help="Path to qwen3_vl_llm.ms"
    )
    parser.add_argument(
        "--processor",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Processor ID",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        help="Image URL or path",
    )
    parser.add_argument(
        "--prompt", type=str, default="Describe this image.", help="Text prompt"
    )
    parser.add_argument("--device-id", type=int, default=0, help="Ascend device ID")

    args = parser.parse_args()

    inferencer = Qwen3VLInferencer(
        args.vision_model, args.llm_model, args.processor, args.device_id
    )
    result = inferencer.infer(args.image, args.prompt)

    print("\n" + "=" * 50)
    print(f"Input Prompt: {args.prompt}")
    print(f"Generated Response: {result}")
    print("=" * 50)


if __name__ == "__main__":
    main()
