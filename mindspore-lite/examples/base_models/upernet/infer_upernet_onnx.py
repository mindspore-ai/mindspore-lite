"""ONNX Runtime inference script for UPerNet model.

Loads the exported ONNX model and runs inference on an input image.
Performs preprocessing (BGR, mean subtraction) and post-processing
(softmax, argmax) to produce segmentation maps.

Usage:
    python infer_upernet_onnx.py --model upernet.onnx --image path/to/image.jpg
"""

import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort


def preprocess_image(image_path, input_size=576):
    """Preprocess image for UPerNet inference.

    Reads image as BGR (cv2 default), resizes to input_size x input_size,
    subtracts BGR mean, and transposes to CHW format.

    Args:
        image_path: path to input image file.
        input_size: target square image size.

    Returns:
        Preprocessed image array [1, 3, H, W] as float32.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')

    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)

    # BGR mean subtraction (matching original UPerNet training)
    mean = np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)
    img -= mean

    # HWC -> CHW -> 1CHW
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def softmax(x, axis=1):
    """Compute softmax along the specified axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def postprocess(scene_logits, object_logits, part_logits, material_logits):
    """Post-process raw logits to produce prediction maps.

    Applies softmax and argmax to produce final predictions.

    Args:
        scene_logits: [1, 365, 1, 1] scene classification logits.
        object_logits: [1, 336, H, W] object segmentation logits.
        part_logits: [1, 427, H, W] part segmentation logits.
        material_logits: [1, 26, H, W] material segmentation logits.

    Returns:
        Dict with scene_pred, object_pred, part_pred, material_pred.
    """
    # Scene: softmax + top-5
    scene_prob = softmax(scene_logits, axis=1)
    scene_prob = scene_prob.squeeze()  # [365]
    scene_top5 = np.argsort(-scene_prob)[:5]

    # Object: softmax + argmax
    object_prob = softmax(object_logits, axis=1)
    object_pred = np.argmax(object_prob, axis=1).squeeze(0)  # [H, W]

    # Part: softmax + argmax
    part_prob = softmax(part_logits, axis=1)
    part_pred = np.argmax(part_prob, axis=1).squeeze(0)  # [H, W]

    # Material: softmax + argmax
    material_prob = softmax(material_logits, axis=1)
    material_pred = np.argmax(material_prob, axis=1).squeeze(0)  # [H, W]

    return {
        'scene_prob': scene_prob,
        'scene_top5': scene_top5,
        'object_pred': object_pred,
        'part_pred': part_pred,
        'material_pred': material_pred,
    }


def run_inference(onnx_path, image_path, input_size=576):
    """Run ONNX inference on a single image.

    Args:
        onnx_path: path to ONNX model file.
        image_path: path to input image.
        input_size: model input size.

    Returns:
        Tuple of (predictions dict, timing dict).
    """
    # Load model
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(onnx_path, sess_options=sess_options,
                                providers=['CPUExecutionProvider'])

    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    print(f'Input: {input_name}')
    print(f'Outputs: {output_names}')

    # Preprocess
    t0 = time.time()
    img_input = preprocess_image(image_path, input_size)
    t_pre = time.time() - t0

    # Inference
    t0 = time.time()
    outputs = sess.run(output_names, {input_name: img_input})
    t_inf = time.time() - t0

    scene_logits, object_logits, part_logits, material_logits = outputs

    # Postprocess
    t0 = time.time()
    preds = postprocess(scene_logits, object_logits, part_logits, material_logits)
    t_post = time.time() - t0

    timing = {
        'preprocess_ms': t_pre * 1000,
        'inference_ms': t_inf * 1000,
        'postprocess_ms': t_post * 1000,
        'total_ms': (t_pre + t_inf + t_post) * 1000,
    }

    return preds, timing


def main():
    """Parse arguments and run ONNX inference."""
    parser = argparse.ArgumentParser(description='UPerNet ONNX inference')
    parser.add_argument('--model', type=str, default='upernet.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--input-size', type=int, default=576,
                        help='Model input size (default: 576)')
    parser.add_argument('--output-dir', type=str, default='./onnx_output',
                        help='Directory to save output visualizations')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    preds, timing = run_inference(args.model, args.image, args.input_size)

    print(f'\n=== UPerNet ONNX Inference Results ===')
    print(f'Scene top-5: {preds["scene_top5"]}')
    print(f'Object pred unique: {np.unique(preds["object_pred"])}')
    print(f'Part pred unique: {np.unique(preds["part_pred"])}')
    print(f'Material pred unique: {np.unique(preds["material_pred"])}')

    print(f'\n=== Timing ===')
    print(f'Preprocess:  {timing["preprocess_ms"]:.2f} ms')
    print(f'Inference:   {timing["inference_ms"]:.2f} ms')
    print(f'Postprocess: {timing["postprocess_ms"]:.2f} ms')
    print(f'Total:       {timing["total_ms"]:.2f} ms')

    # Save visualizations
    np.random.seed(233)
    color_map = np.random.rand(500, 3) * 0.7 + 0.3

    for name in ['object', 'part', 'material']:
        pred = preds[f'{name}_pred']
        colored = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for label in np.unique(pred):
            colored[pred == label] = (color_map[label % 500] * 255).astype(np.uint8)
        out_path = os.path.join(args.output_dir, f'{name}_pred.png')
        cv2.imwrite(out_path, colored)
        print(f'Saved {name} prediction to {out_path}')


if __name__ == '__main__':
    main()
