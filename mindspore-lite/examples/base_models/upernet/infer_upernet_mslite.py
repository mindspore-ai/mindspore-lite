"""MindSpore Lite inference script for UPerNet model.

Loads the converted MindIR model and runs inference on Ascend hardware.
Performs preprocessing (BGR, mean subtraction) and post-processing
(softmax, argmax) using only numpy — no torch dependency.

Usage:
    python infer_upernet_mslite.py --model upernet_mindir.mindir \\
                                   --image path/to/image.jpg
"""

import argparse
import os
import time

import cv2
import mindspore_lite as mslite
import numpy as np


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
    """Compute softmax along the specified axis using numpy."""
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
        Dict with scene_prob, scene_top5, object_pred, part_pred, material_pred.
    """
    scene_prob = softmax(scene_logits, axis=1)
    scene_prob = scene_prob.squeeze()
    scene_top5 = np.argsort(-scene_prob)[:5]

    object_prob = softmax(object_logits, axis=1)
    object_pred = np.argmax(object_prob, axis=1).squeeze(0)

    part_prob = softmax(part_logits, axis=1)
    part_pred = np.argmax(part_prob, axis=1).squeeze(0)

    material_prob = softmax(material_logits, axis=1)
    material_pred = np.argmax(material_prob, axis=1).squeeze(0)

    return {
        'scene_prob': scene_prob,
        'scene_top5': scene_top5,
        'object_pred': object_pred,
        'part_pred': part_pred,
        'material_pred': material_pred,
    }


def build_mslite_inputs(model, feed_dict):
    """Build MSLite input tensor list by matching input names.

    Args:
        model: MSLite Model instance.
        feed_dict: dict mapping input name to numpy array.

    Returns:
        List of MSLite Tensor objects with data set.
    """
    inputs = model.get_inputs()
    result = []
    for inp in inputs:
        if inp.name in feed_dict:
            data = feed_dict[inp.name]
            inp.set_data_from_numpy(data.astype(np.float32))
            result.append(inp)
        else:
            raise ValueError(f'No data provided for input: {inp.name}')
    return result


def load_model(mindir_path):
    """Load MindIR model with Ascend context.

    Args:
        mindir_path: path to .mindir file.

    Returns:
        Configured MSLite Model instance.
    """
    context = mslite.Context()
    context.target = ['ascend']
    context.ascend.device_id = 0

    model = mslite.Model()
    model.build_from_file(mindir_path, mslite.ModelType.MINDIR, context, '')
    return model


def run_inference(model, image_path, input_size=576):
    """Run MSLite inference on a single image.

    Args:
        model: MSLite Model instance.
        image_path: path to input image.
        input_size: model input size.

    Returns:
        Tuple of (predictions dict, timing dict, raw outputs list).
    """
    # Preprocess
    t0 = time.time()
    img_input = preprocess_image(image_path, input_size)
    t_pre = time.time() - t0

    # Prepare inputs
    inputs = build_mslite_inputs(model, {'img': img_input})

    # Inference
    t0 = time.time()
    outputs = model.predict(inputs)
    t_inf = time.time() - t0

    # Convert outputs to numpy
    out_arrays = [out.get_data_to_numpy() for out in outputs]

    # Postprocess
    t0 = time.time()
    preds = postprocess(*out_arrays)
    t_post = time.time() - t0

    timing = {
        'preprocess_ms': t_pre * 1000,
        'inference_ms': t_inf * 1000,
        'postprocess_ms': t_post * 1000,
        'total_ms': (t_pre + t_inf + t_post) * 1000,
    }

    return preds, timing, out_arrays


def save_visualization(preds, output_dir):
    """Save colored segmentation maps as PNG files.

    Args:
        preds: predictions dict from postprocess.
        output_dir: directory to save output images.
    """
    np.random.seed(233)
    color_map = np.random.rand(500, 3) * 0.7 + 0.3

    for name in ['object', 'part', 'material']:
        pred = preds[f'{name}_pred']
        colored = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for label in np.unique(pred):
            colored[pred == label] = (color_map[label % 500] * 255).astype(np.uint8)
        out_path = os.path.join(output_dir, f'{name}_pred.png')
        cv2.imwrite(out_path, colored)
        print(f'Saved {name} prediction to {out_path}')


def main():
    """Parse arguments and run MSLite inference."""
    parser = argparse.ArgumentParser(description='UPerNet MSLite inference')
    parser.add_argument('--model', type=str, default='upernet_mindir.mindir',
                        help='Path to MindIR model file')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--input-size', type=int, default=576,
                        help='Model input size (default: 576)')
    parser.add_argument('--output-dir', type=str, default='./mslite_output',
                        help='Directory to save output visualizations')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading model: {args.model}')
    model = load_model(args.model)

    # Print model info
    inputs = model.get_inputs()
    outputs = model.get_outputs()
    print(f'Input:  {inputs[0].name} {inputs[0].shape} {inputs[0].dtype}')
    for out in outputs:
        print(f'Output: {out.name} {out.shape} {out.dtype}')

    # Warmup run
    dummy = np.random.randn(1, 3, args.input_size, args.input_size).astype(np.float32)
    warmup_inputs = build_mslite_inputs(model, {'img': dummy})
    model.predict(warmup_inputs)
    print('Warmup done.')

    # Real inference
    preds, timing, _ = run_inference(model, args.image, args.input_size)

    print('\n=== UPerNet MSLite Inference Results ===')
    print(f'Scene top-5: {preds["scene_top5"]}')
    print(f'Object pred unique: {np.unique(preds["object_pred"])}')
    print(f'Part pred unique: {np.unique(preds["part_pred"])}')
    print(f'Material pred unique: {np.unique(preds["material_pred"])}')

    print('\n=== Timing ===')
    print(f'Preprocess:  {timing["preprocess_ms"]:.2f} ms')
    print(f'Inference:   {timing["inference_ms"]:.2f} ms')
    print(f'Postprocess: {timing["postprocess_ms"]:.2f} ms')
    print(f'Total:       {timing["total_ms"]:.2f} ms')

    save_visualization(preds, args.output_dir)


if __name__ == '__main__':
    main()
