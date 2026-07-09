"""Verify precision alignment between PyTorch and MSLite MindIR models.

Runs the same random input through both the PyTorch model and the MSLite
MindIR model, then compares outputs using cosine similarity.
Threshold: cos > 0.99.

Usage:
    python verify_mslite_precision.py --weights-dir /path/to/weights \\
                                      --mindir upernet_mindir.mindir
"""

import argparse
import os
import sys

import numpy as np
import torch
from upernet_model import build_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cosine_similarity(a, b):
    """Compute cosine similarity between two flattened arrays."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    return dot / (norm_a * norm_b + 1e-10)


def verify(weights_dir, mindir_path, input_size=576, num_tests=3):
    """Run precision alignment verification between PyTorch and MSLite.

    Args:
        weights_dir: directory containing encoder/decoder weights.
        mindir_path: path to MindIR model.
        input_size: model input size.
        num_tests: number of random test inputs.

    Returns:
        True if all outputs pass cos > 0.99 threshold.
    """
    import mindspore_lite as mslite

    enc_path = os.path.join(weights_dir, 'encoder_epoch_40.pth')
    dec_path = os.path.join(weights_dir, 'decoder_epoch_40.pth')

    # Load PyTorch model
    torch_model = build_model(enc_path, dec_path)
    torch_model.eval()

    # Load MSLite model
    context = mslite.Context()
    context.target = ['ascend']
    context.ascend.device_id = 0
    ms_model = mslite.Model()
    ms_model.build_from_file(mindir_path, mslite.ModelType.MINDIR, context, '')

    ms_inputs = ms_model.get_inputs()

    all_passed = True
    for test_idx in range(num_tests):
        np.random.seed(test_idx * 42 + 123)
        dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

        # PyTorch inference
        with torch.no_grad():
            torch_outputs = torch_model(torch.from_numpy(dummy_input))
        torch_outs = [out.numpy() for out in torch_outputs]

        # MSLite inference
        for inp in ms_inputs:
            inp.set_data_from_numpy(dummy_input.astype(np.float32))
        ms_outputs = ms_model.predict(ms_inputs)
        ms_outs = [out.get_data_to_numpy() for out in ms_outputs]

        names = ['scene', 'object', 'part', 'material']
        print(f'\n--- Test {test_idx + 1}/{num_tests} ---')
        for name, torch_out, ms_out in zip(names, torch_outs, ms_outs):
            cos = cosine_similarity(torch_out, ms_out)
            max_diff = np.max(np.abs(torch_out - ms_out))
            mean_diff = np.mean(np.abs(torch_out - ms_out))
            status = 'PASS' if cos > 0.99 else 'FAIL'
            if cos <= 0.99:
                all_passed = False
            print(f'  {name:10s}: cos={cos:.6f}  max_diff={max_diff:.6e}  '
                  f'mean_diff={mean_diff:.6e}  [{status}]')

    print(f'\n{"=" * 50}')
    print(f'Overall: {"ALL PASSED" if all_passed else "SOME FAILED"}')
    return all_passed


def main():
    """Parse arguments and run verification."""
    parser = argparse.ArgumentParser(
        description='Verify PyTorch vs MSLite MindIR precision')
    parser.add_argument('--weights-dir', type=str, required=True,
                        help='Directory containing encoder/decoder .pth files')
    parser.add_argument('--mindir', type=str, default='upernet_mindir.mindir',
                        help='Path to MindIR model')
    parser.add_argument('--input-size', type=int, default=576,
                        help='Model input size')
    parser.add_argument('--num-tests', type=int, default=3,
                        help='Number of random test inputs')
    args = parser.parse_args()

    passed = verify(args.weights_dir, args.mindir, args.input_size, args.num_tests)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
