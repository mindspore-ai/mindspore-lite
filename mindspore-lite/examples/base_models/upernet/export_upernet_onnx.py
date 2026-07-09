"""Export UPerNet model to ONNX format.

This script loads the pretrained UPerNet PyTorch model and exports it to ONNX.
PrRoIPool2D is replaced with AdaptiveAvgPool2d, and SynchronizedBatchNorm2d
is replaced with nn.BatchNorm2d for ONNX compatibility.

Usage:
    python export_upernet_onnx.py --weights-dir /path/to/upernet/weight/upernet \\
                                  --output upernet.onnx
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from upernet_model import build_model  # noqa: E402


def export_onnx(weights_dir, output_path, input_size=512, opset=17):
    """Export UPerNet model to ONNX.

    Args:
        weights_dir: directory containing encoder_epoch_40.pth and
                     decoder_epoch_40.pth.
        output_path: output ONNX file path.
        input_size: square input image size (default 512).
        opset: ONNX opset version.
    """
    enc_path = os.path.join(weights_dir, 'encoder_epoch_40.pth')
    dec_path = os.path.join(weights_dir, 'decoder_epoch_40.pth')
    assert os.path.exists(enc_path), f'Encoder weights not found: {enc_path}'
    assert os.path.exists(dec_path), f'Decoder weights not found: {dec_path}'

    model = build_model(enc_path, dec_path)
    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Verify forward pass works
    with torch.no_grad():
        outputs = model(dummy_input)
    print(f'Torch forward OK:')
    for name, out in zip(['scene', 'object', 'part', 'material'], outputs):
        print(f'  {name}: {out.shape}')

    input_names = ['img']
    output_names = ['scene_logits', 'object_logits',
                    'part_logits', 'material_logits']

    # Use legacy exporter to avoid torch.export issues
    torch.onnx.utils.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'\nONNX exported to: {output_path}')
    print(f'File size: {file_size_mb:.2f} MB')
    print(f'Opset: {opset}')
    print(f'Input: {input_names[0]} [1, 3, {input_size}, {input_size}]')
    print(f'Outputs: {output_names}')


def main():
    """Parse arguments and run ONNX export."""
    parser = argparse.ArgumentParser(description='Export UPerNet to ONNX')
    parser.add_argument('--weights-dir', type=str, required=True,
                        help='Directory containing encoder/decoder .pth files')
    parser.add_argument('--output', type=str, default='upernet.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--input-size', type=int, default=576,
                        help='Square input image size (default: 576, must be '
                             'divisible by 32 and yield feature map divisible '
                             'by pool scales 1,2,3,6)')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version (default: 17)')
    args = parser.parse_args()

    export_onnx(args.weights_dir, args.output, args.input_size, args.opset)


if __name__ == '__main__':
    main()
