# Adapted from HMOS-Device-AI converter/qwen3.5-35b-a3b; see source_provenance.json.
#!/usr/bin/env python3
"""Export mmap embedding, final norm, and tiled NZ LM Head in FP16."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


VOCAB = 248320
HIDDEN = 2048
TILE = 8192


class TensorRecord:
    """Locate and decode a tensor within a safetensors shard."""
    def __init__(self, model_dir: Path, name: str):
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self.path = model_dir / index["weight_map"][name]
        with self.path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_size))
        info = header[name]
        begin, end = info["data_offsets"]
        self.offset = 8 + header_size + int(begin)
        self.bytes = int(end) - int(begin)
        self.shape = tuple(int(value) for value in info["shape"])
        self.dtype = info["dtype"]

    def bf16_rows(self, first: int, count: int) -> np.ndarray:
        """Read the requested BF16 rows as floating point values."""
        if self.dtype != "BF16" or len(self.shape) != 2:
            raise ValueError(f"expected a BF16 matrix, got {self.dtype} {self.shape}")
        row_bytes = self.shape[1] * 2
        with self.path.open("rb") as stream:
            stream.seek(self.offset + first * row_bytes)
            raw = stream.read(count * row_bytes)
        if len(raw) != count * row_bytes:
            raise EOFError(f"short tensor read from {self.path}")
        bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32).astype(np.float16).reshape(count, self.shape[1])

    def bf16_vector(self) -> np.ndarray:
        if self.dtype != "BF16" or len(self.shape) != 1:
            raise ValueError(f"expected a BF16 vector, got {self.dtype} {self.shape}")
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            raw = stream.read(self.bytes)
        bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32).astype(np.float16)


def pack_nz(weight: np.ndarray) -> np.ndarray:
    n, k = weight.shape
    if n % 16 or k % 16:
        raise ValueError(f"NZ tile must be 16 aligned: {weight.shape}")
    return np.ascontiguousarray(
        weight.T.reshape(k // 16, 16, n // 16, 16).transpose(2, 0, 1, 3)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    embedding = TensorRecord(args.model_dir, "model.language_model.embed_tokens.weight")
    lm_head = TensorRecord(args.model_dir, "lm_head.weight")
    norm = TensorRecord(args.model_dir, "model.language_model.norm.weight")
    if embedding.shape != (VOCAB, HIDDEN) or lm_head.shape != (VOCAB, HIDDEN):
        raise ValueError("unexpected Qwen3.5 embedding/LM Head shape")

    embedding_path = args.output_dir / "embedding_fp16.bin"
    lm_head_path = args.output_dir / "lm_head_nz_fp16.bin"
    tiles = []
    with embedding_path.open("wb") as embedding_out, lm_head_path.open("wb") as head_out:
        for first in range(0, VOCAB, TILE):
            rows = min(TILE, VOCAB - first)
            embedding_tile = embedding.bf16_rows(first, rows)
            embedding_out.write(embedding_tile.tobytes())
            head_tile = lm_head.bf16_rows(first, rows)
            offset = head_out.tell()
            packed = pack_nz(head_tile)
            head_out.write(packed.tobytes())
            tiles.append({"first_token": first, "rows": rows,
                          "offset": offset, "bytes": packed.nbytes})

    norm_value = norm.bf16_vector()
    if norm_value.shape != (HIDDEN,):
        raise ValueError(f"unexpected final norm shape: {norm_value.shape}")
    norm_value.tofile(args.output_dir / "final_norm_fp16.bin")
    manifest = {
        "format": "qwen35-global-fp16-v1",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "embedding": embedding_path.name,
        "final_norm": "final_norm_fp16.bin",
        "lm_head": lm_head_path.name,
        "lm_head_layout": "FRACTAL_NZ_FP16",
        "tiles": tiles,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
