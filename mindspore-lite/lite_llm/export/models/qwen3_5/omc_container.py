"""IMOD parsing adapted from HMOS-Device-AI inspect_omc_abi.py."""
import struct
import sys
from pathlib import Path

HEADER_SIZE = 256
PARTITION_MODEL_DEF = 0

def load_ge_ir(module_dir: Path):
    sys.path.insert(0, str(module_dir))
    import ge_ir_pb2  # pylint: disable=import-error,import-outside-toplevel
    return ge_ir_pb2

def read_model(omc: Path, ge_ir_pb2):
    """Read an OMC container and extract its model-definition partition."""
    with omc.open("rb") as stream:
        header = stream.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE or header[:4] != b"IMOD":
            raise ValueError("not an IMOD container")
        raw_count = stream.read(4)
        if len(raw_count) != 4:
            raise ValueError("truncated partition count")
        partition_count = struct.unpack("<I", raw_count)[0]
        if partition_count == 0 or partition_count > 64:
            raise ValueError(f"invalid partition count {partition_count}")
        partitions = [
            struct.unpack("<III", stream.read(12))
            for _ in range(partition_count)
        ]
        partition_base = stream.tell()
        if partition_base + max(offset + size for _, offset, size in partitions) != omc.stat().st_size:
            raise ValueError("partition table does not cover the OMC file")
        model_parts = [item for item in partitions if item[0] == PARTITION_MODEL_DEF]
        if len(model_parts) != 1:
            raise ValueError(f"expected one MODEL_DEF partition, got {model_parts}")
        _, offset, size = model_parts[0]
        stream.seek(partition_base + offset)
        model_bytes = stream.read(size)
        if len(model_bytes) != size:
            raise ValueError("truncated MODEL_DEF partition")
    model = ge_ir_pb2.ModelDef()
    model.ParseFromString(model_bytes)
    return model, partitions, model_bytes
