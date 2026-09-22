# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Canonical export quantization names and serialized embedding layouts."""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Optional, Tuple


class QuantType(str, Enum):
    """Quantization algorithms; legacy precision names remain input aliases."""

    Q4_0 = "q4_0"
    S16S4 = "s16s4"
    W4A8 = "w4a8"
    W2A16 = "w2a16"

    def __str__(self):
        return self.value

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            name = value.lower()
            if name == "w4a16":
                return cls.Q4_0
            return cls._value2member_map_.get(name)
        return None

    @classmethod
    def parse(cls, value):
        """Normalize quantized and unquantized internal API arguments."""
        if value is None or value == "" or value == "FP16":
            return None
        return cls(value)

    @property
    def embedding_format(self):
        """Keep the established runtime layout identifiers unchanged."""
        return get_quant_preset(self).embedding_format


class EmbeddingFormat(str, Enum):
    """Versioned runtime layout names, distinct from quantization algorithms."""

    W4A16 = "W4A16"
    S16S4_NZ_V1 = "S16S4_NZ_V1"

    def __str__(self):
        return self.value


class QuantDType(str, Enum):
    """Logical arithmetic types, independent of packed byte storage."""

    INT2 = "int2"
    INT4 = "int4"
    INT8 = "int8"
    INT16 = "int16"
    FLOAT16 = "float16"


class PackingLayout(str, Enum):
    """Weight and scale encodings accepted by the custom operators."""

    Q4_0_NZF_COMPACT_PHASE4 = "q4_0_nzf_compact_phase4"
    Q4_NZ = "q4_nz"
    Q2_PLANAR = "q2_planar"
    S16S4_NZ = "s16s4_nz"


class ScaleEncoding(str, Enum):
    """Storage encoding of weight scales, including hardware records."""

    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FIXPIPE = "fixpipe_fp32_pair"

    @property
    def byte_size(self):
        """Bytes stored per weight group (including record padding)."""
        return {self.FLOAT16: 2, self.FLOAT32: 4, self.FIXPIPE: 8}[self]


@dataclass(frozen=True)
class QuantPreset:
    """Fixed packing/kernel contract; group size is not a free parameter."""

    quant_type: QuantType
    bits: int
    group_size: int
    weight_dtype: QuantDType
    activation_dtype: QuantDType
    scale_encoding: ScaleEncoding
    packing_layout: PackingLayout
    operator: str
    embedding_format: Optional[EmbeddingFormat] = None
    export_targets: Tuple[str, ...] = ()

    def validate_target(self, target):
        """Reject presets without a supported one-click export kernel."""
        if target not in self.export_targets:
            targets = ", ".join(self.export_targets) or "none"
            raise ValueError(
                f"{self.quant_type.value.upper()} is supported only with --target {targets}"
            )


# Internal legacy presets remain usable, but only verified export presets have
# target entries. Adding a group size requires a matching packing/kernel contract.
QUANT_PRESETS = MappingProxyType({
    QuantType.Q4_0: QuantPreset(
        quant_type=QuantType.Q4_0, bits=4, group_size=32,
        weight_dtype=QuantDType.INT4, activation_dtype=QuantDType.FLOAT16,
        scale_encoding=ScaleEncoding.FLOAT16, packing_layout=PackingLayout.Q4_0_NZF_COMPACT_PHASE4,
        operator="MsQuant4N0Group32", embedding_format=EmbeddingFormat.W4A16,
        export_targets=("kirin9020", "kirin9030"),
    ),
    QuantType.S16S4: QuantPreset(
        quant_type=QuantType.S16S4, bits=4, group_size=128,
        weight_dtype=QuantDType.INT4, activation_dtype=QuantDType.INT16,
        scale_encoding=ScaleEncoding.FIXPIPE, packing_layout=PackingLayout.S16S4_NZ,
        operator="MsMatmulS16S4", embedding_format=EmbeddingFormat.S16S4_NZ_V1,
        export_targets=("kirin9030",),
    ),
    QuantType.W4A8: QuantPreset(
        quant_type=QuantType.W4A8, bits=4, group_size=128,
        weight_dtype=QuantDType.INT4, activation_dtype=QuantDType.INT8,
        scale_encoding=ScaleEncoding.FLOAT32, packing_layout=PackingLayout.Q4_NZ,
        operator="MsQuant4N0Group128",
    ),
    QuantType.W2A16: QuantPreset(
        quant_type=QuantType.W2A16, bits=2, group_size=32,
        weight_dtype=QuantDType.INT2, activation_dtype=QuantDType.FLOAT16,
        scale_encoding=ScaleEncoding.FLOAT16, packing_layout=PackingLayout.Q2_PLANAR,
        operator="MsQuant2N0Group32",
    ),
})

EXPORT_QUANT_TYPES = tuple(kind for kind, preset in QUANT_PRESETS.items() if preset.export_targets)


def get_quant_preset(quant_type):
    """Resolve canonical names and legacy aliases to an immutable preset."""
    return QUANT_PRESETS[QuantType(quant_type)]
