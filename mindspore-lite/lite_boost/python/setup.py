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
"""
lite_boost setup.py
"""

import glob
import os
import shutil
import sys

from setuptools import setup

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except Exception:
    _bdist_wheel = None
from setuptools.command.build_py import build_py as _build_py


def _read_file(path: str) -> str:
    """
    Read file content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _resolve_top_dir() -> str:
    """
    Resolve top directory.
    """
    if len(sys.argv) >= 2:
        candidate = sys.argv[-1]
        if os.path.isdir(candidate) and os.path.exists(
            os.path.join(candidate, "version.txt")
        ):
            sys.argv.pop()
            return os.path.realpath(candidate)
    return os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def _find_shared_libraries(top_dir: str) -> list[str]:
    """
    Find shared libraries.
    """
    build_dir = os.path.join(top_dir, "build")
    patterns = [
        os.path.join(build_dir, "liblite_boost_ops.so*"),
        os.path.join(build_dir, "lite_boost_ops.so*"),
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    matches = [p for p in matches if os.path.isfile(p)]
    matches.sort()
    return matches


def _vendor_package_data() -> list[str]:
    """AscendC vendor folders (one per SoC) staged by build_all_ops.sh under
    build/custom_ops/<unit>/mslite_custom_ops/. Listed explicitly (setuptools
    package_data has no recursive '**' glob) so the whole tree ships in the wheel.
    Empty when the op build was skipped (non-Ascend/CI)."""
    data = []
    root = os.path.join(TOP_DIR, "build", "custom_ops")
    if os.path.isdir(root):
        for unit in sorted(os.listdir(root)):
            unit_dir = os.path.join(root, unit)
            if not os.path.isdir(unit_dir):
                continue
            for dirpath, _, files in os.walk(unit_dir):
                for fname in files:
                    rel = os.path.relpath(os.path.join(dirpath, fname), root)
                    data.append(os.path.join("custom_ops_vendor", rel))
    return data


def _get_package_data() -> list[str]:
    """
    Get package data.
    """
    return [
        "lib/*.so*",
        # AscendC custom-op vendor folders (one per SoC); the import hook points
        # ASCEND_CUSTOM_OPP_PATH at the matching one (no install to $ASCEND_HOME_PATH).
        *_vendor_package_data(),
    ]


class BuildPy(_build_py):
    """
    Build Python package.
    """

    def run(self):
        """
        Run build command.
        """
        super().run()
        libs = _find_shared_libraries(TOP_DIR)
        if not libs:
            raise RuntimeError(
                "lite_boost_ops shared library not found. "
                "Build lite_boost first to generate build/liblite_boost_ops.so, then run wheel packaging."
            )
        dst_dir = os.path.join(self.build_lib, "lite_boost", "lib")
        os.makedirs(dst_dir, exist_ok=True)
        for lib in libs:
            shutil.copy2(lib, os.path.join(dst_dir, os.path.basename(lib)))

        # Stage the AscendC vendor folders (one per SoC) produced by build.sh.
        # Shipped as folders; the import hook points ASCEND_CUSTOM_OPP_PATH at the
        # matching SoC's folder (no extraction, no install to $ASCEND_HOME_PATH).
        vendor_src = os.path.join(TOP_DIR, "build", "custom_ops")
        vendor_dst = os.path.join(self.build_lib, "lite_boost", "custom_ops_vendor")
        if os.path.isdir(vendor_src):
            for unit in os.listdir(vendor_src):
                unit_src = os.path.join(vendor_src, unit)
                if not os.path.isdir(unit_src):
                    continue
                unit_dst = os.path.join(vendor_dst, unit)
                if os.path.exists(unit_dst):
                    shutil.rmtree(unit_dst)
                shutil.copytree(unit_src, unit_dst)


if _bdist_wheel is not None:

    class BDistWheel(_bdist_wheel):
        """
        Build wheel package.
        """

        def finalize_options(self):
            """
            Finalize options.
            """
            super().finalize_options()
            self.root_is_pure = False


TOP_DIR = _resolve_top_dir()
version = _read_file(os.path.join(TOP_DIR, "version.txt")).strip()


setup(
    name="lite_boost",
    version=version,
    packages=[
        "lite_boost",
        "lite_boost.ops",
        "lite_boost.parallel",
        "lite_boost.layers",
        "lite_boost.model",
        "lite_boost.model.wan2_1",
        "lite_boost.model.wan2_2",
        "lite_boost.model.qwenimage",
        "lite_boost.model.qwen_image_edit",
    ],
    package_dir={
        "lite_boost": ".",
        "lite_boost.ops": "ops",
        "lite_boost.parallel": "parallel",
        "lite_boost.layers": "layers",
        "lite_boost.model": "model",
        "lite_boost.model.wan2_1": "model/wan2_1",
        "lite_boost.model.wan2_2": "model/wan2_2",
        "lite_boost.model.qwenimage": "model/qwenimage",
        "lite_boost.model.qwen_image_edit": "model/qwen_image_edit",
    },
    package_data={"lite_boost": _get_package_data()},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=["torch"],
    cmdclass={
        "build_py": BuildPy,
        **({"bdist_wheel": BDistWheel} if _bdist_wheel is not None else {}),
    },
    zip_safe=False,
)
