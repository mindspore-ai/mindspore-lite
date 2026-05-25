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


def _get_package_data() -> list[str]:
    """
    Get package data.
    """
    return ["lib/*.so*"]


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
    ],
    package_dir={
        "lite_boost": ".",
        "lite_boost.ops": "ops",
        "lite_boost.parallel": "parallel",
        "lite_boost.layers": "layers",
        "lite_boost.model": "model",
        "lite_boost.model.wan2_1": "model/wan2_1",
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
