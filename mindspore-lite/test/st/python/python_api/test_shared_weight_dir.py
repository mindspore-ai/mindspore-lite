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
Test directory conversion with a shared deduplicated external weight file.

Passing a directory as modelFile converts the *.onnx under its rank sub directories into one
*_graph.mindir each (graph only, the "_graph" suffix matching the single-model SplitSave naming),
mirroring the relative layout below the rank directory, and the whole batch shares a single
external weight file ("<outputFile>/variables/data_0") in which byte-identical tensors are stored
once.

The main model set is the context-parallel Qwen3-0.6B export:

    qwen3_0.6b/rank0/prefill/qwen3_llm_prefill.onnx
    qwen3_0.6b/rank0/decode/qwen3_llm_decode.onnx
    qwen3_0.6b/rank1/prefill/qwen3_llm_prefill.onnx
    qwen3_0.6b/rank1/decode/qwen3_llm_decode.onnx

Every rank graph carries the same ~1.4 GB of weights, so the four files have to collapse into a
single ~1.4 GB weight file. Those graphs contain
Custom(AllGather), which cannot be loaded on a single card, so they only cover the conversion,
the mirrored layout and the weight de-duplication. A small model covers the rest: it is
externalized into the shared file just like the big one, and being free of collectives it can
also be built and inferred on one card.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import mindspore_lite as mslite

CONVERTER_LITE = "tools/converter/converter/converter_lite"
SHARED_WEIGHT_REL = Path("variables") / "data_0"
# The header written by SharedWeightStore; anything above it is real tensor data.
SHARED_WEIGHT_HEADER_SIZE = 64

# Folder of the model set inside the ST model directory (--mindir_dir).
MODEL_ROOT_NAME = "qwen3_0.6b"
# Every *.onnx below a rank directory is converted, everything else in the folder is ignored.
MODEL_GLOB = "rank*/**/*.onnx"
# A tiny model (266 B, one 4x4 weight): its batch still gets a shared weight file, and being free
# of collective ops its mindirs can be built on a single card.
SMALL_MODEL = "single_matmul_model.onnx"
SMALL_MODEL_INPUT = ("input", (1, 4))


def mindir_rel(onnx_rel):
    """Output path mirrored from an input onnx: same relative path, "<stem>_graph.mindir"."""
    return onnx_rel.with_name(onnx_rel.stem + "_graph.mindir")


BACKENDS = ("mslite_large_model_inference_arm_ascend910B",)
DEVICE_ID = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
# The rank graphs hold the same weights, so converting the whole batch must not cost much more
# than converting a single model (the remainder is a few KB of rank-specific constants).
DEDUP_MAX_RATIO = 1.1


def run_converter_lite(so_path, model_file, output_file, optimize="none"):
    """Run converter_lite on a model file (or directory) and return the completed process."""
    cmd = [
        Path(so_path) / CONVERTER_LITE,
        "--fmk=ONNX",
        # The shared-weight directory conversion only supports MindIR output.
        "--saveType=MINDIR",
        f"--optimize={optimize}",
        f"--modelFile={model_file}",
        f"--outputFile={output_file}",
    ]
    return subprocess.run(cmd, shell=False, capture_output=True, text=True, check=False)


def build_and_predict(model_path):
    """Build a self-contained mindir on Ascend and run one inference on all-ones input."""
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.provider = "ge"
    context.ascend.device_id = DEVICE_ID
    model = mslite.Model()
    model.build_from_file(str(model_path), mslite.ModelType.MINDIR, context)

    tensor = model.get_inputs()[0]
    tensor.set_data_from_numpy(np.ones(SMALL_MODEL_INPUT[1], dtype=np.float32))
    return [out.get_data_to_numpy() for out in model.predict([tensor])]


def small_model_file(mindir_dir):
    """The small model, used by the cases that do not need the big model set."""
    path = Path(mindir_dir) / SMALL_MODEL
    if not path.is_file():
        pytest.fail(f"{path} not found, set --mindir_dir to the model directory", pytrace=False)
    return path


def discover_models(model_root):
    """Return the onnx paths below the rank directories, relative to model_root."""
    if not model_root.is_dir():
        pytest.fail(f"model set {model_root} not found, set --mindir_dir to its parent", pytrace=False)
    rels = sorted(path.relative_to(model_root) for path in model_root.glob(MODEL_GLOB))
    if not rels:
        pytest.fail(f"no {MODEL_GLOB} found below {model_root}", pytrace=False)
    ranks = sorted({rel.parts[0] for rel in rels})
    assert ranks == [f"rank{index}" for index in range(len(ranks))], \
        f"the model set has no consecutive rank directories: {ranks}"
    return rels


def link_or_copy(src, dst):
    """
    Hardlink the model files so the multi-GB weights are not duplicated on disk.

    A symlink cannot be used here: the onnx parser canonicalizes the external data path and
    rejects it when it resolves outside the model directory (path traversal check).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_batch_input(base_dir, model_root, rels):
    """Hardlink the whole model set into the batch input tree."""
    input_dir = base_dir / "batch_in"
    for rel in rels:
        link_or_copy(model_root / rel, input_dir / rel)
    return input_dir


def prepare_single_input(base_dir, model_root, rels):
    """Reference layout with one model: its weight file holds exactly one copy of the weights."""
    input_dir = base_dir / "single_in"
    link_or_copy(model_root / rels[0], input_dir / rels[0])
    return input_dir


def convert_case(so_path, input_dir, out_dir):
    """Convert one input directory and collect the result."""
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    process = run_converter_lite(so_path, input_dir, out_dir)
    return {
        "returncode": process.returncode,
        # Only the tail of stderr is kept, the log of a real model is very large.
        "stderr": process.stderr[-8000:],
        "output": str(out_dir),
    }


def convert_small_batch(so_path, mindir_dir, base_dir, model_sub_dirs):
    """
    Convert the small model as one batch, one copy per given rank sub directory.

    model_sub_dirs are paths relative to the batch input root, e.g. ("rank0/prefill",
    "rank1/prefill"). The model set is small so it can be converted once per case.
    """
    model_file = small_model_file(mindir_dir)
    input_dir = base_dir / "in"
    for sub_dir in model_sub_dirs:
        link_or_copy(model_file, input_dir / sub_dir / SMALL_MODEL)
    return convert_case(so_path, input_dir, base_dir / "out")


@pytest.fixture(name="shared_weight_cases", scope="module")
def shared_weight_dir_fixture(so_path, mindir_dir, output_dir):
    """
    Convert the model set as one batch and one model alone, and report both results.

    The conversion result is cached in a json file because the ST runner uses pytest-forked,
    which re-executes module-scoped fixtures for every test and would otherwise convert the
    whole model again for each test.
    """
    model_root = Path(mindir_dir) / MODEL_ROOT_NAME
    rels = discover_models(model_root)
    base_dir = Path(output_dir) / "shared_weight_dir"

    result_file = base_dir / "convert_result.json"
    if result_file.exists():
        results = json.loads(result_file.read_text())
    else:
        results = {}
        for name, prepare in (("batch", prepare_batch_input), ("single", prepare_single_input)):
            input_dir = prepare(base_dir, model_root, rels)
            results[name] = convert_case(so_path, input_dir, base_dir / f"{name}_out")
        result_file.write_text(json.dumps(results))

    out_dir = Path(results["batch"]["output"])
    # Required output (relative to --outputFile): one "<same relative path>_graph.mindir" per
    # input onnx, plus the single shared external weight file. See the layout comment in
    # test_shared_weight_dir_mirrors_layout_and_shared_weight_file for the exact rules.
    expected_files = {mindir_rel(rel).as_posix() for rel in rels}
    expected_files.add(SHARED_WEIGHT_REL.as_posix())
    yield {
        "returncode": results["batch"]["returncode"],
        "stderr": results["batch"]["stderr"],
        "output": out_dir,
        "expected_files": expected_files,
        # The layout below the rank directory is mirrored:
        # "rank0/prefill/x.onnx" -> "rank0/prefill/x_graph.mindir".
        "mindirs": [out_dir / mindir_rel(rel) for rel in rels],
        "single_returncode": results["single"]["returncode"],
        "single_shared_weight": Path(results["single"]["output"]) / SHARED_WEIGHT_REL,
    }


@pytest.mark.backend(*BACKENDS)
def test_shared_weight_dir_convert_success(shared_weight_cases):
    """The directory conversion should succeed."""
    result = shared_weight_cases
    assert result["returncode"] == 0, f"directory conversion failed:\n{result['stderr']}"
    assert result["single_returncode"] == 0


@pytest.mark.backend(*BACKENDS)
def test_shared_weight_dir_mirrors_layout_and_shared_weight_file(shared_weight_cases):
    """The output mirrors the rank layout and holds exactly one shared weight file."""
    out_dir = shared_weight_cases["output"]

    # Required output layout (paths below are relative to the --outputFile directory).
    # Every file below a rank directory is mirrored 1:1, only the suffix changes
    # (".onnx" -> "_graph.mindir"); the weights collapse into one shared file:
    #
    #   --modelFile=<input_dir>/                       --outputFile=<output_dir>/
    #   +-- rank0/                                     +-- rank0/
    #   |   +-- prefill/                               |   +-- prefill/
    #   |   |   +-- qwen3_llm_prefill.onnx    ------>  |   |   +-- qwen3_llm_prefill_graph.mindir
    #   |   +-- decode/                                |   +-- decode/
    #   |       +-- qwen3_llm_decode.onnx     ------>  |       +-- qwen3_llm_decode_graph.mindir
    #   +-- rank1/                                     +-- rank1/
    #   |   +-- prefill/                               |   +-- prefill/
    #   |   |   +-- qwen3_llm_prefill.onnx    ------>  |   |   +-- qwen3_llm_prefill_graph.mindir
    #   |   +-- decode/                                |   +-- decode/
    #   |       +-- qwen3_llm_decode.onnx     ------>  |       +-- qwen3_llm_decode_graph.mindir
    #                                                  +-- variables/
    #                                                      +-- data_0  (the one shared weight file)
    #
    # Rules the layout encodes:
    #   * one mindir per onnx, at the same relative path and depth (never flattened);
    #   * exactly one weight file, "<outputFile>/variables/data_0", shared by all mindirs;
    #   * no other file is produced.
    produced_files = {
        path.relative_to(out_dir).as_posix() for path in out_dir.rglob("*") if path.is_file()
    }
    assert produced_files == shared_weight_cases["expected_files"]

    # Larger than the bare header, i.e. the weights were really externalized into this file.
    assert (out_dir / SHARED_WEIGHT_REL).stat().st_size > SHARED_WEIGHT_HEADER_SIZE


@pytest.mark.backend(*BACKENDS)
def test_shared_weight_dir_deduplicates_identical_weights(shared_weight_cases):
    """The identical rank graphs must reuse the weight bytes of the first one."""
    shared_weight = (shared_weight_cases["output"] / SHARED_WEIGHT_REL).stat().st_size
    single_weight = shared_weight_cases["single_shared_weight"].stat().st_size

    # Four rank graphs must not store four times the weights, nor more than the small margin
    # taken by the few rank-specific constants.
    assert shared_weight < single_weight * DEDUP_MAX_RATIO, \
        f"shared weight file {shared_weight} is not deduplicated (single model: {single_weight})"


@pytest.mark.backend(*BACKENDS)
def test_shared_weight_dir_externalizes_weights(shared_weight_cases):
    """Every mindir only holds its graph, the weights live in the shared file."""
    out_dir = shared_weight_cases["output"]
    shared_weight = (out_dir / SHARED_WEIGHT_REL).stat().st_size

    mindirs = shared_weight_cases["mindirs"]
    assert mindirs, "no mindir was produced"
    for mindir in mindirs:
        assert mindir.is_file(), f"{mindir} is missing"
        # The graph of a 0.6B model is a few MB; the multi-GB weights must not be inlined.
        assert mindir.stat().st_size * 10 < shared_weight, \
            f"{mindir} seems to inline its weights (size {mindir.stat().st_size})"


@pytest.mark.backend(*BACKENDS)
def test_shared_weight_dir_small_model_still_shares_weights(so_path, mindir_dir, output_dir):
    """A small model is externalized into the shared weight file too, and can still be built.

    The directory conversion does not look at the model size: every model of the batch records its
    weights in the shared file, so even a 266 B onnx leaves a (small) "variables/data_0" behind
    instead of an inline copy. Such a graph has no collective ops, so it is the one batch here that
    can be built on a single card, which exercises the runtime half of the feature: resolving
    external_data through the weight root recorded in the mindir.
    """
    base_dir = Path(output_dir) / "shared_weight_dir" / "small"
    result = convert_small_batch(so_path, mindir_dir, base_dir, ("rank0/prefill", "rank1/prefill"))
    assert result["returncode"] == 0, f"small-model conversion failed:\n{result['stderr']}"

    out_dir = Path(result["output"])
    stem = Path(SMALL_MODEL).stem
    produced = {path.relative_to(out_dir).as_posix() for path in out_dir.rglob("*") if path.is_file()}
    # One graph-only mindir per input onnx, plus the shared weight file the batch always produces.
    assert produced == {
        f"rank0/prefill/{stem}_graph.mindir",
        f"rank1/prefill/{stem}_graph.mindir",
        SHARED_WEIGHT_REL.as_posix(),
    }

    # Data was really written into the shared file, not just its 64-byte header.
    assert (out_dir / SHARED_WEIGHT_REL).stat().st_size > SHARED_WEIGHT_HEADER_SIZE

    # Both ranks resolve the weights through the shared file and infer to the same result.
    # Compared flattened: the runtime sometimes reports the same [1, 4] output with an extra rank.
    values = [
        np.asarray(build_and_predict(out_dir / rel / f"{stem}_graph.mindir")[0]).reshape(-1)
        for rel in ("rank0/prefill", "rank1/prefill")
    ]
    assert values[0].size == 4, f"unexpected output size {values[0].size}"
    # Zeros would mean the external weights were not applied at all.
    assert not np.allclose(values[0], 0), "the shared weights were not applied"
    np.testing.assert_allclose(values[1], values[0])


@pytest.mark.backend(*BACKENDS)
def test_shared_weight_dir_small_model_matches_single_conversion(so_path, mindir_dir, output_dir):
    """Both batch graphs must infer exactly like the same model converted on its own.

    The batch conversion externalizes the weights into the shared file, while a single-file
    conversion keeps them inline, so equal outputs show that resolving the weights through the
    shared file does not change the numerical result.
    """
    base_dir = Path(output_dir) / "shared_weight_dir" / "batch_vs_single"
    result = convert_small_batch(so_path, mindir_dir, base_dir, ("rank0/prefill", "rank1/prefill"))
    assert result["returncode"] == 0, f"small-model conversion failed:\n{result['stderr']}"
    out_dir = Path(result["output"])
    stem = Path(SMALL_MODEL).stem

    # Reference: the same model converted alone, with its weights inlined in the mindir.
    single_dir = base_dir / "single_out"
    single_dir.mkdir(parents=True, exist_ok=True)
    process = run_converter_lite(so_path, small_model_file(mindir_dir), single_dir / stem)
    assert process.returncode == 0, f"single-model conversion failed:\n{process.stderr[-8000:]}"
    single_mindir = single_dir / f"{stem}.mindir"
    assert single_mindir.is_file()

    expected = np.asarray(build_and_predict(single_mindir)[0]).reshape(-1)
    assert expected.size == 4, f"unexpected output size {expected.size}"
    # Zeros would mean the single conversion lost its weights, leaving nothing to compare against.
    assert not np.allclose(expected, 0), "the single-model output is all zeros"

    # Both ranks of the batch must match the reference, i.e. the shared weights were resolved to
    # the same values as the inline copy.
    for rel in ("rank0/prefill", "rank1/prefill"):
        actual = np.asarray(build_and_predict(out_dir / rel / f"{stem}_graph.mindir")[0]).reshape(-1)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6,
                                   err_msg=f"{rel} differs from the single conversion")


@pytest.mark.backend(*BACKENDS)
def test_shared_weight_dir_accepts_renamed_sub_dir(so_path, mindir_dir, output_dir):
    """Any directory name below a rank directory is allowed; the mindir mirrors it, e.g. prefill -> prefix."""
    base_dir = Path(output_dir) / "shared_weight_dir" / "renamed_sub_dir"
    result = convert_small_batch(so_path, mindir_dir, base_dir, ("rank0/prefix", "rank1/prefix"))
    assert result["returncode"] == 0, f"conversion with renamed sub dirs failed:\n{result['stderr']}"

    out_dir = Path(result["output"])
    stem = Path(SMALL_MODEL).stem
    produced = {path.relative_to(out_dir).as_posix() for path in out_dir.rglob("*") if path.is_file()}
    assert produced == {
        f"rank0/prefix/{stem}_graph.mindir",
        f"rank1/prefix/{stem}_graph.mindir",
        SHARED_WEIGHT_REL.as_posix(),
    }


@pytest.mark.backend(*BACKENDS)
def test_shared_weight_dir_accepts_trailing_slash(so_path, mindir_dir, output_dir):
    """--outputFile may end with "/"; the trailing separator is normalized away."""
    base_dir = Path(output_dir) / "shared_weight_dir" / "trailing_slash"
    model_file = small_model_file(mindir_dir)
    input_dir = base_dir / "in"
    for rank in ("rank0", "rank1"):
        link_or_copy(model_file, input_dir / rank / "prefill" / SMALL_MODEL)

    out_dir = base_dir / "out"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    process = run_converter_lite(so_path, input_dir, str(out_dir) + "/")

    assert process.returncode == 0, f"conversion with a trailing slash failed:\n{process.stderr[-8000:]}"
    stem = Path(SMALL_MODEL).stem
    produced = {path.relative_to(out_dir).as_posix() for path in out_dir.rglob("*") if path.is_file()}
    assert produced == {
        f"rank0/prefill/{stem}_graph.mindir",
        f"rank1/prefill/{stem}_graph.mindir",
        SHARED_WEIGHT_REL.as_posix(),
    }


@pytest.mark.backend(*BACKENDS)
def test_shared_weight_dir_load_fails_when_depth_changed(so_path, mindir_dir, output_dir):
    """Moving a mindir to another depth breaks its recorded weight root, so loading must fail."""
    base_dir = Path(output_dir) / "shared_weight_dir" / "depth_changed"
    result = convert_small_batch(so_path, mindir_dir, base_dir, ("rank0/prefill",))
    assert result["returncode"] == 0, f"small-model conversion failed:\n{result['stderr']}"

    mindir = Path(result["output"]) / "rank0" / "prefill" / f"{Path(SMALL_MODEL).stem}_graph.mindir"
    assert mindir.is_file()
    # One directory deeper: the recorded "../.." now resolves below rank0/prefill, where no
    # weight file exists.
    moved = mindir.parent / "deeper" / mindir.name
    moved.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(mindir), str(moved))

    with pytest.raises(RuntimeError):
        build_and_predict(moved)


@pytest.mark.backend(*BACKENDS)
def test_shared_weight_dir_loads_renamed_mindir(so_path, mindir_dir, output_dir):
    """The weight root is resolved from the mindir directory, so its file name is irrelevant."""
    base_dir = Path(output_dir) / "shared_weight_dir" / "renamed_mindir"
    result = convert_small_batch(so_path, mindir_dir, base_dir, ("rank0/prefill",))
    assert result["returncode"] == 0, f"small-model conversion failed:\n{result['stderr']}"

    mindir = Path(result["output"]) / "rank0" / "prefill" / f"{Path(SMALL_MODEL).stem}_graph.mindir"
    renamed = mindir.with_name("renamed.mindir")
    shutil.copy2(mindir, renamed)

    values = [np.asarray(build_and_predict(path)[0]).reshape(-1) for path in (mindir, renamed)]
    assert values[0].size == 4, f"unexpected output size {values[0].size}"
    assert not np.allclose(values[0], 0), "the shared weights were not applied"
    np.testing.assert_allclose(values[1], values[0])


@pytest.mark.backend(*BACKENDS)
def test_shared_weight_dir_rejects_other_optimize(so_path, mindir_dir, output_dir):
    """Directory input is only supported for ONNX->MindIR with --optimize=none."""
    model_file = small_model_file(mindir_dir)
    rel = Path("rank0") / "prefill" / SMALL_MODEL
    input_dir = Path(output_dir) / "shared_weight_dir" / "unsupported" / "in"
    link_or_copy(model_file, input_dir / rel)

    output_root = Path(output_dir) / "shared_weight_dir" / "unsupported" / "out"
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    process = run_converter_lite(so_path, input_dir, output_root, optimize="general")

    assert process.returncode != 0
    assert "only supports ONNX to MindIR" in (process.stdout + process.stderr)


@pytest.mark.backend(*BACKENDS)
def test_shared_weight_dir_rejects_rank_without_model(so_path, mindir_dir, output_dir):
    """Every rank0..rankn directory has to hold at least one onnx model."""
    model_file = small_model_file(mindir_dir)
    input_dir = Path(output_dir) / "shared_weight_dir" / "empty_rank" / "in"
    link_or_copy(model_file, input_dir / "rank0" / "prefill" / SMALL_MODEL)
    (input_dir / "rank1" / "prefill").mkdir(parents=True, exist_ok=True)

    output_root = Path(output_dir) / "shared_weight_dir" / "empty_rank" / "out"
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    process = run_converter_lite(so_path, input_dir, output_root)

    assert process.returncode != 0
    assert "must contain at least one onnx model" in (process.stdout + process.stderr)


@pytest.mark.backend(*BACKENDS)
def test_shared_weight_dir_rejects_non_consecutive_ranks(so_path, mindir_dir, output_dir):
    """The rank directories have to be named rank0, rank1, ... consecutively."""
    model_file = small_model_file(mindir_dir)
    input_dir = Path(output_dir) / "shared_weight_dir" / "gap" / "in"
    for rank in ("rank0", "rank2"):
        link_or_copy(model_file, input_dir / rank / "prefill" / SMALL_MODEL)

    output_root = Path(output_dir) / "shared_weight_dir" / "gap" / "out"
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    process = run_converter_lite(so_path, input_dir, output_root)

    assert process.returncode != 0
    assert "consecutively" in (process.stdout + process.stderr)
