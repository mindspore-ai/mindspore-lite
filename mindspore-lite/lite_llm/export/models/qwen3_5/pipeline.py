"""Checkpoint-to-paged-package export for the Kirin 9030 MoE backend."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import onnx

from .graphs import export_auxiliary_graphs, run_exporter
from .package import finalize_package, pack_package, validate_checkpoint


def compiler_environment():
    """Use the isolated operator DDK without changing the caller's environment."""
    configured = os.environ.get("DDK_PATH")
    if not configured:
        raise ValueError("Source the DDK environment and set DDK_PATH to the Qwen3.5 operator DDK")
    ddk = Path(configured).absolute()
    platform = ddk / "tools/platform/kirin9030"
    omg_root = ddk / "tools/tools_omg/master"
    for required in (platform / "lib64/libcustom_op.so", omg_root / "omg"):
        if not required.is_file():
            raise FileNotFoundError(required)
    environment = os.environ.copy()
    environment.update(DDK_PATH=str(ddk), TOOLCHAIN_HOME=str(ddk),
                       HIAI_VERSION=str(omg_root), SOC_VERSION="kirin9030")
    for key, paths in {
        "LD_LIBRARY_PATH": [platform / "lib64", omg_root / "lib64"],
        "PYTHONPATH": [platform / "ops/impl"],
        "PATH": [ddk / "ddk/ccec_compiler/bin", ddk / "tools/tools_ascendc/package"],
    }.items():
        environment[key] = os.pathsep.join([*map(str, paths), environment.get(key, "")])
    return ddk, environment


def compile_graph(source, destination, *, dynamic=False):
    """Compile in a short temporary path to avoid OMG's output-path limit."""
    ddk, environment = compiler_environment()
    source, destination = Path(source).resolve(), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="q35-") as scratch:
        raw = Path(scratch) / "model"
        command = [str(ddk / "tools/tools_omg/master/omg"), f"--model={source}",
                   "--framework=5", f"--output={raw}", "--target=omc",
                   "--platform=kirin9030", "--weight_data_type=FP16", "--use_origin_format=true"]
        if dynamic:
            inputs, count = [], 0
            for value in onnx.load(source, load_external_data=False).graph.input:
                dims = [dim.dim_value if dim.HasField("dim_value") else -1
                        for dim in value.type.tensor_type.shape.dim]
                if -1 in dims:
                    inputs.append(f"{value.name}:{','.join(map(str, dims))}")
                    count += dims.count(-1)
            if not count:
                raise ValueError(f"No dynamic dimensions in {source}")
            gears = ";".join(",".join([str(sequence)] * count) for sequence in (1, 64))
            command.extend([f"--input_shape={';'.join(inputs)}", f"--dynamic_dims={gears}",
                            "--save_weights_as_external_data=true"])
        with destination.with_suffix(".compile.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        if dynamic:
            models = list(raw.glob("*.omc"))
            if len(models) != 1:
                raise RuntimeError(f"Expected one compiled resident model, got {models}")
            run_exporter("fix_multishape_omc_gears", models[0], destination, "--onnx", source,
                         "--ge-ir-module-dir", ddk / "tools/tools_ascendc/package/python/opc_tool")
            shutil.copy2(raw / "SubGraph_0.weight", destination.parent / "SubGraph_0.weight")
        else:
            shutil.copy2(raw.with_suffix(".omc"), destination)
    return destination


def export_qwen3_5(args, work_dir):
    """Export all layers/experts, then publish a single .msl package."""
    output = Path(args.output).absolute()
    if output.suffix.lower() != ".msl":
        raise ValueError("Qwen3.5-35B requires an --output path ending in .msl")
    validate_checkpoint(args.model)
    if args.target != "kirin9030" or args.max_length != 8192 or args.chunk_size != 64:
        raise ValueError("Qwen3.5-35B requires --target kirin9030 --max-length 8192 --chunk-size 64")
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    compiler_environment()  # Check operator installation before exporting weights.
    package = Path(work_dir) / "package"
    package.mkdir()
    run_exporter("build_model_manifest", "--model-dir", args.model, "--output", package / "model.json")
    run_exporter("export_global_weights", "--model-dir", args.model, "--output-dir", package / "global")
    for layer in range(40):
        with tempfile.TemporaryDirectory(prefix="q35-layer-", dir=work_dir) as temporary:
            run_exporter("export_resident_layer", "--model-dir", args.model, "--layer", layer,
                         "--sequence", 1, "--dynamic-sequence", "--output-dir", temporary)
            weights = package / "weights" / f"layer_{layer:02d}"
            weights.mkdir(parents=True)
            for source in Path(temporary).glob("weight_*.bin"):
                shutil.move(str(source), weights / source.name)
            shutil.copy2(Path(temporary) / "manifest.json", weights / "manifest.json")
            if layer in (0, 3):
                kind = "linear" if layer == 0 else "full"
                stem = f"qwen35_resident_{kind}_s1_s64"
                compile_graph(Path(temporary) / "model.onnx",
                              package / "omc/resident" / stem / f"{stem}.omc", dynamic=True)
    for module, directory, extra in (
        ("export_paged_weights", "experts", ["--fuse-gate-up"]),
        ("export_s16s4_paged_weights", "experts_s16s4", []),
    ):
        run_exporter(module, "--model-dir", args.model, "--output-dir", package / directory,
                     "--layers", "0-39", "--experts", "0-255", *extra)
    for stem, graph in export_auxiliary_graphs(Path(work_dir) / "onnx").items():
        compile_graph(graph, package / "omc" / f"{stem}.omc")
    finalize_package(args.model, package)
    staged = Path(work_dir) / "model.msl"
    pack_package(package, staged)
    staged.rename(output)
    return str(output)
