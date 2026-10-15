#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Offline-only package writer and model snapshot pruning for BooguImage."""

import json
import logging
import os
import shutil
import time

import torch

logger = logging.getLogger(__name__)
_PKG_FORMAT_VERSION = 3
_PKG_SHARD_BYTES = 4 * 1024 ** 3
_PKG_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")


def _pkg_write_shard(shard_dir, idx, batch):
    fname = f"shard_{idx:03d}.pt"
    torch.save(batch, os.path.join(shard_dir, fname))
    return fname


def _pkg_write_shards(shard_dir, tensors):
    """Write the host tensors into ~4GB shard files; returns file names."""
    os.makedirs(shard_dir, exist_ok=True)
    names = []
    batch, batch_bytes = {}, 0
    for name in sorted(tensors):
        tensor = tensors[name]
        nbytes = tensor.numel() * tensor.element_size()
        if batch and batch_bytes + nbytes > _PKG_SHARD_BYTES:
            names.append(_pkg_write_shard(shard_dir, len(names), batch))
            batch, batch_bytes = {}, 0
        batch[name] = tensor
        batch_bytes += nbytes
    if batch:
        names.append(_pkg_write_shard(shard_dir, len(names), batch))
    return names


def _pkg_safetensors():
    """Load the safetensors I/O used by the offline snapshot writer."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    return load_file, save_file, safe_open


def _pkg_pipeline_modules(pipe):
    """Iterate neural network components of a Diffusers pipeline.

    A pipeline is a container, not an nn.Module. Its registered components
    include modules as well as tokenizers, processors and schedulers.
    """
    return ((name, component) for name, component in pipe.components.items()
            if isinstance(component, torch.nn.Module))


def _pkg_component_of(pipe, path):
    """Pipeline component names whose module tree holds ``path``.

    Attention processor Linears may be regular attributes of a processor
    rather than registered children, so ``get_submodule`` alone cannot find
    them even though their parameters are present in the checkpoint.
    """
    hits = []
    for name, comp in _pkg_pipeline_modules(pipe):
        try:
            comp.get_submodule(path)
        except AttributeError:
            module = comp
            for part in path.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            if not isinstance(module, torch.nn.Module):
                continue
        hits.append(name)
    return hits


def _pkg_dropped_state_keys(pipe, records, modules):
    """state_dict keys whose BF16 tensors the deployed int8 modules replace.

    Returns ``({component: [key, ...]}, [path, ...])``. The keys are read off
    the original modules' own state dicts, so a bias-free Linear asks for its
    weight only. Attribute traversal also resolves joint-attention processor
    Linears, whose parameters appear in the transformer checkpoint without
    being registered as children in the module tree.
    """
    l2_int8 = {meta["path"]: meta.get("has_int8_l2", False)
               for meta in modules}
    dropped, orphan = {}, []
    for unused_parent, unused_attr, original, kind, path in records:
        hits = _pkg_component_of(pipe, path)
        if not hits:
            orphan.append(path)
            continue
        if len(hits) > 1:
            raise ValueError(
                f"{path} resolves in more than one pipeline component "
                f"({hits}); its weights cannot be attributed to one "
                f"component directory of the snapshot")
        if kind == "ffn":
            leaves = ["linear_1", "linear_3"]
            if l2_int8[path]:
                leaves.append("linear_2")
            keys = [f"{path}.{leaf}.{state}" for leaf in leaves
                    for state in getattr(original, leaf).state_dict()]
        else:
            keys = [f"{path}.{state}" for state in original.state_dict()]
        dropped.setdefault(hits[0], []).extend(keys)
    return dropped, orphan


def _pkg_link_or_copy(src, dst):
    """Put one snapshot file in place, hardlinking to save the disk space."""
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _pkg_filter_weight_file(path, drop, api):
    """Rewrite a snapshot weight file without the ``drop`` keys.

    Returns ``(bytes_by_dropped_key, n_dropped)``, or ``None`` when the file
    is not a tensor dict (a ``.bin`` holding e.g. normalization stats) and
    stays as linked. A file left with no tensors is removed here (the only
    owner of snapshot weight files); the index update drops its entries.
    """
    load_file, save_file, safe_open = api
    metadata = None
    if path.endswith(".safetensors"):
        with safe_open(path, framework="pt", device="cpu") as reader:
            metadata = reader.metadata()
        tensors = load_file(path)
    else:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict) or not all(
                torch.is_tensor(v) for v in loaded.values()):
            return None
        tensors = loaded
    removed = {k: t for k, t in tensors.items() if k in drop}
    if not removed:
        return {}, 0
    kept = {k: t for k, t in tensors.items() if k not in drop}
    bytes_by_key = {k: t.numel() * t.element_size()
                    for k, t in removed.items()}
    if os.name == "nt" and path.endswith(".safetensors"):
        # Windows cannot unlink the snapshot while safetensors still maps it.
        kept = {key: tensor.clone() for key, tensor in kept.items()}
        tensors.clear()
        removed.clear()
    os.remove(path)  # break the shared link before writing
    if kept:
        if path.endswith(".safetensors"):
            save_file(kept, path, metadata=metadata)
        else:
            torch.save(kept, path)
    return bytes_by_key, len(bytes_by_key)


def _pkg_update_index(index_path, drop, bytes_by_key):
    """Drop the quantized keys from a weight index and fix its total size.

    ``bytes_by_key`` gives the size of every tensor the component dropped, so
    the index only subtracts the ones its own ``weight_map`` lists (shards
    shared with another index are accounted once per index, correctly).
    Emptied files are already gone (see :func:`_pkg_filter_weight_file`) and
    were referenced solely by dropped keys.
    """
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return
    kept_map = {k: v for k, v in weight_map.items() if k not in drop}
    if len(kept_map) == len(weight_map):
        return
    index["weight_map"] = kept_map
    metadata = index.get("metadata")
    if isinstance(metadata, dict) and "total_size" in metadata:
        lost = sum(bytes_by_key.get(key, 0) for key in weight_map)
        metadata["total_size"] = int(metadata["total_size"]) - lost
    os.remove(index_path)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def _pkg_snapshot_checks(src, dst):
    """Gate the snapshot layout before anything is written."""
    src, dst = os.path.realpath(src), os.path.realpath(dst)
    if not os.path.isdir(src):
        raise ValueError(
            f"the package snapshots the source model directory, which is "
            f"not a directory: {src}")
    same_drive = os.path.splitdrive(src)[0] == os.path.splitdrive(dst)[0]
    if same_drive and os.path.commonpath([src, dst]) == src:
        raise ValueError(
            f"the package directory {dst} is the model directory {src} or "
            f"lives inside it; the snapshot would recurse into itself")
    if os.path.isdir(dst) and os.listdir(dst):
        raise ValueError(
            f"the package directory must be new or empty: {dst}")
    for reserved in ("int8", "manifest.json"):
        if os.path.exists(os.path.join(src, reserved)):
            raise ValueError(
                f"the model directory {src} has a top-level '{reserved}', "
                f"which the package layout reserves")


def _pkg_drop_from_component(comp_dir, drop, api):
    """Drop ``drop`` from one component directory's weight files.

    Returns ``(n_dropped, bytes_by_dropped_key, index_paths)``: the tensors
    removed and their sizes, and the weight indices to update once every file
    is rewritten.
    """
    bytes_by_key, index_files, n_dropped = {}, [], 0
    for root, unused_dirs, files in os.walk(comp_dir):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            if fname.endswith(_PKG_WEIGHT_SUFFIXES):
                result = _pkg_filter_weight_file(fpath, drop, api)
                if result is not None:
                    bytes_by_key.update(result[0])
                    n_dropped += result[1]
            elif fname.endswith(".index.json"):
                index_files.append(fpath)
    return n_dropped, bytes_by_key, index_files


def _pkg_checkpoint_keys(comp_dir, api):
    """Read actual checkpoint names without loading safetensors payloads."""
    keys = set()
    for root, unused_dirs, files in os.walk(comp_dir):
        for filename in sorted(files):
            path = os.path.join(root, filename)
            if filename.endswith(".safetensors"):
                with api[2](path, framework="pt", device="cpu") as reader:
                    keys.update(reader.keys())
            elif filename.endswith((".bin", ".pt")):
                tensors = torch.load(path, map_location="cpu", weights_only=True)
                if isinstance(tensors, dict) and all(torch.is_tensor(value) for value in tensors.values()):
                    keys.update(tensors)
                del tensors
    return keys


def _pkg_resolve_drop_keys(component, comp_dir, drop, api):
    """Resolve MLLM runtime paths to the checkpoint's actual full keys."""
    if component != "mllm":
        return drop
    available = _pkg_checkpoint_keys(comp_dir, api)
    resolved, missing = set(), []
    for key in sorted(drop):
        candidates = (key, f"model.{key}") if key.startswith("language_model.") else (key,)
        matches = [candidate for candidate in candidates if candidate in available]
        if len(matches) > 1:
            raise ValueError(f"ambiguous MLLM snapshot key {key}: {matches}")
        if not matches:
            missing.append(key)
        else:
            resolved.add(matches[0])
    if missing:
        raise ValueError(
            f"W8A8 snapshot: component '{component}' did not find "
            f"{len(missing)}/{len(drop)} expected quantized tensor(s) "
            f"in its weight files (e.g. {missing[:5]})")
    return resolved


def _pkg_snapshot_model_dir(pipe, source_model, out_dir, records, modules):
    """Copy the source model directory into the package minus quantized BF16.

    ``model_index.json``, the per-component configs and any custom code are
    linked as they are; the weight files are rewritten with the tensors the
    int8 modules replace dropped and the ``*.index.json`` files updated to
    match. ``from_pretrained(<out_dir>)`` then rebuilds a pipeline carrying
    only the BF16 still in use, so inference needs this package alone and not
    the original checkpoint. Returns the manifest ``model_snapshot`` dict.
    """
    src = os.path.abspath(os.path.expanduser(source_model))
    dst = os.path.abspath(out_dir)
    _pkg_snapshot_checks(src, dst)
    drop_by_component, orphan = _pkg_dropped_state_keys(pipe, records, modules)
    if orphan:
        raise ValueError(
            f"W8A8 snapshot cannot attribute quantized paths to a model "
            f"component: {orphan[:5]}")
    api = _pkg_safetensors() if drop_by_component else None
    for root, unused_dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        dst_dir = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in sorted(files):
            _pkg_link_or_copy(os.path.join(root, fname),
                              os.path.join(dst_dir, fname))
    components = {}
    for name, keys in sorted(drop_by_component.items()):
        comp_dir = os.path.join(dst, name)
        if not os.path.isdir(comp_dir):
            raise ValueError(
                f"{src} does not store pipeline component '{name}' as its "
                f"own subfolder, so the snapshot cannot drop the tensors "
                f"that int8 replaces")
        drop = _pkg_resolve_drop_keys(name, comp_dir, set(keys), api)
        n_dropped, bytes_by_key, index_files = _pkg_drop_from_component(
            comp_dir, drop, api)
        missing = sorted(drop - bytes_by_key.keys())
        if missing:
            raise ValueError(
                f"W8A8 snapshot: component '{name}' did not find "
                f"{len(missing)}/{len(drop)} expected quantized tensor(s) "
                f"in its weight files: {missing[:5]}")
        for index_path in sorted(index_files):
            _pkg_update_index(index_path, drop, bytes_by_key)
        components[name] = {"dropped_tensors": n_dropped,
                            "dropped_bytes": sum(bytes_by_key.values()),
                            "requested_keys": len(drop),
                            "keys": sorted(drop)}
    return {"path": src, "components": components}


def _pkg_manifest(handle, modules, shard_names, source_model, snapshot):
    """Assemble the manifest content from the handle stats + module meta."""
    stats = handle.stats
    counts = {
        "n_swapped": int(stats.get("n_swapped", len(modules))),
        "n_kept_bf16": int(stats.get("n_kept_bf16", 0)),
        "n_ffn": sum(1 for m in modules if m["kind"] == "ffn"),
        "n_linear": sum(1 for m in modules if m["kind"] == "linear"),
    }
    for key in ("n_l2_int8", "n_qkvo_int8", "n_qkvo_kept_bf16"):
        if key in stats:
            counts[key] = int(stats[key])
    return {
        "format_version": _PKG_FORMAT_VERSION,
        "alg": stats["alg"],
        "alpha": float(stats["alpha"]),
        "calib_prompts": list(stats.get("calib_prompts") or []),
        "source_model": source_model,
        "model_snapshot": snapshot,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "counts": counts,
        "kept_bf16": list(stats.get("kept_bf16_paths") or []),
        "shards": shard_names,
        "modules": modules,
    }


class PackageW8A8:
    """Write an unpadded quantized checkpoint plus its pruned model snapshot."""

    def __init__(self, pipe, source_model, out_dir):
        self.pipe = pipe
        self.source_model = os.path.realpath(os.path.expanduser(source_model))
        self.out_dir = os.path.realpath(os.path.expanduser(out_dir))

    def write(self, result):
        """Persist an offline build result and verify tensor round trips."""
        if not result.records:
            raise ValueError("No calibrated modules to package")
        snapshot = _pkg_snapshot_model_dir(
            self.pipe, self.source_model, self.out_dir,
            result.records, result.modules)
        shards = _pkg_write_shards(os.path.join(self.out_dir, "int8"), result.tensors)
        self._check_shards(shards, result.tensors)
        manifest = _pkg_manifest(result, result.modules, shards,
                                 self.source_model, snapshot)
        manifest["tensor_layout"] = "unpadded_kn"
        manifest["targets"] = result.targets
        with open(os.path.join(self.out_dir, "manifest.json"), "w",
                  encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
        logger.info("W8A8 package ready: %s", self.out_dir)
        return manifest

    def _check_shards(self, shards, expected):
        """Read each shard once and check every written tensor bit for bit."""
        remaining = set(expected)
        for filename in shards:
            tensors = torch.load(os.path.join(self.out_dir, "int8", filename),
                                 map_location="cpu", weights_only=True)
            for name, value in tensors.items():
                if name not in remaining or not torch.equal(value, expected[name]):
                    raise ValueError(f"Package round-trip mismatch: {name}")
                remaining.remove(name)
        if remaining:
            raise ValueError(f"Package omitted tensors: {sorted(remaining)}")
