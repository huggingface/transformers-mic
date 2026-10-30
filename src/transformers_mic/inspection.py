"""Inspect parameter names from headers/indexes, with a bounded binary fallback."""

import json
from pathlib import Path
import re

from huggingface_hub import HfApi


def checkpoint_keys(repo, revision, source):
    source = Path(source)
    index = source / "model.safetensors.index.json"
    if index.exists():
        return set(json.loads(index.read_text())["weight_map"])
    binary_index = source / "pytorch_model.bin.index.json"
    if binary_index.exists():
        return set(json.loads(binary_index.read_text())["weight_map"])
    api = HfApi()
    if not api.file_exists(repo, "model.safetensors", revision=revision):
        # Transformers loads a binary-only repo from the bot's open safetensors
        # conversion PR; its header names the same tensors without a download.
        converted = conversion_pr_keys(api, repo)
        if converted:
            return converted
        import torch
        from huggingface_hub import hf_hub_download

        info = api.model_info(repo, revision=revision, files_metadata=True)
        file = next(
            (f for f in info.siblings if f.rfilename == "pytorch_model.bin"), None
        )
        if file is None or file.size is None or file.size > 1024**3:
            raise ValueError(
                "No parameter index available; binary-only checkpoint exceeds the 1 GiB inspection limit"
            )
        path = hf_hub_download(repo, "pytorch_model.bin", revision=revision)
        return set(torch.load(path, map_location="meta", weights_only=True))
    return set(
        api.parse_safetensors_file_metadata(
            repo, "model.safetensors", revision=revision, timeout=60
        ).tensors
    )


def loads_binary(repo, revision, source):
    """Whether from_pretrained reads this pinned checkpoint's pytorch_model.bin.

    It falls back to .bin when the revision has no safetensors file; the bot's
    conversion PR is only used for use_safetensors=True at revision main.
    """
    if (Path(source) / "model.safetensors.index.json").exists():
        return False
    return not HfApi().file_exists(repo, "model.safetensors", revision=revision)


def conversion_pr_keys(api, repo):
    for discussion in api.get_repo_discussions(
        repo,
        author="SFconvertbot",
        discussion_type="pull_request",
        discussion_status="open",
    ):
        if discussion.title == "Adding `safetensors` variant of this model":
            return set(
                api.get_safetensors_metadata(
                    repo, revision=f"refs/pr/{discussion.num}"
                ).weight_map
            )
    return None


def missing_from_keys(model, keys):
    expected = set(model.state_dict())
    keys = {
        name
        if name in expected
        else name.removesuffix(".gamma") + ".weight"
        if name.endswith(".gamma")
        else name.removesuffix(".beta") + ".bias"
        if name.endswith(".beta")
        else name
        for name in keys
    }
    prefix = getattr(model, "base_model_prefix", "")
    if prefix:
        keys |= {
            prefix + "." + name for name in keys if prefix + "." + name in expected
        }
    # A tied parameter is present if any of its aliases is serialized.
    groups = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), set()).add(name)
    covered = set(keys)
    for aliases in groups.values():
        if aliases & keys:
            covered.update(aliases)
    # Some models expose static ties before aliasing tensors on the meta device.
    for target, source in getattr(model, "all_tied_weights_keys", {}).items():
        if source in covered:
            covered.add(target)
    ignored = getattr(model, "_keys_to_ignore_on_load_missing", None) or []
    missing = sorted(
        name
        for name in expected - covered
        if not name.endswith("num_batches_tracked")
        if not any(re.search(pattern, name) for pattern in ignored)
    )
    # Name conversion / packed quantized weights need a real adapter. Do not
    # silently drop most of a fixture because an old checkpoint names differ.
    if len(missing) > max(8, len(expected) * 0.1):
        raise ValueError(
            f"Checkpoint names need a conversion adapter: {len(missing)}/{len(expected)} missing, examples {missing[:8]}"
        )
    return missing


def inspect_checkpoint(cls, source, repo, revision):
    import torch

    config = cls.config_class.from_pretrained(source)
    keys = checkpoint_keys(repo, revision, source)
    keys = logical_checkpoint_keys(config, keys)
    with torch.device("meta"):
        model = cls._from_config(config)
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import (
        WeightConverter,
        WeightRenaming,
        rename_source_key,
    )

    conversions = get_model_conversion_mapping(model)
    renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
    converters = [c for c in conversions if isinstance(c, WeightConverter)]
    state = model.state_dict()
    converted = set()
    for key in keys:
        if key in state:
            converted.add(key)
            continue
        renamed, matched = rename_source_key(
            key, renamings, converters, model.base_model_prefix, state
        )
        converted.add(renamed)
        if matched:
            for converter in converters:
                # A sub-model's converter is scoped to its renamed prefix
                # (vision_model -> vision_tower.vision_encoder), so it cannot
                # re-match the original key: identify it by the matched pattern.
                scope = getattr(converter, "scope_prefix", None)
                if scope and f"{scope}." not in renamed:
                    continue
                _, source_pattern = converter.rename_source_key(key)
                if (
                    source_pattern == matched or matched in converter.source_patterns
                ) and len(converter.target_patterns) > 1:
                    # Expand the fully renamed key: the converter alone skips
                    # renamings such as encoder.layers -> layers.
                    first = converter.target_patterns[0]
                    converted.update(
                        renamed.replace(first, target)
                        for target in converter.target_patterns
                    )
    return missing_from_keys(model, converted)


def logical_checkpoint_keys(config, keys):
    """Map quantized checkpoint names to the dense slim's names; do not decode weights."""
    quantization = getattr(config, "quantization_config", None)
    if not quantization:
        return keys
    if quantization.get("quant_method") in ("fp8", "mxfp8"):
        # Block-wise FP8: each quantized `.weight` keeps its name and gains a
        # `.weight_scale_inv` sibling; the scales are not parameters of a dense model.
        scales = {k for k in keys if k.endswith(".weight_scale_inv")}
        orphans = [k for k in scales if k.removesuffix("_scale_inv") not in keys]
        if orphans:
            raise ValueError(f"FP8 scale without its weight: {sorted(orphans)[:3]}")
        return set(keys) - scales
    if config.model_type != "gpt_oss" or quantization.get("quant_method") != "mxfp4":
        raise ValueError("Quantized checkpoint needs a reviewed parameter-name adapter")
    logical = set(keys)
    packed = [
        key
        for key in keys
        if key.endswith((".gate_up_proj_blocks", ".down_proj_blocks"))
    ]
    if not packed:
        raise ValueError("GPT-OSS MXFP4 index has no supported packed expert tensors")
    for key in packed:
        prefix = key.removesuffix("_blocks")
        if prefix + "_scales" not in keys:
            raise ValueError(f"GPT-OSS MXFP4 tensor lacks scales: {key}")
        logical.add(prefix)
    return logical
