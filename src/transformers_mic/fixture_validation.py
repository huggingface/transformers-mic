"""Structural checks independent of checkpoint names and generated weight values."""

import copy
import json


def sync_dtype(model):
    """Fixture builders cast the whole model; reflect that in every subconfig.

    Tensor.to() does not update config metadata, and save_pretrained only updates
    the root dtype. Quantized weights keep this as their non-quantized dtype.
    """
    from transformers import PretrainedConfig

    def update(config):
        config.dtype = model.dtype
        for value in vars(config).values():
            if isinstance(value, PretrainedConfig):
                update(value)
            elif isinstance(value, dict):
                for sub in value.values():
                    if isinstance(sub, PretrainedConfig):
                        update(sub)

    update(model.config)


def validate_saved_config(model, path):
    """Check the actual JSON round-trip and its agreement with the built weights."""
    saved = json.loads(path.read_text())
    restored = type(model.config).from_dict(saved)
    resolved = json.loads(restored.to_json_string(use_diff=False))

    def changed(before, after, prefix=""):
        differences = []
        for key, value in before.items():
            if key.startswith("_") or key == "transformers_version":
                continue
            name = f"{prefix}{key}"
            actual = after.get(key)
            if isinstance(value, dict) and isinstance(actual, dict):
                differences.extend(changed(value, actual, name + "."))
            elif value != actual:
                differences.append(f"{name}: saved {value!r}, reloaded {actual!r}")
        return differences

    differences = changed(saved, resolved)
    if differences:
        raise ValueError("Stale fixture config: " + "; ".join(differences))
    validate_model(model, config=restored)


def config_findings(config, prefix="model"):
    findings = []
    rope, nope = config.get("qk_rope_head_dim"), config.get("qk_nope_head_dim")
    if isinstance(rope, int) and isinstance(nope, int):
        total = config.get("qk_head_dim")
        if total is not None and total != rope + nope:
            findings.append(f"{prefix}: qk_head_dim {total} differs from rotary + non-rotary width {rope + nope}")
        index = config.get("index_head_dim")
        if isinstance(index, int) and rope > index:
            findings.append(f"{prefix}: rotary width {rope} exceeds indexer head width {index}")
        if rope < 0 or rope % 2:
            findings.append(f"{prefix}: rotary width {rope} must be nonnegative and even")
    flags, sources = config.get("no_rope_layers"), config.get("kv_source_layers")
    if isinstance(flags, list) and isinstance(sources, list):
        for layer, source in enumerate(sources):
            if not isinstance(source, int) or not 0 <= source <= layer or layer >= len(flags):
                findings.append(f"{prefix}: invalid KV source {source} for layer {layer}")
            elif flags[layer] != flags[source]:
                findings.append(f"{prefix}: layer {layer} and its KV source {source} disagree on RoPE")
    for name, value in config.items():
        if isinstance(value, dict) and (name.endswith("_config") or "model_type" in value):
            findings.extend(config_findings(value, f"{prefix}.{name}"))
    return findings


def validate_model(model, config=None):
    """Reject weights inconsistent with their config, before writing an artifact.

    Loading can replace parameter storage without reporting a shape mismatch.
    Compare against a fresh meta model instead of trusting loading-info alone.
    Fixture fitting uses dense models, so this check runs before quantization.
    """
    import torch

    config = copy.deepcopy(config if config is not None else model.config)
    findings = config_findings(config.to_dict())
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    with torch.device("meta"):
        expected = type(model)._from_config(config)
    actual = dict(model.named_parameters())
    for name, parameter in expected.named_parameters():
        if name not in actual:
            findings.append(f"missing parameter {name}")
        elif actual[name].shape != parameter.shape:
            findings.append(f"{name}: weight shape {tuple(actual[name].shape)} differs from config {tuple(parameter.shape)}")
    if findings:
        raise ValueError("Invalid fixture: " + "; ".join(findings))
