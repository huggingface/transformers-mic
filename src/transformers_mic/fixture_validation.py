"""Structural checks independent of checkpoint names and generated weight values."""

import copy
import json


def validate_preservation(original, fixture, quantization=False):
    """A smaller fixture must keep sparse routing and the source's layer kinds."""
    from .shrink_view import configs

    after_configs = dict(configs(fixture))
    failures = []
    for path, before in configs(original):
        after = after_configs.get(path, {})
        if quantization and before.get("quantization_config"):
            method = before["quantization_config"].get("quant_method")
            if (after.get("quantization_config") or {}).get("quant_method") != method:
                failures.append(f"{path}: lost {method} quantization")
        experts = next((key for key in ("n_routed_experts", "num_local_experts", "num_experts", "moe_num_experts")
                        if isinstance(before.get(key), int) and before[key] > 1), None)
        if experts:
            count = after.get(experts, 0)
            if not isinstance(count, int) or count <= 1:
                failures.append(f"{path}: MoE experts became dense")
            else:
                for top_key in ("num_experts_per_tok", "top_k_experts", "moe_k"):
                    top = before.get(top_key)
                    if isinstance(top, int) and 0 < top < before[experts]:
                        new_top = after.get(top_key, count)
                        if not isinstance(new_top, int) or not 0 < new_top < count:
                            failures.append(f"{path}: sparse {top_key} routing became dense")
            dense, depth = before.get("first_k_dense_replace"), before.get("num_hidden_layers")
            if isinstance(dense, int) and isinstance(depth, int) and dense < depth:
                if after.get("first_k_dense_replace", 0) >= after.get("num_hidden_layers", 0):
                    failures.append(f"{path}: no MoE layer remains")
        for key in ("layer_types", "mlp_layer_types"):
            kinds, new_kinds = before.get(key), after.get(key)
            if isinstance(kinds, list) and all(isinstance(kind, str) for kind in kinds):
                if not isinstance(new_kinds, list) or set(kinds) - set(new_kinds):
                    failures.append(f"{path}: lost {key} kinds")
    if failures:
        raise ValueError("Lossy fixture: " + "; ".join(failures))


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


def compact_saved_config(model, path):
    """Omit redundant derived fields and inherited nested dtype metadata.

    Keep explicit dimensions and layer patterns. A derived/default field is
    removed only when rebuilding the config reproduces the same effective value.
    """
    saved = json.loads(path.read_text())

    def compact(values, config, inherited_dtype=None):
        dtype = values.get("dtype", inherited_dtype)
        if inherited_dtype is not None and values.get("dtype") == inherited_dtype:
            values.pop("dtype", None)
        for key in ("head_dim", "qk_head_dim", "no_rope_layer_interval"):
            if key not in values:
                continue
            if key == "head_dim" and "qk_rope_head_dim" not in values:
                continue
            if key == "no_rope_layer_interval" and not isinstance(values.get("no_rope_layers"), list):
                continue
            candidate = {name: value for name, value in values.items() if name != key}
            restored = type(config).from_dict(candidate)
            if getattr(restored, key, None) == values[key]:
                values.pop(key)
        for key, value in values.items():
            subconfig = getattr(config, key, None)
            if isinstance(value, dict) and hasattr(subconfig, "to_dict"):
                compact(value, subconfig, dtype)

    compact(saved, model.config)
    path.write_text(json.dumps(saved, indent=2) + "\n")


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
