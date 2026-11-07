"""Quantize fixture weights according to the source config and serialized scales."""

import math
from types import SimpleNamespace


def require_supported_source(config, path="model"):
    """Refuse lossy dense fallback, including quantization hidden in sub-configs."""
    quantization = config.get("quantization_config")
    if quantization:
        method = quantization.get("quant_method", "unspecified")
        if method != "fp8" or path != "model":
            raise ValueError(f"{path}: fixture serialization does not support {method} quantization here; refusing a dense fixture")
    for key, value in config.items():
        if isinstance(value, dict) and (key.endswith("_config") or "model_type" in value):
            require_supported_source(value, f"{path}.{key}")


def scale_names(name):
    """Supported serialized inverse-scale names, including fused expert weights."""
    if name.endswith(".weight"):
        prefix = name.removesuffix(".weight")
        return [prefix + ".weight_scale_inv", prefix + ".qscale_weight"]
    return [name + "_scale_inv"]


def quantize(tensors, config, source_keys, block_size=None):
    """Encode dynamic FP8 projections selected by the pinned source's scales.

    Run after restoring source tensor names. Cap block dimensions at 8 to keep
    local scaling in small fixtures, then reduce them to divide every selected
    matrix. An explicit recipe block size is used unchanged and must divide every
    selected matrix; incompatible dimensions fail rather than relaxing it.
    Leading dimensions (e.g. fused experts) do not affect block sizes.
    Unsupported quantization methods fail instead of producing dense weights.
    """
    if not config:
        if block_size is not None:
            raise ValueError("An FP8 block override requires an FP8 source")
        return tensors, None
    require_supported_source({"quantization_config": config})
    if config.get("activation_scheme", "dynamic") != "dynamic":
        raise ValueError("Fixture FP8 serialization requires dynamic activation quantization")
    from transformers import FineGrainedFP8Config
    from transformers.integrations.finegrained_fp8 import Fp8Quantize

    selected = {}
    for name, tensor in tensors.items():
        matches = [key for key in scale_names(name) if key in source_keys]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous FP8 scale names for {name}: {matches}")
        if matches:
            if tensor.ndim < 2:
                raise ValueError(f"Expected a matrix or stacked matrices for FP8 projection {name}")
            selected[name] = matches[0]
    if not selected:
        raise ValueError("FP8 source has no matching quantization scales")
    config = dict(config)
    block = config.get("weight_block_size") if block_size is None else block_size
    if block is None:
        raise ValueError("Fixture FP8 serialization requires blockwise weight quantization")
    if not isinstance(block, (list, tuple)) or len(block) != 2 or any(type(n) is not int or n <= 0 for n in block):
        raise ValueError("FP8 weight_block_size must contain two positive integers")
    block = [min(n, 8) for n in block] if block_size is None else list(block)
    for name in selected:
        for axis in range(2):
            width = tensors[name].shape[-2 + axis]
            if block_size is not None:
                if width % block[axis]:
                    raise ValueError(f"Explicit FP8 block {block} does not divide {name} shape {tuple(tensors[name].shape)}")
            else:
                block[axis] = math.gcd(block[axis], width)
    config["weight_block_size"] = block
    quantizer = Fp8Quantize(SimpleNamespace(quantization_config=FineGrainedFP8Config(**config)))
    result = dict(tensors)
    for name, scale_name in selected.items():
        converted = quantizer.convert({name: tensors[name]})
        scale = converted.get(scale_names(name)[0])
        if scale is None:
            raise ValueError(f"FP8 quantization produced no scales for {name}")
        result[name] = converted[name]
        result[scale_name] = scale
    return result, config
