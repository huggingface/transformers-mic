"""Quantize fixture weights according to the source config and serialized scales."""

import math
from types import SimpleNamespace


def scale_names(name):
    """Supported serialized inverse-scale names, including fused expert weights."""
    if name.endswith(".weight"):
        prefix = name.removesuffix(".weight")
        return [prefix + ".weight_scale_inv", prefix + ".qscale_weight"]
    return [name + "_scale_inv"]


def quantize(tensors, config, source_keys):
    """Encode dynamic FP8 projections selected by the pinned source's scales.

    Run after restoring source tensor names. Cap block dimensions at 8 to keep
    local scaling in small fixtures, then reduce them to divide every selected
    matrix. Leading dimensions (e.g. fused experts) do not affect block sizes.
    Unsupported quantization methods retain the existing dense-fixture behavior.
    """
    if not config or config.get("quant_method") != "fp8":
        return tensors, None
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
    block = config.get("weight_block_size")
    if block is None:
        raise ValueError("Fixture FP8 serialization requires blockwise weight quantization")
    if len(block) != 2 or any(not isinstance(n, int) or n <= 0 for n in block):
        raise ValueError("FP8 weight_block_size must contain two positive integers")
    block = [min(n, 8) for n in block]
    for name in selected:
        for axis in range(2):
            block[axis] = math.gcd(block[axis], tensors[name].shape[-2 + axis])
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
