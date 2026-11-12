import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers_mic.fixture_quantization import quantize


def source(**options):
    return {
        "quant_method": "fp8", "activation_scheme": "dynamic",
        "weight_block_size": [128, 128], "modules_to_not_convert": ["lm_head"], **options,
    }


def test_quantized_weights_round_trip_and_exclusions():
    torch.manual_seed(17)
    weight = torch.randn(16, 24).bfloat16()
    weight[:8, :8] = 0
    tensors = {"layers.0.attention.wq.weight": weight, "output.weight": weight.clone()}
    config = source()
    result, quantization = quantize(tensors, config, {"layers.0.attention.wq.qscale_weight"})
    encoded = result["layers.0.attention.wq.weight"]
    scale = result["layers.0.attention.wq.qscale_weight"]
    assert encoded.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32 and scale.shape == (2, 3)
    assert torch.isfinite(scale).all() and (scale > 0).all()
    decoded = encoded.float() * scale.repeat_interleave(8, 0).repeat_interleave(8, 1)
    torch.testing.assert_close(decoded, weight.float(), rtol=0.07, atol=0.001)
    assert torch.count_nonzero(decoded[:8, :8]) == 0
    assert result["output.weight"] is tensors["output.weight"]
    assert tensors["layers.0.attention.wq.weight"].dtype == torch.bfloat16
    assert quantization["weight_block_size"] == [8, 8]
    assert quantization["modules_to_not_convert"] == ["lm_head"]
    assert config["weight_block_size"] == [128, 128]


def test_smaller_projections_get_divisible_blocks():
    result, config = quantize({"p.weight": torch.ones(4, 12)}, source(), {"p.qscale_weight"})
    assert config["weight_block_size"] == [4, 4]
    assert result["p.qscale_weight"].shape == (1, 3)


def test_missing_scale_names_fail_instead_of_saving_dense_weights():
    with pytest.raises(ValueError, match="no matching quantization scales"):
        quantize({"p.weight": torch.ones(8, 8)}, source(), set())


@pytest.mark.parametrize("config", [None, {"quant_method": "mxfp4"}])
def test_other_quantization_methods_are_unchanged(config):
    tensors = {"p.weight": torch.ones(8, 8)}
    result, quantization = quantize(tensors, config, {"p.qscale_weight"})
    assert result is tensors and quantization is None


@pytest.mark.parametrize("name,scale_name,shape", [
    ("attention.q_proj.weight", "attention.q_proj.weight_scale_inv", (16, 24)),
    ("experts.gate_up_proj", "experts.gate_up_proj_scale_inv", (4, 32, 16)),
])
def test_standard_and_fused_serialization(name, scale_name, shape):
    result, config = quantize({name: torch.ones(shape)}, source(), {scale_name})
    assert result[name].dtype == torch.float8_e4m3fn
    assert result[scale_name].shape == (*shape[:-2], shape[-2] // 8, shape[-1] // 8)
    assert set(result) == {name, scale_name}


def test_source_blocks_smaller_than_fixture_cap_are_preserved():
    _, config = quantize({"p.weight": torch.ones(16, 16)}, source(weight_block_size=[4, 2]), {"p.weight_scale_inv"})
    assert config["weight_block_size"] == [4, 2]


def test_ambiguous_scale_names_are_rejected():
    with pytest.raises(ValueError, match="Ambiguous"):
        quantize({"p.weight": torch.ones(8, 8)}, source(), {"p.weight_scale_inv", "p.qscale_weight"})


def test_static_activation_quantization_is_not_silently_changed():
    with pytest.raises(ValueError, match="dynamic activation"):
        quantize({"p.weight": torch.ones(8, 8)}, source(activation_scheme="static"), {"p.weight_scale_inv"})
