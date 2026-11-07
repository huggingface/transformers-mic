import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers_mic.fixture_validation import config_findings, validate_model  # noqa: E402
from transformers_mic import shrink_view  # noqa: E402


def test_inconsistent_attention_and_shared_cache_are_findings():
    config = {"text_config": {"qk_rope_head_dim": 16, "qk_nope_head_dim": 16,
              "qk_head_dim": 128, "no_rope_layers": [1, 0], "kv_source_layers": [0, 0]}}
    findings = config_findings(config)
    assert len(findings) == 2
    assert "rotary + non-rotary width 32" in findings[0]
    assert "layer 1 and its KV source 0 disagree on RoPE" in findings[1]
    assert shrink_view.verdict({"config_findings": findings}) == findings


def test_rotary_width_cannot_exceed_indexer_or_be_odd():
    findings = config_findings({"qk_rope_head_dim": 17, "qk_nope_head_dim": 16, "index_head_dim": 16})
    assert len(findings) == 2


def test_weight_shape_is_checked_against_a_fresh_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    validate_model(model)
    model.model.layers[0].self_attn.q_proj.weight = torch.nn.Parameter(torch.zeros(8, 16))
    with pytest.raises(ValueError, match="weight shape .* differs from config"):
        validate_model(model)


def test_saved_config_rejects_stale_derived_values(tmp_path):
    import json
    from transformers import MistralLarge4TextConfig, MistralLarge4TextModel
    from transformers_mic.fixture_validation import validate_saved_config

    config = MistralLarge4TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        qk_rope_head_dim=16, qk_nope_head_dim=16, kv_lora_rank=16,
        v_head_dim=16, index_head_dim=16, index_n_heads=2)
    with torch.device("meta"):
        model = MistralLarge4TextModel(config)
    saved = config.to_dict()
    saved["head_dim"] = 8
    path = tmp_path / "config.json"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Stale fixture config: head_dim"):
        validate_saved_config(model, path)


def test_compact_saved_config_preserves_effective_dimensions_and_patterns(tmp_path):
    import json
    from types import SimpleNamespace
    from transformers import MistralLarge4Config
    from transformers_mic.fixture_validation import compact_saved_config

    config = MistralLarge4Config(text_config={"num_hidden_layers": 2,
        "qk_rope_head_dim": 16, "qk_nope_head_dim": 16,
        "no_rope_layers": [1, 0], "kv_source_layers": [0, 1]})
    config.dtype = config.text_config.dtype = config.vision_config.dtype = torch.bfloat16
    path = tmp_path / "config.json"
    path.write_text(config.to_json_string())
    compact_saved_config(SimpleNamespace(config=config), path)
    saved = json.loads(path.read_text())
    text = saved["text_config"]
    assert "qk_head_dim" not in text
    assert "no_rope_layer_interval" not in text
    assert "dtype" not in text and "dtype" not in saved["vision_config"]
    assert saved["dtype"] == "bfloat16"
    assert "head_dim" not in text and text["no_rope_layers"] == [1, 0]
    restored = MistralLarge4Config.from_dict(saved)
    assert restored.text_config.qk_head_dim == 32
    assert restored.text_config.head_dim == 16
    assert restored.text_config.no_rope_layers == [1, 0]


@pytest.mark.parametrize("fixture", [
    {"num_experts": 1, "num_experts_per_tok": 1},
    {"num_experts": 4, "num_experts_per_tok": 4},
    {"num_experts": 4, "num_experts_per_tok": 2, "first_k_dense_replace": 2, "num_hidden_layers": 2},
])
def test_sparse_source_cannot_become_dense(fixture):
    from transformers_mic.fixture_validation import validate_preservation

    original = {"num_experts": 64, "num_experts_per_tok": 8, "first_k_dense_replace": 1, "num_hidden_layers": 48}
    with pytest.raises(ValueError, match="Lossy fixture"):
        validate_preservation(original, fixture)


def test_quantization_and_layer_kinds_cannot_be_lost():
    from transformers_mic.fixture_validation import validate_preservation

    original = {"text_config": {"quantization_config": {"quant_method": "fp8"},
        "mlp_layer_types": ["dense", "sparse"]}}
    with pytest.raises(ValueError, match="lost fp8 quantization"):
        validate_preservation(original, {"text_config": {"mlp_layer_types": ["dense"]}}, quantization=True)
    validate_preservation({"num_experts": 64, "num_experts_per_tok": 8},
                          {"num_experts": 4, "num_experts_per_tok": 2})


@pytest.mark.parametrize("top_key", ["num_experts_per_tok", "top_k_experts", "moe_k"])
def test_sparse_routing_aliases_cannot_select_all_experts(top_key):
    from transformers_mic.fixture_validation import validate_preservation

    with pytest.raises(ValueError, match="routing became dense"):
        validate_preservation({"moe_num_experts": 64, top_key: 8}, {"moe_num_experts": 4, top_key: 4})
