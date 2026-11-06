import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers_mic.fixture_validation import config_findings, validate_model
from transformers_mic import shrink_view


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
