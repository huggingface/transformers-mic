import pytest

from transformers_mic.fixture_review import merge_profile
from transformers_mic.serving_profiles import PROFILES, overrides, profile, violations

VLLM = profile("vllm-cpu")
# Attention fields of published fixtures (the generic shrink rule: 4 heads of 16).
QWEN2 = {"hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2}
LLAMA = {**QWEN2, "head_dim": 16}
MISTRAL = {"hidden_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16}
GEMMA = {"hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 16}
VAULTGEMMA = {"hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 4, "head_dim": 16}


def fitted(config, fields):
    out = {**config, "text_config": dict(config.get("text_config", {}))} if "text_config" in config else dict(config)
    for path, value in fields.items():
        *parents, leaf = path.split(".")
        node = out
        for key in parents:
            node = node[key]
        node[leaf] = value
    return out


@pytest.mark.parametrize(
    "config, expected, keeps_kv",
    [
        (QWEN2, {"num_attention_heads": 2, "num_key_value_heads": 1}, True),
        (LLAMA, {"num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32}, True),
        (MISTRAL, {"num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32}, True),
        (VAULTGEMMA, {"num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 32}, True),
        # MQA keeps its single KV head, so the k/v projections grow.
        (GEMMA, {"num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32}, False),
    ],
)
def test_vllm_cpu_widens_heads_to_a_supported_width(config, expected, keeps_kv):
    assert violations(config, VLLM)
    fields, reason = overrides(config, VLLM)
    assert fields == expected
    assert ("k/v projections grow" not in reason) == keeps_kv
    assert "serving profile vllm-cpu v1" in reason
    # Applying them makes the config fit, so a rerun adds nothing.
    assert violations(fitted(config, fields), VLLM) == []
    assert overrides(fitted(config, fields), VLLM) is None


def test_a_fitting_config_needs_nothing():
    wide = {"hidden_size": 128, "num_attention_heads": 2, "num_key_value_heads": 1}
    assert violations(wide, VLLM) == [] and overrides(wide, VLLM) is None


def test_multimodal_configs_change_the_text_config():
    config = {"model_type": "x", "text_config": LLAMA, "vision_config": {"hidden_size": 64, "num_attention_heads": 4}}
    fields, _ = overrides(config, VLLM)
    assert fields == {
        "text_config.num_attention_heads": 2,
        "text_config.num_key_value_heads": 1,
        "text_config.head_dim": 32,
    }


def test_layers_without_attention_are_reported_not_widened():
    hybrid = {**LLAMA, "layer_types": ["linear_attention", "full_attention"]}
    assert any("linear_attention" in v for v in violations(hybrid, VLLM))
    with pytest.raises(ValueError, match="attention in every layer"):
        overrides(hybrid, VLLM)


@pytest.mark.parametrize("name", ["transformers", "transformers.js", "sentence_transformers"])
def test_integrations_that_run_the_fixture_as_built_have_no_constraints(name):
    assert violations(GEMMA, PROFILES[name]) == [] and overrides(GEMMA, PROFILES[name]) is None


def test_unknown_profiles_are_named():
    with pytest.raises(ValueError, match="known: "):
        profile("sglang")


def test_merge_profile_adds_reviewed_overrides_once():
    recipe = {"seed": 17}
    assert merge_profile(recipe, LLAMA, "vllm-cpu") == ["head_dim", "num_attention_heads", "num_key_value_heads"]
    assert recipe["config_overrides"]["fields"] == {"num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32}
    assert "vllm-cpu" in recipe["config_overrides"]["reason"]
    # The shrunk config the next build gives already fits.
    assert merge_profile(recipe, fitted(LLAMA, recipe["config_overrides"]["fields"]), "vllm-cpu") == []
    assert merge_profile({"seed": 17}, LLAMA, "transformers") == []
