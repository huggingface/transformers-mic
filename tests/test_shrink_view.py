import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers_mic import shrink_view  # noqa: E402


def gemma(layer_types, heads=4, kv=1):
    from transformers import Gemma3TextConfig

    return Gemma3TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        head_dim=8,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        num_hidden_layers=len(layer_types),
        layer_types=layer_types,
        sliding_window=4,
    ).to_dict()


def report(original, shrunk):
    data = {
        "fixture": "f",
        "source": "s",
        "build": {"model_class": "Gemma3ForCausalLM"},
        "original": original,
        "shrunk": shrunk,
    }
    return shrink_view.analyze(data)


def test_a_layer_kind_the_fixture_lost_is_a_finding():
    original = gemma(["sliding_attention"] * 5 + ["full_attention"])
    result = report(original, gemma(["sliding_attention"] * 2, heads=4, kv=2))

    [stack] = result["stacks"]
    assert [(k["label"], k["original"], k["fixture"]) for k in stack["kinds"]] == [
        ("sliding_attention", 5, 2),
        ("full_attention", 1, 0),
    ]
    findings = shrink_view.verdict(result)
    assert any("full_attention' is not in the fixture" in f for f in findings)
    assert any("MQA (4 heads, 1 KV) became GQA (4 heads, 2 KV)" in f for f in findings)
    assert "⚠ missing" in shrink_view.render(result)


def test_a_faithful_shrink_has_no_findings():
    original = gemma(["sliding_attention"] * 5 + ["full_attention"])
    result = report(original, gemma(["sliding_attention", "full_attention"]))

    assert shrink_view.verdict(result) == []
    assert {"field": "num_hidden_layers", "original": "6", "fixture": "2"} in result[
        "config_changes"
    ]
    assert result["parameters"]["original"] > result["parameters"]["fixture"]
    assert "Same architecture, smaller" in shrink_view.render(result)
