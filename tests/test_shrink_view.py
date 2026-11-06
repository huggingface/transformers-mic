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


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("method", [None, "fp8", "mxfp4"])
def test_quantization_loss_is_a_finding(nested, method):
    original = {"quantization_config": {"quant_method": "fp8"}}
    shrunk = {"quantization_config": {"quant_method": method}} if method else {}
    if nested:
        original, shrunk = {"text_config": original}, {"text_config": shrunk}
    findings = shrink_view.verdict({"properties": shrink_view.properties(original, shrunk)})
    if method == "fp8":
        assert findings == []
    else:
        assert len(findings) == 1
        assert f"quantization method fp8 became {method or 'missing'}" in findings[0]


def test_the_page_is_titled_by_the_model_and_links_both_pins():
    original = gemma(["sliding_attention"] * 5 + ["full_attention"])
    result = report(original, gemma(["sliding_attention", "full_attention"]))
    result |= {"fixture": "huggingface-mic/demo-mic@" + "a" * 40, "source": "creator/Demo-7B@" + "b" * 40}
    page = shrink_view.render(result)
    assert "<h1>creator/Demo-7B</h1>" in page
    assert f'href="https://huggingface.co/huggingface-mic/demo-mic/tree/{"a" * 40}"' in page
    assert f'href="https://huggingface.co/creator/Demo-7B/tree/{"b" * 40}"' in page


def test_a_barely_shrunk_fixture_and_a_test_checkpoint_source_are_findings():
    original = gemma(["sliding_attention"] * 5 + ["full_attention"])
    result = report(original, original)
    assert any("keeps 100% of the original" in f for f in shrink_view.verdict(result))
    assert not any("keeps" in f for f in shrink_view.verdict(result | {"build_shrink": "none"}))
    result["source"] = "optimum-intel-internal-testing/tiny-random-demo@" + "c" * 40
    assert any("is a test checkpoint" in f for f in shrink_view.verdict(result))


def test_model_tester_values_and_advisory_findings():
    from transformers_mic import tester

    code = """
class FooModelTester(VLMModelTester):
    def __init__(self, parent, batch_size=13, hidden_size=32, n_routed_experts=8, **kwargs):
        kwargs.setdefault("index_topk", 4)
        kwargs.setdefault("no_rope_layers", [1, 0, 1, 0])
        kwargs.setdefault("num_hidden_layers", 4)
        kwargs.setdefault("depths", [1, 2])
        self.kv_lora_rank = 16
        self.parent = parent
"""
    found = tester.values(code)
    assert found["no_rope_layers"] == [1, 0, 1, 0] and found["kv_lora_rank"] == 16 and found["hidden_size"] == 32
    original = {"text_config": {"kv_lora_rank": 512, "index_topk": 2048, "no_rope_layers": [1, 1, 1, 0]}}
    shrunk = {"text_config": {"kv_lora_rank": 512, "index_topk": 2, "no_rope_layers": [1, 1], "hidden_size": 64, "depths": [1, 1]}}
    notes = tester.compare(found, original, shrunk)
    assert notes == [
        "kv_lora_rank kept at the checkpoint's 512; the model tester uses 16",
        "no_rope_layers: the model tester covers 0, the fixture has only 1",
    ]
    # A stage list is not a kind; a fixture kept whole has nothing to shrink.
    assert tester.compare(found, original, shrunk, kept_whole=True) == [
        "no_rope_layers: the model tester covers 0, the fixture has only 1"]



def test_static_small_config_defaults_are_extracted_without_importing():
    from transformers_mic import tester

    code = '''
raise AssertionError("Do not import this test module")
class FooModelTester(VLMModelTester):
    @staticmethod
    def get_small_config_defaults():
        """Defaults shared by the tests and fixture tools."""
        return {"qk_rope_head_dim": 32, "index_head_dim": 32, "no_rope_layers": [1, 0]}
'''
    assert tester.values(code) == {
        "qk_rope_head_dim": 32, "index_head_dim": 32, "no_rope_layers": [1, 0],
    }


def test_static_defaults_preserve_constructor_overrides():
    from transformers_mic import tester

    code = '''
class FooModelTester(VLMModelTester):
    @staticmethod
    def get_small_config_defaults():
        return {"hidden_size": 32, "index_head_dim": 32}
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("hidden_size", 64)
        for key, value in self.get_small_config_defaults().items():
            kwargs.setdefault(key, value)
'''
    assert tester.values(code) == {"hidden_size": 64, "index_head_dim": 32}


def test_computed_static_defaults_fall_back_to_legacy_literals():
    from transformers_mic import tester

    code = '''
class FooModelTester(VLMModelTester):
    @staticmethod
    def get_small_config_defaults():
        return expensive_or_unsafe_function()
    def __init__(self, parent, hidden_size=64):
        pass
'''
    assert tester.values(code) == {"hidden_size": 64}
