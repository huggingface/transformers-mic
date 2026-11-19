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


def test_model_tester_static_small_config(monkeypatch):
    """transformers#50462: literal `small_config_overrides` over the bases' `defaults` chain."""
    from transformers_mic import tester

    files = {
        "tests/model_tester_config.py": """
_TEXT = {"hidden_size": 32, "num_attention_heads": 2, "num_hidden_layers": 2, "seq_length": 7, "batch_size": 13}
class ModelTesterConfigMixin:
    defaults = _TEXT
    small_config_overrides = {}
""",
        "tests/multimodal_tester.py": """
class MultiModalModelTester(ModelTesterConfigMixin):
    defaults = {**ModelTesterConfigMixin.defaults, "batch_size": 3, "scope": None}
    def __init__(self, parent, **kwargs):
        for key, value in self.get_small_config_defaults(**kwargs).items():
            setattr(self, key, value)
""",
        "tests/vlm_tester.py": """
class VLMModelTester(MultiModalModelTester):
    defaults = {**MultiModalModelTester.defaults, "image_size": 8, "patch_size": 4, "num_channels": 3}
""",
    }
    monkeypatch.setattr(tester, "fetch", lambda url, timeout=20: files[url.split("/main/")[1]])
    code = """
class FooTester(VLMModelTester):
    small_config_overrides = {"image_size": 16, "kv_lora_rank": 16, "no_rope_layers": [1, 0, 1, 0], "hidden_size": 64}
"""
    found, own, static = tester.gather(code)
    assert static and own == {"image_size": 16, "kv_lora_rank": 16, "no_rope_layers": [1, 0, 1, 0], "hidden_size": 64}
    assert found["batch_size"] == 3 and found["num_channels"] == 3 and found["scope"] is None
    # Derived as get_small_config_defaults() derives them.
    assert found["num_image_tokens"] == 16 and found["seq_length"] == 23 and found["head_dim"] == 32
    # What __init__ still sets wins over the class overrides, as at runtime.
    legacy = code + """
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("kv_lora_rank", 8)
        super().__init__(parent, **kwargs)
"""
    assert tester.gather(legacy)[0]["kv_lora_rank"] == 8


def test_model_tester_own_defaults_derivations_and_classes(monkeypatch):
    """transformers#50916 shapes: a class's own `defaults`, simple derivations in its own
    `get_small_config_defaults()`, and per-class precedence in a file of several testers."""
    from transformers_mic import tester

    mixin = """
class ModelTesterConfigMixin:
    defaults = {"hidden_size": 32, "num_experts": 8}
    small_config_overrides = {}
"""
    monkeypatch.setattr(tester, "fetch", lambda url, timeout=20: mixin)
    code = """
class BertModelTester(ModelTesterConfigMixin):
    defaults = {}
    small_config_overrides = {"hidden_size": 48, "num_attention_heads": 4, "image_size": 16, "patch_size": 4}

    @classmethod
    def get_small_config_defaults(cls, **kwargs):
        values = super().get_small_config_defaults(**kwargs)
        values["out_hidden_size"] = values["hidden_size"]
        values.setdefault("head_dim", values["hidden_size"] // values["num_attention_heads"])
        if "grid" not in values:
            values["grid"] = values["image_size"] // values["patch_size"]
        values["layers"] = {i: i for i in range(2)}
        values["missing"] = values["nope"]  # skipped
        return values

class BertVisionTester:
    def __init__(self, parent, hidden_size=99, depth=3):
        self.parent = parent
"""
    found, own, static = tester.gather(code)
    assert static and "num_experts" not in found  # the class's own defaults replace the mixin's
    assert found["out_hidden_size"] == 48 and found["head_dim"] == 12 and found["grid"] == 4
    assert found["layers"] == {0: 0, 1: 1} and "missing" not in found
    # The first class wins across classes; the second still adds what the first lacks.
    assert found["hidden_size"] == 48 and found["depth"] == 3
    assert tester.gather(code, only="BertVisionTester")[0] == {"hidden_size": 99, "depth": 3}


def test_model_tester_override_forms_from_transformers_50944(monkeypatch):
    """Overrides that fill kwargs before super(), use locals, len(), displays and comprehensions."""
    from transformers_mic import tester

    files = {
        "tests/model_tester_config.py": """
class ModelTesterConfigMixin:
    defaults = {"hidden_size": 32, "num_attention_heads": 2, "seq_length": 7}
""",
        "tests/vlm_tester.py": """
class VLMModelTester(ModelTesterConfigMixin):
    defaults = {**ModelTesterConfigMixin.defaults, "image_size": 8, "patch_size": 4}
""",
    }
    monkeypatch.setattr(tester, "fetch", lambda url, timeout=20: files[url.split("/main/")[1]])
    code = """
class LlavaLikeTester(VLMModelTester):
    small_config_overrides = {"layer_types": ["full", "sliding", "full"], "num_hidden_layers": 2}

    @classmethod
    def get_small_config_defaults(cls, **kwargs):
        image_size = kwargs.get("image_size", cls.defaults["image_size"])
        height = width = image_size // 4
        kwargs.setdefault("num_image_tokens", height * (width + 1))
        values = super().get_small_config_defaults(**kwargs)
        values["num_hidden_layers"] = len(values["layer_types"])
        values["per_layer"] = {i: {"head_dim": 2 * values["head_dim"]} for i, t in enumerate(values["layer_types"]) if t == "full"}
        values["sub_config"] = {"hidden_size": values["hidden_size"] * 2, **{"depth": 1}}
        values["device"] = torch_device  # a runtime value: skipped
        __import__("os").system("false")  # never run
        if "seq_length" not in kwargs:
            values["seq_length"] = 7 + 4 * values["num_image_tokens"]
        return values
"""
    found, _, _ = tester.gather(code)
    # kwargs filled before super() win and feed the bases' derivations.
    assert found["num_image_tokens"] == 6 and found["seq_length"] == 31
    assert found["num_hidden_layers"] == 3 and found["head_dim"] == 16
    assert found["per_layer"] == {0: {"head_dim": 32}, 2: {"head_dim": 32}}
    assert found["sub_config"] == {"hidden_size": 64, "depth": 1}
    assert "device" not in found


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

